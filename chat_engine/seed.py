"""The seeded first step.

Before the first model call the engine runs one search on the words of the
question and one lookup of the names in it. The results go into the prompt,
so a simple recall question ends in one model call. With a search by meaning,
that search joins the exchanges found by meaning and by words. With a
reranker, its hits are ranked against the whole question (`rerank.py`).

A question about a time ("in 2021", "last summer") gets its word search
narrowed to that time, and the figures of that time: who the user wrote with,
and how much. A question about a period is seldom answered by a message that
holds its words, so the model starts from the pattern. A time worked out from
today can be a wrong guess, so one more search, with half the hits, looks at
every day of the scope.

A question that names one person also gets the first messages with that
person. How two people met, and how it began, is seldom in a message that
holds the words of the question.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Collection, Optional, Sequence

from .directory import Directory, Entry
from .labeler import handle_spans
from .ports import ArchiveSummary
from .question_time import Period, archive_years, question_time
from .question_names import names_in_question
from .search import SearchRequest
from .textnorm import fold, surface_words
from .tool_args import ToolInputError
from .tool_browse import BrowseRequest, run_browse
from .tool_context import Prepared, ToolContext
from .tool_figures import list_entries, run_count
from .tool_meaning import MeaningRequest, meaning_prepared
from .tool_people import SEED_PEOPLE, seeded_people
from .tool_search import search_lookup
from .tools import ToolBox

SEED_HITS = 8
SEED_RADIUS = 3
SEED_NAMES = 6
_MAX_UNITS = 12
_MAX_WORD = 40
_BOOSTED = 8
_LIST_TOKENS = 700
_FIGURES_TOKENS = 1500  # room for every row a count by conversation shows
SEED_OPENING = 10  # the first messages with the one person a question names
_OPENING_TOKENS = 1200
_SEARCHING = "Searching your messages"
_BACKUP_SHARE = 2  # the search over every day gets this part of the hits and of the room


@dataclass(frozen=True)
class SeedCall:
    tool: str
    label: str  # how the lookup reads in the prompt
    prepared: Prepared


@dataclass(frozen=True)
class SeedPlan:
    """The lookups to run, and the days the question is about, inside the scope. None: it names no time."""

    calls: tuple[SeedCall, ...]
    period: Optional[Period]


def question_units(question: str, redact: Callable[[str], str], drop: Collection[str] = ()) -> list[str]:
    """Words of the question worth searching for, as the user wrote them.

    A handle the labeler would mask is left out, so its digits never become
    search words that the model reads. `drop` holds folded words to leave
    out: the words that name the time of the question.
    """
    kept = question
    for start, end in reversed(handle_spans(question)):
        span = question[start:end]
        if redact(span) != span:
            kept = kept[:start] + " " + kept[end:]
    first: dict[str, str] = {}
    for token in surface_words(kept):
        first.setdefault(fold(token), token)
    tokens = [token for key, token in first.items() if _worth_a_search(key) and key not in drop]
    if len(tokens) > _MAX_UNITS:
        longest = set(sorted(tokens, key=len, reverse=True)[:_MAX_UNITS])
        tokens = [token for token in tokens if token in longest]
    return tokens


def _is_year(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and 1990 <= int(token) <= 2099


def _worth_a_search(token: str) -> bool:
    """Drop years, very short words and runs too long to be a word.

    A short word with a digit or in another script stays.
    """
    if _is_year(token) or len(token) > _MAX_WORD:
        return False
    if len(token) >= 3:
        return True
    return len(token) == 2 and (not token.isascii() or any(ch.isdigit() for ch in token))


def named_entries(directory: Directory, names: Sequence[str]) -> list[Entry]:
    """Entries any name matches; those that match more names come first."""
    scored: dict[str, tuple[int, Entry]] = {}
    for name in names:
        for entry in directory.find(name, loose_digits=True):
            key = str(entry.conversation["id"])
            count, _ = scored.get(key, (0, entry))
            scored[key] = (count + 1, entry)
    position = {str(entry.conversation["id"]): index for index, entry in enumerate(directory.entries)}
    ranked = sorted(
        scored.items(),
        key=lambda item: (-item[1][0], bool(item[1][1].conversation.get("is_group")), position[item[0]]),
    )
    return [entry for _, (_, entry) in ranked]


def plan_seed(box: ToolBox, question: str, summary: ArchiveSummary, now: datetime) -> SeedPlan:
    """The lookups to run before the first model call. `summary` is the summary of the archive."""
    ctx = box.context
    period = _inside_scope(ctx, question_time(question, now, archive_years(summary), ctx.labeler.redact))
    directory: Directory = box.guard(ctx.directory)
    total = int(summary.get("messages") or 0)
    names: list[str] = box.guard(lambda: names_in_question(directory, ctx.archive, question, total))
    entries = named_entries(directory, names)
    calls: list[SeedCall] = []
    units = question_units(question, ctx.labeler.redact, period.words if period else ())
    if units:
        calls.extend(_search_calls(box, ctx, units, entries, question, period))
    if entries:
        calls.append(_names_call(box, ctx, names, entries))
        opening = _opening_call(box, ctx, entries[0])
        if opening is not None:
            calls.append(opening)
    elif ctx.scope.conversations:
        calls.append(_scope_call(box, ctx, directory))
    if ctx.memory.people is not None:
        label = f"find_people(description=the question, limit={SEED_PEOPLE})"
        calls.append(SeedCall("find_people", label, box.wrap("find_people", seeded_people(ctx, question))))
    if period is not None:
        calls.append(_figures_call(box, ctx, period))
    return SeedPlan(tuple(calls), period)


def _inside_scope(ctx: ToolContext, period: Optional[Period]) -> Optional[Period]:
    """The period cut to the days the user chose. None when it lies outside them."""
    if period is None:
        return None
    try:
        low, high = ctx.days(period.since, period.until)
    except ToolInputError:
        return None
    return replace(period, since=low or period.since, until=high or period.until)


def _search_calls(
    box: ToolBox,
    ctx: ToolContext,
    units: list[str],
    entries: Sequence[Entry],
    question: str,
    period: Optional[Period],
) -> list[SeedCall]:
    """The search on the words of the question, narrowed to its period.

    A period worked out from today can be a wrong guess. So a second search
    looks at every day of the scope, with part of the hits and of the room.
    """
    if period is None:
        return [_search_call(box, ctx, units, entries, question)]
    narrowed = _search_call(box, ctx, units, entries, question, (period.since, period.until))
    if not period.relative:
        return [narrowed]
    return [narrowed, _search_call(box, ctx, units, entries, question, share=_BACKUP_SHARE)]


def _search_call(
    box: ToolBox,
    ctx: ToolContext,
    units: list[str],
    entries: Sequence[Entry],
    question: str,
    days: Optional[tuple[str, str]] = None,
    share: int = 1,
) -> SeedCall:
    """One search on the words of the question. With a search by meaning: the joined ranking for the question.

    days: the period it is narrowed to; None for every day of the scope.
    share: it gets 1/share of the hits and of the room of one lookup.
    """
    since, until = days or ctx.days(None, None)
    request = SearchRequest(
        query=" ".join(units),
        match="any",
        conversations=ctx.conversations(None),
        since=since,
        until=until,
        limit=SEED_HITS // share,
        boost=tuple(str(entry.conversation["id"]) for entry in entries[:_BOOSTED]),
    )
    shown = "" if days is None else f', since="{since}", until="{until}"'
    if ctx.memory.meaning is not None:
        joined = MeaningRequest(question, request, "the question")
        prepared = _shared(meaning_prepared(ctx, joined, _SEARCHING, radius=SEED_RADIUS, rank_by=question), share)
        label = f'search_messages(match="meaning", query=the question{shown})'
        return SeedCall("search_messages", label, box.wrap("search_messages", prepared))
    prepared = _shared(search_lookup(ctx, request, _SEARCHING, radius=SEED_RADIUS, rank_by=question), share)
    label = f'search_messages(query="{request.query}", match="any"{shown})'
    return SeedCall("search_messages", label, box.wrap("search_messages", prepared))


def _shared(prepared: Prepared, share: int) -> Prepared:
    """The lookup with 1/share of the room of one lookup."""
    if share <= 1:
        return prepared
    return replace(prepared, run=lambda budget: prepared.run(budget // share))


def _names_call(box: ToolBox, ctx: ToolContext, names: Sequence[str], entries: Sequence[Entry]) -> SeedCall:
    spoken = [name for name in names if not name.isdigit()]
    shown = ", ".join(spoken) if spoken else "a number in the question"
    status = f"Looking up “{', '.join(spoken)}”" if spoken else "Looking up a number from your question"
    prepared = Prepared(
        status,
        lambda budget: list_entries(ctx, entries, SEED_NAMES, min(budget, _LIST_TOKENS), name=shown),
    )
    label = f'list_conversations(name="{shown}")'
    return SeedCall("list_conversations", label, box.wrap("list_conversations", prepared))


def _opening_call(box: ToolBox, ctx: ToolContext, entry: Entry) -> Optional[SeedCall]:
    """The first messages with the person the question names first. None for a group or a conversation out of scope."""
    if entry.conversation.get("is_group"):
        return None
    conversation = str(entry.conversation["id"])
    allowed = ctx.conversations(None)
    if allowed is not None and conversation not in allowed:
        return None
    try:
        since, until = ctx.days(None, None)
    except ToolInputError:
        return None
    request = BrowseRequest(conversation, "oldest", SEED_OPENING, since, until)
    label = ctx.book.label(conversation)
    prepared = Prepared(
        f"Reading how it began with {label}",
        lambda budget: run_browse(ctx, request, min(budget, _OPENING_TOKENS)),
    )
    return SeedCall("search_messages", f'search_messages(order="oldest", one conversation: {label})', box.wrap("search_messages", prepared))


def _scope_call(box: ToolBox, ctx: ToolContext, directory: Directory) -> SeedCall:
    prepared = Prepared(
        "Reading the chosen conversation",
        lambda budget: list_entries(ctx, directory.top(SEED_NAMES), SEED_NAMES, min(budget, _LIST_TOKENS)),
    )
    return SeedCall("list_conversations", "list_conversations()", box.wrap("list_conversations", prepared))


def _figures_call(box: ToolBox, ctx: ToolContext, period: Period) -> SeedCall:
    """Who the user wrote with in the period of the question. The period is already inside the scope."""
    low, high = period.since, period.until
    filters: dict[str, Any] = {
        "conversations": ctx.conversations(None),
        "since": low,
        "until": high,
        "from_me": None,
        "text": None,
        "group_by": "conversation",
    }
    prepared = Prepared(
        f"Counting who you wrote with {_when(low, high)}",
        lambda budget: run_count(ctx, filters, min(budget, _FIGURES_TOKENS)),
    )
    label = f'count_messages(group_by="conversation", since="{low}", until="{high}")'
    return SeedCall("count_messages", label, box.wrap("count_messages", prepared))


def _when(low: str, high: str) -> str:
    """The period for the status line: whole years by their numbers, other days by their dates."""
    if low[5:] != "01-01" or high[5:] != "12-31":
        return f"from {low} to {high}"
    return f"in {low[:4]}" if low[:4] == high[:4] else f"from {low[:4]} to {high[:4]}"
