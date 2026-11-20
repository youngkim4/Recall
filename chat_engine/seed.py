"""The seeded first step.

Before the first model call the engine runs one search on the words of the
question and one lookup of the names in it. The results go into the prompt,
so a simple recall question ends in one model call. With a search by meaning,
that search joins the exchanges found by meaning and by words. With a
reranker, its hits are ranked against the whole question (`rerank.py`).

A question that names years also gets the figures of that period: who the
user wrote with, and how much. A question about a period is seldom answered
by a message that holds its words, so the model starts from the pattern.

A question that names one person also gets the first messages with that
person. How two people met, and how it began, is seldom in a message that
holds the words of the question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, cast

from .directory import Directory, Entry
from .handles import DIGITS_RE, MIN_PHONE_DIGITS, YEAR_RE, is_dates
from .labeler import handle_spans
from .ports import ArchiveSummary
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
ALL_YEARS = (1990, 2099)
_SEARCHING = "Searching your messages"


@dataclass(frozen=True)
class SeedCall:
    tool: str
    label: str  # how the lookup reads in the prompt
    prepared: Prepared


def question_units(question: str, redact: Callable[[str], str]) -> list[str]:
    """Words of the question worth searching for, as the user wrote them.

    A handle the labeler would mask is left out, so its digits never become
    search words that the model reads.
    """
    kept = question
    for start, end in reversed(handle_spans(question)):
        span = question[start:end]
        if redact(span) != span:
            kept = kept[:start] + " " + kept[end:]
    first: dict[str, str] = {}
    for token in surface_words(kept):
        first.setdefault(fold(token), token)
    tokens = [token for key, token in first.items() if _worth_a_search(key)]
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


def archive_years(summary: ArchiveSummary) -> tuple[int, int]:
    """The first and the last year the archive holds. With no dated message: every year."""
    first, last = (str(summary.get(key) or "")[:4] for key in ("first_timestamp", "last_timestamp"))
    if YEAR_RE.fullmatch(first) and YEAR_RE.fullmatch(last):
        return int(first), int(last)
    return ALL_YEARS


def question_period(
    question: str, held: tuple[int, int] = ALL_YEARS, redact: Optional[Callable[[str], str]] = None
) -> tuple[Optional[str], Optional[str]]:
    """The first and the last day of the years the question names. No year: (None, None).

    Only a year the archive holds counts, so an amount such as 2000 is seldom
    taken for a year. Digits of a phone number or an email address never are:
    a handle that `redact` masks gives no year, whatever its shape.
    """
    kept = question
    for start, end in reversed(handle_spans(question)):
        span = question[start:end]
        masked = redact is not None and redact(span) != span
        kept = f"{kept[:start]} {'' if masked else _years_beside_a_handle(span)} {kept[end:]}"
    years = sorted(year for year in map(int, YEAR_RE.findall(kept)) if held[0] <= year <= held[1])
    if not years:
        return None, None
    return f"{years[0]}-01-01", f"{years[-1]}-12-31"


def _years_beside_a_handle(span: str) -> str:
    """What a span that looks like a phone number or an email address keeps for the search of years.

    A range of years or a date looks like a phone number: it is kept whole. A
    year written beside a phone number is kept; the number is not.
    """
    if "@" in span or "+" in span:
        return ""
    if is_dates(span):
        return span
    parts = span.split()
    years: list[str] = []
    for end in (-1, 0):
        while len(parts) > 1 and YEAR_RE.fullmatch(parts[end]) and _digit_count(parts) - 4 >= MIN_PHONE_DIGITS:
            years.append(parts.pop(end))
    return " ".join(years)


def _digit_count(parts: Sequence[str]) -> int:
    return sum(len(run) for part in parts for run in DIGITS_RE.findall(part))


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


def plan_seed(box: ToolBox, question: str, summary: ArchiveSummary) -> list[SeedCall]:
    """The lookups to run before the first model call. `summary` is the summary of the archive."""
    ctx = box.context
    held = archive_years(summary)
    directory: Directory = box.guard(ctx.directory)
    total = int(summary.get("messages") or 0)
    names: list[str] = box.guard(lambda: names_in_question(directory, ctx.archive, question, total))
    entries = named_entries(directory, names)
    calls: list[SeedCall] = []
    units = question_units(question, ctx.labeler.redact)
    if units:
        calls.append(_search_call(box, ctx, units, entries, question))
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
    figures = _figures_call(box, ctx, *question_period(question, held, ctx.labeler.redact))
    if figures is not None:
        calls.append(figures)
    return calls


def _search_call(
    box: ToolBox, ctx: ToolContext, units: list[str], entries: Sequence[Entry], question: str
) -> SeedCall:
    """The search on the words of the question. With a search by meaning: the joined ranking for the question."""
    since, until = ctx.days(None, None)
    request = SearchRequest(
        query=" ".join(units),
        match="any",
        conversations=ctx.conversations(None),
        since=since,
        until=until,
        limit=SEED_HITS,
        boost=tuple(str(entry.conversation["id"]) for entry in entries[:_BOOSTED]),
    )
    if ctx.memory.meaning is not None:
        joined = MeaningRequest(question, request, "the question")
        prepared = meaning_prepared(ctx, joined, _SEARCHING, radius=SEED_RADIUS, rank_by=question)
        label = 'search_messages(match="meaning", query=the question)'
        return SeedCall("search_messages", label, box.wrap("search_messages", prepared))
    prepared = search_lookup(ctx, request, _SEARCHING, radius=SEED_RADIUS, rank_by=question)
    label = f'search_messages(query="{request.query}", match="any")'
    return SeedCall("search_messages", label, box.wrap("search_messages", prepared))


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


def _figures_call(
    box: ToolBox, ctx: ToolContext, since: Optional[str], until: Optional[str]
) -> Optional[SeedCall]:
    """Who the user wrote with in the period of the question. None when it names no period."""
    if since is None or until is None:
        return None
    try:
        low, high = cast(tuple[str, str], ctx.days(since, until))
    except ToolInputError:
        return None  # the period is outside the days the user chose
    filters: dict[str, Any] = {
        "conversations": ctx.conversations(None),
        "since": low,
        "until": high,
        "from_me": None,
        "text": None,
        "group_by": "conversation",
    }
    years = f"in {low[:4]}" if low[:4] == high[:4] else f"from {low[:4]} to {high[:4]}"
    prepared = Prepared(
        f"Counting who you wrote with {years}",
        lambda budget: run_count(ctx, filters, min(budget, _FIGURES_TOKENS)),
    )
    label = f'count_messages(group_by="conversation", since="{low}", until="{high}")'
    return SeedCall("count_messages", label, box.wrap("count_messages", prepared))
