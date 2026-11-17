"""The seeded first step.

Before the first model call the engine runs one search on the words of the
question and one lookup of the names in it. The results go into the prompt,
so a simple recall question ends in one model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .directory import Directory, Entry
from .labeler import handle_spans
from .search import SearchRequest
from .textnorm import fold, surface_words
from .tool_context import Prepared, ToolContext
from .tool_figures import list_entries
from .tool_search import run_search_tool
from .tools import ToolBox

SEED_HITS = 8
SEED_RADIUS = 3
SEED_NAMES = 6
_MAX_UNITS = 12
_MAX_WORD = 40
_BOOSTED = 8
_LIST_TOKENS = 700


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


def plan_seed(box: ToolBox, question: str) -> list[SeedCall]:
    ctx = box.context
    directory: Directory = box.guard(ctx.directory)
    names = directory.named_in(question)
    entries = named_entries(directory, names)
    calls: list[SeedCall] = []
    units = question_units(question, ctx.labeler.redact)
    if units:
        calls.append(_search_call(box, ctx, units, entries))
    if entries:
        calls.append(_names_call(box, ctx, names, entries))
    elif ctx.scope.conversations:
        calls.append(_scope_call(box, ctx, directory))
    return calls


def _search_call(box: ToolBox, ctx: ToolContext, units: list[str], entries: Sequence[Entry]) -> SeedCall:
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
    prepared = Prepared(
        "Searching your messages",
        lambda budget: run_search_tool(ctx, request, budget, radius=SEED_RADIUS),
    )
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


def _scope_call(box: ToolBox, ctx: ToolContext, directory: Directory) -> SeedCall:
    prepared = Prepared(
        "Reading the chosen conversation",
        lambda budget: list_entries(ctx, directory.top(SEED_NAMES), SEED_NAMES, min(budget, _LIST_TOKENS)),
    )
    return SeedCall("list_conversations", "list_conversations()", box.wrap("list_conversations", prepared))
