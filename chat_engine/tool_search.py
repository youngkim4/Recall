"""search_messages and read_around: find messages, and read the exchange around one."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from .ports import Message
from .registry import normalize_ref
from .render import ResultWriter, add_block, fit_messages, time_key
from .rerank import RERANK_RADIUS, candidate_text, pool_size, ranked_lookup, reranks
from .search import Found, SearchOutcome, SearchRequest, run_search
from .tool_args import ToolInputError, bool_arg, choice_arg, day_arg, int_arg, list_arg, text_arg
from .tool_browse import prepare_browse
from .tool_context import OUTSIDE_SCOPE, Prepared, ToolContext, ToolOutcome, month_year, plural
from .tool_limits import ORDERS, SEARCH_LIMIT_DEFAULT, SEARCH_LIMIT_MAX

SEARCH_RADIUS = 2
READ_RADIUS_DEFAULT = 6
READ_RADIUS_MAX = 15
_ORDER_WORDS = {"newest": "newest first", "oldest": "oldest first", "relevance": "best match first"}
_EXCHANGE = timedelta(hours=36)
_ECHO_CHARS = 40
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class Window:
    """Messages of one exchange in time order, with the hits among them."""

    messages: tuple[Message, ...]
    hits: frozenset[str]
    focus: Mapping[str, int]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(str(m["id"]) for m in self.messages)

    @property
    def conversation(self) -> str:
        return str(self.messages[0]["conversation"])


def prepare_search(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    query = text_arg(arguments, "query")
    if not query:
        return prepare_browse(ctx, arguments)
    match = choice_arg(arguments, "match", ("all", "any"), "all") or "all"
    order = choice_arg(arguments, "order", ORDERS, "newest") or "newest"
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    from_me = bool_arg(arguments, "from_me")
    limit = int_arg(arguments, "limit", default=SEARCH_LIMIT_DEFAULT, low=1, high=SEARCH_LIMIT_MAX)
    conversations = ctx.conversations(list_arg(arguments, "conversations"))
    since, until = ctx.days(since, until)
    request = SearchRequest(
        query=query, match=match, conversations=conversations, since=since, until=until,
        from_me=from_me, order=order, limit=limit,
    )
    rank_by = query if reranks(ctx, arguments.get("order")) else None
    return search_lookup(ctx, request, search_status(ctx, request), rank_by=rank_by)


def search_status(ctx: ToolContext, request: SearchRequest) -> str:
    names = ctx.names(request.conversations)
    return f"Searching for “{request.query}”" + (f" in {names}" if names else "")


def search_lookup(
    ctx: ToolContext, request: SearchRequest, status: str, *, radius: int = SEARCH_RADIUS,
    rank_by: Optional[str] = None,
) -> Prepared:
    """A search with words. With `rank_by` and a reranker, its hits are ranked against `rank_by` (`rerank.py`)."""
    if rank_by is None or ctx.memory.reranker is None:
        return Prepared(status, lambda budget: run_search_tool(ctx, request, budget, radius))

    def gather() -> tuple[SearchOutcome, list[str]]:
        outcome = run_search(ctx.archive, replace(request, limit=pool_size(request.limit)))
        return outcome, [_candidate(ctx, found) for found in outcome.hits]

    def write(outcome: SearchOutcome, order: tuple[int, ...], budget: int) -> ToolOutcome:
        ranked = replace(outcome, hits=tuple(outcome.hits[place] for place in order))
        return write_search(ctx, replace(request, order="relevance"), ranked, budget, radius)

    run, fetch = ranked_lookup(ctx, rank_by, request.limit, gather, write)
    return Prepared(status, run, fetch=fetch)


def _candidate(ctx: ToolContext, found: Found) -> str:
    message_id = str(found.message["id"])
    near = _around(ctx, found.message, RERANK_RADIUS, within=_EXCHANGE)
    return candidate_text(ctx, near.messages, {message_id}, {} if found.focus is None else {message_id: found.focus})


def run_search_tool(
    ctx: ToolContext, request: SearchRequest, budget: int, radius: int = SEARCH_RADIUS
) -> ToolOutcome:
    return write_search(ctx, request, run_search(ctx.archive, request), budget, radius)


def write_search(
    ctx: ToolContext, request: SearchRequest, outcome: SearchOutcome, budget: int, radius: int = SEARCH_RADIUS
) -> ToolOutcome:
    """The result of a search whose hits are found. The order of `request` is the one the header names."""
    windows = build_windows(ctx, outcome.hits, radius)
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(_header(ctx, request, outcome), force=True)
    visible: set[str] = set()
    fitted = True
    for window in windows:
        shown = _add_window(ctx, writer, window, outcome.matched)
        if not shown:
            fitted = False
            break
        visible |= shown
    writer.add(_footer(request, outcome, len(visible), fitted), force=True)
    return ToolOutcome(writer.text().rstrip(), _summary(request, outcome, len(visible)))


def _add_window(ctx: ToolContext, writer: ResultWriter, window: Window, matched: frozenset[str]) -> set[str]:
    """Add one exchange, or only its hits when the exchange does not fit. The ids of the matches shown."""
    for messages in (window.messages, tuple(m for m in window.messages if str(m["id"]) in window.hits)):
        marks = {str(m["id"]): "match" for m in messages if str(m["id"]) in matched | window.hits}
        fits = add_block(
            writer, messages, book=ctx.book, registry=ctx.registry, limits=ctx.limits,
            marks=marks, focus=window.focus,
        )
        if fits:
            return set(marks)
    return set()


def build_windows(ctx: ToolContext, hits: Sequence[Found], radius: int) -> list[Window]:
    """The exchange around each hit. Windows that share a message become one."""
    windows: list[Window] = []
    for found in hits:
        message_id = str(found.message["id"])
        focus = {} if found.focus is None else {message_id: found.focus}
        near = _around(ctx, found.message, radius, within=_EXCHANGE)
        merged = Window(near.messages, frozenset({message_id}), focus)
        touching = [w for w in windows if w.conversation == merged.conversation and w.ids & merged.ids]
        for window in touching:
            merged = Window(
                _merged(window.messages, merged.messages),
                window.hits | merged.hits,
                {**window.focus, **merged.focus},
            )
        if not touching:
            windows = [*windows, merged]
            continue
        rest = [w for w in windows if all(w is not t for t in touching[1:])]
        windows = [merged if w is touching[0] else w for w in rest]
    return windows


def _around(
    ctx: ToolContext, anchor: Message, radius: int, within: Optional[timedelta] = None
) -> Window:
    """Up to `radius` messages on each side of the anchor, inside the scope.

    `within` drops neighbours that are far away in time: they are another exchange.
    """
    near = ctx.archive.context(str(anchor["id"]), radius=radius) if radius > 0 else []
    kept = [m for m in near if m["conversation"] == anchor["conversation"] and ctx.allows(m)]
    if within is not None:
        kept = [m for m in kept if _close(m, anchor, within)]
    if str(anchor["id"]) not in {str(m["id"]) for m in kept}:
        kept = sorted([*kept, anchor], key=time_key)  # the archive gave the neighbours without the message
    return Window(tuple(kept), frozenset(), {})


def _close(message: Message, anchor: Message, within: timedelta) -> bool:
    try:
        one = datetime.strptime(str(message.get("timestamp"))[:19], _TIME_FORMAT)
        other = datetime.strptime(str(anchor.get("timestamp"))[:19], _TIME_FORMAT)
    except ValueError:
        return True
    return abs(one - other) <= within


def _merged(first: Sequence[Message], second: Sequence[Message]) -> tuple[Message, ...]:
    by_id = {str(m["id"]): m for m in [*first, *second]}
    return tuple(sorted(by_id.values(), key=time_key))


def _header(ctx: ToolContext, request: SearchRequest, outcome: SearchOutcome) -> str:
    place = f"{ctx.where(request.conversations)}{ctx.when(request.since, request.until)}{ctx.who(request.from_me)}"
    if outcome.total is None:
        return _any_header(request, outcome, place)
    if outcome.total == 0:
        text = f'Search "{request.query}" in {place}: no message matches.'
        if outcome.word_counts:
            alone = ", ".join(f'"{word}" {count:,}' for word, count in outcome.word_counts)
            text += f" All words must be in one message. Messages per word: {alone}."
        return text
    text = f'Search "{request.query}" in {place}: {plural(outcome.total, "message")} '
    text += "matches." if outcome.total == 1 else f"match. {_ORDER_WORDS[request.order].capitalize()}."
    return text


def _any_header(request: SearchRequest, outcome: SearchOutcome, place: str) -> str:
    counts = dict(outcome.word_counts)
    parts = [f'Search for any of "{request.query}" in {place}.']
    if outcome.used:
        used = ", ".join(f'"{word}" {counts.get(word, 0):,}' for word in outcome.used)
        parts.append(f"Messages per word: {used}. Rare words weigh more.")
    if outcome.common:
        common = ", ".join(f'"{word}" {counts.get(word, 0):,}' for word in outcome.common)
        parts.append(f"Too common to rank by: {common}.")
    if outcome.absent:
        parts.append("No message has: " + ", ".join(f'"{word}"' for word in outcome.absent) + ".")
    if not outcome.hits:
        parts.append("Nothing to show.")
    return " ".join(parts)


def _footer(request: SearchRequest, outcome: SearchOutcome, visible: int, fitted: bool) -> str:
    """What the result leaves out, and what would show it. Empty when nothing is left out."""
    known = len(outcome.matched) if outcome.total is None else outcome.total
    left = known - visible
    if left <= 0:
        return ""
    cut = "Result cut to fit. " if not fitted else ""
    if outcome.total is None:
        return (
            f"\n[{cut}At least {plural(left, 'more message')} hold one of the words. "
            "Raise limit, or search one word alone to list them all.]"
        )
    hints = ["Narrow the search with dates, conversations or more words"]
    if fitted and len(outcome.hits) >= request.limit and request.limit < SEARCH_LIMIT_MAX:
        hints.append("or raise limit")
    text = f"\n[{cut}{plural(left, 'more match', 'more matches')} not shown. {', '.join(hints)}."
    if len(outcome.hits) < min(request.limit, len(outcome.matched)):
        text += " One exchange opens at most 2 matches: use read_around to read on."
    return text + "]"


def _summary(request: SearchRequest, outcome: SearchOutcome, visible: int) -> str:
    if outcome.total is None:
        return f"Searched for any of “{request.query}”: {plural(visible, 'match', 'matches')} shown"
    text = f"Searched for “{request.query}”: {plural(outcome.total, 'match', 'matches')}"
    return text if visible >= outcome.total else f"{text}, {visible} shown"


def prepare_read(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    raw = arguments.get("reference")
    ref = normalize_ref(raw)
    anchor = ctx.registry.get(ref) if ref else None
    if anchor is None:
        shown = " ".join(str(raw).split())[:_ECHO_CHARS]
        raise ToolInputError(f'"{shown}" is not a known reference. Use one like m12 from an earlier result.')
    if not ctx.allows(anchor):
        raise ToolInputError(OUTSIDE_SCOPE)
    radius = int_arg(arguments, "radius", default=READ_RADIUS_DEFAULT, low=1, high=READ_RADIUS_MAX)
    status = f"Reading a conversation from {month_year(anchor['timestamp'])}"
    return Prepared(status, lambda budget: run_read(ctx, ref, anchor, radius, budget))


def run_read(ctx: ToolContext, ref: str, anchor: Message, radius: int, budget: int) -> ToolOutcome:
    window = _around(ctx, anchor, radius)
    anchor_id = str(anchor["id"])
    writer, shown = fit_messages(
        window.messages,
        header=lambda count: f"{plural(count, 'message')} around {ref}:",
        drop=lambda rows: _trimmed(rows, anchor_id),
        budget=budget,
        book=ctx.book,
        registry=ctx.registry,
        limits=ctx.limits,
        marks={anchor_id: ref},
    )
    if len(shown) < len(window.messages):
        writer.add(f"[Result cut to fit: {len(shown)} of {len(window.messages)} messages shown.]", force=True)
    summary = f"Read {plural(len(shown), 'message')} from {month_year(anchor['timestamp'])}"
    return ToolOutcome(writer.text(), summary)


def _trimmed(messages: list[Message], anchor_id: str) -> list[Message]:
    """Drop the message farthest from the anchor."""
    index = next((i for i, m in enumerate(messages) if str(m["id"]) == anchor_id), 0)
    return messages[1:] if index >= len(messages) - 1 - index else messages[:-1]
