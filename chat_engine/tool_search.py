"""search_messages and read_around: find messages, and read the exchange around one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .ports import Message
from .registry import normalize_ref
from .render import ResultWriter, render_block
from .search import MAX_FETCH, Found, SearchOutcome, SearchRequest, run_search
from .tool_args import ToolInputError, bool_arg, choice_arg, day_arg, int_arg, list_arg, text_arg
from .tool_context import OUTSIDE_SCOPE, Prepared, ToolContext, ToolOutcome, month_year, plural

SEARCH_LIMIT_DEFAULT = 8
SEARCH_LIMIT_MAX = MAX_FETCH // 3
SEARCH_RADIUS = 2
READ_RADIUS_DEFAULT = 6
READ_RADIUS_MAX = 15
_ORDERS = ("newest", "oldest", "relevance")
_ORDER_WORDS = {"newest": "newest first", "oldest": "oldest first", "relevance": "best match first"}
_NARROW_HINT = "Narrow the search with dates, conversations or more words, or raise limit."


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
        raise ToolInputError("query needs at least one word.")
    match = choice_arg(arguments, "match", ("all", "any"), "all") or "all"
    order = choice_arg(arguments, "order", _ORDERS, "newest") or "newest"
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    from_me = bool_arg(arguments, "from_me")
    limit = int_arg(arguments, "limit", default=SEARCH_LIMIT_DEFAULT, low=1, high=SEARCH_LIMIT_MAX)
    conversations = ctx.conversations(list_arg(arguments, "conversations"))
    since, until = ctx.days(since, until)
    request = SearchRequest(
        query=query, match=match, conversations=conversations, since=since, until=until,
        from_me=from_me, order=order, limit=limit,
    )
    return Prepared(search_status(ctx, request), lambda budget: run_search_tool(ctx, request, budget))


def search_status(ctx: ToolContext, request: SearchRequest) -> str:
    names = ctx.names(request.conversations)
    return f"Searching for “{request.query}”" + (f" in {names}" if names else "")


def run_search_tool(
    ctx: ToolContext, request: SearchRequest, budget: int, radius: int = SEARCH_RADIUS
) -> ToolOutcome:
    outcome = run_search(ctx.archive, request)
    windows = build_windows(ctx, outcome.hits, radius)
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(_header(ctx, request, outcome), force=True)
    shown = 0
    for window in windows:
        mark = ctx.registry.checkpoint()
        block = render_block(
            window.messages, book=ctx.book, registry=ctx.registry, limits=ctx.limits,
            marks={message_id: "match" for message_id in window.hits}, focus=window.focus,
        )
        if not writer.add("\n" + "\n".join(block)):
            ctx.registry.rollback(mark)
            break
        shown += len(window.hits)
    writer.add(_footer(outcome, shown), force=True)
    return ToolOutcome(writer.text().rstrip(), _summary(request, outcome, shown))


def build_windows(ctx: ToolContext, hits: Sequence[Found], radius: int) -> list[Window]:
    """The exchange around each hit. Hits of one exchange share one window."""
    windows: list[Window] = []
    for found in hits:
        message_id = str(found.message["id"])
        focus = {} if found.focus is None else {message_id: found.focus}
        holder = _holding(windows, found.message)
        if holder is None:
            holder = _around(ctx, found.message, radius)
            overlap = next(
                (w for w in windows if w.conversation == holder.conversation and w.ids & holder.ids), None
            )
            if overlap is None:
                windows = [*windows, Window(holder.messages, frozenset({message_id}), focus)]
                continue
            holder = Window(_merged(overlap.messages, holder.messages), overlap.hits, overlap.focus)
            windows = [holder if w is overlap else w for w in windows]
        updated = Window(holder.messages, holder.hits | {message_id}, {**holder.focus, **focus})
        windows = [updated if w is holder else w for w in windows]
    return windows


def _holding(windows: Sequence[Window], message: Message) -> Optional[Window]:
    for window in windows:
        if window.conversation == str(message["conversation"]) and str(message["id"]) in window.ids:
            return window
    return None


def _around(ctx: ToolContext, anchor: Message, radius: int) -> Window:
    near = ctx.archive.context(str(anchor["id"]), radius=radius) if radius > 0 else []
    kept = [m for m in near if m["conversation"] == anchor["conversation"] and ctx.allows(m)]
    if str(anchor["id"]) not in {str(m["id"]) for m in kept}:
        kept = [anchor]
    return Window(tuple(kept), frozenset(), {})


def _merged(first: Sequence[Message], second: Sequence[Message]) -> tuple[Message, ...]:
    by_id = {str(m["id"]): m for m in [*first, *second]}
    return tuple(sorted(by_id.values(), key=lambda m: (str(m.get("timestamp") or ""), m.get("sent_at") or 0)))


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
    text += "matches." if outcome.total == 1 else "match."
    if len(outcome.hits) < outcome.total:
        text += f" Showing {len(outcome.hits)}, {_ORDER_WORDS[request.order]}, at most 2 per exchange."
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


def _footer(outcome: SearchOutcome, shown: int) -> str:
    planned = len(outcome.hits)
    if shown < planned:
        left = (outcome.total or planned) - shown
        return f"\n[Result cut to fit. {plural(left, 'more match', 'more matches')} not shown. {_NARROW_HINT}]"
    if outcome.total is not None and outcome.total > shown:
        left = outcome.total - shown
        return f"\n[{plural(left, 'more match', 'more matches')} not shown. {_NARROW_HINT}]"
    return ""


def _summary(request: SearchRequest, outcome: SearchOutcome, shown: int) -> str:
    if outcome.total is None:
        return f"Searched for any of “{request.query}”: {plural(shown, 'match', 'matches')} shown"
    text = f"Searched for “{request.query}”: {plural(outcome.total, 'match', 'matches')}"
    return text if shown == outcome.total else f"{text}, {shown} shown"


def prepare_read(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    raw = arguments.get("reference")
    ref = normalize_ref(raw)
    anchor = ctx.registry.get(ref) if ref else None
    if anchor is None:
        raise ToolInputError(f'"{raw}" is not a known reference. Use one like m12 from an earlier result.')
    if not ctx.allows(anchor):
        raise ToolInputError(OUTSIDE_SCOPE)
    radius = int_arg(arguments, "radius", default=READ_RADIUS_DEFAULT, low=1, high=READ_RADIUS_MAX)
    status = f"Reading a conversation from {month_year(anchor['timestamp'])}"
    return Prepared(status, lambda budget: run_read(ctx, ref, anchor, radius, budget))


def run_read(ctx: ToolContext, ref: str, anchor: Message, radius: int, budget: int) -> ToolOutcome:
    window = _around(ctx, anchor, radius)
    messages = list(window.messages)
    anchor_id = str(anchor["id"])
    while True:
        writer = ResultWriter(budget, ctx.limits.count_tokens)
        writer.add(f"{plural(len(messages), 'message')} around {ref}:", force=True)
        mark = ctx.registry.checkpoint()
        block = "\n".join(
            render_block(
                messages, book=ctx.book, registry=ctx.registry, limits=ctx.limits, marks={anchor_id: ref}
            )
        )
        if writer.fits(block) or len(messages) <= 1:
            writer.add(block, force=True)
            break
        ctx.registry.rollback(mark)
        messages = _trimmed(messages, anchor_id)
    if len(messages) < len(window.messages):
        writer.add(f"[Result cut to fit: {len(messages)} of {len(window.messages)} messages shown.]", force=True)
    summary = f"Read {plural(len(messages), 'message')} from {month_year(anchor['timestamp'])}"
    return ToolOutcome(writer.text(), summary)


def _trimmed(messages: list[Message], anchor_id: str) -> list[Message]:
    """Drop the message farthest from the anchor."""
    index = next((i for i, m in enumerate(messages) if str(m["id"]) == anchor_id), 0)
    return messages[1:] if index >= len(messages) - 1 - index else messages[:-1]
