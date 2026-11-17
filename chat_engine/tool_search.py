"""search_messages and read_around: find messages, and read the exchange around one."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from .browse import browse
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
_EXCHANGE = timedelta(hours=36)
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
        kept = [anchor]
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


def prepare_browse(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    """No search word: read one conversation from its start or its end."""
    if bool_arg(arguments, "from_me") is not None:
        raise ToolInputError("from_me needs a search word in query.")
    edge = "oldest" if choice_arg(arguments, "order", _ORDERS, "newest") == "oldest" else "newest"
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    limit = int_arg(arguments, "limit", default=SEARCH_LIMIT_DEFAULT, low=1, high=SEARCH_LIMIT_MAX)
    conversations = ctx.conversations(list_arg(arguments, "conversations"))
    if conversations is None or len(conversations) != 1:
        raise ToolInputError(
            "without a search word, give exactly one conversation to read. Or put a word in query."
        )
    since, until = ctx.days(since, until)
    conversation_id = conversations[0]
    joiner = "in" if ctx.book.is_group(conversation_id) else "with"
    which = "first" if edge == "oldest" else "latest"
    words = f"the {which} messages {joiner} {ctx.book.label(conversation_id)}"
    return Prepared(
        f"Reading {words}",
        lambda budget: run_browse(ctx, conversation_id, edge, limit, since, until, words, budget),
    )


def run_browse(
    ctx: ToolContext,
    conversation_id: str,
    edge: str,
    limit: int,
    since: Optional[str],
    until: Optional[str],
    words: str,
    budget: int,
) -> ToolOutcome:
    found = browse(ctx.archive, conversation_id, edge=edge, limit=limit, since=since, until=until)
    title = f"{ctx.book.title(conversation_id)}{ctx.when(since, until)}"
    if not found.anchored:
        return ToolOutcome(
            f"Cannot read {title} without a search word: no way into that period was found. "
            'Use count_messages with group_by "day" to see the active days, or search for a word.',
            f"Could not read {words}",
            ok=False,
        )
    messages = [m for m in found.messages if ctx.allows(m)]
    if not messages:
        return ToolOutcome(f"No messages in {title}.", f"Read {words}: nothing there")
    backwards = edge == "oldest"
    facts = ctx.book.get(conversation_id)
    whole = not (since or until) and facts is not None and len(messages) >= int(facts.get("message_count") or 0)
    while True:
        writer = ResultWriter(budget, ctx.limits.count_tokens)
        lead = "First" if backwards else "Latest"
        writer.add(f"{lead} {plural(len(messages), 'message')} of {title}{' (all of it)' if whole else ''}:", force=True)
        mark = ctx.registry.checkpoint()
        block = "\n".join(
            render_block(messages, book=ctx.book, registry=ctx.registry, limits=ctx.limits, marks={})[1:]
        )
        if writer.fits(block) or len(messages) <= 1:
            writer.add(block, force=True)
            break
        ctx.registry.rollback(mark)
        messages = messages[:-1] if backwards else messages[1:]
        whole = False
    if len(messages) < len(found.messages):
        writer.add("[Result cut to fit. Ask again with a smaller limit, or read_around a line.]", force=True)
    if not found.reached_edge:
        side = "start" if backwards else "end"
        writer.add(f"[This conversation is very long: its {side} was not reached. These are the nearest read.]", force=True)
    which = "first" if backwards else "latest"
    summary = "Read " + words.replace(f"the {which} messages", f"the {which} {plural(len(messages), 'message')}", 1)
    return ToolOutcome(writer.text(), summary)


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
