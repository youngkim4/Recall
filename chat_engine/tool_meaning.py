"""search_messages with match "meaning": find exchanges by what they are about.

Two rankings of exchanges are joined by rank (see `fusion.py`):

- the hits of the search by meaning (`ports.MeaningSearch`), and
- the hits of the word search (match "any") for the same query. A word hit
  counts for the exchange that holds it. A word hit outside every exchange
  of the first ranking is an exchange of its own: the messages around it.

The conversations and the days of the request limit both rankings. They come
from `ToolContext.conversations` and `ToolContext.days`, as for every search,
so the scope of the user holds. Each message that is shown passes
`ToolContext.allows` again. A failure of the port leaves the word ranking.

The word ranking is always by relevance. The order of the call is not used.
After a cancel that comes while the port searches, the archive is not read.
When the archive holds none of the hits of the port, the index is older than
the archive: the first line of the result says so.

With a reranker, the joined ranking reads a pool of exchanges, and the
reranker orders them (`rerank.py`). The order of the call is not used either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from .fusion import fuse
from .ports import MeaningHit, Message
from .render import ResultWriter, add_block, time_key
from .rerank import candidate_text, pool_size, ranked_lookup
from .search import MAX_FETCH, Found, SearchRequest, run_search
from .tool_args import ToolInputError, bool_arg, choice_arg, day_arg, int_arg, list_arg, text_arg
from .tool_context import Prepared, ToolContext, ToolOutcome, plural
from .tool_limits import ORDERS, SEARCH_LIMIT_DEFAULT, SEARCH_LIMIT_MAX
from .tool_search import SEARCH_RADIUS, build_windows, prepare_search

logger = logging.getLogger(__name__)
MATCHES = ("all", "any", "meaning")
FETCH_FOR_EACH = 3  # hits read from each ranking for each exchange that is asked for
SEGMENT_MESSAGES = 40  # the most messages of one exchange that are read
_NOT_AVAILABLE = "The search by meaning was not available: these were found by their words only."
_STALE = (
    "The search by meaning found nothing that the archive still holds: its index must be brought up to date. "
    "These were found by their words only."
)
Key = tuple[str, str, str]  # conversation, first message id, last message id ("" for a word hit alone)


@dataclass(frozen=True)
class MeaningRequest:
    """One search by meaning.

    description: what goes to the port. words: the word ranking; its
    conversations, its days and its limit hold for both rankings.
    what: how the result names the search.
    """

    description: str
    words: SearchRequest
    what: str


@dataclass(frozen=True)
class Segment:
    """An exchange the port found, with its two ends read from the archive."""

    first: Message
    last: Message

    @property
    def key(self) -> Key:
        return str(self.first["conversation"]), str(self.first["id"]), str(self.last["id"])

    def holds(self, message: Message) -> bool:
        same = message["conversation"] == self.first["conversation"]
        return same and time_key(self.first) <= time_key(message) <= time_key(self.last)


@dataclass(frozen=True)
class Exchange:
    """The messages of one exchange in time order, as far as the scope allows them."""

    messages: tuple[Message, ...]
    hits: frozenset[str]  # ids of the messages that hold a word of the query
    focus: Mapping[str, int]
    more: bool = False  # the exchange is longer than what was read


@dataclass(frozen=True)
class MeaningFound:
    """The exchanges a search by meaning found, best first.

    known: the exchanges that were built, and those past the limit.
    note: why only the words were searched, or "".
    """

    exchanges: tuple[Exchange, ...]
    known: int
    note: str


def prepare_search_or_meaning(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    """search_messages of an engine that has a search by meaning."""
    if choice_arg(arguments, "match", MATCHES, "all") != "meaning":
        return prepare_search(ctx, arguments)
    query = text_arg(arguments, "query")
    if not query:
        raise ToolInputError('match "meaning" needs a short description in query.')
    if bool_arg(arguments, "from_me") is not None:
        raise ToolInputError('from_me does not work with match "meaning": an exchange holds both sides.')
    ctx.listed()  # the labeler knows every handle of the archive before the status line is masked
    choice_arg(arguments, "order", ORDERS, "newest")  # checked, not used: the result is best first
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    limit = int_arg(arguments, "limit", default=SEARCH_LIMIT_DEFAULT, low=1, high=SEARCH_LIMIT_MAX)
    conversations = ctx.conversations(list_arg(arguments, "conversations"))
    since, until = ctx.days(since, until)
    words = SearchRequest(
        query=query, match="any", conversations=conversations, since=since, until=until, limit=limit
    )
    names = ctx.names(conversations)
    status = f"Searching for moments like “{query}”" + (f" in {names}" if names else "")
    return meaning_prepared(ctx, MeaningRequest(query, words, f"“{query}”"), status, rank_by=query)


def meaning_prepared(
    ctx: ToolContext, request: MeaningRequest, status: str, *, radius: int = SEARCH_RADIUS,
    rank_by: Optional[str] = None,
) -> Prepared:
    """A search by meaning. With `rank_by` and a reranker, its exchanges are ranked against `rank_by`."""
    if rank_by is None or ctx.memory.reranker is None:
        return Prepared(status, meaning_lookup(ctx, request, radius))
    limit = request.words.limit

    def gather() -> tuple[Optional[MeaningFound], list[str]]:
        hits = ask_port(ctx, request)
        if ctx.memory.cancel.cancelled:  # the cancel came while the port searched: nothing more is read
            return None, []
        found = find_exchanges(ctx, request, hits, radius, pool_size(limit))
        return found, [candidate_text(ctx, e.messages, e.hits, e.focus) for e in found.exchanges]

    def write(found: Optional[MeaningFound], order: tuple[int, ...], budget: int) -> ToolOutcome:
        if found is None:
            return ToolOutcome("", "Stopped before the messages were read", ok=False)
        chosen = replace(found, exchanges=tuple(found.exchanges[place] for place in order))
        return write_meaning(ctx, request, chosen, budget)

    run, fetch = ranked_lookup(ctx, rank_by, limit, gather, write)
    return Prepared(status, run, fetch=fetch)


def meaning_lookup(
    ctx: ToolContext, request: MeaningRequest, radius: int = SEARCH_RADIUS
) -> Callable[[int], ToolOutcome]:
    """The lookup of one search by meaning. The port is asked one time, also when the result is written twice."""
    asked: list[Optional[list[MeaningHit]]] = []

    def run(budget: int) -> ToolOutcome:
        if not asked:
            asked.append(ask_port(ctx, request))
        if ctx.memory.cancel.cancelled:  # the cancel came while the port searched: nothing more is read
            return ToolOutcome("", "Stopped before the messages were read", ok=False)
        return run_meaning_search(ctx, request, asked[0], budget, radius)

    return run


def ask_port(ctx: ToolContext, request: MeaningRequest) -> Optional[list[MeaningHit]]:
    """The hits of the port that have the right form, best first. None when the port fails.

    A port that gives more hits than were asked for is cut to that number.
    """
    port = ctx.memory.meaning
    words = request.words
    wanted = _fetch(words.limit)
    ctx.listed()  # the labeler knows every handle of the archive before the description is masked
    description = ctx.labeler.redact(request.description)
    try:
        found = port.search(  # type: ignore[union-attr]
            description,
            conversations=None if words.conversations is None else list(words.conversations),
            since=words.since,
            until=words.until,
            limit=wanted,
        )
        if not isinstance(found, (list, tuple)):
            raise TypeError("the port gave no list")
    except Exception as error:  # the question goes on with the word ranking
        logger.warning("the search by meaning failed: %s", type(error).__name__)
        return None
    return [row for row in found[:wanted] if _is_hit(row)]


def _fetch(limit: int) -> int:
    return min(MAX_FETCH, max(1, limit) * FETCH_FOR_EACH)


def _is_hit(row: object) -> bool:
    keys = ("conversation", "first_message_id", "last_message_id")
    return isinstance(row, Mapping) and all(isinstance(row.get(key), str) and row[key] for key in keys)


def run_meaning_search(
    ctx: ToolContext,
    request: MeaningRequest,
    hits: Optional[Sequence[MeaningHit]],
    budget: int,
    radius: int = SEARCH_RADIUS,
) -> ToolOutcome:
    return write_meaning(ctx, request, find_exchanges(ctx, request, hits, radius, request.words.limit), budget)


def find_exchanges(
    ctx: ToolContext, request: MeaningRequest, hits: Optional[Sequence[MeaningHit]], radius: int, limit: int
) -> MeaningFound:
    """The best `limit` exchanges of both rankings joined. `hits` is None when the port failed."""
    words = request.words
    segments, stale = _segments(ctx, words, hits or ())
    note = _NOT_AVAILABLE if hits is None else _STALE if stale else ""
    # by relevance, whatever the order of the call: the newest messages must not hide an old strong match
    found = run_search(ctx.archive, replace(words, limit=_fetch(words.limit), order="relevance")).hits
    keys, by_key = _joined(segments, found)
    by_segment = {segment.key: segment for segment in segments}
    exchanges, used = _exchanges(ctx, keys, by_key, by_segment, limit, radius)
    known = len(exchanges) + len(keys) - used  # the exchanges that were built, and those past the limit
    return MeaningFound(tuple(exchanges), known, note)


def write_meaning(ctx: ToolContext, request: MeaningRequest, found: MeaningFound, budget: int) -> ToolOutcome:
    """The result, from the exchanges in the order they are given."""
    words = request.words
    known, exchanges, note = found.known, found.exchanges, found.note
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    place = f"{ctx.where(words.conversations)}{ctx.when(words.since, words.until)}"
    writer.add(_header(request, place, known, note), force=True)
    shown = 0
    for exchange in exchanges:
        if not _add_exchange(ctx, writer, exchange):
            break
        shown += 1
    writer.add(_footer(known, len(exchanges), shown, words.limit), force=True)
    summary = f"Searched by meaning for {request.what}: {plural(shown, 'exchange')} shown"
    return ToolOutcome(writer.text().rstrip(), summary + (", by words only" if note else ""))


def _segments(
    ctx: ToolContext, words: SearchRequest, hits: Sequence[MeaningHit]
) -> tuple[list[Segment], bool]:
    """The hits inside the conversations and the days of the request, each one time, best first.

    Also: True when hits were looked up and the archive holds none of them.
    Then the index is older than the archive.
    """
    segments: dict[Key, Segment] = {}
    looked = held = 0
    for row in hits:
        conversation = row["conversation"]
        if words.conversations is not None and conversation not in words.conversations:
            continue
        looked += 1
        first = ctx.archive.message(row["first_message_id"])
        last = None if first is None else ctx.archive.message(row["last_message_id"])
        if first is None or last is None:
            continue
        held += 1
        if time_key(first) > time_key(last):
            continue
        if first["conversation"] != conversation or last["conversation"] != conversation:
            continue
        if _day(last) < (words.since or "") or (words.until and _day(first) > words.until):
            continue
        segment = Segment(first, last)
        segments.setdefault(segment.key, segment)
    stale = looked > 0 and held == 0
    if stale:
        logger.warning("the search by meaning gave %d hits that the archive does not hold", looked)
    return list(segments.values()), stale


def _day(message: Message) -> str:
    return str(message.get("timestamp") or "")[:10]


def _joined(segments: Sequence[Segment], found: Sequence[Found]) -> tuple[list[Key], dict[Key, list[Found]]]:
    """The exchanges of both rankings, best first, and the word hits of each."""
    by_key: dict[Key, list[Found]] = {}
    word_keys: list[Key] = []
    for hit in found:
        holder = next((segment for segment in segments if segment.holds(hit.message)), None)
        alone = (str(hit.message["conversation"]), str(hit.message["id"]), "")
        key = holder.key if holder is not None else alone
        word_keys.append(key)
        by_key.setdefault(key, []).append(hit)
    fused = fuse([[segment.key for segment in segments], word_keys])
    return [item.key for item in fused], by_key


def _exchanges(
    ctx: ToolContext,
    keys: Sequence[Key],
    by_key: Mapping[Key, Sequence[Found]],
    segments: Mapping[Key, Segment],
    limit: int,
    radius: int,
) -> tuple[list[Exchange], int]:
    """The best `limit` exchanges, and how many keys they took.

    Two exchanges that share a message become one, at the place of the better.
    """
    exchanges: list[Exchange] = []
    used = 0
    for key in keys:
        if len(exchanges) >= limit:
            break
        used += 1
        hits = by_key.get(key, ())
        if key in segments:
            exchange = _read_segment(ctx, segments[key], hits)
        else:
            window = build_windows(ctx, hits[:1], radius)[0]
            exchange = Exchange(window.messages, window.hits, window.focus)
        if not exchange.messages:
            continue  # nothing of it is inside the scope
        touching = [other for other in exchanges if _share(other, exchange)]
        for other in touching:
            exchange = _merged(other, exchange)
        if touching:
            kept = [other for other in exchanges if all(other is not t for t in touching[1:])]
            exchanges = [exchange if other is touching[0] else other for other in kept]
        else:
            exchanges.append(exchange)
    return exchanges, used


def _read_segment(ctx: ToolContext, segment: Segment, hits: Sequence[Found]) -> Exchange:
    conversation, first_id, last_id = segment.key
    rows = [segment.first]
    if first_id != last_id:
        page = ctx.archive.messages(conversation, after=first_id, limit=SEGMENT_MESSAGES - 1)
        rows.extend(sorted(page["messages"], key=time_key))
    ids = [str(row["id"]) for row in rows]
    reached = last_id in ids
    if reached:
        rows = rows[:ids.index(last_id) + 1]
    kept = tuple(row for row in rows if row["conversation"] == conversation and ctx.allows(row))
    inside = {str(row["id"]) for row in kept}
    focus = {str(hit.message["id"]): hit.focus for hit in hits if hit.focus is not None}
    return Exchange(kept, frozenset(str(hit.message["id"]) for hit in hits) & inside, focus, more=not reached)


def _share(one: Exchange, other: Exchange) -> bool:
    if one.messages[0]["conversation"] != other.messages[0]["conversation"]:
        return False
    return bool({str(m["id"]) for m in one.messages} & {str(m["id"]) for m in other.messages})


def _merged(one: Exchange, other: Exchange) -> Exchange:
    by_id = {str(m["id"]): m for m in [*one.messages, *other.messages]}
    messages = tuple(sorted(by_id.values(), key=time_key))
    return Exchange(messages, one.hits | other.hits, {**one.focus, **other.focus}, one.more or other.more)


def _add_exchange(ctx: ToolContext, writer: ResultWriter, exchange: Exchange) -> bool:
    """Add one exchange. A long one loses messages until it fits, and a note says so. False when nothing fits."""
    kept = exchange.messages
    total = f"{'more than ' if exchange.more else ''}{len(exchange.messages)}"
    while kept:
        cut = len(kept) < len(exchange.messages) or exchange.more
        note = f"[Exchange cut to fit: {len(kept)} of {total} messages shown. Use read_around to read on.]"
        marks = {str(m["id"]): "match" for m in kept if str(m["id"]) in exchange.hits}
        fits = add_block(
            writer, kept, book=ctx.book, registry=ctx.registry, limits=ctx.limits, marks=marks,
            focus=exchange.focus, tail=(note,) if cut else (),
        )
        if fits:
            return True
        kept = _fewer(kept, exchange.hits)
    return False


def _fewer(messages: Sequence[Message], hits: frozenset[str]) -> tuple[Message, ...]:
    """Half of the messages: those around the first word hit, or the first ones."""
    size = len(messages) // 2
    at = next((index for index, m in enumerate(messages) if str(m["id"]) in hits), 0)
    start = max(0, min(at - size // 2, len(messages) - size))
    return tuple(messages[start:start + size])


def _header(request: MeaningRequest, place: str, known: int, note: str) -> str:
    """The first line. `note` says why only the words were searched, or is empty."""
    found = f"{plural(known, 'exchange')}, best first." if known else "nothing found."
    text = f"Search by meaning for {request.what} in {place}: {found}"
    return f"{text} {note}" if note else text


def _footer(known: int, read: int, shown: int, limit: int) -> str:
    """How many exchanges the result leaves out, and what would show them. Empty when none."""
    left = known - shown
    if left <= 0:
        return ""
    cut = "Result cut to fit. " if shown < read else ""
    hint = "Narrow the search with dates or conversations"
    if not cut and limit < SEARCH_LIMIT_MAX:
        hint += ", or raise limit"
    return f"\n[{cut}{plural(left, 'more exchange')} not shown. {hint}.]"
