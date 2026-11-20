"""Ranking the hits of a search with the reranker port (`ports.Reranker`).

A search that ranks reads a pool of candidates: `pool_size(limit)`, three
times the limit, at most RERANK_MOST. Each candidate goes to the port as one
short text: the conversation, the day, the hit and the message before it and
after it, masked, at most RERANK_CHARS. The port gives the ids best first. The
best `limit` are shown. The result is then written as without a ranking: the
windows, the references and the evidence registry do not change.

Which searches rank: a search with words ("all" or "any"), and the joined
search by meaning and by words. The seeded search ranks against the whole
question. A search with order "oldest" or "newest" does not rank: the order
is the point. A reading of one conversation (no query) does not rank.

A lookup that ranks has three parts (`ranked_lookup`):
- `start`, on the thread of the turn: read the pool and mask its texts.
- `work`, beside the stream: ask the port. It tells the stream RANKING first.
- `run`: write the result from the pool in the new order.
`run` does each part that nobody ran. The port is asked one time, also when
the result is written twice.

A failure of the port never fails the question: the order of the search stays.
"""

from __future__ import annotations

import logging
import math
import re
from typing import AbstractSet, Callable, Generic, Mapping, Optional, Sequence, TypeVar

from .ports import Message, RerankItem
from .render import Limits, message_body
from .tool_context import Fetch, Report, ToolContext, ToolOutcome
from .turn_parts import Event, status

logger = logging.getLogger(__name__)

RERANK_POOL = 3  # candidates read for each hit that is shown
RERANK_MOST = 30  # the most candidates of one ranking
RERANK_CHARS = 600  # the longest text of one candidate
RERANK_RADIUS = 1  # messages on each side of a word hit in the text of its candidate
QUERY_CHARS = 1000  # the longest query: a seeded search ranks against the whole question
HIT_CHARS = 300
NEAR_CHARS = 150
RANKING = "Ranking what I found"
FIRST_OR_LAST = ("oldest", "newest")
_ID_RE = re.compile(r"[0-9]{1,6}")

Pool = TypeVar("Pool")
Gather = Callable[[], tuple[Pool, Sequence[str]]]
Write = Callable[[Pool, tuple[int, ...], int], ToolOutcome]


def pool_size(limit: int) -> int:
    """How many candidates a search with this limit reads. Never fewer than the limit."""
    return max(limit, min(RERANK_MOST, limit * RERANK_POOL))


def reranks(ctx: ToolContext, order: object) -> bool:
    """True when a search with words ranks: a reranker is there, and the call did not ask for the first or last."""
    return ctx.memory.reranker is not None and order not in FIRST_OR_LAST


def kept_order(count: int, given: Sequence[str], keep: int) -> tuple[int, ...]:
    """The places of the candidates to show, best first, at most `keep`.

    Id "n" is the place n - 1. An id that is not a known one is dropped; so is
    a repeat. The candidates the ids leave out follow, in their order.
    """
    order: dict[int, None] = {}
    for value in given:
        if isinstance(value, str) and _ID_RE.fullmatch(value.strip()):
            place = int(value.strip()) - 1
            if 0 <= place < count:
                order.setdefault(place, None)
    order.update(dict.fromkeys(range(count)))  # a known place keeps where it is; the others follow
    return tuple(order)[:max(0, keep)]


def candidate_text(
    ctx: ToolContext, messages: Sequence[Message], hits: AbstractSet[str], focus: Mapping[str, int]
) -> str:
    """The text of one candidate: the conversation and the day, then the lines from just before the first hit.

    A hit line starts with "* ". It gives no reference, so nothing is registered.
    """
    if not messages:
        return ""
    first = next((place for place, m in enumerate(messages) if str(m["id"]) in hits), 0)
    shown = messages[max(0, first - 1):]
    book = ctx.book
    lines = [f"{book.label(str(shown[0]['conversation']))}, {str(shown[0].get('timestamp') or '')[:10]}"]
    for message in shown:
        message_id = str(message["id"])
        hit = message_id in hits
        speaker = book.speaker(message)  # first: the labeler learns the handle before the text is masked
        limits = Limits(max_message_chars=HIT_CHARS if hit else NEAR_CHARS)
        body = message_body(message, limits, focus.get(message_id) if hit else None, book.redact)
        lines.append(f"{'* ' if hit else ''}{speaker}: {body}")
    return "\n".join(lines)[:RERANK_CHARS]


def ranked_lookup(
    ctx: ToolContext, query: str, keep: int, gather: Gather[Pool], write: Write[Pool]
) -> tuple[Callable[[int], ToolOutcome], Fetch]:
    """The `run` and the `fetch` of a search that ranks its pool against `query`.

    gather: reads the pool and gives the raw text of each candidate.
    write: writes the result from the pool, the places to show and a budget.
    """
    lookup: _Lookup[Pool] = _Lookup(ctx, query, keep, gather, write)
    return lookup.run, Fetch(lookup.work, _told, lookup.start)


class _Lookup(Generic[Pool]):
    """The state of one search that ranks. Each part runs at most one time."""

    def __init__(self, ctx: ToolContext, query: str, keep: int, gather: Gather[Pool], write: Write[Pool]) -> None:
        self._ctx = ctx
        self._query = query
        self._keep = keep
        self._gather = gather
        self._write = write
        self._pool: list[tuple[Pool, tuple[str, ...], str]] = []
        self._order: list[tuple[int, ...]] = []

    def start(self) -> None:
        if self._pool:
            return
        ctx = self._ctx
        ctx.listed()  # the labeler knows every handle of the archive before anything is masked
        found, texts = self._gather()
        redact = ctx.labeler.redact
        masked = tuple(redact(text)[:RERANK_CHARS] for text in texts)  # masked first: a cut never splits a handle
        self._pool.append((found, masked, redact(self._query)[:QUERY_CHARS]))

    def work(self, report: Report) -> None:
        """Ask the port. It touches no archive and no labeler, so it can run on another thread."""
        if self._order:
            return
        _found, texts, query = self._pool[0]
        if len(texts) < 2 or self._ctx.memory.cancel.cancelled:
            self._order.append(kept_order(len(texts), (), self._keep))
            return
        report(RANKING)
        self._order.append(_ask(self._ctx, query, texts, self._keep))

    def run(self, budget: int) -> ToolOutcome:
        self.start()
        if not self._order:
            self.work(_silent)
        return self._write(self._pool[0][0], self._order[0], budget)


def _ask(ctx: ToolContext, query: str, texts: Sequence[str], keep: int) -> tuple[int, ...]:
    """The places best first. Any failure of the port keeps the order of the search."""
    items = tuple(RerankItem(str(number), text) for number, text in enumerate(texts, 1))
    try:
        result = ctx.memory.reranker.rerank(query, items, keep, ctx.memory.cancel)  # type: ignore[union-attr]
        ids, usd, failed = tuple(result.ids), result.usd, bool(result.failed)
    except Exception as error:  # the question goes on with the order of the search
        logger.warning("the reranker failed: %s", type(error).__name__)
        return kept_order(len(texts), (), keep)
    ctx.rerank_usd += _amount(usd)  # counted at once: what was paid is not lost
    if failed:
        logger.warning("the ranking failed; the order of the search is kept")
        return kept_order(len(texts), (), keep)
    return kept_order(len(texts), ids, keep)


def _amount(usd: object) -> float:
    known = isinstance(usd, (int, float)) and not isinstance(usd, bool) and math.isfinite(usd) and usd >= 0
    return float(usd) if known else 0.0  # type: ignore[arg-type]


def _told(raw: object) -> Optional[Event]:
    return status(RANKING) if raw == RANKING else None


def _silent(_raw: object) -> None:
    return None
