"""conversation_digest: the story of one conversation, period by period.

The digest comes from `ports.DigestSource`. This tool guards the cost, keeps
the scope, reads the messages of each moment from the archive and registers
them as evidence. The result has a budget of its own: it does not spend from
the budget of the other lookups.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

from .digest_text import Moment, Notes, Period, write_digest
from .ports import DigestCost
from .render import time_key
from .scope import parse_day
from .tool_args import ToolInputError, day_arg, text_arg
from .tool_context import Prepared, ToolContext, ToolOutcome, plural

logger = logging.getLogger(__name__)
DIGESTS_PER_QUESTION = 3  # each digest can cost money: one question builds at most this many
MOMENT_MESSAGES = 6  # the most messages of one moment that are read and shown
_FAILED = "The digest of {title} could not be built now. Go on with the other tools."
_AGAIN = "This digest is above already. Read around one of its moments, or answer."
_TOO_MANY = "This question already read {count} conversations whole. Answer from them, or read around a moment."
_TOO_COSTLY = (
    "Reading {title} whole would cost about ${usd:.2f}: {periods} not built yet. That is above the limit "
    "of ${limit:.2f}, so nothing was built. Narrow the days with since and until, or use the other tools."
)


@dataclass(frozen=True)
class DigestRequest:
    conversation: str
    since: Optional[str]
    until: Optional[str]

    @property
    def key(self) -> tuple[str, Optional[str], Optional[str]]:
        return self.conversation, self.since, self.until


@dataclass(frozen=True)
class Built:
    """What the source gave. state: "ok", "failed" or "stopped" (the question was cancelled)."""

    state: str
    periods: tuple[Period, ...] = ()
    failed: int = 0


def prepare_digest(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    value = text_arg(arguments, "conversation")
    if not value:
        raise ToolInputError("conversation needs an alias like c3.")
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    allowed = ctx.conversations([value])
    since, until = ctx.days(since, until)
    request = DigestRequest(allowed[0] if allowed else "", since, until)
    own = ctx.memory.digest_tokens
    if request.key in ctx.digests_asked:
        again = ToolOutcome(_AGAIN, "Did not read the same conversation again", ok=False)
        return Prepared("", lambda budget: again, own_budget=own, repeated=True)
    ctx.digests_asked.add(request.key)
    label = ctx.book.label(request.conversation)
    if ctx.digests_built >= DIGESTS_PER_QUESTION:
        text = _TOO_MANY.format(count=DIGESTS_PER_QUESTION)
        refused = ToolOutcome(text, "Did not read another conversation whole", ok=False)
        return Prepared("", lambda budget: refused, own_budget=own)
    cost = _estimate(ctx, request)
    if cost is None:
        return Prepared("", lambda budget: _answer(ctx, request, "failed"), own_budget=own)
    if cost["usd"] > ctx.memory.digest_usd_limit:
        return Prepared("", lambda budget: _too_costly(ctx, request, cost), own_budget=own)
    if cost["periods"] <= 0:
        return Prepared("", lambda budget: _written(ctx, request, Built("ok"), budget), own_budget=own)
    cached = cost["periods_cached"] >= cost["periods"]
    status = f"Recalling your conversation with {label}" if cached else f"Reading your whole conversation with {label}"
    return Prepared(status, _lookup(ctx, request), own_budget=own)


def _title(ctx: ToolContext, request: DigestRequest) -> str:
    return f"{ctx.book.title(request.conversation)}{ctx.when(request.since, request.until)}"


def _estimate(ctx: ToolContext, request: DigestRequest) -> Optional[DigestCost]:
    """The cost of the digest, in the right form. None when the source fails."""
    try:
        cost = ctx.memory.digests.estimate(  # type: ignore[union-attr]
            request.conversation, since=request.since, until=request.until
        )
        checked: DigestCost = {
            "periods": int(cost["periods"]),
            "periods_cached": int(cost["periods_cached"]),
            "usd": float(cost["usd"]),
        }
        if not (math.isfinite(checked["usd"]) and checked["usd"] >= 0):
            raise ValueError("the cost is not an amount")
    except Exception as error:  # the question goes on with the other tools
        logger.warning("the estimate of a digest failed: %s", type(error).__name__)
        return None
    return checked


def _too_costly(ctx: ToolContext, request: DigestRequest, cost: DigestCost) -> ToolOutcome:
    missing = max(1, cost["periods"] - cost["periods_cached"])
    periods = f"{plural(missing, 'period')} {'is' if missing == 1 else 'are'}"
    text = _TOO_COSTLY.format(
        title=_title(ctx, request), usd=cost["usd"], periods=periods, limit=ctx.memory.digest_usd_limit
    )
    label = ctx.book.label(request.conversation)
    summary = f"Did not read the conversation with {label}: it would cost about ${cost['usd']:.2f}"
    return ToolOutcome(text, summary, ok=False)


def _lookup(ctx: ToolContext, request: DigestRequest) -> Callable[[int], ToolOutcome]:
    """The lookup of one digest. The source is asked one time, also when the result is written twice."""
    built: list[Built] = []

    def run(budget: int) -> ToolOutcome:
        if not built:
            built.append(_build(ctx, request))
        return _written(ctx, request, built[0], budget)

    return run


def _build(ctx: ToolContext, request: DigestRequest) -> Built:
    """Ask the source. After a cancel nothing more is read: the question ends."""
    cancel = ctx.memory.cancel
    if cancel.cancelled:
        return Built("stopped")
    ctx.digests_built += 1
    try:
        result = ctx.memory.digests.digest(  # type: ignore[union-attr]
            request.conversation, since=request.since, until=request.until, cancel=cancel
        )
    except Exception as error:
        if cancel.cancelled:
            return Built("stopped")
        logger.warning("the digest source failed: %s", type(error).__name__)
        return Built("failed")
    if cancel.cancelled:
        return Built("stopped")
    if not isinstance(result, Mapping) or not isinstance(result.get("periods"), (list, tuple)):
        logger.warning("the digest source gave an answer in the wrong form")
        return Built("failed")
    periods = [_period(row) for row in result["periods"]]
    kept = tuple(period for period in periods if period is not None)
    return Built("ok", kept, _whole(result.get("periods_failed")) + len(periods) - len(kept))


def _answer(ctx: ToolContext, request: DigestRequest, state: str) -> ToolOutcome:
    """The outcome of a digest that gave no periods."""
    if state == "stopped":
        return ToolOutcome("", "Stopped before the conversation was read", ok=False)
    label = ctx.book.label(request.conversation)
    text = _FAILED.format(title=ctx.book.title(request.conversation))
    return ToolOutcome(text, f"Could not read the story of your conversation with {label}", ok=False)


def _written(ctx: ToolContext, request: DigestRequest, built: Built, budget: int) -> ToolOutcome:
    if built.state != "ok":
        return _answer(ctx, request, built.state)
    ctx.listed()  # the labeler knows every handle of the archive before a summary is masked
    inside = [period for period in built.periods if _in_scope(ctx, period.first_day, period.last_day)]
    notes = Notes(failed=built.failed, outside=len(built.periods) - len(inside))
    periods = [_with_evidence(ctx, request, period) for period in inside]
    text, count = write_digest(ctx, _title(ctx, request), periods, notes, budget)
    label = ctx.book.label(request.conversation)
    return ToolOutcome(text, f"Read the story of your conversation with {label}: {plural(count, 'period')}")


def _in_scope(ctx: ToolContext, *days: str) -> bool:
    return all(ctx.scope.allows_time(day) for day in days)


def _with_evidence(ctx: ToolContext, request: DigestRequest, period: Period) -> Period:
    """The period with the messages of its moments read from the archive. A moment outside the scope goes."""
    moments = []
    for moment in period.moments:
        if not _in_scope(ctx, moment.date):
            continue
        found = {}
        for message_id in moment.message_ids[:MOMENT_MESSAGES]:
            message = ctx.archive.message(message_id)
            if message is None or message["conversation"] != request.conversation or not ctx.allows(message):
                continue  # dropped without a word
            found[str(message["id"])] = message
        moments.append(replace(moment, messages=tuple(sorted(found.values(), key=time_key))))
    return replace(period, moments=tuple(moments))


def _whole(value: object) -> int:
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _period(row: object) -> Optional[Period]:
    """One period of the source in the form the tool needs. None when it has another form."""
    if not isinstance(row, Mapping):
        return None
    try:
        first, last = parse_day(row.get("first_day")), parse_day(row.get("last_day"))
    except (TypeError, ValueError):
        return None
    label, summary = row.get("label"), row.get("summary")
    if not (first and last and isinstance(label, str) and isinstance(summary, str)):
        return None
    listed = row.get("moments")
    moments = [_moment(item) for item in (listed if isinstance(listed, (list, tuple)) else ())]
    kept = tuple(moment for moment in moments if moment is not None)
    return Period(label, first, last, _whole(row.get("messages")), summary, kept)


def _moment(row: object) -> Optional[Moment]:
    if not isinstance(row, Mapping):
        return None
    texts = [row.get(key, "") for key in ("title", "detail", "quote")]
    if not all(isinstance(text, str) for text in texts) or not (texts[0] or texts[1]):
        return None
    try:
        date = parse_day(row.get("date")) or ""
    except (TypeError, ValueError):
        date = ""
    ids = row.get("message_ids")
    message_ids = tuple(item for item in (ids if isinstance(ids, (list, tuple)) else ()) if isinstance(item, str))
    return Moment(date, texts[0], texts[1], texts[2], message_ids)
