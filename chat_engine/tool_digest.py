"""conversation_digest: the story of one conversation, period by period.

The digest comes from `ports.DigestSource`. This tool guards the time and the
cost, keeps the scope, reads the messages of each moment from the archive and
registers them as evidence. The result has a budget of its own: it does not
spend from the budget of the other lookups.

The guards, in their order: the scope, a repeat, the estimate, the number of
periods that are not read yet, the cost. A conversation with more periods to
read than one question reads is not read here: the source is asked to read it
in the background (`prepare`), and the model is told to go on without it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

from .digest_text import Moment, Notes, Period, write_digest
from .ports import DigestCost, Message
from .render import time_key
from .scope import parse_day
from .tool_args import ToolInputError, day_arg, text_arg
from .tool_context import Prepared, ToolContext, ToolOutcome, plural
from .tool_limits import DIGESTS_PER_QUESTION
from .verify import is_quote_of

logger = logging.getLogger(__name__)
MOMENT_MESSAGES = 6  # the most messages of one moment that are shown
MOMENT_IDS_READ = 24  # the most message ids of one moment that are checked against the scope
_FAILED = "The digest of {title} could not be built now. Go on with the other tools."
_FAILED_BEFORE = (
    "The digest of {title} could not be built earlier in this question. It is not tried again. "
    "Go on with the other tools."
)
_AGAIN = "This digest is above already. Use what it shows, or answer."
_TOO_MANY = "This question already read {count} conversations whole. Answer from them, or read around a moment."
_TOO_COSTLY = (
    "Reading {title} whole would cost about ${usd:.2f}{missing}. That is above the limit of ${limit:.2f}, "
    "so nothing was built. Narrow the days with since and until, or use the other tools."
)
_LONG = "The story of {title} is long: {missing} not read yet"
_STARTED = (
    f"{_LONG}. Recall reads them now in the background, so the story is not ready for this question. "
    "Answer from the other tools, and tell the user that a fuller answer is possible in a few minutes."
)
_NOT_STARTED = (
    f"{_LONG}, and one question reads at most {{limit}}. "
    "Narrow the days with since and until, or use the other tools."
)
_SHOWN, _FAILED_STATE = "shown", "failed"


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
    usd: Optional[float] = None  # what the call cost; None when the source did not say


def prepare_digest(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    """Check a call. A repeat of a call comes here again: the tool knows what the first call gave."""
    value = text_arg(arguments, "conversation")
    if not value:
        raise ToolInputError("conversation needs an alias like c3.")
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    allowed = ctx.conversations([value])
    since, until = ctx.days(since, until)
    request = DigestRequest(allowed[0] if allowed else "", since, until)
    before = ctx.digests_asked.get(request.key)
    if before == _SHOWN:
        again = ToolOutcome(_AGAIN, "Did not read the same conversation again", ok=False)
        return _prepared(ctx, "", lambda budget: again, repeated=True)
    if before == _FAILED_STATE:
        return _prepared(ctx, "", lambda budget: _answer(ctx, request, "failed before"))
    if ctx.digests_built >= DIGESTS_PER_QUESTION:
        text = _TOO_MANY.format(count=DIGESTS_PER_QUESTION)
        refused = ToolOutcome(text, "Did not read another conversation whole", ok=False)
        return _prepared(ctx, "", lambda budget: refused)
    cost = _estimate(ctx, request)
    if cost is None:
        ctx.digests_asked[request.key] = _FAILED_STATE
        return _prepared(ctx, "", lambda budget: _answer(ctx, request, "failed"))
    # the two refusals below are not kept: the reading in the background or a narrower digest changes the estimate
    if cost["periods"] - cost["periods_cached"] > ctx.memory.digest_parts_limit:
        return _too_long(ctx, request, cost)
    if cost["usd"] > ctx.memory.digest_usd_limit:
        return _prepared(ctx, "", lambda budget: _too_costly(ctx, request, cost))
    if cost["periods"] <= 0:
        ctx.digests_asked[request.key] = _SHOWN
        return _prepared(ctx, "", lambda budget: _written(ctx, request, Built("ok"), budget))
    label = ctx.book.label(request.conversation)
    cached = cost["periods_cached"] >= cost["periods"]
    status = f"Recalling your conversation with {label}" if cached else f"Reading your whole conversation with {label}"
    return _prepared(ctx, status, _lookup(ctx, request))


def _prepared(
    ctx: ToolContext, status: str, run: Callable[[int], ToolOutcome], repeated: bool = False
) -> Prepared:
    """A digest has a budget of its own, and the tool answers a repeat itself."""
    return Prepared(status, run, own_budget=ctx.memory.digest_tokens, repeated=repeated, answers_repeats=True)


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
    """The cost line. It counts the periods to build only when the estimate gives a count that makes sense."""
    missing = cost["periods"] - cost["periods_cached"]
    count = f": {plural(missing, 'period')} {'is' if missing == 1 else 'are'} not built yet" if missing >= 1 else ""
    text = _TOO_COSTLY.format(
        title=_title(ctx, request), usd=cost["usd"], missing=count, limit=ctx.memory.digest_usd_limit
    )
    label = ctx.book.label(request.conversation)
    summary = f"Did not read the conversation with {label}: it would cost about ${cost['usd']:.2f}"
    return ToolOutcome(text, summary, ok=False)


def _too_long(ctx: ToolContext, request: DigestRequest, cost: DigestCost) -> Prepared:
    """A story with more periods to read than one question reads. Its reading is started outside the question."""
    count = cost["periods"] - cost["periods_cached"]
    missing = f"{plural(count, 'period')} {'is' if count == 1 else 'are'}"
    label = ctx.book.label(request.conversation)
    if _start_reading(ctx, request):
        text = _STARTED.format(title=_title(ctx, request), missing=missing)
        words = f"your whole conversation with {label} in the background"
        started = ToolOutcome(text, f"Started to read {words}", ok=False)
        return _prepared(ctx, f"Starting to read {words}", lambda budget: started)
    limit = plural(ctx.memory.digest_parts_limit, "period")
    text = _NOT_STARTED.format(title=_title(ctx, request), missing=missing, limit=limit)
    refused = ToolOutcome(text, f"Did not read the conversation with {label}: {missing} not read yet", ok=False)
    status = f"Your conversation with {label} is too long to read inside one question"
    return _prepared(ctx, status, lambda budget: refused)


def _start_reading(ctx: ToolContext, request: DigestRequest) -> bool:
    """Ask the source to read the conversation outside the question. False when it has no way, or fails."""
    start = getattr(ctx.memory.digests, "prepare", None)
    if not callable(start):
        return False
    try:
        return start(request.conversation, since=request.since, until=request.until) is True
    except Exception as error:  # the question goes on with the other tools
        logger.warning("the reading of a digest in the background did not start: %s", type(error).__name__)
        return False


def _lookup(ctx: ToolContext, request: DigestRequest) -> Callable[[int], ToolOutcome]:
    """The lookup of one digest. The source is asked one time, also when the result is written twice."""
    built: list[Built] = []

    def run(budget: int) -> ToolOutcome:
        if not built:
            built.append(_build(ctx, request))
            if built[0].state != "stopped":
                ctx.digests_asked[request.key] = _SHOWN if built[0].state == "ok" else _FAILED_STATE
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
    usd = _spent(result)
    ctx.digest_usd += usd or 0.0  # counted at once: a cancel or a failure after this does not lose what was paid
    if cancel.cancelled:
        return Built("stopped")
    if not isinstance(result, Mapping) or not isinstance(result.get("periods"), (list, tuple)):
        logger.warning("the digest source gave an answer in the wrong form")
        return Built("failed")
    periods = [_period(row) for row in result["periods"]]
    kept = tuple(period for period in periods if period is not None)
    return Built("ok", kept, _whole(result.get("periods_failed")) + len(periods) - len(kept), usd)


def _spent(result: object) -> Optional[float]:
    """The cost that a digest names, when it is an amount. None when the source did not say."""
    usd = result.get("usd") if isinstance(result, Mapping) else None
    known = isinstance(usd, (int, float)) and not isinstance(usd, bool) and math.isfinite(usd) and usd >= 0
    return float(usd) if known else None  # type: ignore[arg-type]


def _answer(ctx: ToolContext, request: DigestRequest, state: str) -> ToolOutcome:
    """The outcome of a digest that gave no periods: "stopped", "failed" or "failed before"."""
    if state == "stopped":
        return ToolOutcome("", "Stopped before the conversation was read", ok=False)
    label = ctx.book.label(request.conversation)
    line = _FAILED if state == "failed" else _FAILED_BEFORE
    text = line.format(title=ctx.book.title(request.conversation))
    return ToolOutcome(text, f"Could not read the story of your conversation with {label}", ok=False)


def _written(ctx: ToolContext, request: DigestRequest, built: Built, budget: int) -> ToolOutcome:
    """The digest as a result. Before every read of the archive the cancel token is looked at."""
    if built.state != "ok" or ctx.memory.cancel.cancelled:
        return _answer(ctx, request, built.state if built.state != "ok" else "stopped")
    ctx.listed()  # the labeler knows every handle of the archive before a summary is masked
    inside = [period for period in built.periods if _in_scope(ctx, period.first_day, period.last_day)]
    periods, moments_outside = [], 0
    for period in inside:
        read = _with_evidence(ctx, request, period)
        if read is None:
            return _answer(ctx, request, "stopped")
        periods.append(read)
        moments_outside += len(period.moments) - len(read.moments)
    notes = Notes(built.failed, len(built.periods) - len(inside), moments_outside)
    written = write_digest(ctx, _title(ctx, request), periods, notes, budget)
    label = ctx.book.label(request.conversation)
    count = plural(written.periods, "period")
    shown = count if written.shown == written.periods else f"{written.shown:,} of {count}"
    return ToolOutcome(written.text, f"Read the story of your conversation with {label}: {shown}{_cost(built.usd)}")


def _cost(usd: Optional[float]) -> str:
    """What a digest that was read now cost, for the step line. Empty when nothing was paid or the cost is not known."""
    if not usd:
        return ""
    return f", about ${usd:.2f}" if usd >= 0.005 else ", less than $0.01"


def _in_scope(ctx: ToolContext, *days: str) -> bool:
    return all(ctx.scope.allows_time(day) for day in days)


def _with_evidence(ctx: ToolContext, request: DigestRequest, period: Period) -> Optional[Period]:
    """The period with the messages of its moments read from the archive. None after a cancel.

    A moment goes when its day is outside the scope, or when the scope
    refuses one of its messages: its text is then about something outside the
    scope. A quote stays only when a message of its moment holds it.
    """
    moments = []
    for moment in period.moments:
        messages = _messages_of(ctx, request, moment) if _in_scope(ctx, moment.date) else None
        if ctx.memory.cancel.cancelled:
            return None
        if messages is None:
            continue
        shown = messages[:MOMENT_MESSAGES]
        held = is_quote_of(moment.quote, [str(message.get("text") or "") for message in shown])
        moments.append(replace(moment, messages=shown, quote=moment.quote if shown and held else ""))
    return replace(period, moments=tuple(moments))


def _messages_of(
    ctx: ToolContext, request: DigestRequest, moment: Moment
) -> Optional[tuple[Message, ...]]:
    """The messages of a moment that the archive holds, in time order. None when the scope refuses one.

    An id that the archive does not hold is passed over. So is a message of
    another conversation that the scope allows. A moment with more ids than
    are checked cannot be proved to lie inside a scope.
    """
    if len(moment.message_ids) > MOMENT_IDS_READ and ctx.scope.is_limited:
        return None
    found = {}
    for message_id in moment.message_ids[:MOMENT_IDS_READ]:
        if ctx.memory.cancel.cancelled:
            break
        message = ctx.archive.message(message_id)
        if message is None:
            continue
        if not ctx.allows(message):
            return None
        if message["conversation"] == request.conversation:
            found[str(message["id"])] = message
    return tuple(sorted(found.values(), key=time_key))


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
