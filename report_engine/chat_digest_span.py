"""Which periods a conversation digest reads, with which cache keys, and what it may spend.

The periods are the periods of a report on all dates: `build_plan` lays out
the whole conversation, and a span selects the periods that overlap it. A
selected period is read whole, so its text is the text that a report reads.

`reading_of` decides the keys. A digest has the keys of a report only when
its prompts are the prompts of a report. A span that starts after periods
that the cache does not hold is read with no notes from before it, under
keys of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from report_engine.cache import CacheEntry, period_key
from report_engine.chunking import Period
from report_engine.dates import DAY_LENGTH
from report_engine.digest import DigestError, PeriodDigest, parse_digest
from report_engine.errors import (
    ConfigError,
    EmptyConversationError,
    ReportEngineError,
    StoreError,
)
from report_engine.estimate import ReadableCache, known_keys
from report_engine.model_info import ModelInfo
from report_engine.options import ReportOptions
from report_engine.plan import Plan, PlannedPeriod, build_plan
from report_engine.ports import Labeler, ReportSource
from report_engine.prompts import MAP_PROMPT_VERSION, Note
from report_engine.scope import coerce_scope
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

SPAN_MARK = "/span-from/"


class DigestTooLarge(ReportEngineError):
    """More periods need a model call than the caller allows. No call was made."""

    def __init__(self, periods: int, periods_cached: int, limit: int) -> None:
        super().__init__(
            f"the digest needs {periods - periods_cached} model calls; the limit is {limit}"
        )
        self.periods = periods
        self.periods_cached = periods_cached
        self.limit = limit

    @property
    def new_periods(self) -> int:
        return self.periods - self.periods_cached


@dataclass(frozen=True)
class Span:
    """The plan of the whole conversation and the place of the selected periods in it."""

    whole: Plan
    start: int
    stop: int

    @property
    def plan(self) -> Plan:
        """The same plan with the selected periods only. Its lines are all the
        lines of the conversation: the quote check reads them."""
        return replace(self.whole, periods=self.whole.periods[self.start : self.stop])

    @property
    def earlier(self) -> tuple[PlannedPeriod, ...]:
        return self.whole.periods[: self.start]


def _span_days(since: str | None, until: str | None) -> tuple[str | None, str | None]:
    """The first and the last day of the span. A bound that a report refuses is refused here."""
    scope = coerce_scope({"since": since, "until": until})
    first = scope.since[:DAY_LENGTH] if scope.since else None
    last = scope.until[:DAY_LENGTH] if scope.until else None
    return first, last


def _overlaps(period: Period, first: str | None, last: str | None) -> bool:
    if first is not None and period.last_day < first:
        return False
    return last is None or period.first_day <= last


def span_plan(
    source: ReportSource,
    conversation: str,
    since: str | None,
    until: str | None,
    labeler: Labeler,
    model_info: ModelInfo,
    options: ReportOptions,
    counter: TokenCounter | None,
) -> Span:
    """Lay out the whole conversation as a report on all dates does, then select."""
    first, last = _span_days(since, until)
    whole = build_plan(
        source=source,
        conversation=conversation,
        scope=None,
        labeler=labeler,
        model_info=model_info,
        options=options,
        counter=counter,
    )
    chosen = [
        index
        for index, planned in enumerate(whole.periods)
        if _overlaps(planned.period, first, last)
    ]
    if not chosen:
        raise EmptyConversationError("no message with text in this conversation and span")
    return Span(whole=whole, start=chosen[0], stop=chosen[-1] + 1)


@dataclass(frozen=True)
class Reading:
    """How a digest reads the periods of its span.

    `plan` holds the selected periods with the cache keys that this digest
    uses. `notes` are the notes of the periods before the span, and
    `prompt_version` is written with each new entry. `shared` says that the
    keys are the keys of a report."""

    plan: Plan
    notes: tuple[Note, ...]
    prompt_version: str
    shared: bool


def _held(entry: CacheEntry | None) -> PeriodDigest | None:
    """The result of a period as a run would read it from the cache, or None."""
    if entry is None:
        return None
    try:
        return parse_digest(entry.payload)
    except DigestError:
        return None


def _notes_before(
    cache: ReadableCache | None, earlier: Sequence[PlannedPeriod]
) -> tuple[Note, ...] | None:
    """The notes of every period before the span, oldest first: what a report
    sends with the first period of the span. None when the cache does not
    hold each of those periods: a report would read them first."""
    if not earlier:
        return ()
    if cache is None:
        return None
    try:
        entries = cache.get_many([planned.key for planned in earlier])
    except StoreError as exc:
        logger.warning(
            "the period cache could not be read for the periods before the span: %s",
            type(exc).__name__,
        )
        return None
    held = [_held(entries.get(planned.key)) for planned in earlier]
    if any(digest is None for digest in held):
        return None
    notes = [
        (planned.period.label, digest.carry_over)
        for planned, digest in zip(earlier, held, strict=True)
        if digest is not None and digest.carry_over
    ]
    return tuple(notes)


def own_version(first_label: str) -> str:
    """The prompt version of a span that is read with no notes from before it.
    It names the period where the span starts. No report has this version."""
    return f"{MAP_PROMPT_VERSION}{SPAN_MARK}{first_label}"


def reading_of(span: Span, cache: ReadableCache | None, model_id: str) -> Reading:
    """Decide with which keys and which notes a digest reads its span.

    The key of a period does not hold the notes of earlier periods, but the
    prompt does. So a digest uses the keys of a report only when its prompts
    are the prompts of a report: the span starts at the first period, or the
    cache holds every period before the span, and their notes are sent.

    In every other case the digest sends no notes from before the span, and
    its periods have keys of their own, made from `own_version`. A digest of
    a span with the same start finds them. A report, and a digest of a span
    with another start, never do."""
    plan = span.plan
    notes = _notes_before(cache, span.earlier)
    if notes is not None:
        return Reading(plan=plan, notes=notes, prompt_version=MAP_PROMPT_VERSION, shared=True)
    version = own_version(plan.periods[0].period.label)
    own = tuple(
        replace(planned, key=period_key(version, model_id, planned.labels, planned.transcript))
        for planned in plan.periods
    )
    return Reading(plan=replace(plan, periods=own), notes=(), prompt_version=version, shared=False)


def checked_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ConfigError("max_new_periods must be a whole number, zero or more, or None")
    return limit


def check_limit(limit: int | None, plan: Plan, cache: ReadableCache | None) -> None:
    """Refuse a digest that would make more calls than the caller allows."""
    if limit is None:
        return
    keys = [planned.key for planned in plan.periods]
    known = known_keys(cache, keys)
    cached = sum(key in known for key in keys)
    if len(keys) - cached > limit:
        raise DigestTooLarge(len(keys), cached, limit)
