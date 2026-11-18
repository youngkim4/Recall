"""Which periods a conversation digest reads, and what it may spend on them.

The periods are the periods of a report on all dates: `build_plan` lays out
the whole conversation, and a span selects the periods that overlap it. A
selected period is read whole, so its text and its cache key are the ones of
the report.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from report_engine.cache import CacheEntry
from report_engine.chunking import Period
from report_engine.dates import DAY_LENGTH
from report_engine.digest import DigestError, parse_digest
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
from report_engine.prompts import Note
from report_engine.scope import coerce_scope
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)


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


def _carry_over(entry: CacheEntry | None) -> str:
    if entry is None:
        return ""
    try:
        return parse_digest(entry.payload).carry_over
    except DigestError:
        return ""


def earlier_notes(
    cache: ReadableCache | None, earlier: Sequence[PlannedPeriod]
) -> tuple[Note, ...]:
    """The notes of the periods before the span, from the cache, oldest first.
    A report gives a period the notes of every period before it. A digest of a
    span runs no period before the span, so it gives what earlier runs left."""
    if cache is None or not earlier:
        return ()
    try:
        entries = cache.get_many([planned.key for planned in earlier])
    except StoreError as exc:
        logger.warning(
            "the period cache could not be read for the notes of earlier periods: %s",
            type(exc).__name__,
        )
        return ()
    notes = [(planned.period.label, _carry_over(entries.get(planned.key))) for planned in earlier]
    return tuple(note for note in notes if note[1])


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
