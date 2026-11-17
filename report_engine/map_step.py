"""Map step: one call per period, oldest first.

A period whose cache entry exists costs no call. Every other period waits for
its turn, because it reads the notes of the periods before it. A period that
fails is reported and skipped; it leaves no notes and no cache entry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from report_engine.cache import CacheEntry
from report_engine.calls import Caller, Emit, usage_of
from report_engine.control import ProgressEvent
from report_engine.digest import DigestError, PeriodDigest, digest_payload, parse_digest
from report_engine.errors import LLMError, StoreError
from report_engine.model_info import ModelInfo
from report_engine.models import Usage
from report_engine.plan import Plan, PlannedPeriod
from report_engine.ports import LLMResult, LLMUsage
from report_engine.prompts import MAP_PROMPT_VERSION, MAP_SCHEMA, Note, map_messages, notes_block
from report_engine.tokens import TokenMeter

logger = logging.getLogger(__name__)

STATUS_REASONS = {
    "incomplete": "the reply was cut off at the output limit",
    "refused": "the model refused to answer",
    "failed": "the model reported a failure",
}


class PeriodStore(Protocol):
    """What the map step needs from a cache."""

    def get_many(self, keys: Sequence[str], touch: bool = False) -> Mapping[str, CacheEntry]: ...

    def put(
        self, entry: CacheEntry, *, conversation: str, label: str, model: str, prompt_version: str
    ) -> None: ...


@dataclass(frozen=True)
class PeriodOutcome:
    """What one period gave. `digest` is None when the period failed."""

    planned: PlannedPeriod
    digest: PeriodDigest | None
    usage: Usage = Usage()
    from_cache: bool = False
    error: str | None = None


@dataclass(frozen=True)
class MapResult:
    outcomes: tuple[PeriodOutcome, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _SizedNote:
    note: Note
    tokens: int


def _fitting_notes(notes: Sequence[_SizedNote], budget: int) -> tuple[Note, ...]:
    """The newest notes that fit the budget, oldest first."""
    kept: list[Note] = []
    total = 0
    for sized in reversed(notes):
        if total + sized.tokens > budget:
            break
        kept.append(sized.note)
        total += sized.tokens
    return tuple(reversed(kept))


def _read_cache(cache: PeriodStore | None, keys: Sequence[str], warnings: list[str]) -> Mapping[str, CacheEntry]:
    if cache is None:
        return {}
    try:
        return cache.get_many(keys, touch=True)
    except StoreError as exc:
        warnings.append(f"The period cache could not be read; every period was run again ({exc}).")
        return {}


def _from_cache(planned: PlannedPeriod, entry: CacheEntry | None) -> PeriodOutcome | None:
    if entry is None:
        return None
    try:
        digest = parse_digest(entry.payload)
    except DigestError:
        logger.warning("cache entry for period %s does not parse; it is run again", planned.period.label)
        return None
    return PeriodOutcome(planned=planned, digest=digest, usage=_recorded_usage(entry), from_cache=True)


def _recorded_usage(entry: CacheEntry) -> Usage:
    """What the call cost when it was made. The price of that day stays."""
    input_tokens = max(0, entry.usage.input_tokens)
    return Usage(
        input_tokens=input_tokens,
        cached_input_tokens=min(max(0, entry.usage.cached_input_tokens), input_tokens),
        output_tokens=max(0, entry.usage.output_tokens),
        cost_usd=max(0.0, entry.cost_usd),
        calls=1,
    )


def _digest_of(result: LLMResult) -> PeriodDigest:
    if result.status != "completed":
        raise DigestError(STATUS_REASONS.get(result.status, f"the reply has status {result.status}"))
    return parse_digest(result.parsed)


def _run_period(
    planned: PlannedPeriod,
    notes: tuple[Note, ...],
    plan: Plan,
    caller: Caller,
    model_info: ModelInfo,
    events: Mapping[str, ProgressEvent],
) -> PeriodOutcome:
    """Call the model for one period. A failure comes back as an outcome, not an error."""
    limits = plan.limits
    estimate = planned.request_tokens + plan.meter.count(notes_block(notes))
    if estimate > limits.map_input_cap:
        reason = f"the period is too large for the model ({estimate} tokens, the limit is {limits.map_input_cap})"
        return PeriodOutcome(planned=planned, digest=None, error=reason)
    try:
        result = caller.send(
            map_messages(planned.transcript, planned.labels, notes),
            schema=MAP_SCHEMA,
            max_output_tokens=limits.map_max_output,
            input_estimate=estimate,
            waiting=events["waiting"],
            starting=events["start"],
        )
    except LLMError as exc:
        return PeriodOutcome(planned=planned, digest=None, error=str(exc) or type(exc).__name__)
    spent = usage_of(model_info, result.usage)
    try:
        digest = _digest_of(result)
    except DigestError as exc:
        return PeriodOutcome(planned=planned, digest=None, usage=spent, error=str(exc))
    return PeriodOutcome(planned=planned, digest=digest, usage=spent)


def _save(
    cache: PeriodStore | None, outcome: PeriodOutcome, conversation: str, model_info: ModelInfo
) -> str | None:
    """Store a fresh result. Returns the reason when the cache refuses it."""
    if cache is None or outcome.digest is None:
        return None
    entry = CacheEntry(
        key=outcome.planned.key,
        payload=digest_payload(outcome.digest),
        usage=_llm_usage(outcome.usage),
        cost_usd=outcome.usage.cost_usd,
    )
    try:
        cache.put(
            entry,
            conversation=conversation,
            label=outcome.planned.period.label,
            model=model_info.id,
            prompt_version=MAP_PROMPT_VERSION,
        )
    except StoreError as exc:
        return str(exc)
    return None


def _llm_usage(usage: Usage) -> LLMUsage:
    return LLMUsage(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
    )


def _events(planned: PlannedPeriod, index: int, total: int) -> dict[str, ProgressEvent]:
    def event(state: str) -> ProgressEvent:
        return ProgressEvent(step="map", state=state, period=planned.period.label, index=index, total=total)  # type: ignore[arg-type]

    return {state: event(state) for state in ("start", "cached", "waiting", "done", "failed")}


def map_periods(
    *,
    plan: Plan,
    caller: Caller,
    cache: PeriodStore | None,
    model_info: ModelInfo,
    conversation: str,
    notes_budget: int,
    emit: Emit,
) -> MapResult:
    """Run every period in order. Raises ReportCancelled; never LLMError."""
    warnings: list[str] = []
    cached = _read_cache(cache, [planned.key for planned in plan.periods], warnings)
    notes: list[_SizedNote] = []
    outcomes: list[PeriodOutcome] = []
    unsaved: list[str] = []
    total = len(plan.periods)
    for index, planned in enumerate(plan.periods, start=1):
        caller.check_cancel()
        events = _events(planned, index, total)
        outcome = _from_cache(planned, cached.get(planned.key))
        if outcome is not None:
            emit(events["cached"])
        else:
            outcome = _run_period(
                planned, _fitting_notes(notes, notes_budget), plan, caller, model_info, events
            )
            refused = _save(cache, outcome, conversation, model_info)
            if refused is not None:
                unsaved.append(refused)
            emit(events["done"] if outcome.digest is not None else events["failed"])
        outcomes.append(outcome)
        if outcome.digest is not None and outcome.digest.carry_over:
            note = (planned.period.label, outcome.digest.carry_over)
            notes.append(_SizedNote(note, plan.meter.count(f"- {note[0]}: {note[1]}")))
    if unsaved:
        many = len(unsaved)
        warnings.append(
            f"{many} period result{'s' if many != 1 else ''} could not be saved to the cache ({unsaved[0]})."
        )
    return MapResult(outcomes=tuple(outcomes), warnings=tuple(warnings))
