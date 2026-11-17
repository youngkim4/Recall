"""Write the report on one conversation.

    prepare -> map (one call per period, oldest first) -> select -> reduce

A run reads the messages once, calls the model only for periods that are not
in the cache, checks every date and quote against the messages, and returns a
Report. It writes nothing but cache entries; saving the report is the caller's
step (`ReportStore.save`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC
from typing import Any

from report_engine.calls import Caller, Emit
from report_engine.control import CancelToken, ProgressCallback, ProgressEvent, Runtime
from report_engine.errors import AllPeriodsFailed, ReduceError, RunError
from report_engine.jsonsafe import jsonable
from report_engine.map_step import MapResult, PeriodOutcome, PeriodStore, map_periods
from report_engine.model_info import ModelInfo
from report_engine.models import PeriodSummary, Report, Scope
from report_engine.options import ReportOptions
from report_engine.plan import Plan, build_plan
from report_engine.ports import LLM, Labeler, ReportSource
from report_engine.reduce_step import write_sections
from report_engine.selection import select_moments
from report_engine.tokens import TokenCounter
from report_engine.verify import Checked, MessageIndex, check_moments

logger = logging.getLogger(__name__)


def _emitter(callback: ProgressCallback | None) -> Emit:
    """Progress goes to foreign code. Its errors are logged, not raised."""

    def emit(event: ProgressEvent) -> None:
        if callback is None:
            return
        try:
            callback(event)
        except Exception:  # noqa: BLE001 - a broken listener must not end a paid run
            logger.warning("progress callback raised; the run goes on", exc_info=True)

    return emit


def _failure_warning(outcome: PeriodOutcome) -> str:
    return f'Period "{outcome.planned.period.label}" was not analysed: {outcome.error}'


def _counted(count: int, one: str, many: str) -> list[str]:
    if count == 0:
        return []
    return [f"1 {one}" if count == 1 else f"{count} {many}"]


def _check_warnings(checks: list[Checked], skipped: int) -> list[str]:
    cleared = sum(check.quotes_cleared for check in checks)
    repaired = sum(check.dates_repaired for check in checks)
    dropped = sum(check.dropped for check in checks) + skipped
    return [
        *_counted(
            cleared,
            "quote was not found in the messages and was removed.",
            "quotes were not found in the messages and were removed.",
        ),
        *_counted(
            repaired,
            "moment had a date outside its period; the date was set from the cited messages.",
            "moments had a date outside their period; the date was set from the cited messages.",
        ),
        *_counted(
            dropped,
            "moment from the model could not be used and was dropped.",
            "moments from the model could not be used and were dropped.",
        ),
    ]


def _summary(outcome: PeriodOutcome, checked: Checked | None) -> PeriodSummary:
    period = outcome.planned.period
    digest = outcome.digest
    return PeriodSummary(
        label=period.label,
        first_day=period.first_day,
        last_day=period.last_day,
        summary=digest.summary if digest else "",
        carry_over=digest.carry_over if digest else "",
        moments=checked.moments if checked else (),
        usage=outcome.usage,
        cache_key=outcome.planned.key,
        from_cache=outcome.from_cache,
        error=outcome.error,
        message_count=len(outcome.planned.lines),
    )


def _summaries(plan: Plan, mapped: MapResult) -> tuple[tuple[PeriodSummary, ...], list[str]]:
    """Verify each period's moments against the whole conversation."""
    index = MessageIndex(plan.lines)
    checks = [
        check_moments(outcome.digest.moments, outcome.planned.lines, index)
        if outcome.digest
        else None
        for outcome in mapped.outcomes
    ]
    summaries = tuple(
        _summary(outcome, check) for outcome, check in zip(mapped.outcomes, checks, strict=True)
    )
    skipped = sum(outcome.digest.skipped for outcome in mapped.outcomes if outcome.digest)
    return summaries, _check_warnings([check for check in checks if check is not None], skipped)


def _after_prepare(
    *,
    plan: Plan,
    caller: Caller,
    cache: PeriodStore | None,
    model_info: ModelInfo,
    conversation: str,
    options: ReportOptions,
    stats: dict[str, Any],
    clock: Runtime,
    emit: Emit,
) -> Report:
    """Map, check, select, reduce and put the report together."""
    total = len(plan.periods)
    mapped = map_periods(
        plan=plan,
        caller=caller,
        cache=cache,
        model_info=model_info,
        conversation=conversation,
        notes_budget=options.carry_over_token_budget,
        emit=emit,
    )
    failures = [_failure_warning(outcome) for outcome in mapped.outcomes if outcome.digest is None]
    if len(failures) == total:
        raise AllPeriodsFailed(
            f"no period could be analysed; the first failure: {failures[0]}", usage=caller.usage
        )

    periods, check_warnings = _summaries(plan, mapped)
    moments = select_moments(
        [period.moments for period in periods],
        score_floor=options.score_floor,
        max_moments=options.max_moments,
    )
    try:
        reduced = write_sections(plan=plan, caller=caller, periods=periods, moments=moments)
    except ReduceError as exc:
        exc.usage = caller.usage
        emit(ProgressEvent(step="reduce", state="failed", total=total))
        raise
    emit(ProgressEvent(step="reduce", state="done", total=total))

    created = clock.now()
    return Report(
        id=clock.new_id(created),
        conversation=conversation,
        scope=plan.scope,
        model=model_info.id,
        created_at=created.astimezone(UTC).isoformat(timespec="seconds"),
        status="partial" if failures else "complete",
        warnings=(*failures, *mapped.warnings, *check_warnings, *reduced.warnings),
        sections=reduced.sections,
        moments=moments,
        periods=periods,
        usage=caller.usage,
        stats=stats,
    )


def run_report(
    *,
    source: ReportSource,
    conversation: str,
    llm: LLM,
    model_info: ModelInfo,
    labeler: Labeler,
    cache: PeriodStore | None = None,
    scope: Scope | Mapping[str, str | None] | None = None,
    stats: Mapping[str, Any] | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
    on_progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    runtime: Runtime | None = None,
) -> Report:
    """Run one report. A returned Report always has all four sections.

    Before any model call it can raise EmptyConversationError, ConfigError or
    SourceError. After that it raises only RunError and its kinds
    (ReportCancelled, AllPeriodsFailed, ReduceError), each with the `usage` of
    the run so far. Period results that were paid for stay in the cache."""
    chosen = options or ReportOptions()
    clock = runtime or Runtime()
    safe_stats = jsonable(dict(stats or {}))
    emit = _emitter(on_progress)
    caller = Caller(
        llm=llm,
        model_info=model_info,
        runtime=clock,
        cancel=cancel,
        emit=emit,
        reasoning=chosen.reasoning,
    )
    caller.check_cancel()

    emit(ProgressEvent(step="prepare", state="start"))
    plan = build_plan(
        source=source,
        conversation=conversation,
        scope=scope,
        labeler=labeler,
        model_info=model_info,
        options=chosen,
        counter=counter,
    )
    emit(ProgressEvent(step="prepare", state="done", total=len(plan.periods)))

    try:
        report = _after_prepare(
            plan=plan,
            caller=caller,
            cache=cache,
            model_info=model_info,
            conversation=conversation,
            options=chosen,
            stats=safe_stats,
            clock=clock,
            emit=emit,
        )
    except RunError:
        raise
    except Exception as exc:
        raise RunError(
            f"the run stopped on an unexpected error: {type(exc).__name__}", usage=caller.usage
        ) from exc
    emit(ProgressEvent(step="done", state="done", total=len(plan.periods)))
    return report
