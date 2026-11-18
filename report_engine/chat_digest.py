"""A digest of one conversation for chat: the map step of a report, and no more.

    prepare -> map (one call per period, oldest first) -> check

A digest gives the summary of each period and the moments of each period,
with the ids of the messages that back them. It writes no report and makes
no reduce call.

A digest and a report use one period cache. The digest lays out the periods
of the whole conversation with `build_plan`, as a report on all dates does,
and runs them with `map_periods`.

`since` and `until` do not cut the conversation. They select the periods
that overlap the span, and each selected period is read whole.

A digest has the cache keys of a report when its prompts are the prompts of
a report: the span starts at the first period, or the cache holds every
period before the span. Then a period that one of them read costs the other
no call. In every other case the digest sends no notes from before the span
and keeps entries of its own, which no report reads (`reading_of`).

Every moment passes `period_summaries`, the check of a report. The cache
holds a reply as the model gave it, so the check runs on each read. A digest
then gives only messages of the period of the moment, and text with no line
reference of the engine (`chat_digest_refs`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from report_engine.calls import Caller, Emit
from report_engine.chat_digest_refs import without_refs
from report_engine.chat_digest_span import (
    DigestTooLarge,
    Reading,
    check_limit,
    checked_limit,
    reading_of,
    span_plan,
)
from report_engine.control import CancelToken, ProgressCallback, ProgressEvent, Runtime
from report_engine.errors import (
    AllPeriodsFailed,
    CallLimitReached,
    EmptyConversationError,
    RunError,
)
from report_engine.estimate import ReadableCache, known_keys, map_calls
from report_engine.map_step import PeriodStore, map_periods
from report_engine.model_info import ModelInfo
from report_engine.models import Moment, PeriodSummary, Usage
from report_engine.options import ReportOptions
from report_engine.pipeline import emitter, period_summaries
from report_engine.ports import LLM, Labeler, LLMUsage, ReportSource
from report_engine.pricing import call_cost_usd
from report_engine.tokens import TokenCounter
from report_engine.transcript import Line

__all__ = [
    "ConversationDigest",
    "DigestEstimate",
    "DigestTooLarge",
    "MomentNote",
    "PeriodNote",
    "conversation_digest",
    "estimate_digest",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MomentNote:
    """One moment of a period. `quote` is in a message of `message_ids`, or it
    is empty. `message_ids` are ids of the source. Each one passed the check
    and is a message of the period of the moment."""

    date: str
    title: str
    detail: str
    quote: str
    message_ids: tuple[str, ...]


@dataclass(frozen=True)
class PeriodNote:
    """What the map step found in one period. `messages` counts its messages with text."""

    label: str
    first_day: str
    last_day: str
    messages: int
    summary: str
    moments: tuple[MomentNote, ...]


@dataclass(frozen=True)
class ConversationDigest:
    """The periods that could be read, oldest first. `periods_built` counts the
    periods that a model call of this run read, `periods_cached` the periods
    from the cache, and `periods_failed` the periods that gave no result: they
    are not in `periods`. `usage` is the tokens and the cost of this run."""

    conversation_id: str
    periods: tuple[PeriodNote, ...]
    periods_built: int
    periods_cached: int
    periods_failed: int
    usage: Usage


@dataclass(frozen=True)
class DigestEstimate:
    """What a digest that starts now would send. A period in the cache is
    counted in `periods_cached` and adds no tokens and no cost."""

    periods: int
    periods_cached: int
    input_tokens: int
    output_tokens: int
    usd: float


def _moment_note(moment: Moment, own: frozenset[str]) -> MomentNote:
    """A moment with the messages of its own period only. The check of a
    report finds a quote up to three days from the moment, also in a message
    of another period. A digest does not give that message, and then it does
    not give the quote."""
    kept = tuple(mid for mid in moment.message_ids if mid in own)
    return MomentNote(
        date=moment.date,
        title=without_refs(moment.title),
        detail=without_refs(moment.detail),
        quote=moment.quote if moment.quote_message_id in kept else "",
        message_ids=kept,
    )


def _period_note(period: PeriodSummary, lines: Sequence[Line]) -> PeriodNote:
    """A period for chat. Its text holds no line reference of the engine: chat
    has references of the same form for other messages. The cache keeps them.
    A moment whose title was a reference only has no title, and is not given."""
    own = frozenset(line.id for line in lines)
    moments = (_moment_note(moment, own) for moment in period.moments)
    return PeriodNote(
        label=period.label,
        first_day=period.first_day,
        last_day=period.last_day,
        messages=period.message_count,
        summary=without_refs(period.summary),
        moments=tuple(moment for moment in moments if moment.title),
    )


def _quotes(moments: Iterable[Moment | MomentNote]) -> int:
    return sum(bool(moment.quote) for moment in moments)


def _log_warnings(of_cache: Sequence[str], of_checks: Sequence[str], outside: int) -> None:
    """A digest has no place for warnings. The log gets them as counts only."""
    if of_cache:
        logger.warning("the period cache failed during a digest (%d warnings)", len(of_cache))
    for line in of_checks:
        logger.info("digest check: %s", line)
    if outside:
        logger.info("digest check: %d quotes stood outside their period and were removed", outside)


def _mapped_digest(
    reading: Reading,
    caller: Caller,
    cache: PeriodStore | None,
    model_info: ModelInfo,
    conversation: str,
    options: ReportOptions,
    emit: Emit,
) -> ConversationDigest:
    """Map the selected periods, check their moments and put the digest together."""
    plan = reading.plan
    mapped = map_periods(
        plan=plan,
        caller=caller,
        cache=cache,
        model_info=model_info,
        conversation=conversation,
        notes_budget=options.carry_over_token_budget,
        emit=emit,
        earlier_notes=reading.notes,
        prompt_version=reading.prompt_version,
    )
    failed = [outcome for outcome in mapped.outcomes if outcome.digest is None]
    if len(failed) == len(mapped.outcomes):
        raise AllPeriodsFailed(
            f"no period could be analysed; the first failure: {failed[0].error}",
            usage=caller.usage,
        )
    summaries, check_warnings = period_summaries(plan, mapped)
    read = [
        (period, planned.lines)
        for period, planned in zip(summaries, plan.periods, strict=True)
        if period.error is None
    ]
    notes = tuple(_period_note(period, lines) for period, lines in read)
    checked = _quotes(moment for period, _ in read for moment in period.moments)
    given = _quotes(moment for note in notes for moment in note.moments)
    _log_warnings(mapped.warnings, check_warnings, checked - given)
    return ConversationDigest(
        conversation_id=conversation,
        periods=notes,
        periods_built=sum(not period.from_cache for period, _ in read),
        periods_cached=sum(period.from_cache for period, _ in read),
        periods_failed=len(failed),
        usage=caller.usage,
    )


def _guarded_digest(
    reading: Reading,
    caller: Caller,
    cache: PeriodStore | None,
    model_info: ModelInfo,
    conversation: str,
    options: ReportOptions,
    emit: Emit,
) -> ConversationDigest:
    """`_mapped_digest`. A run that reaches its limit of calls ends as
    DigestTooLarge. An error that nobody planned for becomes a RunError that
    names its type only and holds the usage of the run."""
    try:
        return _mapped_digest(reading, caller, cache, model_info, conversation, options, emit)
    except CallLimitReached as exc:
        # the cache held less than the check of the limit counted
        total = len(reading.plan.periods)
        cached = max(0, total - exc.limit - 1)
        raise DigestTooLarge(total, cached, exc.limit, usage=caller.usage) from exc
    except RunError:
        raise
    except Exception as exc:
        raise RunError(
            f"the digest stopped on an unexpected error: {type(exc).__name__}", usage=caller.usage
        ) from exc


def conversation_digest(
    *,
    source: ReportSource,
    conversation: str,
    llm: LLM,
    model_info: ModelInfo,
    labeler: Labeler,
    cache: PeriodStore | None = None,
    since: str | None = None,
    until: str | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
    on_progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
    runtime: Runtime | None = None,
    max_new_periods: int | None = None,
) -> ConversationDigest:
    """Build the digest of one conversation. Give it what a report run gets:
    the same source, labeler, model facts, options, counter and cache.

    Before any model call it can raise EmptyConversationError, ConfigError,
    SourceError or DigestTooLarge. After that it raises only RunError and its
    kinds (ReportCancelled, AllPeriodsFailed), each with the `usage` of the
    run so far. Period results that were paid for stay in the cache."""
    chosen = options or ReportOptions()
    limit = checked_limit(max_new_periods)
    emit = emitter(on_progress)
    caller = Caller(
        llm=llm,
        model_info=model_info,
        runtime=runtime or Runtime(),
        cancel=cancel,
        emit=emit,
        reasoning=chosen.reasoning,
        max_calls=limit,
    )
    caller.check_cancel()

    emit(ProgressEvent(step="prepare", state="start"))
    span = span_plan(source, conversation, since, until, labeler, model_info, chosen, counter)
    total = span.stop - span.start
    emit(ProgressEvent(step="prepare", state="done", total=total))
    reading = reading_of(span, cache, model_info.id)
    check_limit(limit, reading.plan, cache)

    digest = _guarded_digest(reading, caller, cache, model_info, conversation, chosen, emit)
    emit(ProgressEvent(step="done", state="done", total=total))
    return digest


def estimate_digest(
    *,
    source: ReportSource,
    conversation: str,
    model_info: ModelInfo,
    labeler: Labeler,
    cache: ReadableCache | None = None,
    since: str | None = None,
    until: str | None = None,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
) -> DigestEstimate:
    """Tokens and cost of a digest that starts now. Makes no model call and
    writes nothing. A conversation or a span with no message gives zeros.

    A period is counted as in the cache with the key that the digest would
    use. The numbers of a period are the numbers of its map call in the
    estimate of a report on all dates, less the notes that a digest of a
    span does not send."""
    chosen = options or ReportOptions()
    try:
        span = span_plan(source, conversation, since, until, labeler, model_info, chosen, counter)
    except EmptyConversationError:
        return DigestEstimate(periods=0, periods_cached=0, input_tokens=0, output_tokens=0, usd=0.0)
    reading = reading_of(span, cache, model_info.id)
    known = known_keys(cache, [planned.key for planned in reading.plan.periods])
    # the notes of earlier periods are counted only when the digest sends them
    if reading.shared:
        calls = map_calls(span.whole, chosen, known)[span.start : span.stop]
    else:
        calls = map_calls(reading.plan, chosen, known)
    to_send = [call for call in calls if not call.cached]
    costs = (
        call_cost_usd(
            model_info, LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens)
        )
        for call in to_send
    )
    return DigestEstimate(
        periods=len(calls),
        periods_cached=len(calls) - len(to_send),
        input_tokens=sum(call.input_tokens for call in to_send),
        output_tokens=sum(call.output_tokens for call in to_send),
        usd=float(sum(costs)),
    )
