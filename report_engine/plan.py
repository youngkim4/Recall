"""Read a conversation and lay out the periods of a run.

`estimate` and `run_report` both start here, so they agree on where periods
end and on every cache key.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from report_engine.cache import period_key
from report_engine.chunking import Period, line_sizer, measure_days, plan_periods
from report_engine.errors import ConfigError, EmptyConversationError
from report_engine.figures import conversation_figures
from report_engine.model_info import ModelInfo
from report_engine.models import Scope
from report_engine.options import ReportOptions
from report_engine.ports import Labeler, ReportSource
from report_engine.prompts import (
    MAP_PROMPT_VERSION,
    map_overhead_text,
    notes_frame_text,
    reduce_overhead_text,
)
from report_engine.scope import coerce_scope
from report_engine.tokens import TokenCounter, TokenMeter, meter_for
from report_engine.transcript import Line, build_lines, render_period

logger = logging.getLogger(__name__)

MIN_PERIOD_LINE_TOKENS = 60


@dataclass(frozen=True)
class Limits:
    """Token limits for this model and these options, in the meter's units."""

    transcript_budget: int
    map_overhead: int
    map_max_output: int
    reduce_max_output: int
    map_input_cap: int
    reduce_input_cap: int
    reduce_overhead: int


@dataclass(frozen=True)
class PlannedPeriod:
    """One period ready to send. `request_tokens` is the request without notes:
    the transcript, the fixed prompt text and the speaker names."""

    period: Period
    lines: tuple[Line, ...]
    transcript: str
    labels: tuple[str, ...]
    key: str
    request_tokens: int


@dataclass(frozen=True)
class Plan:
    """`names` are all speakers of the conversation; `figures` is the block of
    exact numbers that the reduce step reads."""

    scope: Scope
    lines: tuple[Line, ...]
    periods: tuple[PlannedPeriod, ...]
    meter: TokenMeter
    limits: Limits
    names: tuple[str, ...]
    figures: str


def _input_cap(model_info: ModelInfo, max_output: int) -> int:
    """Largest input one request may carry: the model's limit, the window less
    the reply, and one minute of the rate limit less the reply."""
    caps = [model_info.max_input_tokens, model_info.context_window - max_output]
    if model_info.tokens_per_minute is not None:
        caps.append(model_info.tokens_per_minute - max_output)
    return min(caps)


def resolve_limits(
    model_info: ModelInfo, options: ReportOptions, meter: TokenMeter, names: Sequence[str]
) -> Limits:
    """Limits for one conversation. A map request is the transcript, the fixed
    prompt text, the speaker names and the notes: the transcript budget is what
    is left of the input cap when the other three are at their largest."""
    map_out = min(options.map_max_output_tokens, model_info.max_output_tokens)
    reduce_out = min(options.reduce_max_output_tokens, model_info.max_output_tokens)
    map_text, notes_text, reduce_text, name_text = meter.count_many(
        [map_overhead_text(), notes_frame_text(), reduce_overhead_text(), ", ".join(names)]
    )
    map_cap = _input_cap(model_info, map_out)
    reduce_cap = _input_cap(model_info, reduce_out)
    room = map_cap - map_text - name_text - notes_text - options.carry_over_token_budget
    if room <= 0:
        raise ConfigError(
            f"model {model_info.id} leaves no room for a period: input cap {map_cap} tokens, "
            f"prompt {map_text + name_text + notes_text}, notes {options.carry_over_token_budget}"
        )
    return Limits(
        transcript_budget=min(options.period_token_budget, room),
        map_overhead=map_text,
        map_max_output=map_out,
        reduce_max_output=reduce_out,
        map_input_cap=map_cap,
        reduce_input_cap=reduce_cap,
        reduce_overhead=reduce_text + name_text,
    )


def _check_reduce_fits(limits: Limits, figure_tokens: int, periods: int, model_id: str) -> None:
    """The reduce request holds one line for each period. When even the shortest
    lines can not fit, say so before a single call is paid for."""
    least = limits.reduce_overhead + figure_tokens + periods * MIN_PERIOD_LINE_TOKENS
    if least > limits.reduce_input_cap:
        raise ConfigError(
            f"model {model_id} leaves no room to write the report from {periods} periods: "
            f"the request needs {least} tokens at least, the limit is {limits.reduce_input_cap}. "
            "Use a smaller reduce_max_output_tokens or a larger period_token_budget."
        )


def _planned_periods(
    periods: Sequence[Period],
    lines: tuple[Line, ...],
    model_id: str,
    max_chars: int,
    meter: TokenMeter,
    overhead: int,
) -> tuple[PlannedPeriod, ...]:
    sliced = [lines[period.start : period.stop] for period in periods]
    rendered = [render_period(own, max_chars) for own in sliced]
    name_tokens = meter.count_many([", ".join(item.labels) for item in rendered])
    return tuple(
        PlannedPeriod(
            period=period,
            lines=own,
            transcript=item.text,
            labels=item.labels,
            key=period_key(MAP_PROMPT_VERSION, model_id, item.labels, item.text),
            request_tokens=period.tokens + overhead + names,
        )
        for period, own, item, names in zip(periods, sliced, rendered, name_tokens, strict=True)
    )


def build_plan(
    *,
    source: ReportSource,
    conversation: str,
    scope: Scope | Mapping[str, str | None] | None,
    labeler: Labeler,
    model_info: ModelInfo,
    options: ReportOptions,
    counter: TokenCounter | None = None,
) -> Plan:
    """Read the messages once and split them. Raises EmptyConversationError when
    the scope holds no message with text."""
    wanted = coerce_scope(scope)
    lines = build_lines(source.messages(conversation, wanted.since, wanted.until), labeler)
    if not lines:
        raise EmptyConversationError("no message with text in this conversation and scope")
    meter = meter_for(model_info, counter, options.tokenizer_mismatch_margin)
    names = tuple(sorted({line.label for line in lines}))
    limits = resolve_limits(model_info, options, meter, names)
    days = measure_days(lines, meter, options.max_message_chars)
    periods = plan_periods(
        days, limits.transcript_budget, line_sizer(lines, meter, options.max_message_chars)
    )
    figures = conversation_figures(lines)
    _check_reduce_fits(limits, meter.count(figures), len(periods), model_info.id)
    logger.info(
        "%d line%s in %d period%s of at most %d tokens, counted with %s",
        len(lines),
        "" if len(lines) == 1 else "s",
        len(periods),
        "" if len(periods) == 1 else "s",
        limits.transcript_budget,
        meter.describe(),
    )
    return Plan(
        scope=wanted,
        lines=lines,
        periods=_planned_periods(
            periods, lines, model_info.id, options.max_message_chars, meter, limits.map_overhead
        ),
        meter=meter,
        limits=limits,
        names=names,
        figures=figures,
    )
