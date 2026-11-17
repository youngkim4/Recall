"""What a run would send and cost, before it runs.

The estimate lays out the same periods as `run_report` and looks each one up
in the period cache. A period that is in the cache is not counted: its tokens
go to `cached_tokens`. The reduce call is always counted.

Counting uses the engine's tokenizer (see `tokens`). When that is not the
model's tokenizer family, every count carries the mismatch margin from the
options. Reply sizes can not be counted; they come from the options and are
capped by the model's output limit.

Pass the same labeler, options and counter that the run will get. Other
values give other periods and other cache keys.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from report_engine.errors import EmptyConversationError, StoreError
from report_engine.model_info import ModelInfo
from report_engine.models import Scope
from report_engine.options import ReportOptions
from report_engine.plan import Plan, build_plan
from report_engine.ports import Labeler, LLMUsage, ReportSource
from report_engine.pricing import call_cost_usd
from report_engine.prompts import reduce_overhead_text
from report_engine.tokens import TokenCounter

logger = logging.getLogger(__name__)

# sizes the estimate can not know yet, in the engine tokenizer's tokens
NOTE_TOKENS = 110
SUMMARY_TOKENS = 400
MOMENT_TOKENS = 60
FIGURES_TOKENS = 120
MOMENTS_PER_PERIOD = 4


class KnownKeys(Protocol):
    """What the estimate needs from a cache."""

    def known(self, keys: Sequence[str]) -> frozenset[str]: ...


@dataclass(frozen=True)
class PlannedCall:
    """One model call a run would make, or would skip because it is cached."""

    kind: Literal["map", "reduce"]
    label: str
    input_tokens: int
    output_tokens: int
    cached: bool = False


def _known(cache: KnownKeys | None, keys: Sequence[str]) -> frozenset[str]:
    if cache is None:
        return frozenset()
    try:
        return cache.known(keys)
    except StoreError as exc:
        logger.warning("the period cache could not be read for the estimate: %s", exc)
        return frozenset()


def _calls(plan: Plan, options: ReportOptions, known: frozenset[str]) -> tuple[PlannedCall, ...]:
    limits = plan.limits

    def padded(tokens: int) -> int:
        return math.ceil(tokens * plan.meter.margin)

    map_output = min(options.map_output_estimate_tokens, limits.map_max_output)
    maps = [
        PlannedCall(
            kind="map",
            label=planned.period.label,
            input_tokens=planned.request_tokens
            + min(options.carry_over_token_budget, padded(NOTE_TOKENS) * position),
            output_tokens=map_output,
            cached=planned.key in known,
        )
        for position, planned in enumerate(plan.periods)
    ]
    count = len(plan.periods)
    reduce_input = (
        plan.meter.count(reduce_overhead_text())
        + padded(FIGURES_TOKENS)
        + count * padded(SUMMARY_TOKENS)
        + min(options.max_moments, count * MOMENTS_PER_PERIOD) * padded(MOMENT_TOKENS)
    )
    reduce = PlannedCall(
        kind="reduce",
        label="reduce",
        input_tokens=reduce_input,
        output_tokens=min(options.reduce_output_estimate_tokens, limits.reduce_max_output),
    )
    return (*maps, reduce)


def planned_calls(
    source: ReportSource,
    conversation: str,
    scope: Scope | Mapping[str, str | None] | None,
    model_info: ModelInfo,
    cache: KnownKeys | None,
    *,
    labeler: Labeler,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
) -> tuple[PlannedCall, ...]:
    """Every call of a run, in order. Empty when the scope holds no message."""
    chosen = options or ReportOptions()
    try:
        plan = build_plan(
            source=source,
            conversation=conversation,
            scope=scope,
            labeler=labeler,
            model_info=model_info,
            options=chosen,
            counter=counter,
        )
    except EmptyConversationError:
        return ()
    return _calls(plan, chosen, _known(cache, [planned.key for planned in plan.periods]))


def estimate(
    source: ReportSource,
    conversation: str,
    scope: Scope | Mapping[str, str | None] | None,
    model_info: ModelInfo,
    cache: KnownKeys | None,
    *,
    labeler: Labeler,
    options: ReportOptions | None = None,
    counter: TokenCounter | None = None,
) -> dict[str, int | float]:
    """Tokens and cost of a run that starts now. Makes no model call."""
    calls = planned_calls(
        source,
        conversation,
        scope,
        model_info,
        cache,
        labeler=labeler,
        options=options,
        counter=counter,
    )
    to_send = [call for call in calls if not call.cached]
    maps = [call for call in calls if call.kind == "map"]
    return {
        "input_tokens": sum(call.input_tokens for call in to_send),
        "cached_tokens": sum(call.input_tokens for call in calls if call.cached),
        "output_tokens": sum(call.output_tokens for call in to_send),
        "cost_usd": float(
            sum(
                call_cost_usd(
                    model_info,
                    LLMUsage(input_tokens=call.input_tokens, output_tokens=call.output_tokens),
                )
                for call in to_send
            )
        ),
        "periods": len(maps),
        "periods_cached": sum(call.cached for call in maps),
    }
