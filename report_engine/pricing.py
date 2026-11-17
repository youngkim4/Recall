"""Cost of one call, from real usage and ModelInfo prices."""

from __future__ import annotations

from report_engine.model_info import ModelInfo
from report_engine.ports import LLMUsage

TOKENS_PER_PRICE_UNIT = 1_000_000


def call_cost_usd(model_info: ModelInfo, usage: LLMUsage) -> float:
    """Price one call. The long-context rule is judged per call, on its input size."""
    input_tokens = max(0, usage.input_tokens)
    cached = min(max(0, usage.cached_input_tokens), input_tokens)
    uncached = input_tokens - cached
    output_tokens = max(0, usage.output_tokens)

    input_mult = cached_mult = output_mult = 1.0
    rule = model_info.long_context
    if rule is not None and input_tokens > rule.threshold_tokens:
        input_mult = rule.input_multiplier
        output_mult = rule.output_multiplier
        cached_mult = (
            rule.input_multiplier
            if rule.cached_input_multiplier is None
            else rule.cached_input_multiplier
        )

    total = (
        uncached * model_info.input_usd_per_mtok * input_mult
        + cached * model_info.cached_input_usd_per_mtok * cached_mult
        + output_tokens * model_info.output_usd_per_mtok * output_mult
    )
    return total / TOKENS_PER_PRICE_UNIT
