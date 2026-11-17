"""Choices a caller can make for a run. None of them is a model fact."""

from __future__ import annotations

import math
from dataclasses import dataclass

from report_engine.errors import ConfigError
from report_engine.transcript import MAX_MESSAGE_CHARS

DEFAULT_PERIOD_TOKEN_BUDGET = 80_000
POSITIVE_COUNTS = (
    "period_token_budget",
    "map_max_output_tokens",
    "reduce_max_output_tokens",
    "max_message_chars",
)
COUNTS_FROM_ZERO = (
    "max_moments",
    "carry_over_token_budget",
    "map_output_estimate_tokens",
    "reduce_output_estimate_tokens",
)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class ReportOptions:
    """Defaults aim at reading quality, not at filling the context window.

    The same options must go to `estimate` and to `run_report`: the period
    budget and the message cap decide where periods end, and so which cache
    entries match.
    """

    period_token_budget: int = DEFAULT_PERIOD_TOKEN_BUDGET
    map_max_output_tokens: int = 16_000
    reduce_max_output_tokens: int = 24_000
    reasoning: str | None = None
    score_floor: float = 0.4
    max_moments: int = 40
    max_message_chars: int = MAX_MESSAGE_CHARS
    carry_over_token_budget: int = 4_000
    tokenizer_mismatch_margin: float = 1.35
    map_output_estimate_tokens: int = 4_000
    reduce_output_estimate_tokens: int = 8_000

    def __post_init__(self) -> None:
        for name in POSITIVE_COUNTS:
            if not _is_count(getattr(self, name)) or getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be a whole number above zero")
        for name in COUNTS_FROM_ZERO:
            if not _is_count(getattr(self, name)) or getattr(self, name) < 0:
                raise ConfigError(f"{name} must be a whole number, zero or more")
        if not _is_number(self.score_floor) or not 0.0 <= self.score_floor <= 1.0:
            raise ConfigError("score_floor must be a number between 0 and 1")
        if not _is_number(self.tokenizer_mismatch_margin) or self.tokenizer_mismatch_margin < 1.0:
            raise ConfigError("tokenizer_mismatch_margin must be a number, 1.0 or more")
        if self.reasoning is not None and (
            not isinstance(self.reasoning, str) or not self.reasoning
        ):
            raise ConfigError("reasoning must be a word such as low, or None")
