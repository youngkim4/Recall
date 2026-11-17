"""Choices a caller can make for a run. None of them is a model fact."""

from __future__ import annotations

from dataclasses import dataclass

from report_engine.errors import ConfigError

DEFAULT_PERIOD_TOKEN_BUDGET = 80_000


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
    max_message_chars: int = 6_000
    carry_over_token_budget: int = 4_000
    tokenizer_mismatch_margin: float = 1.35
    map_output_estimate_tokens: int = 4_000
    reduce_output_estimate_tokens: int = 8_000

    def __post_init__(self) -> None:
        positive = (
            "period_token_budget",
            "map_max_output_tokens",
            "reduce_max_output_tokens",
            "max_message_chars",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        not_negative = (
            "max_moments",
            "carry_over_token_budget",
            "map_output_estimate_tokens",
            "reduce_output_estimate_tokens",
        )
        for name in not_negative:
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} can not be negative")
        if not 0.0 <= self.score_floor <= 1.0:
            raise ConfigError("score_floor must be between 0 and 1")
        if not self.tokenizer_mismatch_margin >= 1.0:
            raise ConfigError("tokenizer_mismatch_margin must be 1.0 or more")
