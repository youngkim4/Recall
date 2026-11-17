"""Model facts. The engine holds no model id, price or limit of its own."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LongContextRule(BaseModel):
    """Price change for a request whose input is larger than the threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    threshold_tokens: int = Field(gt=0)
    input_multiplier: float = Field(gt=0)
    output_multiplier: float = Field(gt=0)
    cached_input_multiplier: float | None = Field(default=None, gt=0)


class ModelInfo(BaseModel):
    """Everything the engine knows about one model. Prices are USD per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1)
    context_window: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    input_usd_per_mtok: float = Field(ge=0)
    cached_input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    long_context: LongContextRule | None = None
    tokens_per_minute: int | None = Field(default=None, gt=0)
    tokenizer_family: str = Field(min_length=1)

    @model_validator(mode="after")
    def _input_fits_window(self) -> ModelInfo:
        if self.max_input_tokens > self.context_window:
            raise ValueError("max_input_tokens is larger than context_window")
        return self
