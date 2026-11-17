"""The report as data. One shape for every size of conversation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from report_engine.dates import DAY_LENGTH, parse_day, parse_stamp

SCHEMA_VERSION = 1

SectionKey = Literal["arc", "dynamics", "defining_moments", "big_picture"]
SECTION_KEYS: tuple[SectionKey, ...] = ("arc", "dynamics", "defining_moments", "big_picture")
SECTION_TITLES: dict[SectionKey, str] = {
    "arc": "The Arc",
    "dynamics": "Dynamics",
    "defining_moments": "Defining Moments",
    "big_picture": "The Big Picture",
}

Category = Literal[
    "milestone", "conflict", "reconciliation", "turning_point", "intimacy", "external_event"
]
CATEGORIES: tuple[Category, ...] = (
    "milestone",
    "conflict",
    "reconciliation",
    "turning_point",
    "intimacy",
    "external_event",
)

ReportStatus = Literal["complete", "partial"]


class _Record(BaseModel):
    """Immutable; unknown fields from a newer writer are dropped on read."""

    model_config = ConfigDict(frozen=True, extra="ignore", allow_inf_nan=False)


def _check_bound(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) == DAY_LENGTH:
        parse_day(value)
    else:
        parse_stamp(value)
    return value


class Scope(_Record):
    """The dates a report covers. A day means the whole day; both ends are included."""

    since: str | None = None
    until: str | None = None

    _since_ok = field_validator("since")(_check_bound)
    _until_ok = field_validator("until")(_check_bound)

    @model_validator(mode="after")
    def _ordered(self) -> "Scope":
        if self.since and self.until and self.since[:DAY_LENGTH] > self.until[:DAY_LENGTH]:
            raise ValueError("scope: since is after until")
        return self


class Usage(_Record):
    """Tokens and cost. `input_tokens` includes the cached ones."""

    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    calls: int = Field(default=0, ge=0)

    def plus(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )


class Section(_Record):
    key: SectionKey
    title: str
    markdown: str


class Moment(_Record):
    """One moment. `message_ids` open the exchange; `quote` is empty unless verified."""

    date: str
    title: str
    detail: str
    category: Category
    score: float = Field(ge=0, le=1)
    quote: str = ""
    message_ids: tuple[str, ...] = ()
    quote_verified: bool = False

    @field_validator("date")
    @classmethod
    def _date_is_a_day(cls, value: str) -> str:
        parse_day(value)
        return value


class PeriodSummary(_Record):
    """What the map step found in one period. `error` is set when the period failed."""

    label: str
    first_day: str
    last_day: str
    summary: str = ""
    carry_over: str = ""
    moments: tuple[Moment, ...] = ()
    usage: Usage = Usage()
    cache_key: str = ""
    from_cache: bool = False
    error: str | None = None
    message_count: int = Field(default=0, ge=0)


class Report(_Record):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(min_length=1)
    conversation: str
    scope: Scope = Scope()
    model: str
    created_at: str
    status: ReportStatus
    warnings: tuple[str, ...] = ()
    sections: tuple[Section, ...]
    moments: tuple[Moment, ...] = ()
    periods: tuple[PeriodSummary, ...] = ()
    usage: Usage = Usage()
    stats: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_iso(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _fixed_sections(self) -> "Report":
        keys = tuple(section.key for section in self.sections)
        if keys != SECTION_KEYS:
            raise ValueError(f"sections must be {SECTION_KEYS} in this order, got {keys}")
        return self
