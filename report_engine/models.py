"""The report as data. One shape for every size of conversation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from report_engine.dates import DAY_LENGTH, lower_bound, parse_day, parse_stamp, upper_bound
from report_engine.jsonsafe import jsonable

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
    """The dates a report covers. A day means the whole day; both ends are included.
    An unknown field is an error: a misspelled bound must not mean "all dates"."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    since: str | None = None
    until: str | None = None

    _since_ok = field_validator("since")(_check_bound)
    _until_ok = field_validator("until")(_check_bound)

    @model_validator(mode="after")
    def _ordered(self) -> Scope:
        low, high = lower_bound(self.since), upper_bound(self.until)
        if low is not None and high is not None and low > high:
            raise ValueError("scope: since is after until")
        return self


class Usage(_Record):
    """Tokens and cost. `input_tokens` includes the cached ones."""

    input_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    calls: int = Field(default=0, ge=0)

    def plus(self, other: Usage) -> Usage:
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
    """One moment. `message_ids` open the exchange.

    A quote is kept only when a message holds it. Then `quote_verified` is true,
    `quote_message_id` is that message and `quote_speaker` is the name of the
    person who wrote it. With no verified quote all three are empty.
    `title`, `detail` and `quote` are plain text, not markdown.
    """

    date: str
    title: str
    detail: str
    category: Category
    score: float = Field(ge=0, le=1)
    quote: str = ""
    message_ids: tuple[str, ...] = ()
    quote_verified: bool = False
    quote_message_id: str = ""
    quote_speaker: str = ""

    @field_validator("date")
    @classmethod
    def _date_is_a_day(cls, value: str) -> str:
        parse_day(value)
        return value

    @model_validator(mode="after")
    def _quote_has_a_source(self) -> Moment:
        if self.quote_verified:
            if not self.quote.strip() or not self.quote_message_id:
                raise ValueError("a verified quote needs its text and the id of its message")
            if self.quote_message_id not in self.message_ids:
                raise ValueError("the message of a quote must be one of message_ids")
        elif self.quote or self.quote_message_id or self.quote_speaker:
            raise ValueError("a quote that is not verified must be cleared")
        return self


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

    @model_validator(mode="after")
    def _days_in_order(self) -> PeriodSummary:
        if parse_day(self.first_day) > parse_day(self.last_day):
            raise ValueError("first_day is after last_day")
        return self


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
    def _created_at_in_utc(cls, value: str) -> str:
        """One instant, always written in UTC, so that text order is time order."""
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            raise ValueError("created_at needs a UTC offset")
        return moment.astimezone(UTC).isoformat(timespec="seconds")

    @field_validator("stats", mode="before")
    @classmethod
    def _stats_as_json(cls, value: Any) -> Any:
        return jsonable(value) if isinstance(value, dict) else value

    @model_validator(mode="after")
    def _fixed_sections(self) -> Report:
        keys = tuple(section.key for section in self.sections)
        if keys != SECTION_KEYS:
            raise ValueError(f"sections must be {SECTION_KEYS} in this order, got {keys}")
        return self
