"""The report shape: one fixed set of sections, checked on the way in."""

import pytest
from pydantic import ValidationError

from report_engine.dates import lower_bound, upper_bound
from report_engine.models import (
    SCHEMA_VERSION,
    SECTION_KEYS,
    SECTION_TITLES,
    Moment,
    PeriodSummary,
    Report,
    Scope,
    Section,
    Usage,
)


def sections() -> tuple[Section, ...]:
    return tuple(
        Section(key=key, title=SECTION_TITLES[key], markdown=f"text for {key}")
        for key in SECTION_KEYS
    )


def report(**overrides: object) -> Report:
    values: dict[str, object] = {
        "id": "20240701T101500Z-abc123",
        "conversation": "+12025550143",
        "scope": Scope(),
        "model": "test-model-1",
        "created_at": "2024-07-01T10:15:00+00:00",
        "status": "complete",
        "sections": sections(),
    }
    values.update(overrides)
    return Report(**values)  # type: ignore[arg-type]


def test_section_keys_are_the_four_fixed_ones() -> None:
    assert SECTION_KEYS == ("arc", "dynamics", "defining_moments", "big_picture")
    assert set(SECTION_TITLES) == set(SECTION_KEYS)


def test_report_defaults() -> None:
    made = report()
    assert made.schema_version == SCHEMA_VERSION
    assert made.warnings == ()
    assert made.moments == ()
    assert made.periods == ()
    assert made.usage == Usage()
    assert made.stats == {}


def test_report_rejects_a_missing_section() -> None:
    with pytest.raises(ValidationError):
        report(sections=sections()[:3])


def test_report_rejects_sections_out_of_order() -> None:
    with pytest.raises(ValidationError):
        report(sections=tuple(reversed(sections())))


def test_report_rejects_an_unknown_status() -> None:
    with pytest.raises(ValidationError):
        report(status="failed")


def test_report_round_trips_through_json() -> None:
    moment = Moment(
        date="2024-06-03",
        title="A plan",
        detail="You two set a date.",
        category="milestone",
        score=0.7,
        quote="see you at the harbor",
        message_ids=("11", "12"),
        quote_verified=True,
        quote_message_id="11",
        quote_speaker="Mina",
    )
    period = PeriodSummary(
        label="Jun 2024",
        first_day="2024-06-01",
        last_day="2024-06-30",
        summary="A calm month.",
        carry_over="A trip is planned.",
        moments=(moment,),
        usage=Usage(input_tokens=10, output_tokens=2, cost_usd=0.01, calls=1),
        cache_key="k" * 64,
    )
    made = report(moments=(moment,), periods=(period,), stats={"totalMessages": 40})
    again = Report.model_validate_json(made.model_dump_json())
    assert again == made


def test_report_ignores_fields_from_a_newer_writer() -> None:
    data = report().model_dump(mode="json")
    data["added_later"] = {"x": 1}
    assert Report.model_validate(data).id == "20240701T101500Z-abc123"


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_moment_score_must_be_between_zero_and_one(score: float) -> None:
    with pytest.raises(ValidationError):
        Moment(date="2024-06-03", title="t", detail="d", category="conflict", score=score)


def test_moment_rejects_an_unknown_category() -> None:
    with pytest.raises(ValidationError):
        Moment(date="2024-06-03", title="t", detail="d", category="gossip", score=0.5)


def test_moment_rejects_a_date_that_is_not_a_day() -> None:
    with pytest.raises(ValidationError):
        Moment(date="June 3rd", title="t", detail="d", category="conflict", score=0.5)


def test_usage_adds_up() -> None:
    first = Usage(input_tokens=5, cached_input_tokens=1, output_tokens=2, cost_usd=0.5, calls=1)
    second = Usage(input_tokens=7, cached_input_tokens=0, output_tokens=3, cost_usd=0.25, calls=2)
    total = first.plus(second)
    assert total == Usage(
        input_tokens=12, cached_input_tokens=1, output_tokens=5, cost_usd=0.75, calls=3
    )
    assert first.input_tokens == 5


@pytest.mark.parametrize(
    "since, until",
    [("2024-6-1", None), (None, "tomorrow"), ("2024-02-30", None), ("2024-06-10", "2024-06-01")],
)
def test_scope_rejects_bad_dates(since: str | None, until: str | None) -> None:
    with pytest.raises(ValidationError):
        Scope(since=since, until=until)


def test_scope_bounds_cover_whole_days() -> None:
    whole = Scope(since="2024-06-01", until="2024-06-30")
    assert (lower_bound(whole.since), upper_bound(whole.until)) == (
        "2024-06-01 00:00:00",
        "2024-06-30 23:59:59",
    )
    assert (lower_bound(None), upper_bound(None)) == (None, None)
    assert lower_bound("2024-06-01 08:30:00") == "2024-06-01 08:30:00"
    assert upper_bound("2024-06-30 08:30:00") == "2024-06-30 08:30:00"
