"""Boundary values: what the data models and helpers refuse, and what they repair."""

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from report_engine.calls import usage_of
from report_engine.dates import is_stamp, parse_day, shift_day
from report_engine.errors import ConfigError
from report_engine.jsonsafe import jsonable
from report_engine.models import Moment, PeriodSummary, Report, Scope
from report_engine.options import ReportOptions
from report_engine.ports import LLMUsage
from report_engine.scope import coerce_scope

from .synth import model_info
from .test_models import report as make_report


@pytest.mark.parametrize(
    "value",
    [
        {"sinse": "2024-01-01"},
        {"since": "2024-01-01", "extra": None},
        "2024-01-01",
        7,
        {"since": "tomorrow"},
    ],
)
def test_a_scope_that_is_not_understood_is_refused_not_ignored(value: Any) -> None:
    with pytest.raises(ConfigError):
        coerce_scope(value)


def test_scope_refuses_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Scope(sincee="2024-01-01")  # type: ignore[call-arg]


def test_scope_order_is_checked_to_the_second() -> None:
    with pytest.raises(ValidationError):
        Scope(since="2024-01-02 10:00:00", until="2024-01-02 09:00:00")
    assert Scope(since="2024-01-02 10:00:00", until="2024-01-02").until == "2024-01-02"
    assert Scope(since="2024-01-02", until="2024-01-02").since == "2024-01-02"


def test_days_and_timestamps_must_use_plain_digits() -> None:
    wide = "".join(chr(0xFF10 + int(digit)) for digit in "2024") + "-02-03"
    with pytest.raises(ValueError):
        parse_day(wide)
    assert not is_stamp(wide + " 10:00:00")
    assert is_stamp("2024-02-03 10:00:00")
    assert not is_stamp("2024-02-03T10:00:00")
    assert not is_stamp("2024-02-03 10:00:00.5")
    assert not is_stamp(None)


@pytest.mark.parametrize("day, step", [("9999-12-31", 1), ("0001-01-01", -1)])
def test_a_shift_past_the_calendar_is_a_value_error(day: str, step: int) -> None:
    with pytest.raises(ValueError):
        shift_day(day, step)


def test_jsonable_keeps_arrays_and_tables_as_data() -> None:
    safe = jsonable(
        {
            "array": np.array([1, 2, 3]),
            "matrix": np.array([[1.5, np.nan]]),
            "series": pd.Series([4, 5], index=["a", "b"]),
            "frame": pd.DataFrame({"n": [1, 2]}),
            "missing": pd.NA,
            "flag": np.bool_(True),
        }
    )
    assert safe == {
        "array": [1, 2, 3],
        "matrix": [[1.5, None]],
        "series": {"a": 4, "b": 5},
        "frame": {"n": {"0": 1, "1": 2}},
        "missing": None,
        "flag": True,
    }


def test_jsonable_says_when_two_keys_become_one(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert jsonable({1: "a", "1": "b"}) == {"1": "b"}
    assert "appears twice" in caplog.text


def test_jsonable_survives_a_structure_that_holds_itself() -> None:
    loop: dict[str, Any] = {}
    loop["self"] = loop
    safe = jsonable(loop)
    assert isinstance(safe, dict)


def test_report_stats_are_made_json_safe_on_the_way_in() -> None:
    made = make_report(stats={"pair": (1, 2), "when": pd.Timestamp("2024-06-01")})
    assert made.stats == {"pair": [1, 2], "when": "2024-06-01T00:00:00"}
    assert Report.model_validate_json(made.model_dump_json()) == made


@pytest.mark.parametrize(
    "change",
    [
        {"period_token_budget": True},
        {"period_token_budget": 1.5},
        {"max_moments": 2.0},
        {"score_floor": True},
        {"score_floor": float("nan")},
        {"tokenizer_mismatch_margin": float("inf")},
        {"reasoning": ""},
        {"reasoning": 3},
    ],
)
def test_options_refuse_the_wrong_kind_of_value(change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        replace(ReportOptions(), **change)


def test_model_info_checks_the_reply_size_and_trims_names() -> None:
    with pytest.raises(ValidationError):
        model_info(context_window=100, max_input_tokens=80, max_output_tokens=500)
    with pytest.raises(ValidationError):
        model_info(id="   ")
    assert model_info(id="  spaced  ").id == "spaced"


def test_period_days_must_be_days_in_order() -> None:
    with pytest.raises(ValidationError):
        PeriodSummary(label="x", first_day="x", last_day="2024-06-01")
    with pytest.raises(ValidationError):
        PeriodSummary(label="x", first_day="2024-06-02", last_day="2024-06-01")


def test_created_at_needs_an_offset_and_is_kept_in_utc() -> None:
    with pytest.raises(ValidationError):
        make_report(created_at="2024-07-01T10:15:00")
    with pytest.raises(ValidationError):
        make_report(created_at="2024-07-01")
    assert (
        make_report(created_at="2024-07-01T12:15:00+02:00").created_at
        == "2024-07-01T10:15:00+00:00"
    )


def moment(**overrides: Any) -> Moment:
    values: dict[str, Any] = {
        "date": "2024-06-03",
        "title": "T",
        "detail": "D",
        "category": "conflict",
        "score": 0.5,
    }
    values.update(overrides)
    return Moment(**values)


def test_a_quote_and_its_source_go_together() -> None:
    good = moment(
        quote="exact words",
        quote_verified=True,
        quote_message_id="7",
        quote_speaker="Mina",
        message_ids=("7",),
    )
    assert good.quote_message_id == "7"
    bad: list[dict[str, Any]] = [
        {"quote": "left over"},
        {"quote_verified": True},
        {"quote": "words", "quote_verified": True},
        {"quote": "words", "quote_verified": True, "quote_message_id": "7"},
        {"quote_message_id": "7", "message_ids": ("7",)},
        {"quote_speaker": "Mina"},
    ]
    for fields in bad:
        with pytest.raises(ValidationError):
            moment(**fields)


def test_usage_that_can_not_be_true_is_clamped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        spent = usage_of(
            model_info(), LLMUsage(input_tokens=10, cached_input_tokens=50, output_tokens=-4)
        )
    assert (spent.input_tokens, spent.cached_input_tokens, spent.output_tokens) == (10, 10, 0)
    assert "usage" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        usage_of(model_info(), LLMUsage(input_tokens=10, cached_input_tokens=5, output_tokens=1))
    assert caplog.text == ""
