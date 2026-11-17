"""Figures for the reduce prompt, and stats made safe for JSON."""

import json
import math
from datetime import date, datetime

import pandas as pd
import pytest

from report_engine.figures import conversation_figures
from report_engine.jsonsafe import jsonable
from report_engine.transcript import build_lines

from .synth import MINA, RAVI, labeler, message


def test_figures_count_people_days_years_starts_and_the_longest_silence() -> None:
    lines = build_lines(
        [
            message(1, "2023-12-30 09:00:00", "a"),
            message(2, "2023-12-30 09:05:00", "b", is_from_me=True),
            message(3, "2023-12-30 21:00:00", "c", is_from_me=True),
            message(4, "2024-01-02 08:00:00", "d"),
            message(5, "2024-03-01 08:00:00", "e", sender=RAVI),
            message(6, "2024-03-01 08:01:00", "f", is_from_me=True),
        ],
        labeler,
    )
    assert conversation_figures(lines) == (
        "Messages with text: 6 (You 3, Mina 2, Ravi 1).\n"
        "First message: 2023-12-30. Last message: 2024-03-01. Days with messages: 3.\n"
        "By year: 2023 3 (You 67%); 2024 3 (You 33%).\n"
        "Conversation starts, counted as the first message after 6 quiet hours: You 1, Mina 2, Ravi 1.\n"
        "Longest silence: 59 days, from 2024-01-02 to 2024-03-01."
    )


def test_figures_for_one_message() -> None:
    lines = build_lines([message(1, "2024-06-01 09:00:00", "hi")], labeler)
    assert conversation_figures(lines) == (
        "Messages with text: 1 (Mina 1).\n"
        "First message: 2024-06-01. Last message: 2024-06-01. Days with messages: 1.\n"
        "By year: 2024 1 (You 0%).\n"
        "Conversation starts, counted as the first message after 6 quiet hours: Mina 1."
    )


def test_figures_fold_a_long_tail_of_speakers() -> None:
    names = {f"+1202555{number:04d}": f"Person{number}" for number in range(12)}
    messages = [
        message(number, f"2024-06-01 09:{number:02d}:00", "x", sender=sender)
        for number, sender in enumerate(names)
    ]
    text = conversation_figures(build_lines(messages, lambda sender: names[sender]))
    assert "others 4" in text.splitlines()[0]


def test_figures_hold_no_raw_sender() -> None:
    lines = build_lines([message(1, "2024-06-01 09:00:00", "hi", sender=MINA)], lambda sender: "")
    assert MINA not in conversation_figures(lines)


def test_figures_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        conversation_figures(())


def test_jsonable_keeps_plain_values() -> None:
    value = {"a": 1, "b": [1.5, "x", None, True], "c": {"d": ()}}
    assert jsonable(value) == {"a": 1, "b": [1.5, "x", None, True], "c": {"d": []}}


def test_jsonable_handles_what_a_stats_payload_holds() -> None:
    stats = {
        "firstTimestamp": pd.Timestamp("2024-06-01 09:00:00"),
        "missing": pd.NaT,
        "count": pd.Series([1, 2]).sum(),
        "ratio": pd.Series([0.5]).mean(),
        "when": datetime(2024, 6, 1, 9, 0, 0),
        "day": date(2024, 6, 1),
        "nan": math.nan,
        "inf": math.inf,
        7: {"tags": {"b"}},
        "thing": object,
    }
    safe = jsonable(stats)
    assert safe["firstTimestamp"] == "2024-06-01T09:00:00"
    assert safe["missing"] is None
    assert safe["count"] == 3 and isinstance(safe["count"], int)
    assert safe["ratio"] == 0.5 and isinstance(safe["ratio"], float)
    assert safe["when"] == "2024-06-01T09:00:00"
    assert safe["day"] == "2024-06-01"
    assert safe["nan"] is None and safe["inf"] is None
    assert safe["7"] == {"tags": ["b"]}
    assert isinstance(safe["thing"], str)
    json.dumps(safe, allow_nan=False)


def test_jsonable_does_not_change_its_input() -> None:
    inner = [1, 2]
    stats = {"list": inner}
    safe = jsonable(stats)
    safe["list"].append(3)
    assert inner == [1, 2]
