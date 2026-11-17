"""inputs.py: values as a route receives them."""

from __future__ import annotations

import pytest

from archive import Archive
from recall_services import day, inputs, search
from recall_services.names import Names
from tests.recall_services import world


@pytest.mark.parametrize(("given", "number"), [("12", 12), (" 7 ", 7), ("-3", -3), ("+4", 4), (5, 5), (0, 0)])
def test_a_whole_number(given: object, number: int) -> None:
    assert inputs.whole("limit", given, 60) == number


@pytest.mark.parametrize("nothing", [None, ""])
def test_nothing_given_is_the_default(nothing: object) -> None:
    assert inputs.whole("limit", nothing, 60) == 60
    assert inputs.whole("offset", nothing, None) is None


@pytest.mark.parametrize("given", ["1_000", "٣", "１２", "1e3", "0x10", "1.0", "twelve", True, 1.5, b"3", [3]])
def test_only_plain_digits_are_a_whole_number(given: object) -> None:
    """A query string can hold digits of other scripts and the underscore of Python. A route means none of them."""
    with pytest.raises(ValueError, match="limit must be a whole number"):
        inputs.whole("limit", given, 60)


def test_a_number_beyond_any_place_is_held_to_the_largest_one() -> None:
    assert inputs.whole("offset", 10**30, None) == inputs.LARGEST
    assert inputs.whole("offset", "9" * 5_000, None) == inputs.LARGEST
    assert inputs.whole("offset", -(10**30), None) == -inputs.LARGEST


def test_the_error_does_not_send_a_long_value_back() -> None:
    with pytest.raises(ValueError) as caught:
        inputs.whole("limit", "x" * 100_000, 60)
    assert len(str(caught.value)) < 120


def test_a_place_past_the_end_of_a_day_gives_no_message(archive: Archive, saved: Names) -> None:
    """ "A day of a conversation" in CONTRACTS.md: a place past the end gives none."""
    found = day.payload(archive, saved, world.INES_PHONE, "2024-03-05", offset=10**30)
    assert found["messages"] == [] and found["total"] > 0
    assert day.payload(archive, saved, world.INES_PHONE, "2024-03-05", limit=10**30)["messages"]


def test_a_place_past_the_end_of_a_search_gives_no_result(archive: Archive, saved: Names) -> None:
    found = search.payload(archive, saved, "lemons", offset=10**30)
    assert found["results"] == [] and found["total"] > 0
    assert search.payload(archive, saved, "", world.INES_PHONE, offset=10**30)["results"] == []


@pytest.mark.parametrize("around", ["2024-03-05T07:50", "2024-03-05 07:50", "2024-03-05 07:50:00", "2024-03-05T07:50:00.123Z"])
def test_the_time_to_open_a_day_at_can_be_written_without_seconds(around: str) -> None:
    assert day._clock(around) == "2024-03-05 07:50:00"


@pytest.mark.parametrize("around", ["07:50", "2024-03-05", "2024-03-05 7:50", "soon", "", None, 5])
def test_a_time_that_is_no_time_is_no_place(around: object) -> None:
    assert day._clock(around) is None
