"""day.py: the /api/day payload from Archive.day, with the paging rule of CONTRACTS.md."""

from __future__ import annotations

import json

import pytest

from archive import Archive
from day_view import DAY_LIMIT, DAY_LIMIT_MAX
from recall_services import day
from recall_services.names import Names
from tests.recall_services import world
from tests.recall_services.test_messages import RESULT_KEYS

# DayMessages in app/src/types.ts at 89ab504
DAY_KEYS = {"contact", "day", "total", "offset", "previousDay", "nextDay", "messages"}
INES = world.INES_PHONE


def texts(found: dict[str, object]) -> list[str]:
    return [str(message["text"]) for message in found["messages"]]  # type: ignore[union-attr]


def test_the_payload_has_the_keys_the_app_reads(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, "2024-03-01")
    assert set(found) == DAY_KEYS
    assert all(set(message) == RESULT_KEYS for message in found["messages"])
    json.dumps(found)


def test_a_short_day_is_given_whole_with_its_neighbours(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, "2024-03-01")
    assert texts(found) == [
        "Are you bringing the ladder on Saturday?",
        "Yes, and the pruning saw",
        "",
        "Picked the first lemons today",
    ]
    assert (found["contact"], found["day"], found["total"], found["offset"]) == (INES, "2024-03-01", 4, 0)
    assert (found["previousDay"], found["nextDay"]) == (None, world.LONG_DAY)
    assert found["messages"][2]["attachment"] == "Photo"
    assert {message["displayName"] for message in found["messages"]} == {"Ines Calder"}


def test_the_neighbours_are_the_nearest_days_with_messages(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, "2024-03-05")
    assert (found["previousDay"], found["nextDay"]) == (world.LONG_DAY, None)


def test_a_day_with_no_message_has_its_neighbours(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, world.QUIET_DAY)
    assert (found["total"], found["offset"], found["messages"]) == (0, 0, [])
    assert (found["previousDay"], found["nextDay"]) == (world.LONG_DAY, "2024-03-05")


def test_a_long_day_is_given_in_part_from_its_start(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY)
    assert (found["total"], found["offset"], len(found["messages"])) == (world.LONG_DAY_MESSAGES, 0, DAY_LIMIT)
    assert texts(found)[:2] == ["pruning note 0", "pruning note 1"]


def test_a_place_given_as_offset_is_used_as_it_is(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY, offset=400, limit=200)
    # the answer can hold fewer rows than the limit
    assert (found["offset"], len(found["messages"])) == (400, 50)
    assert texts(found)[0] == "pruning note 400"
    at_the_end = day.payload(archive, saved, INES, world.LONG_DAY, offset=world.LONG_DAY_MESSAGES)
    assert (at_the_end["offset"], at_the_end["messages"]) == (world.LONG_DAY_MESSAGES, [])
    past = day.payload(archive, saved, INES, world.LONG_DAY, offset=9_999)
    assert (past["offset"], past["messages"], past["total"]) == (world.LONG_DAY_MESSAGES, [], world.LONG_DAY_MESSAGES)


@pytest.mark.parametrize("offset", [-1, -400, "-3"])
def test_an_offset_below_zero_is_the_start(archive: Archive, saved: Names, offset: object) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY, offset=offset, limit=5)  # type: ignore[arg-type]
    assert (found["offset"], texts(found)[0]) == (0, "pruning note 0")


def test_a_limit_above_the_largest_part_is_the_largest_part(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY, limit=5_000)
    assert len(found["messages"]) == DAY_LIMIT_MAX == 400


@pytest.mark.parametrize("limit", [0, -1, "0"])
def test_a_limit_below_one_is_refused(archive: Archive, saved: Names, limit: object) -> None:
    with pytest.raises(ValueError):
        day.payload(archive, saved, INES, world.LONG_DAY, limit=limit)  # type: ignore[arg-type]


def test_the_part_around_a_time_never_runs_past_either_end(archive: Archive, saved: Names) -> None:
    # note 300 was sent at 13:00; a window of 100 around it starts 50 before
    middle = day.payload(archive, saved, INES, world.LONG_DAY, around="2024-03-02 13:00:00", limit=100)
    assert (middle["offset"], texts(middle)[0], len(middle["messages"])) == (250, "pruning note 250", 100)
    early = day.payload(archive, saved, INES, world.LONG_DAY, around="2024-03-02 08:01:00", limit=100)
    assert early["offset"] == 0
    late = day.payload(archive, saved, INES, world.LONG_DAY, around="2024-03-02 15:29:00", limit=100)
    assert (late["offset"], len(late["messages"])) == (350, 100)


@pytest.mark.parametrize(
    "around",
    ["2024-03-02T13:00:00", "2024-03-02 13:00:00.250", "2024-03-02T13:00:00Z", " 2024-03-02 13:00:00 ", "2024-03-02T13:00:00-08:00"],
)
def test_a_time_in_another_form_is_read_as_the_time_on_the_clock(archive: Archive, saved: Names, around: str) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY, around=around, limit=100)
    assert found["offset"] == 250


@pytest.mark.parametrize("around", ["", "noon", "13:00", None, 1300])
def test_a_time_that_can_not_be_read_starts_the_day_at_its_start(archive: Archive, saved: Names, around: object) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY, around=around, limit=100)  # type: ignore[arg-type]
    assert found["offset"] == 0


def test_an_offset_wins_over_a_time(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES, world.LONG_DAY, around="2024-03-02 13:00:00", offset=10, limit=5)
    assert found["offset"] == 10


def test_the_parts_of_a_long_day_join_with_no_gap_and_no_message_twice(archive: Archive, saved: Names) -> None:
    """partToAsk in app/src/lib/dayParts.ts asks for the rows after what is shown, and no more."""
    shown = day.payload(archive, saved, INES, world.LONG_DAY, around="2024-03-02 13:00:00")
    ids = [message["messageId"] for message in shown["messages"]]
    start = shown["offset"]
    while start > 0:
        ask = max(0, start - 200)
        part = day.payload(archive, saved, INES, world.LONG_DAY, offset=ask, limit=start - ask)
        assert part["offset"] == ask
        ids = [message["messageId"] for message in part["messages"]] + ids
        start = ask
    end = shown["offset"] + len(shown["messages"])
    while end < shown["total"]:
        part = day.payload(archive, saved, INES, world.LONG_DAY, offset=end, limit=min(200, shown["total"] - end))
        ids += [message["messageId"] for message in part["messages"]]
        end += len(part["messages"])
    assert len(ids) == len(set(ids)) == world.LONG_DAY_MESSAGES


def test_messages_of_one_second_keep_the_order_of_the_source(archive: Archive, saved: Names) -> None:
    found = texts(day.payload(archive, saved, INES, "2024-03-05"))
    assert found.index("Same second, first") + 1 == found.index("Same second, second")


def test_in_a_group_each_message_names_who_wrote_it(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, world.GARDEN, "2024-04-01")
    assert [(message["senderName"], message["isFromMe"]) for message in found["messages"]] == [
        ("Ines Calder", False),
        ("Tomas Reyes", False),
        ("", False),
        ("", True),
    ]
    # the event of the group is not a message of the day
    assert found["total"] == 4


def test_a_conversation_is_never_matched_by_a_part_of_its_id(archive: Archive, saved: Names) -> None:
    found = day.payload(archive, saved, INES[:-1], "2024-03-01")
    assert (found["contact"], found["total"], found["messages"]) == (INES[:-1], 0, [])
    assert (found["previousDay"], found["nextDay"]) == (None, None)


@pytest.mark.parametrize("contact", ["", None, "   "])
def test_a_conversation_is_needed(archive: Archive, saved: Names, contact: object) -> None:
    with pytest.raises(ValueError, match="A conversation is needed"):
        day.payload(archive, saved, contact, "2024-03-01")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "2024-3-1", "yesterday", "2024-02-30", "2024-03-01 09:00:00", None])
def test_text_that_is_not_a_day_is_refused(archive: Archive, saved: Names, bad: object) -> None:
    with pytest.raises(ValueError):
        day.payload(archive, saved, INES, bad)  # type: ignore[arg-type]


def test_a_message_with_no_conversation_is_in_no_day(archive: Archive, saved: Names) -> None:
    for contact in (INES, world.GARDEN, world.UNSAVED):
        assert day.payload(archive, saved, contact, "2024-06-01")["total"] == 0
