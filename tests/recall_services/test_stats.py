"""stats.py: the /api/stats payload, counts with the filters of search over the whole archive.

The Orchard history is in tests/recall_services/world.py: 479 messages, 159 of
them sent, from 2024-03-01 to 2024-05-01.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from archive import Archive
from recall_services import stats
from recall_services.names import Names
from tests.recall_services import world

NOTHING_APPLIED = {"contacts": [], "since": "", "until": "", "from": "", "kind": "", "attachments": ""}
PAYLOAD_KEYS = {"by", "query", "prefix", "filters", "totals", "first", "last", "rows", "rowsTotal"}
INES = "Ines Calder"


def asked(archive: Archive, saved: Names, **params: str) -> dict[str, Any]:
    return stats.answer(archive, saved, params)


def table(found: dict[str, Any]) -> list[tuple[str, str, int, int, int]]:
    return [(row["key"], row["label"], row["total"], row["sent"], row["received"]) for row in found["rows"]]


def test_one_total_for_the_whole_archive(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved)
    assert set(found) == PAYLOAD_KEYS
    assert found["totals"] == {"total": 479, "sent": 159, "received": 320}
    assert (found["first"], found["last"], found["rows"], found["rowsTotal"]) == ("2024-03-01", "2024-05-01", [], 0)
    assert (found["by"], found["query"], found["prefix"], found["filters"]) == ("", "", False, NOTHING_APPLIED)
    json.dumps(found)


def test_every_day_of_the_range_has_a_row(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="day", since="2024-03-01", until="2024-03-08")
    assert table(found) == [
        ("2024-03-01", "2024-03-01", 4, 2, 2),
        ("2024-03-02", "2024-03-02", 450, 150, 300),
        ("2024-03-03", "2024-03-03", 0, 0, 0),
        ("2024-03-04", "2024-03-04", 0, 0, 0),
        ("2024-03-05", "2024-03-05", 5, 2, 3),
        ("2024-03-06", "2024-03-06", 2, 1, 1),
        ("2024-03-07", "2024-03-07", 3, 2, 1),
        ("2024-03-08", "2024-03-08", 0, 0, 0),
    ]
    assert (found["first"], found["last"], found["totals"]["total"]) == ("2024-03-01", "2024-03-07", 464)


def test_months_and_years_from_the_first_to_the_last_message(archive: Archive, saved: Names) -> None:
    months = asked(archive, saved, by="month")
    assert table(months) == [
        ("2024-03", "2024-03", 464, 157, 307),
        ("2024-04", "2024-04", 5, 1, 4),
        ("2024-05", "2024-05", 10, 1, 9),
    ]
    assert table(asked(archive, saved, by="year")) == [("2024", "2024", 479, 159, 320)]


def test_weeks_start_on_monday(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="week", until="2024-03-31")
    # 2024-03-01 is a Friday: its week starts on 2024-02-26
    assert table(found) == [
        ("2024-02-26", "2024-02-26", 454, 152, 302),
        ("2024-03-04", "2024-03-04", 10, 5, 5),
        ("2024-03-11", "2024-03-11", 0, 0, 0),
        ("2024-03-18", "2024-03-18", 0, 0, 0),
        ("2024-03-25", "2024-03-25", 0, 0, 0),
    ]


def test_a_range_wider_than_the_archive_stops_at_its_ends(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="month", since="1990-01-01", until="2099-12-31")
    assert [row["key"] for row in found["rows"]] == ["2024-03", "2024-04", "2024-05"]


def test_weekdays_hours_and_both_have_every_bucket(archive: Archive, saved: Names) -> None:
    weekdays = asked(archive, saved, by="weekday")
    assert [row["label"] for row in weekdays["rows"]] == [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    # 2024-03-02 (the long day) is a Saturday, 2024-05-01 a Wednesday
    assert [row["total"] for row in weekdays["rows"]] == [4, 6, 12, 3, 4, 450, 0]
    hours = asked(archive, saved, by="hour")
    assert (len(hours["rows"]), hours["rows"][9]["key"], hours["rows"][9]["label"]) == (24, "09", "09:00")
    both = asked(archive, saved, by="weekday_hour")
    assert len(both["rows"]) == 168
    assert (both["rows"][0]["key"], both["rows"][0]["label"]) == ("0-00", "Monday 00:00")
    assert sum(row["total"] for row in both["rows"]) == 479


def test_the_filters_and_the_words_narrow_every_figure(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, query="pruning", by="day", **{"from": "me"})
    assert found["totals"] == {"total": 151, "sent": 151, "received": 0}
    assert (found["first"], found["last"]) == ("2024-03-01", "2024-03-02")
    typing = asked(archive, saved, query="wheelb", prefix="true", kind="group")
    assert (typing["totals"]["total"], typing["prefix"], typing["filters"]["kind"]) == (2, True, "group")
    assert asked(archive, saved, query="wheelb")["totals"]["total"] == 0
    images = asked(archive, saved, attachments="image", contacts=f"{world.INES_PHONE},{world.GARDEN}")
    assert images["totals"] == {"total": 2, "sent": 1, "received": 1}


def test_nothing_found(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, query="zeppelin", by="month")
    assert (found["totals"]["total"], found["first"], found["last"], found["rows"]) == (0, None, None, [])
    hours = asked(archive, saved, query="zeppelin", by="hour")
    assert len(hours["rows"]) == 24 and not any(row["total"] for row in hours["rows"])
    days = asked(archive, saved, query="zeppelin", by="day", since="2024-03-01", until="2024-03-02")
    assert table(days) == [("2024-03-01", "2024-03-01", 0, 0, 0), ("2024-03-02", "2024-03-02", 0, 0, 0)]


def test_conversations_keep_the_threads_of_one_person_apart(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="conversation")
    assert table(found) == [
        (world.INES_PHONE, INES, 459, 154, 305),
        (world.UNTITLED, world.UNTITLED_NAME, 10, 1, 9),
        (world.GARDEN, world.GARDEN_TITLE, 5, 1, 4),
        (world.UNSAVED, "Number ending in 9999", 3, 2, 1),
        (world.INES_EMAIL, INES, 2, 1, 1),
    ]
    assert found["rows"][0]["conversations"] == [world.INES_PHONE]
    assert found["rowsTotal"] == 5


def test_a_person_joins_the_phone_and_the_email_thread(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="person")
    assert table(found) == [
        (world.INES_PHONE, INES, 461, 155, 306),
        (world.UNTITLED, world.UNTITLED_NAME, 10, 1, 9),
        (world.GARDEN, world.GARDEN_TITLE, 5, 1, 4),
        (world.UNSAVED, "Number ending in 9999", 3, 2, 1),
    ]
    assert found["rows"][0]["conversations"] == [world.INES_PHONE, world.INES_EMAIL]
    # the busiest thread in the filtered set gives the key
    email_first = asked(archive, saved, by="person", since="2024-03-06")
    assert email_first["rows"][-1]["key"] == world.INES_EMAIL


def test_an_unsaved_handle_is_masked_as_the_app_masks_it(archive: Archive) -> None:
    found = asked(archive, Names(), by="person")
    labels = {row["key"]: row["label"] for row in found["rows"]}
    assert labels[world.INES_PHONE] == "Number ending in 0001"
    assert labels[world.UNTITLED] == "Unnamed group"
    assert not any("555" in label for label in labels.values())
    by_thread = asked(archive, Names(), by="conversation")
    # an address is shown as it is, as the app shows it
    assert {row["key"]: row["label"] for row in by_thread["rows"]}[world.INES_EMAIL] == world.INES_EMAIL


def test_the_busiest_come_first_up_to_the_limit(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="conversation", limit="2")
    assert [row["key"] for row in found["rows"]] == [world.INES_PHONE, world.UNTITLED]
    assert found["rowsTotal"] == 5
    assert len(asked(archive, saved, by="person", limit="9999")["rows"]) == 4
    assert len(asked(archive, saved, by="person", limit="-5")["rows"]) == 1


@pytest.mark.parametrize(
    ("params", "sentence"),
    [
        ({"by": "minute"}, "by must be day, week, month, year, weekday, hour, weekday_hour, person, conversation or empty."),
        ({"limit": "many"}, "limit must be a whole number, got 'many'"),
        ({"kind": "self"}, "kind must be direct, group or empty."),
        ({"prefix": "maybe"}, "prefix must be true or false."),
    ],
)
def test_a_value_that_is_not_allowed_is_refused(
    archive: Archive, saved: Names, params: dict[str, str], sentence: str
) -> None:
    with pytest.raises(ValueError) as refused:
        asked(archive, saved, **params)
    assert str(refused.value) == sentence


def test_days_in_the_wrong_order_give_no_time_row(archive: Archive, saved: Names) -> None:
    found = asked(archive, saved, by="day", since="2024-03-05", until="2024-03-01")
    assert (found["rows"], found["totals"]["total"]) == ([], 0)


def test_an_empty_archive_gives_no_time_row(tmp_path: Path) -> None:
    with Archive.open(tmp_path / "empty.sqlite3", readonly=False) as empty:
        found = asked(empty, Names(), by="day", since="2024-01-01", until="2024-12-31")
        people = asked(empty, Names(), by="person")
    assert (found["rows"], found["first"], found["totals"]["total"]) == ([], None, 0)
    assert (people["rows"], people["rowsTotal"]) == ([], 0)
