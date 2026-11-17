"""count() and stats() against figures computed by hand for the Harbor history.

The history is in tests/archive/worlds.py. Local time is America/Los_Angeles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from archive import Archive, import_chat_db
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder
from tests.archive.conftest import Harbor
from tests.archive.legacy import load_legacy_conversation

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID

BY_DAY = [
    ("2023-12-31", 1, 0, 1),
    ("2024-01-01", 2, 1, 1),
    ("2024-02-14", 2, 1, 1),
    ("2024-03-08", 3, 1, 2),
    ("2024-03-09", 1, 1, 0),
    ("2024-03-10", 8, 2, 6),
    ("2024-03-11", 1, 1, 0),
    ("2024-03-25", 1, 1, 0),
]
BY_MONTH = [("2023-12", 1, 0, 1), ("2024-01", 2, 1, 1), ("2024-02", 2, 1, 1), ("2024-03", 14, 6, 8)]
BY_YEAR = [("2023", 1, 0, 1), ("2024", 18, 8, 10)]
# weekday 0 is Monday; 2024-03-08 is a Friday, 2024-03-10 a Sunday
BY_WEEKDAY_HOUR = [
    ("0-00", 2, 1, 1),
    ("0-07", 1, 1, 0),
    ("0-08", 1, 1, 0),
    ("2-18", 2, 1, 1),
    ("4-09", 2, 1, 1),
    ("4-21", 1, 0, 1),
    ("5-23", 1, 1, 0),
    ("6-00", 1, 0, 1),
    ("6-01", 1, 0, 1),
    ("6-03", 2, 1, 1),
    ("6-12", 4, 1, 3),
    ("6-23", 1, 0, 1),
]
BY_CONVERSATION = [(A, 10, 5, 5), (G, 7, 2, 5), (B, 2, 1, 1)]
BY_SENDER = [
    ("", 8, 8, 0),
    (worlds.ROWAN_PHONE, 5, 0, 5),
    (worlds.MIRA_PHONE, 3, 0, 3),
    (worlds.TESSA_EMAIL, 2, 0, 2),
    (worlds.ROWAN_EMAIL, 1, 0, 1),
]
A_STATS = {
    "conversation": A,
    "total_messages": 10,
    "sent_count": 5,
    "received_count": 5,
    "first_timestamp": "2024-03-08 09:00:00",
    "last_timestamp": "2024-03-25 08:00:00",
    "active_days": 4,
    "avg_messages_per_day": 2.5,
    "busiest_day": "2024-03-10",
    "busiest_day_count": 5,
    # 2024-03-10 12:01 to 2024-03-25 08:00, both daylight time: 14 days, 19 hours, 59 minutes
    "longest_gap_days": (14 * 86400 + 19 * 3600 + 59 * 60) / 86400,
    "longest_gap_start": "2024-03-10 12:01:00",
    "longest_gap_end": "2024-03-25 08:00:00",
}
EMPTY_STATS = {
    "total_messages": 0,
    "sent_count": 0,
    "received_count": 0,
    "first_timestamp": None,
    "last_timestamp": None,
    "active_days": 0,
    "avg_messages_per_day": 0.0,
    "busiest_day": None,
    "busiest_day_count": 0,
    "longest_gap_days": 0.0,
    "longest_gap_start": None,
    "longest_gap_end": None,
}


def rows(result: Any) -> list[tuple[str, int, int, int]]:
    assert all(list(row) == ["key", "total", "sent", "received"] for row in result)
    return [(row["key"], row["total"], row["sent"], row["received"]) for row in result]


class TestCount:
    def test_all_messages(self, harbor: Harbor) -> None:
        assert harbor.archive.count() == 19
        assert type(harbor.archive.count()) is int

    def test_group_events_and_tapbacks_are_not_counted(self, harbor: Harbor) -> None:
        assert harbor.archive.count(include_system=True) == 21
        assert harbor.archive.count(conversations=[A]) == 10
        assert harbor.archive.count(conversations=[A], include_system=True) == 11

    @pytest.mark.parametrize(
        ("group_by", "expected"),
        [
            ("day", BY_DAY),
            ("month", BY_MONTH),
            ("year", BY_YEAR),
            ("weekday_hour", BY_WEEKDAY_HOUR),
            ("conversation", BY_CONVERSATION),
            ("sender", BY_SENDER),
        ],
    )
    def test_group_by(self, harbor: Harbor, group_by: str, expected: list[tuple[str, int, int, int]]) -> None:
        result = harbor.archive.count(group_by=group_by)
        assert rows(result) == expected
        assert sum(row["total"] for row in result) == 19
        assert all(row["total"] == row["sent"] + row["received"] for row in result)
        json.dumps(result)

    def test_direction(self, harbor: Harbor) -> None:
        assert harbor.archive.count(from_me=True) == 8
        assert harbor.archive.count(from_me=False) == 11
        assert rows(harbor.archive.count(from_me=True, group_by="year")) == [("2024", 8, 8, 0)]

    def test_days_include_all_of_the_first_and_the_last_day(self, harbor: Harbor) -> None:
        assert harbor.archive.count(since="2024-03-10", until="2024-03-10") == 8
        assert harbor.archive.count(since="2024-01-01", until="2024-02-29") == 4
        assert harbor.archive.count(since="2024-03-11") == 2
        assert harbor.archive.count(until="2023-12-31") == 1
        assert harbor.archive.count(since="2024-01-01", until="2023-12-31") == 0
        assert harbor.archive.count(since="2025-01-01") == 0

    def test_each_day_bucket_equals_a_count_of_that_day(self, harbor: Harbor) -> None:
        for day, total, sent, _received in BY_DAY:
            assert harbor.archive.count(since=day, until=day) == total
            assert harbor.archive.count(since=day, until=day, from_me=True) == sent

    def test_conversations(self, harbor: Harbor) -> None:
        assert harbor.archive.count(conversations=[A, B]) == 12
        assert harbor.archive.count(conversations=[G], since="2024-03-10") == 2
        assert harbor.archive.count(conversations=[]) == 0
        assert harbor.archive.count(conversations=["+15550109999"]) == 0
        assert harbor.archive.count(conversations=A) == 10
        assert rows(harbor.archive.count(conversations=[A, B], group_by="conversation")) == [(A, 10, 5, 5), (B, 2, 1, 1)]
        assert harbor.archive.count(conversations=[], group_by="day") == []

    def test_text_uses_the_search_rules(self, harbor: Harbor) -> None:
        assert harbor.archive.count(text="happy new year") == 3
        assert harbor.archive.count(text='"year almost"') == 1
        assert harbor.archive.count(text="clo*") == 2
        assert harbor.archive.count(text="cafe") == 1
        assert harbor.archive.count(text="🎉") == 1
        assert harbor.archive.count(text="학교") == 1
        assert harbor.archive.count(text="brunch AND (") == 0
        assert harbor.archive.count(text="") == 0
        assert rows(harbor.archive.count(text="happy", group_by="year")) == [("2023", 1, 0, 1), ("2024", 2, 1, 1)]
        # equal totals: the key decides, and "+" sorts before "c"
        assert rows(harbor.archive.count(text="clocks", group_by="conversation")) == [(A, 1, 1, 0), (G, 1, 0, 1)]

    def test_count_with_text_equals_the_search_total(self, harbor: Harbor) -> None:
        for query in ("happy", "clocks change", '"new year"', "bru*", "🎉", "寿司", "?", "nothing-here", "the"):
            for options in ({}, {"from_me": True}, {"conversations": [G]}, {"since": "2024-01-01", "until": "2024-03-09"}):
                assert harbor.archive.count(text=query, **options) == harbor.archive.search(query, **options)["total"]

    @pytest.mark.parametrize("bad", [{"group_by": "week"}, {"since": "03/10/2024"}, {"until": "2024-3-1"}, {"from_me": 1}])
    def test_bad_arguments_raise_value_error(self, harbor: Harbor, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            harbor.archive.count(**bad)  # type: ignore[arg-type]


class TestStats:
    def test_hand_computed_figures(self, harbor: Harbor) -> None:
        found = harbor.archive.stats(A)
        assert found == A_STATS
        assert list(found) == list(A_STATS)
        json.dumps(found)

    def test_group_conversation(self, harbor: Harbor) -> None:
        found = harbor.archive.stats(G)
        assert (found["total_messages"], found["sent_count"], found["received_count"]) == (7, 2, 5)
        assert (found["active_days"], found["avg_messages_per_day"]) == (4, 7 / 4)
        # 2023-12-31, 2024-01-01, 2024-02-14 and 2024-03-10 each... the first day with the top count wins
        assert (found["busiest_day"], found["busiest_day_count"]) == ("2024-01-01", 2)
        # 2024-01-01 00:00:30 to 2024-02-14 18:00:00, both standard time
        assert found["longest_gap_days"] == pytest.approx((44 * 86400 + 17 * 3600 + 59 * 60 + 30) / 86400)
        assert (found["longest_gap_start"], found["longest_gap_end"]) == ("2024-01-01 00:00:30", "2024-02-14 18:00:00")

    def test_day_range(self, harbor: Harbor) -> None:
        found = harbor.archive.stats(A, since="2024-03-09", until="2024-03-10")
        assert (found["total_messages"], found["sent_count"], found["received_count"]) == (6, 3, 3)
        assert (found["first_timestamp"], found["last_timestamp"]) == ("2024-03-09 23:30:00", "2024-03-10 12:01:00")
        assert (found["active_days"], found["busiest_day"], found["busiest_day_count"]) == (2, "2024-03-10", 5)
        # 03:30 daylight time to 12:00: 8 hours and 30 minutes
        assert found["longest_gap_days"] == pytest.approx(8.5 / 24)
        assert (found["longest_gap_start"], found["longest_gap_end"]) == ("2024-03-10 03:30:00", "2024-03-10 12:00:00")

    def test_gap_is_elapsed_time_across_the_clock_change(self, harbor: Harbor) -> None:
        # g16 is 01:59:59 standard time, g17 is 03:00:00 daylight time: one second passed
        night = harbor.archive.stats(G, since="2024-03-10", until="2024-03-10")
        assert night["total_messages"] == 2
        assert night["longest_gap_days"] == pytest.approx(1 / 86400)
        assert (night["longest_gap_start"], night["longest_gap_end"]) == ("2024-03-10 01:59:59", "2024-03-10 03:00:00")

    def test_one_message_has_no_gap(self, harbor: Harbor) -> None:
        found = harbor.archive.stats(A, since="2024-03-25")
        assert found["total_messages"] == 1
        assert (found["longest_gap_days"], found["longest_gap_start"], found["longest_gap_end"]) == (0.0, None, None)
        assert (found["active_days"], found["avg_messages_per_day"], found["busiest_day_count"]) == (1, 1.0, 1)

    def test_empty_and_unknown_conversations(self, harbor: Harbor) -> None:
        assert harbor.archive.stats(worlds.HARBOR_EMPTY_ID) == {"conversation": worlds.HARBOR_EMPTY_ID, **EMPTY_STATS}
        assert harbor.archive.stats("+15550109999") == {"conversation": "+15550109999", **EMPTY_STATS}
        assert harbor.archive.stats(A, since="2030-01-01") == {"conversation": A, **EMPTY_STATS}

    def test_group_events_only_on_request(self, harbor: Harbor) -> None:
        with_events = harbor.archive.stats(A, include_system=True)
        assert with_events["total_messages"] == 11
        assert with_events["last_timestamp"] == "2024-03-25 08:05:00"

    def test_merged_thread_of_one_person(self, harbor: Harbor) -> None:
        found = harbor.archive.stats([A, B])
        assert found["conversation"] == [A, B]
        assert (found["total_messages"], found["sent_count"], found["received_count"]) == (12, 6, 6)
        assert (found["active_days"], found["busiest_day"], found["busiest_day_count"]) == (5, "2024-03-10", 6)
        # b12 2024-03-11 07:45 to a10 2024-03-25 08:00
        assert found["longest_gap_days"] == pytest.approx((14 * 86400 + 15 * 60) / 86400)

    def test_stats_agree_with_count(self, harbor: Harbor) -> None:
        for conversation in (A, B, G):
            found = harbor.archive.stats(conversation)
            assert found["total_messages"] == harbor.archive.count(conversations=[conversation])
            assert found["sent_count"] == harbor.archive.count(conversations=[conversation], from_me=True)
            days = harbor.archive.count(conversations=[conversation], group_by="day")
            assert found["active_days"] == len(days)
            assert found["busiest_day_count"] == max(row["total"] for row in days)


class TestParityWithLegacyComputeStats:
    """stats() gives the same figures as compute_stats in conversation.py."""

    @pytest.mark.parametrize("conversation", [A, B, G])
    def test_same_figures(self, harbor: Harbor, monkeypatch: pytest.MonkeyPatch, conversation: str) -> None:
        compute_stats = load_legacy_conversation(monkeypatch).compute_stats
        messages = list(harbor.archive.iter_messages(conversation))
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([message["timestamp"] for message in messages]),
                "is_from_me": [int(message["is_from_me"]) for message in messages],
                "chat_id": [conversation] * len(messages),
                "text": [message["text"] for message in messages],
            }
        )
        legacy = compute_stats(frame)
        found = harbor.archive.stats(conversation)
        assert found["total_messages"] == legacy.total_messages
        assert found["sent_count"] == legacy.sent_count
        assert found["received_count"] == legacy.received_count
        assert found["first_timestamp"] == str(legacy.first_timestamp)
        assert found["last_timestamp"] == str(legacy.last_timestamp)
        assert found["active_days"] == legacy.active_days
        assert found["avg_messages_per_day"] == pytest.approx(legacy.avg_messages_per_day)
        assert found["busiest_day"] == str(legacy.busiest_day.date())
        assert found["busiest_day_count"] == legacy.busiest_day_count
        # no longest gap here crosses the clock change, so wall-clock and elapsed time agree
        assert found["longest_gap_days"] == pytest.approx(legacy.longest_gap_days)
        assert found["longest_gap_start"] == str(legacy.longest_gap_start)
        assert found["longest_gap_end"] == str(legacy.longest_gap_end)


class TestTies:
    def test_earliest_day_and_earliest_gap_win(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        for day in (1, 3, 5):
            builder.message(chat, when=worlds.la(2024, 6, day, 10), text=f"day {day} first", handle=rowan)
            builder.message(chat, when=worlds.la(2024, 6, day, 11), text=f"day {day} second", handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            found = archive.stats(worlds.ROWAN_PHONE)
        assert (found["busiest_day"], found["busiest_day_count"]) == ("2024-06-01", 2)
        assert found["longest_gap_days"] == pytest.approx(47 / 24)
        assert (found["longest_gap_start"], found["longest_gap_end"]) == ("2024-06-01 11:00:00", "2024-06-03 10:00:00")
