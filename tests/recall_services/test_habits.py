"""habits.py and `habits=true` of /api/stats: the habits of a set as the screen reads them.

The figures are counted by archive/habits.py (tests/archive/test_habits.py); here they are
put together, and joined for the threads of one person. The Orchard history is in
tests/recall_services/world.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from archive import Archive
from archive.habits import Habits
from recall_services import habits, stats
from recall_services.names import Names
from tests.recall_services import world

HABIT_KEYS = {
    "messages",
    "youReply",
    "theyReply",
    "replies",
    "youStart",
    "talks",
    "streak",
    "lateNight",
    "sentLength",
    "receivedLength",
}


def asked(archive: Archive, saved: Names, **params: str) -> dict[str, Any]:
    return stats.answer(archive, saved, params)


class TestFigures:
    def test_nothing_to_count_is_null_not_zero(self) -> None:
        found = habits.figures([])
        assert set(found) == HABIT_KEYS
        assert found["messages"] == 0 and found["talks"] == 0
        for name in ("youReply", "theyReply", "youStart", "streak", "lateNight", "sentLength", "receivedLength"):
            assert found[name] is None, name

    def test_a_reply_at_once_is_zero(self) -> None:
        found = habits.figures([Habits(you_replied=(0,), they_started=1)])
        assert found["youReply"] == 0
        assert found["replies"] == {"you": 1, "they": 0}

    def test_parts_are_joined_before_the_median(self) -> None:
        # the median of 10, 20, 400 is 20; a median of the two medians (15 and 400) would not be
        found = habits.figures([Habits(you_replied=(10, 20)), Habits(you_replied=(400,))])
        assert found["youReply"] == 20

    def test_shares_and_means(self) -> None:
        found = habits.figures(
            [
                Habits(messages=8, late=2, sent_chars=30, sent_texts=3, you_started=1, they_started=3),
                Habits(messages=2, late=0, received_chars=10, received_texts=2, you_started=1),
            ]
        )
        assert found["lateNight"] == pytest.approx(0.2)
        assert found["youStart"] == pytest.approx(2 / 5)
        assert (found["sentLength"], found["receivedLength"]) == (10, 5)


class TestLongestStreak:
    def test_no_day_is_no_streak(self) -> None:
        assert habits.longest_streak([]) is None

    def test_one_day_is_a_streak_of_one(self) -> None:
        assert habits.longest_streak([0]) == {"days": 1, "first": "1970-01-01", "last": "1970-01-01"}

    def test_the_longest_run_wins_and_the_earliest_of_a_tie(self) -> None:
        days = [10, 11, 20, 21, 22, 30, 31, 32]
        found = habits.longest_streak(days)
        assert found == {"days": 3, "first": "1970-01-21", "last": "1970-01-23"}

    def test_the_order_of_the_days_does_not_matter(self) -> None:
        assert habits.longest_streak([5, 3, 4]) == habits.longest_streak([3, 4, 5])


class TestStatsWithHabits:
    def test_habits_are_only_given_on_request(self, archive: Archive, saved: Names) -> None:
        assert "habits" not in asked(archive, saved)
        assert "habits" not in asked(archive, saved, by="person")["rows"][0]

    def test_the_habits_of_the_set(self, archive: Archive, saved: Names) -> None:
        found = asked(archive, saved, habits="true")
        assert set(found["habits"]) == HABIT_KEYS
        assert found["habits"]["messages"] == found["totals"]["total"]
        json.dumps(found)

    def test_a_person_joins_the_habits_of_their_threads(self, archive: Archive, saved: Names) -> None:
        found = asked(archive, saved, by="person", habits="true")
        parts = archive.habits()
        for row in found["rows"]:
            assert set(row["habits"]) == HABIT_KEYS
            assert row["habits"] == habits.figures(parts[key] for key in row["conversations"] if key in parts)
            assert row["habits"]["messages"] == row["total"]

    def test_ines_has_two_threads_and_one_row(self, archive: Archive, saved: Names) -> None:
        found = asked(archive, saved, by="person", habits="true")
        ines = next(row for row in found["rows"] if world.INES_PHONE in row["conversations"])
        assert set(ines["conversations"]) == {world.INES_PHONE, world.INES_EMAIL}
        assert ines["habits"]["talks"] >= 2

    def test_the_filters_narrow_the_habits(self, archive: Archive, saved: Names) -> None:
        mine = asked(archive, saved, habits="true", **{"from": "me"})
        assert mine["habits"]["messages"] == mine["totals"]["total"] == 159
        assert mine["habits"]["receivedLength"] is None

    def test_a_flag_that_is_not_true_or_false_is_refused(self, archive: Archive, saved: Names) -> None:
        with pytest.raises(ValueError, match="habits must be true or false"):
            asked(archive, saved, habits="yes")
