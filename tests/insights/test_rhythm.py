from __future__ import annotations

from datetime import timedelta

import pytest

from insights.rhythm import balance, drift
from insights.thread import Note, Thread
from tests.insights.kit import NOW, at, every, note, thread


class TestDrift:
    def test_a_weekly_friend_silent_for_five_months_drifts(self):
        notes = every("2024-06-01", "2025-07-20", 7)
        found = drift(thread("sam", notes), NOW)
        assert found is not None
        assert found.kind == "drift"
        assert found.figure("usual_gap_days") == 7
        assert found.figure("silent_days") == (NOW.date() - notes[-1].when.date()).days

    def test_the_last_message_is_cited(self):
        notes = every("2024-06-01", "2025-07-20", 7)
        found = drift(thread("sam", notes), NOW)
        assert found is not None
        assert found.message_ids == (notes[-1].id,)

    def test_a_weekly_friend_who_wrote_three_days_ago_does_not_drift(self):
        assert drift(thread("sam", every("2024-06-01", "2025-12-17", 7)), NOW) is None

    def test_a_friend_seen_every_two_months_is_not_drifting_after_forty_days(self):
        assert drift(thread("ana", every("2023-01-10", "2025-11-10", 60)), NOW) is None

    def test_a_monthly_friend_needs_four_quiet_months(self):
        assert drift(thread("ana", every("2024-01-01", "2025-10-28", 30)), NOW) is None
        assert drift(thread("ana", every("2024-01-01", "2025-08-01", 30)), NOW) is not None

    def test_a_friend_gone_for_years_is_not_a_nudge(self):
        assert drift(thread("old", every("2020-01-01", "2022-06-01", 7)), NOW) is None

    def test_a_silence_over_a_year_is_not_drift(self):
        assert drift(thread("old", every("2023-06-01", "2024-11-30", 7)), NOW) is None
        assert drift(thread("old", every("2024-06-01", "2025-01-10", 7)), NOW) is not None

    def test_a_one_sided_thread_is_not_a_relationship(self):
        assert drift(thread("bot", every("2024-06-01", "2025-07-20", 7, starter="them", reply=False)), NOW) is None

    def test_six_quiet_weeks_are_the_least(self):
        assert drift(thread("sam", every("2024-06-01", "2025-11-10", 7)), NOW) is None
        assert drift(thread("sam", every("2024-06-01", "2025-11-01", 7)), NOW) is not None

    def test_a_handful_of_messages_is_not_a_relationship(self):
        notes = [note("2025-01-02", "me", "hi"), note("2025-01-03", "them", "hey"), note("2025-02-01", "me", "yo")]
        assert drift(thread("few", notes), NOW) is None

    def test_an_empty_thread_has_nothing(self):
        assert drift(Thread("none", "None", ()), NOW) is None

    def test_the_summary_names_the_person_and_the_silence(self):
        found = drift(thread("sam", every("2024-06-01", "2025-07-20", 7), name="Sam Ito"), NOW)
        assert found is not None
        assert "Sam Ito" in found.summary
        assert "week" in found.summary


class TestBalance:
    def _sessions(self, starters: list[str]) -> list[Note]:
        notes = []
        when = at("2025-08-01 09:00")
        for place, starter in enumerate(starters):
            other = "them" if starter == "me" else "me"
            notes.append(Note(f"s{place}", when, starter == "me", "hey"))
            notes.append(Note(f"r{place}", when + timedelta(minutes=5), other == "me", "hi"))
            when += timedelta(days=4)
        return notes

    def test_you_start_most_conversations(self):
        found = balance(thread("jo", self._sessions(["me"] * 17 + ["them"] * 3)), NOW)
        assert found is not None
        assert found.kind == "balance"
        assert found.figure("started_by_me") == 17
        assert found.figure("conversations") == 20
        assert "17 of the last 20" in found.summary

    def test_they_start_most_conversations(self):
        found = balance(thread("jo", self._sessions(["them"] * 18 + ["me"] * 2)), NOW)
        assert found is not None
        assert found.figure("started_by_me") == 2

    def test_an_even_share_says_nothing(self):
        assert balance(thread("jo", self._sessions(["me", "them"] * 10)), NOW) is None

    def test_too_few_conversations_say_nothing(self):
        assert balance(thread("jo", self._sessions(["me"] * 5)), NOW) is None

    def test_messages_close_in_time_are_one_conversation(self):
        notes = []
        for day in range(12):
            start = at("2025-09-01 08:00") + timedelta(days=day * 3)
            notes.append(Note(f"a{day}", start, True, "morning"))
            notes.append(Note(f"b{day}", start + timedelta(hours=7), False, "late reply"))
        found = balance(thread("jo", notes), NOW)
        assert found is not None
        assert found.figure("conversations") == 12
        assert found.figure("started_by_me") == 12

    def test_only_the_last_half_year_counts(self):
        old = self._sessions(["me"] * 20)
        shifted = [Note(item.id, item.when - timedelta(days=400), item.from_me, item.text) for item in old]
        assert balance(thread("jo", shifted), NOW) is None


@pytest.mark.parametrize("kind", ["drift", "balance"])
def test_no_insight_reads_the_future(kind):
    notes = every("2024-06-01", "2025-07-20", 7) + [note("2026-01-05", "me", "from the future")]
    made = thread("sam", notes)
    finder = drift if kind == "drift" else balance
    found = finder(made, NOW)
    if found is not None:
        assert all(ref != notes[-1].id for ref in found.message_ids)
