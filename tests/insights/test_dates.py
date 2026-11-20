from __future__ import annotations

from datetime import datetime

from insights.dates import birthday, unanswered
from tests.insights.kit import NOW, note, thread


class TestUnanswered:
    def test_a_question_you_never_answered(self):
        made = thread("lee", [note("2025-12-10", "me", "see you"), note("2025-12-16", "them", "are you coming saturday?", "q1")])
        found = unanswered(made, NOW)
        assert found is not None
        assert found.kind == "unanswered"
        assert found.message_ids == ("q1",)
        assert found.figure("waiting_days") == 4

    def test_the_question_can_come_before_their_last_message(self):
        made = thread("lee", [note("2025-12-15 10:00", "them", "dinner friday?", "q1"), note("2025-12-15 10:02", "them", "my treat", "q2")])
        found = unanswered(made, NOW)
        assert found is not None
        assert found.message_ids == ("q1",)

    def test_a_question_from_this_morning_is_not_late_yet(self):
        assert unanswered(thread("lee", [note("2025-12-20 08:00", "them", "lunch?")]), NOW) is None

    def test_a_question_from_last_month_is_stale(self):
        assert unanswered(thread("lee", [note("2025-11-01", "them", "lunch?")]), NOW) is None

    def test_your_reply_answers_it(self):
        made = thread("lee", [note("2025-12-15", "them", "lunch?"), note("2025-12-16", "me", "yes")])
        assert unanswered(made, NOW) is None

    def test_no_question_no_insight(self):
        assert unanswered(thread("lee", [note("2025-12-15", "them", "ok sounds good")]), NOW) is None


class TestBirthday:
    def test_a_birthday_you_wished_twice_is_coming(self):
        made = thread("ivy", [note("2023-12-28", "me", "happy birthday!!", "b1"), note("2024-12-28", "me", "Happy bday ivy", "b2")])
        found = birthday(made, NOW)
        assert found is not None
        assert found.kind == "birthday"
        assert found.figure("days_until") == 8
        assert found.figure("years_seen") == 2
        assert found.message_ids == ("b2",)
        assert "December 28" in found.summary

    def test_hbd_counts(self):
        assert birthday(thread("ivy", [note("2024-12-24", "me", "hbd!")]), NOW) is not None

    def test_their_wish_to_you_is_your_birthday(self):
        assert birthday(thread("ivy", [note("2024-12-24", "them", "happy birthday sam!")]), NOW) is None

    def test_a_wish_for_someone_else_does_not_count(self):
        assert birthday(thread("ivy", [note("2024-12-24", "me", "happy birthday to your mom!")]), NOW) is None
        assert birthday(thread("ivy", [note("2024-12-24", "me", "happy birthday to you")]), NOW) is not None

    def test_a_birthday_far_away_waits(self):
        assert birthday(thread("ivy", [note("2024-03-02", "me", "happy birthday")]), NOW) is None

    def test_the_year_turns(self):
        now = datetime(2025, 12, 28, 9, 0)
        found = birthday(thread("ivy", [note("2025-01-03", "me", "happy birthday")]), now)
        assert found is not None
        assert found.figure("days_until") == 6

    def test_today_counts(self):
        found = birthday(thread("ivy", [note("2024-12-20", "me", "happy birthday")]), NOW)
        assert found is not None
        assert found.figure("days_until") == 0

    def test_a_wish_a_day_late_keeps_the_most_common_day(self):
        made = thread("ivy", [note("2022-12-27", "me", "happy birthday"), note("2023-12-27", "me", "happy birthday"), note("2024-12-28", "me", "happy belated birthday")])
        found = birthday(made, NOW)
        assert found is not None
        assert found.figure("days_until") == 7

    def test_no_card_once_you_wished_them_today(self):
        made = thread("ivy", [note("2024-12-20", "me", "happy birthday"), note("2025-12-20 09:00", "me", "happy birthday!!")])
        assert birthday(made, datetime(2025, 12, 20, 18, 0)) is None

    def test_leap_day_birthdays_fall_on_february_28(self):
        now = datetime(2025, 2, 20, 9, 0)
        found = birthday(thread("ivy", [note("2024-02-29", "me", "happy birthday")]), now)
        assert found is not None
        assert found.figure("days_until") == 8
