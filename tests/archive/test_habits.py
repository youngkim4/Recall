"""habits() against figures computed by hand for the Harbor history.

The history is in tests/archive/worlds.py. Local time is America/Los_Angeles.
A talk starts after 6 hours of quiet in a one-to-one conversation.

Rowan's phone (A), talks in order:
  1. a1 09:00 (them), a2 09:05 (you: a reply after 300 s)
  2. a3 21:15 (them), not answered before 6 hours of quiet
  3. a4 23:30 (you), a5 00:30 (them: a reply after 3600 s), a6 03:30 (you, across the clock change)
  4. a7 12:00 (them), a8 12:00 (you: a reply after 0 s), a9 12:01 (them)
  5. a10 03-25 08:00 (you)
Rowan's email (B): b11 (them), then b12 19 hours later (you): two talks, no reply.
"""

from __future__ import annotations

from archive.habits import Habits
from tests.archive import worlds
from tests.archive.conftest import Harbor

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID
# 2024-03-08 is day 19790 since 1970
MARCH_8 = 19790


def _days(*offsets: int) -> frozenset[int]:
    return frozenset(MARCH_8 + offset for offset in offsets)


class TestHabits:
    def test_every_conversation_with_a_message_is_a_key(self, harbor: Harbor) -> None:
        assert set(harbor.archive.habits()) == {A, B, G}
        assert all(isinstance(found, Habits) for found in harbor.archive.habits().values())

    def test_talks_and_replies_of_one_conversation(self, harbor: Harbor) -> None:
        a = harbor.archive.habits()[A]
        assert (a.you_started, a.they_started) == (2, 3)
        assert a.you_replied == (300, 0)
        assert a.they_replied == (3600,)

    def test_a_talk_after_six_hours_of_quiet_is_a_new_talk(self, harbor: Harbor) -> None:
        b = harbor.archive.habits()[B]
        assert (b.you_started, b.they_started) == (1, 1)
        assert b.you_replied == () and b.they_replied == ()

    def test_a_group_has_no_talks(self, harbor: Harbor) -> None:
        g = harbor.archive.habits()[G]
        assert (g.you_started, g.they_started, g.you_replied, g.they_replied) == (0, 0, (), ())
        assert g.messages == 7

    def test_late_night_is_midnight_to_five_local(self, harbor: Harbor) -> None:
        found = harbor.archive.habits()
        # a5 00:30 and a6 03:30 in A; g14, g15 at midnight, g16 01:59 and g17 03:00 in the group
        assert (found[A].late, found[B].late, found[G].late) == (2, 0, 4)
        assert sum(part.messages for part in found.values()) == 19

    def test_days_are_local_days(self, harbor: Harbor) -> None:
        found = harbor.archive.habits()
        # a4 at 23:30 on the 9th; a5 and a6 after midnight are the 10th
        assert found[A].days == _days(0, 1, 2, 17)
        assert found[B].days == _days(2, 3)

    def test_length_counts_the_characters_of_texts(self, harbor: Harbor) -> None:
        a = harbor.archive.habits()[A]
        sent = ["Yes, at ten.", "Clocks change tonight.", "Can't sleep.", "On my way", "Long time."]
        # a9 is an attachment with no text: it is not counted
        received = ["Morning. Coffee later?", "That was fun.", "Good night.", "Brunch?"]
        assert (a.sent_chars, a.sent_texts) == (sum(map(len, sent)), len(sent))
        assert (a.received_chars, a.received_texts) == (sum(map(len, received)), len(received))

    def test_words_narrow_the_messages_but_not_the_talks(self, harbor: Harbor) -> None:
        found = harbor.archive.habits(text="night")
        # "Good night." is the one match in A, at 00:30
        assert (found[A].messages, found[A].late) == (1, 1)
        assert found[A].you_replied == (300, 0)
        assert B not in found or found[B].messages == 0

    def test_direction_narrows_the_messages_but_not_the_talks(self, harbor: Harbor) -> None:
        found = harbor.archive.habits(from_me=True)
        assert found[A].messages == 5
        assert found[A].received_texts == 0
        assert (found[A].you_started, found[A].they_started) == (2, 3)

    def test_days_narrow_the_talks_too(self, harbor: Harbor) -> None:
        found = harbor.archive.habits(since="2024-03-10", until="2024-03-10")
        # a5 opens the day, from them; a6 is your reply at 03:30, 2 hours later across the clock change.
        # a7 at noon opens a new talk, and a8 answers it at once.
        assert (found[A].you_started, found[A].they_started) == (0, 2)
        assert found[A].you_replied == (7200, 0)

    def test_conversations_narrow_everything(self, harbor: Harbor) -> None:
        assert set(harbor.archive.habits(conversations=[B])) == {B}

    def test_nothing_matches_gives_nothing(self, harbor: Harbor) -> None:
        assert harbor.archive.habits(conversations=[]) == {}
