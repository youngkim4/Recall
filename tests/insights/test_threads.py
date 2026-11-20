"""Threads from the archive: merged conversations, the night clocks go back, and names that are handles."""

from __future__ import annotations

from collections.abc import Iterable

from recall_services.insight_threads import threads
from recall_services.names import Names


class FakeArchive:
    """The two reads `threads` makes. Messages stream in UTC order, as the archive gives them."""

    def __init__(self, conversations: list[dict], messages: dict[str, list[dict]]) -> None:
        self._conversations = conversations
        self._messages = messages
        self.streamed: list[tuple[str, ...]] = []

    def conversations(self) -> list[dict]:
        return self._conversations

    def iter_messages(self, ids: Iterable[str]):
        chosen = tuple(ids)
        self.streamed.append(chosen)
        rows = [row for key in chosen for row in self._messages.get(key, [])]
        return iter(sorted(rows, key=lambda row: row["utc"]))


def _chat(cid: str, person: int | None, group: bool = False) -> dict:
    return {"id": cid, "is_group": group, "person_id": person}


def _msg(mid: str, utc: int, local: str, mine: bool, text: str = "hi") -> dict:
    return {"id": mid, "utc": utc, "timestamp": local, "is_from_me": mine, "text": text, "attachments": [], "unsent": False}


NAMES = Names.from_records([
    {"name": "Ana Ruiz", "phones": ["+15105550101"], "emails": ["ana@example.org"]},
    {"name": "+1 415 555 0102", "phones": ["+14155550102"], "emails": []},
])


def test_a_phone_and_an_email_join_into_one_thread_in_time_order():
    archive = FakeArchive(
        [_chat("+15105550101", 1), _chat("ana@example.org", 1)],
        {
            "+15105550101": [_msg("a", 1, "2025-03-01 10:00:00", True), _msg("c", 3, "2025-03-01 12:00:00", True)],
            "ana@example.org": [_msg("b", 2, "2025-03-01 11:00:00", False)],
        },
    )
    made = threads(archive, NAMES)
    assert [(thread.name, [note.id for note in thread.notes]) for thread in made] == [("Ana Ruiz", ["a", "b", "c"])]


def test_the_night_clocks_go_back_does_not_break_a_thread():
    archive = FakeArchive(
        [_chat("+15105550101", 1)],
        {"+15105550101": [_msg("x", 1, "2025-11-02 01:30:00", True), _msg("y", 2, "2025-11-02 01:00:00", False)]},
    )
    made = threads(archive, NAMES)
    assert [note.id for note in made[0].notes] == ["y", "x"]


def test_unsaved_names_that_are_handles_and_groups_are_left_out_unread():
    archive = FakeArchive(
        [_chat("+14155550102", 2), _chat("+19995550199", None), _chat("chat1", None, group=True)],
        {"+14155550102": [_msg("p", 1, "2025-03-01 10:00:00", True)], "+19995550199": [_msg("q", 1, "2025-03-01 10:00:00", True)]},
    )
    assert threads(archive, NAMES) == []
    assert archive.streamed == []


def test_a_thread_where_i_never_wrote_is_left_out():
    archive = FakeArchive([_chat("+15105550101", 1)], {"+15105550101": [_msg("a", 1, "2025-03-01 10:00:00", False)]})
    assert threads(archive, NAMES) == []
