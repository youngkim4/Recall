"""frame.py at its edges: words that look like a label, tapbacks of other kinds, an unsent message, a late write."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from conversation import compute_reaction_stats
from message_text import display_text
from recall_services import frame, messages
from recall_services.archive_state import ArchiveState
from recall_services.names import Names
from tests.archive.builders import ChatDbBuilder
from tests.recall_services import state_data as data

PERSON = data.PERSON
START = data.START


def build(path: Path) -> Path:
    """One conversation: typed words, a message with two tapbacks, and a message that was unsent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    builder = ChatDbBuilder(path)
    person = builder.handle(PERSON)
    thread = builder.chat(PERSON, members=[person])
    builder.message(thread, when=START, text="[Attachment]", handle=person, guid="typed-label")
    builder.message(thread, when=START + timedelta(minutes=1), text="the ferry leaves at nine", handle=person, guid="tapped")
    builder.tapback(thread, "tapped", 2000, when=START + timedelta(minutes=2), from_me=True)
    builder.tapback(
        thread, "tapped", 2006, when=START + timedelta(minutes=3), handle=person, emoji="\U0001f6f6", text="Reacted"
    )
    builder.message(
        thread,
        when=START + timedelta(minutes=4),
        text=None,
        handle=person,
        guid="unsent",
        retracted=START + timedelta(minutes=5),
    )
    builder.message(thread, when=START + timedelta(minutes=6), text="after it", from_me=True, guid="after")
    builder.close()
    return path


@pytest.fixture
def state(tmp_path: Path) -> Iterator[ArchiveState]:
    made = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    made.pull_from_messages(build(tmp_path / "copy" / "chat.db"), None)
    yield made
    made.close()


def test_the_typed_word_of_the_old_label_stays_a_word(state: ArchiveState) -> None:
    """A person typed exactly `[Attachment]`. The message holds no file, so its text is not the placeholder."""
    messages = frame.messages_frame(state)
    assert messages[messages["message_id"] == "typed-label"].iloc[0]["text"] == "[Attachment]"


def test_only_the_six_kinds_that_the_figures_know_are_tapbacks_of_the_frame(state: ArchiveState) -> None:
    """The exporter of today reads the six classic kinds. A tapback with any emoji is a kind of its own in the archive."""
    kinds = {tapback["kind"] for found in state.archive().reactions(["tapped"]).values() for tapback in found}
    assert kinds == {"love", "emoji"}
    found = frame.reactions_frame(state, PERSON)
    assert list(found["reaction_type"]) == ["love"]
    figures = compute_reaction_stats(found, PERSON)
    assert figures.total == figures.loves == 1


def test_an_unsent_message_is_a_row_of_the_archive_and_of_the_frame(state: ArchiveState) -> None:
    """One count on every screen: the list, a day and the figures on pandas all hold a message that was unsent.

    It has no words. The frame says so in `unsent`, and a payload gives it a label.
    """
    archive = state.archive()
    assert archive.conversation(PERSON)["message_count"] == 4
    day = archive.day(PERSON, START.strftime("%Y-%m-%d"))
    assert day["total"] == 4
    made = frame.messages_frame(state)
    assert len(made) == 4
    taken_back = made[made["unsent"]]
    assert list(taken_back["message_id"]) == ["unsent"]
    assert display_text(taken_back.iloc[0]["text"]) == ""
    shown = messages.payloads(archive, Names(), [message for message in day["messages"] if message["unsent"]])
    assert [(row["text"], row["attachment"]) for row in shown] == [("", "Unsent message")]


def test_a_write_that_is_not_in_the_one_file_yet_makes_a_new_frame(state: ArchiveState) -> None:
    """SQLite keeps new writes beside the database for a time. The key of the frame reads that file too."""
    first = frame.messages_frame(state)
    in_the_file = (state.path.stat().st_mtime_ns, state.path.stat().st_size)
    mark = state.file_mark()
    other = sqlite3.connect(state.path, isolation_level=None)  # stands for another process
    try:
        other.execute("PRAGMA wal_autocheckpoint=0")
        other.execute("CREATE TABLE written_by_another_process (word TEXT)")
        assert (state.path.stat().st_mtime_ns, state.path.stat().st_size) == in_the_file
        assert state.file_mark() != mark
        assert frame.messages_frame(state) is not first
    finally:
        other.close()


def test_the_mark_of_the_file_is_the_same_for_the_same_content(state: ArchiveState) -> None:
    assert state.file_mark() == state.file_mark()
    assert state.file_mark()[0] == str(state.path)


def test_a_read_does_not_change_the_mark_of_the_archive(tmp_path: Path) -> None:
    """SQLite makes an empty file beside the archive at the first read. A cache that was keyed before it is still found."""
    first = ArchiveState(tmp_path / "saves" / "archive.sqlite3")
    first.pull_from_messages(build(tmp_path / "copy" / "chat.db"), None)
    first.close()
    for beside in ("-wal", "-shm"):
        first.path.with_name(first.path.name + beside).unlink(missing_ok=True)
    made = ArchiveState(first.path)  # as after a new start of the server: nothing has read the file yet
    try:
        before = made.file_mark()
        frame.messages_frame(made)
        assert made.file_mark() == before
        assert frame.messages_frame(made) is frame.messages_frame(made)
    finally:
        made.close()
