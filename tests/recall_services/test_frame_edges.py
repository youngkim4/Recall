"""frame.py at its edges: words that look like a label, tapbacks of other kinds, an unsent message, a late write."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from conversation import compute_reaction_stats
from recall_services import frame
from recall_services.archive_state import ArchiveState
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


def test_an_unsent_message_is_a_row_of_the_archive_and_no_row_of_the_frame(state: ArchiveState) -> None:
    """A seam that is open: the archive counts a message that was unsent, and the export frame has no row for it.

    After the hook-up the list of conversations says 4 and the figures on
    pandas say 3. The archive or `archive.compat` must change to close it.
    """
    archive = state.archive()
    assert archive.conversation(PERSON)["message_count"] == 4
    assert archive.day(PERSON, START.strftime("%Y-%m-%d"))["total"] == 4
    assert len(frame.messages_frame(state)) == 3


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
