"""archive_state when another writer or a file fault meets a load or a pull.

Another process can write the archive too (the command line importer, a second
server). The lock of this module holds only inside one process, so each case
here uses a second SQLite connection, or an importer that is replaced, to
stand for the other process.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import archive
from archive import ImportBusyError
from recall_services import archive_files, archive_state, files
from recall_services.archive_state import ArchiveState, PullBusy, PullDone, PullFailed
from tests.recall_services import state_data as data


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    made = tmp_path / "saves"
    made.mkdir()
    return made


@pytest.fixture
def state(folder: Path) -> ArchiveState:
    made = ArchiveState(folder / "archive.sqlite3")
    yield made
    made.close()


@pytest.fixture
def export(tmp_path: Path) -> Path:
    return data.write_export(tmp_path / "messages.csv", ["first", "second", "third"])


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return data.write_database(tmp_path / "copy" / "chat.db", ["one", "two", "three"])


@pytest.fixture
def short_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_files, "SWAP_WAIT_SECONDS", 0.05)


def fail_to_write(*_given: Any, **_named: Any) -> None:
    raise OSError("the disk is full")


# --- a pull ------------------------------------------------------------------


def test_a_first_pull_that_meets_another_writer_removes_nothing(
    state: ArchiveState, database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There was no archive when the pull began. Another process made one and writes it: it is not ours to remove."""
    theirs = data.write_database(tmp_path / "theirs" / "chat.db", ["made", "elsewhere"])

    def another_process_got_there_first(_source: object, target: Path, **_more: object) -> None:
        archive.import_chat_db(theirs, target, full=True)
        raise ImportBusyError("another import is writing to this archive")

    monkeypatch.setattr(archive_state, "import_chat_db", another_process_got_there_first)
    assert isinstance(state.pull_from_messages(database, None), PullBusy)
    assert data.texts_of(state.archive()) == ["made", "elsewhere"]


def test_a_first_pull_that_breaks_leaves_no_archive(
    state: ArchiveState, database: Path, folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def breaks(_source: object, target: Path, **_more: object) -> None:
        Path(target).write_bytes(b"")
        raise RuntimeError("a fault that has no type of the archive")

    monkeypatch.setattr(archive_state, "import_chat_db", breaks)
    with pytest.raises(RuntimeError):
        state.pull_from_messages(database, None)
    assert data.files_in(folder) == []


def test_a_pull_is_done_when_only_its_note_can_not_be_written(
    state: ArchiveState, database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive changed. The caches must know it, and the caller must hear that the pull ran."""
    before = state.generation
    monkeypatch.setattr(files, "write_text_atomic", fail_to_write)
    result = state.pull_from_messages(database, None)
    assert isinstance(result, PullDone) and result.messages == 3
    assert state.generation > before


def test_a_pull_whose_note_is_lost_is_still_kept_from_an_export(
    state: ArchiveState, database: Path, export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        patched.setattr(files, "write_text_atomic", fail_to_write)
        state.pull_from_messages(database, None)
    assert state.ensure_from_export(export).status == "kept"


@pytest.mark.parametrize("named", ["a\0b", "~no-such-user-of-this-mac/chat.db"])
def test_a_path_that_is_no_path_is_a_database_that_is_not_there(state: ArchiveState, named: str, folder: Path) -> None:
    result = state.pull_from_messages(named, None)
    assert isinstance(result, PullFailed) and result.reason
    assert data.files_in(folder) == []


# --- a load of an export -------------------------------------------------------


def test_a_load_does_not_replace_an_archive_that_another_process_writes(
    state: ArchiveState, export: Path, short_wait: None
) -> None:
    state.ensure_from_export(export)
    data.write_export(export, ["alpha", "beta"])
    writer = sqlite3.connect(state.path, isolation_level=None)
    try:
        writer.execute("BEGIN IMMEDIATE")
        result = state.ensure_from_export(export)
        assert (result.status, result.changed) == ("busy", False)
        assert data.texts_of(state.archive()) == ["first", "second", "third"]
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    assert state.ensure_from_export(export).status == "built"
    assert data.texts_of(state.archive()) == ["alpha", "beta"]


def test_a_load_does_not_replace_an_archive_that_another_process_reads(
    state: ArchiveState, export: Path, short_wait: None
) -> None:
    """A reader in the middle of a query holds the old pages. The move waits for it and then says busy."""
    state.ensure_from_export(export)
    reader = sqlite3.connect(state.path, isolation_level=None)
    writer = sqlite3.connect(state.path, isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM messages").fetchone()
        writer.execute("CREATE TABLE written_after_the_read (word TEXT)")
        data.write_export(export, ["alpha", "beta"])
        assert state.ensure_from_export(export).status == "busy"
    finally:
        reader.close()
        writer.close()
    assert state.ensure_from_export(export).status == "built"


def test_what_the_old_archive_holds_outside_its_one_file_survives_a_move_that_fails(
    state: ArchiveState, export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite keeps the last writes beside the database. They must be in the file before anything is removed."""
    state.ensure_from_export(export)
    state.close()
    other = sqlite3.connect(state.path, isolation_level=None)
    try:
        other.execute("CREATE TABLE written_by_another_process (word TEXT)")
        other.execute("INSERT INTO written_by_another_process VALUES ('kept')")
        assert state.path.with_name("archive.sqlite3-wal").stat().st_size > 0
        data.write_export(export, ["alpha", "beta"])
        monkeypatch.setattr(os, "replace", fail_to_write)
        result = state.ensure_from_export(export)
        assert result.status == "failed" and result.reason.endswith("The archive was not changed.")
    finally:
        other.close()
    monkeypatch.undo()
    check = sqlite3.connect(state.path)
    try:
        assert check.execute("SELECT word FROM written_by_another_process").fetchall() == [("kept",)]
    finally:
        check.close()
    assert data.texts_of(state.archive()) == ["first", "second", "third"]


def test_a_load_is_built_when_only_its_note_can_not_be_written(
    state: ArchiveState, export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive holds the export. Without the note, the next call loads the export one more time."""
    with monkeypatch.context() as patched:
        patched.setattr(files, "write_text_atomic", fail_to_write)
        result = state.ensure_from_export(export)
    assert (result.status, result.changed) == ("built", True)
    assert data.texts_of(state.archive()) == ["first", "second", "third"]
    assert state.ensure_from_export(export).status == "built"
    assert state.ensure_from_export(export).status == "current"


def test_a_file_of_a_killed_run_that_can_not_be_removed_is_a_load_that_failed(
    state: ArchiveState, export: Path, folder: Path
) -> None:
    (folder / ".archive.sqlite3.building").mkdir()
    result = state.ensure_from_export(export)
    assert result.status == "failed" and not state.exists()


@pytest.mark.parametrize("named", ["a\0b", "~no-such-user-of-this-mac/messages.csv"])
def test_a_path_that_is_no_path_is_no_export(state: ArchiveState, named: str, folder: Path) -> None:
    result = state.ensure_from_export(named)
    assert result.status == "no_export" and result.reason
    assert data.files_in(folder) == []
