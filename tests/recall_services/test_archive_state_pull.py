"""archive_state.pull_from_messages: a full pull of a Messages database into the archive.

The rules of "Import and update" in CONTRACTS.md: the caller names the
database, a database that can not be read gives the reason, a failure leaves
the archive as it was, and nothing from a second source is merged.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from recall_services import archive_state
from recall_services.archive_state import (
    ArchiveState,
    PullBusy,
    PullCancelled,
    PullDone,
    PullFailed,
    PullRefused,
)
from tests.recall_services import state_data as data

KEPT = "The archive was not changed."


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
def database(tmp_path: Path) -> Path:
    return data.write_database(tmp_path / "copy" / "chat.db", ["one", "two", "three"])


def test_a_pull_fills_the_archive(state: ArchiveState, database: Path) -> None:
    result = state.pull_from_messages(database, None)
    assert isinstance(result, PullDone)
    assert (result.status, result.changed) == ("done", True)
    assert result.counts["messages_added"] == 3 and result.counts["messages_removed"] == 0
    assert result.messages == 3 and result.conversations == 1
    assert data.texts_of(state.archive()) == ["one", "two", "three"]


def test_the_result_is_ready_for_json(state: ArchiveState, database: Path) -> None:
    result = state.pull_from_messages(database, None)
    found = result.as_dict()
    assert found["status"] == "done" and found["messages"] == 3
    json.dumps(found)


def test_a_second_pull_makes_the_archive_equal_to_the_database(state: ArchiveState, database: Path) -> None:
    state.pull_from_messages(database, None)
    old = state.archive()
    data.write_database(database, ["one", "two", "three", "four", "five"])
    result = state.pull_from_messages(database, None)
    assert isinstance(result, PullDone) and result.counts["messages_added"] == 2
    assert data.texts_of(state.archive()) == ["one", "two", "three", "four", "five"]
    # a pull writes into the same file: a reader that kept its handle reads the new rows too
    assert old.summary()["messages"] == 5


def test_no_path_of_a_database_is_kept(state: ArchiveState, database: Path) -> None:
    """CONTRACTS.md: the server keeps no path to a database."""
    state.pull_from_messages(database, None)
    note = state.source_path.read_text(encoding="utf-8")
    assert json.loads(note) == {"source": "messages"}
    assert str(database) not in note and "chat.db" not in note


def test_names_come_in_as_data(state: ArchiveState, database: Path) -> None:
    state.pull_from_messages(database, [{"name": "Petra Lund", "phones": [data.PERSON], "emails": []}])
    person = state.archive().conversation(data.PERSON)["person_id"]
    assert state.archive().person(person)["name"] == "Petra Lund"


def test_a_pull_over_an_archive_from_an_export_replaces_every_row(
    state: ArchiveState, database: Path, tmp_path: Path
) -> None:
    """Nothing of a second source is merged: after the pull the archive holds what the database holds."""
    export = data.write_export(tmp_path / "messages.csv", ["one", "two", "only in the export"])
    assert state.ensure_from_export(export).status == "built"
    result = state.pull_from_messages(database, None)
    assert isinstance(result, PullDone)
    assert data.texts_of(state.archive()) == ["one", "two", "three"]
    assert data.ids_of(state.archive()) == ["guid-one", "guid-two", "guid-three"]
    # and the export does not come back
    assert state.ensure_from_export(export).status == "kept"


def test_a_database_that_lacks_much_of_the_archive_is_refused(state: ArchiveState, database: Path) -> None:
    state.pull_from_messages(database, None)
    data.write_database(database, ["one"])
    result = state.pull_from_messages(database, None)
    assert isinstance(result, PullRefused)
    assert (result.status, result.changed) == ("removal_limit", False)
    assert (result.would_remove, result.stored, result.max_share) == (2, 3, 0.2)
    assert result.reason == (
        "This database lacks 2 of the 3 messages in the archive. That is more than 20%. " + KEPT
    )
    assert data.texts_of(state.archive()) == ["one", "two", "three"]
    json.dumps(result.as_dict())


def test_the_caller_can_take_the_limit_away(state: ArchiveState, database: Path) -> None:
    state.pull_from_messages(database, None)
    data.write_database(database, ["one"])
    result = state.pull_from_messages(database, None, max_removed_share=1.0)
    assert isinstance(result, PullDone) and result.counts["messages_removed"] == 2
    assert data.texts_of(state.archive()) == ["one"]


def test_a_second_pull_at_the_same_time_is_refused(state: ArchiveState, database: Path) -> None:
    inner: list[object] = []

    def during(_event: dict[str, object]) -> None:
        if not inner:
            inner.append(state.pull_from_messages(database, None))

    assert isinstance(state.pull_from_messages(database, None, progress=during), PullDone)
    assert isinstance(inner[0], PullBusy)
    assert (inner[0].status, inner[0].changed) == ("busy", False)
    assert inner[0].reason == "An import is already running."


def test_a_load_of_an_export_is_refused_while_a_pull_runs(state: ArchiveState, database: Path, tmp_path: Path) -> None:
    export = data.write_export(tmp_path / "messages.csv", ["one"])
    inner: list[str] = []
    state.pull_from_messages(database, None, progress=lambda _event: inner.append(state.ensure_from_export(export).status))
    assert set(inner) == {"busy"}


def test_another_writer_of_the_archive_is_an_import_that_runs(
    state: ArchiveState, database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from archive import run_control

    monkeypatch.setattr(run_control, "LOCK_WAIT_SECONDS", 0.05)
    state.pull_from_messages(database, None)
    writer = sqlite3.connect(state.path, isolation_level=None)
    try:
        writer.execute("BEGIN IMMEDIATE")
        assert isinstance(state.pull_from_messages(database, None), PullBusy)
    finally:
        writer.execute("ROLLBACK")
        writer.close()


def test_a_cancelled_pull_leaves_the_archive_as_it_was(state: ArchiveState, database: Path) -> None:
    state.pull_from_messages(database, None)
    data.write_database(database, ["one", "two", "three", "four"])
    result = state.pull_from_messages(database, None, cancel=lambda: True)
    assert isinstance(result, PullCancelled)
    assert (result.status, result.changed) == ("cancelled", False)
    assert data.texts_of(state.archive()) == ["one", "two", "three"]


@pytest.mark.parametrize("nothing", [None, "", "   "])
def test_the_caller_must_name_the_database(state: ArchiveState, nothing: object) -> None:
    """The service never chooses the Messages database of the Mac by itself: database_for in the server does."""
    with pytest.raises(ValueError, match="name the database"):
        state.pull_from_messages(nothing, None)  # type: ignore[arg-type]


def test_a_database_that_is_not_there(state: ArchiveState, tmp_path: Path, folder: Path) -> None:
    missing = tmp_path / "nowhere" / "chat.db"
    result = state.pull_from_messages(missing, None)
    assert isinstance(result, PullFailed)
    assert (result.status, result.changed) == ("failed", False)
    assert result.reason == f"There is no file at {missing.resolve()}. {KEPT}"
    # a first pull that fails leaves no archive behind
    assert data.files_in(folder) == []
    assert state.exists() is False


def test_a_file_that_is_no_database(state: ArchiveState, tmp_path: Path) -> None:
    wrong = tmp_path / "notes.db"
    wrong.write_text("not a database", encoding="utf-8")
    result = state.pull_from_messages(wrong, None)
    assert isinstance(result, PullFailed)
    assert result.reason == f"The copy at {wrong.resolve()} is not a Messages database (not a SQLite database). {KEPT}"


def test_a_database_with_no_message(state: ArchiveState, tmp_path: Path) -> None:
    empty = data.write_database(tmp_path / "empty.db", [])
    result = state.pull_from_messages(empty, None)
    assert isinstance(result, PullFailed)
    assert result.reason == f"The copy at {empty.resolve()} holds no messages. {KEPT}"


def test_a_database_that_the_importer_can_not_read(state: ArchiveState, tmp_path: Path, folder: Path) -> None:
    """The probe finds a message table with a row, and the importer then finds that it is no Messages database."""
    odd = tmp_path / "odd.db"
    made = sqlite3.connect(odd)
    made.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, note TEXT)")
    made.execute("INSERT INTO message (note) VALUES ('a row')")
    made.commit()
    made.close()
    result = state.pull_from_messages(odd, None)
    assert isinstance(result, PullFailed)
    assert result.reason == f"This is not a Messages database that Recall can read. {KEPT}"
    assert result.detail  # for the log of the job; it is not shown
    assert state.exists() is False
    assert data.files_in(folder) == []


def test_a_pull_that_fails_leaves_an_archive_as_it_was(state: ArchiveState, database: Path, tmp_path: Path) -> None:
    state.pull_from_messages(database, None)
    inode = state.path.stat().st_ino
    wrong = tmp_path / "notes.db"
    wrong.write_text("not a database", encoding="utf-8")
    assert isinstance(state.pull_from_messages(wrong, None), PullFailed)
    assert state.path.stat().st_ino == inode
    assert data.texts_of(state.archive()) == ["one", "two", "three"]
    assert json.loads(state.source_path.read_text(encoding="utf-8")) == {"source": "messages"}


def test_progress_is_passed_on(state: ArchiveState, database: Path) -> None:
    seen: list[dict[str, object]] = []
    state.pull_from_messages(database, None, progress=seen.append)
    assert {event["phase"] for event in seen} >= {"copy", "messages", "finish"}


def test_nothing_is_left_in_the_folder_of_the_database(state: ArchiveState, database: Path) -> None:
    before = data.files_in(database.parent)
    state.pull_from_messages(database, None)
    assert data.files_in(database.parent) == before


def test_the_folder_of_the_archive_holds_the_archive_and_its_note(state: ArchiveState, database: Path, folder: Path) -> None:
    state.pull_from_messages(database, None)
    assert data.kept_files(folder) == ["archive.sqlite3", "archive.sqlite3.source.json"]


# --- the shared state of the server ------------------------------------------


def test_the_module_has_one_shared_state_at_the_default_place(tmp_path: Path) -> None:
    shared = archive_state.shared()
    assert shared is archive_state.shared()
    export = data.write_export(tmp_path / "messages.csv", ["one"])
    assert archive_state.ensure_from_export(export).status == "built"
    assert archive_state.archive() is shared.archive()
    assert archive_state.archive().summary()["messages"] == 1
    result = archive_state.pull_from_messages(data.write_database(tmp_path / "chat.db", ["x", "y"]), None)
    assert isinstance(result, PullDone)
    assert archive_state.archive().summary()["messages"] == 2
    archive_state.reopen()
    shared.close()
