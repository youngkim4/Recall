"""archive_state: where the archive is, the shared read handle, and the bridge from an export."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from archive import Archive, ArchiveError, ArchiveNotFoundError, default_archive_path, import_chat_db
from recall_services.archive_state import ArchiveState, ExportSignature
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


# --- where the archive is, and the read handle -------------------------------


def test_the_default_place_is_the_one_of_the_archive_package() -> None:
    state = ArchiveState()
    assert state.path == default_archive_path()
    assert state.path.name == "archive.sqlite3" and state.path.parent.name == "saves"


def test_the_default_place_follows_the_data_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import recall_paths

    state = ArchiveState()
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "elsewhere")
    assert state.path == tmp_path / "elsewhere" / "archive.sqlite3"


def test_no_archive_is_the_error_of_the_archive_package(state: ArchiveState) -> None:
    with pytest.raises(ArchiveNotFoundError):
        state.archive()
    assert state.exists() is False


def test_one_handle_is_shared(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    assert state.archive() is state.archive()
    assert state.exists() is True


def test_reopen_gives_a_new_handle_and_closes_the_old_one(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    old = state.archive()
    before = state.generation
    state.reopen()
    assert state.generation == before + 1
    assert state.archive() is not old
    with pytest.raises(ArchiveError):
        old.summary()
    assert state.archive().summary()["messages"] == 3


def test_a_handle_follows_a_change_of_the_default_place(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import recall_paths

    state = ArchiveState()
    for name, texts in (("one", ["a"]), ("two", ["a", "b"])):
        monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / name)
        state.ensure_from_export(data.write_export(tmp_path / f"{name}.csv", texts))
        assert state.archive().summary()["messages"] == len(texts)
        assert state.archive().path == (tmp_path / name / "archive.sqlite3").resolve()
    state.close()


def test_close_can_be_called_at_any_time(state: ArchiveState, export: Path) -> None:
    state.close()
    state.ensure_from_export(export)
    state.archive()
    state.close()
    state.close()
    assert state.archive().summary()["messages"] == 3


# --- the bridge from an export ------------------------------------------------


def test_with_no_archive_the_export_is_loaded(state: ArchiveState, export: Path, folder: Path) -> None:
    result = state.ensure_from_export(export)
    assert (result.status, result.changed) == ("built", True)
    assert result.counts["messages_added"] == 3
    assert data.texts_of(state.archive()) == ["first", "second", "third"]
    assert data.ids_of(state.archive()) == ["legacy:1", "legacy:2", "legacy:3"]
    # nothing but the archive and the note of its source is left in the folder
    assert data.kept_files(folder) == ["archive.sqlite3", "archive.sqlite3.source.json"]


def test_the_signature_of_the_export_is_recorded_beside_the_archive(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    found = export.stat()
    recorded = json.loads(state.source_path.read_text(encoding="utf-8"))
    assert recorded == {
        "source": "export",
        "export": {"path": str(export.resolve()), "mtime_ns": found.st_mtime_ns, "size": found.st_size},
    }
    assert ExportSignature.of(export) == ExportSignature(str(export.resolve()), found.st_mtime_ns, found.st_size)


def test_the_archive_and_the_note_are_for_the_owner_only(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    for path in (state.path, state.source_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_the_same_export_is_not_loaded_again(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    handle, inode = state.archive(), state.path.stat().st_ino
    again = state.ensure_from_export(export)
    assert (again.status, again.changed) == ("current", False)
    assert state.archive() is handle and state.path.stat().st_ino == inode


def test_a_changed_export_is_loaded_into_a_new_archive(state: ArchiveState, export: Path, tmp_path: Path) -> None:
    state.ensure_from_export(export)
    old = state.archive()
    generation = state.generation
    data.write_export(export, ["first", "second", "third", "fourth"])
    result = state.ensure_from_export(export)
    assert result.status == "built"
    assert state.generation > generation
    assert data.texts_of(state.archive()) == ["first", "second", "third", "fourth"]
    # a reader that kept the old handle is told that it is closed
    with pytest.raises(ArchiveError):
        old.summary()


def test_a_touched_export_with_the_same_text_is_loaded_again(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    found = export.stat()
    os.utime(export, ns=(found.st_atime_ns, found.st_mtime_ns + 1_000_000_000))
    assert state.ensure_from_export(export).status == "built"


def test_another_export_file_is_loaded(state: ArchiveState, export: Path, tmp_path: Path) -> None:
    state.ensure_from_export(export)
    other = data.write_export(tmp_path / "other" / "messages.csv", ["first", "second", "third"])
    assert state.ensure_from_export(other).status == "built"


def test_no_export_is_said_and_nothing_is_made(state: ArchiveState, tmp_path: Path, folder: Path) -> None:
    result = state.ensure_from_export(tmp_path / "nowhere.csv")
    assert (result.status, result.changed) == ("no_export", False)
    assert "nowhere.csv" in result.reason
    assert data.files_in(folder) == []


def test_a_file_that_is_no_export_with_no_archive_yet_says_nothing_of_an_archive(
    state: ArchiveState, tmp_path: Path, folder: Path
) -> None:
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    result = state.ensure_from_export(wrong)
    assert result.status == "failed"
    assert result.reason == "The file wrong.csv is not an export of messages that Recall can read."
    assert state.exists() is False and data.files_in(folder) == []


def test_a_file_that_is_no_export_fails_and_leaves_the_archive_as_it_was(
    state: ArchiveState, export: Path, tmp_path: Path, folder: Path
) -> None:
    state.ensure_from_export(export)
    inode = state.path.stat().st_ino
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    result = state.ensure_from_export(wrong)
    assert (result.status, result.changed) == ("failed", False)
    assert result.reason == "The file wrong.csv is not an export of messages that Recall can read. The archive was not changed."
    assert state.path.stat().st_ino == inode
    assert data.texts_of(state.archive()) == ["first", "second", "third"]
    assert data.kept_files(folder) == ["archive.sqlite3", "archive.sqlite3.source.json"]


def test_a_first_load_that_fails_leaves_no_archive(state: ArchiveState, tmp_path: Path, folder: Path) -> None:
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    assert state.ensure_from_export(wrong).status == "failed"
    assert data.files_in(folder) == []
    assert state.exists() is False


def test_a_cancelled_load_leaves_the_archive_as_it_was(state: ArchiveState, export: Path, folder: Path) -> None:
    state.ensure_from_export(export)
    data.write_export(export, ["first", "second", "third", "fourth"])
    result = state.ensure_from_export(export, cancel=lambda: True)
    assert (result.status, result.changed) == ("cancelled", False)
    assert data.texts_of(state.archive()) == ["first", "second", "third"]
    assert data.kept_files(folder) == ["archive.sqlite3", "archive.sqlite3.source.json"]
    # the export is still newer than the archive: the next call loads it
    assert state.ensure_from_export(export).status == "built"


def test_progress_is_passed_on(state: ArchiveState, export: Path) -> None:
    seen: list[dict[str, object]] = []
    state.ensure_from_export(export, progress=seen.append)
    assert {event["phase"] for event in seen} >= {"messages", "finish"}
    assert all(set(event) == {"phase", "done", "total"} for event in seen)


def test_a_second_load_at_the_same_time_is_refused(state: ArchiveState, export: Path) -> None:
    inner: list[str] = []

    def during(_event: dict[str, object]) -> None:
        if not inner:
            inner.append(state.ensure_from_export(export).status)

    assert state.ensure_from_export(export, progress=during).status == "built"
    assert inner == ["busy"]


def test_a_file_left_by_a_run_that_was_killed_is_removed(state: ArchiveState, export: Path, folder: Path) -> None:
    for name in (".archive.sqlite3.building", ".archive.sqlite3.building-wal", ".archive.sqlite3.building-shm"):
        (folder / name).write_bytes(b"left by a run that was killed")
    assert state.ensure_from_export(export).status == "built"
    assert data.kept_files(folder) == ["archive.sqlite3", "archive.sqlite3.source.json"]


def test_files_that_stand_beside_the_old_archive_are_not_read_into_the_new_one(
    state: ArchiveState, export: Path, folder: Path
) -> None:
    """SQLite reads `-wal` beside a database by its name. One left by the old archive must not meet the new file."""
    state.ensure_from_export(export)
    state.close()
    (folder / "archive.sqlite3-wal").write_bytes(b"\x00" * 4096)
    (folder / "archive.sqlite3-shm").write_bytes(b"\x00" * 32768)
    data.write_export(export, ["alpha", "beta"])
    assert state.ensure_from_export(export).status == "built"
    assert data.texts_of(state.archive()) == ["alpha", "beta"]
    state.close()
    with Archive.open(state.path) as fresh:
        assert fresh.summary()["messages"] == 2


def test_a_note_that_can_not_be_read_means_that_the_export_is_loaded_again(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    for broken in ("{not json", "[]", '{"source": "export"}', '{"source": "export", "export": {"path": 3}}', ""):
        state.source_path.write_text(broken, encoding="utf-8")
        assert state.ensure_from_export(export).status == "built", broken
        assert state.ensure_from_export(export).status == "current", broken


def test_an_archive_from_a_pull_is_never_replaced_by_an_export(
    state: ArchiveState, export: Path, tmp_path: Path, folder: Path
) -> None:
    """The archive mirrors Messages. An export is older than a pull, whatever its date."""
    import_chat_db(data.write_database(tmp_path / "chat.db", ["from messages"]), state.path)
    result = state.ensure_from_export(export)
    assert (result.status, result.changed) == ("kept", False)
    assert data.texts_of(state.archive()) == ["from messages"]
    assert data.ids_of(state.archive()) == ["guid-from messages"]


def test_an_archive_that_a_pull_emptied_is_not_filled_from_an_export(
    state: ArchiveState, export: Path, tmp_path: Path
) -> None:
    state.pull_from_messages(data.write_database(tmp_path / "chat.db", ["one"]), None)
    emptied = state.pull_from_messages(
        data.write_database_of_events(tmp_path / "chat.db"), None, max_removed_share=1.0
    )
    assert emptied.status == "done" and emptied.counts["messages_removed"] == 1
    assert state.archive().summary()["messages"] == 0
    assert state.ensure_from_export(export).status == "kept"
    assert state.archive().summary()["messages"] == 0


def test_a_file_that_is_no_archive_is_not_replaced(state: ArchiveState, export: Path) -> None:
    state.path.write_text("these are not the bytes of an archive, and they may be something to keep", encoding="utf-8")
    result = state.ensure_from_export(export)
    assert result.status == "failed"
    assert result.reason == "The file archive.sqlite3 is not an archive that Recall can open. It was not changed."
    assert state.path.read_text(encoding="utf-8").startswith("these are not the bytes")


def test_readers_in_other_threads_never_see_a_broken_archive(state: ArchiveState, export: Path) -> None:
    state.ensure_from_export(export)
    seen: list[int] = []
    problems: list[BaseException] = []
    stop = threading.Event()

    def read() -> None:
        while not stop.is_set():
            try:
                seen.append(state.archive().summary()["messages"])
            except ArchiveError:
                continue  # the handle was closed for a swap: the next call opens the new file
            except BaseException as error:  # noqa: BLE001 - any other error is the finding
                problems.append(error)
                return

    readers = [threading.Thread(target=read) for _ in range(4)]
    for reader in readers:
        reader.start()
    for size in (4, 5, 6):
        data.write_export(export, [f"text {number}" for number in range(size)])
        assert state.ensure_from_export(export).status == "built"
    stop.set()
    for reader in readers:
        reader.join(timeout=10)
    assert problems == []
    assert set(seen) <= {3, 4, 5, 6}
    assert state.archive().summary()["messages"] == 6
