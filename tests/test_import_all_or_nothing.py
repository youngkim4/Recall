"""An import changes the whole archive, or nothing of it.

An update makes the archive hold what the database of Messages holds: new
messages come in, changed ones change, and those that Messages no longer has
go. A pull that fails at any step leaves the archive as it was, row for row,
and a first pull that fails leaves no archive file. Nothing of a second source
is merged. The export of an earlier build is never written and never removed.
"""
import errno
import sqlite3

import pytest

import ui_server
from archive import directory, importer, ledger, runs, source
from recall_services import archive_state
from setup_status import DbProbe
from tests.archive_kit import (
    KEPT,
    beside,
    bridged,
    held,
    import_from,
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    picture,
    pulled,
    put_in_messages,
    run,
    unchanged,
    write_database,
    write_export,
    write_rows,
)

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives", "messages_home")

FIRST = ["one", "two", "three", "four", "five"]
FIRST_HELD = {f"guid-{text}": text for text in FIRST}
# against FIRST: one message changed, one is gone, one is new
LATER = [("guid-one", "one, said again"), ("guid-two", "two"), ("guid-three", "three"), ("guid-four", "four"), ("guid-six", "six")]
QUERY = "SELECT m.ROWID FROM message m"


@pytest.fixture
def filled(tmp_path):
    """An archive that a pull filled with the messages of FIRST."""
    return pulled(tmp_path / "saves" / "archive.sqlite3", FIRST)


def later_database(folder):
    return write_rows(folder / "later" / "chat.db", LATER)


# ---- an update makes the archive hold what the database holds ----


def test_an_update_brings_in_the_messages_that_are_new(filled, tmp_path):
    job = import_from(write_database(tmp_path / "chat.db", [*FIRST, "six"]), filled)

    assert job["status"] == "completed", job["error"]
    assert held(filled) == {**FIRST_HELD, "guid-six": "six"}


def test_an_update_changes_a_message_that_changed_in_messages(filled, tmp_path):
    rows = [(guid, "two, said again" if guid == "guid-two" else text) for guid, text in FIRST_HELD.items()]

    job = import_from(write_rows(tmp_path / "chat.db", rows), filled)

    assert job["status"] == "completed", job["error"]
    assert held(filled) == {**FIRST_HELD, "guid-two": "two, said again"}


def test_an_update_removes_a_message_that_messages_no_longer_has(filled, tmp_path):
    job = import_from(write_database(tmp_path / "chat.db", FIRST[:4]), filled)

    assert job["status"] == "completed", job["error"]
    assert held(filled) == {guid: text for guid, text in FIRST_HELD.items() if text != "five"}


def test_after_an_update_the_archive_holds_what_the_database_holds(filled, tmp_path):
    before = picture(filled)

    job = import_from(later_database(tmp_path), filled)

    assert job["status"] == "completed", job["error"]
    assert held(filled) == dict(LATER)
    assert not unchanged(filled, before)


def test_an_update_from_the_same_database_changes_no_message(filled, tmp_path):
    job = import_from(write_database(tmp_path / "chat.db", FIRST), filled)

    assert job["status"] == "completed", job["error"]
    assert held(filled) == FIRST_HELD


def test_nothing_of_an_export_is_merged_into_the_archive_that_a_pull_fills(tmp_path):
    export = bridged(tmp_path / "messages.csv", ["one", "only in the export"])

    job = import_from(write_database(tmp_path / "chat.db", ["one", "two"]), export)

    assert job["status"] == "completed", job["error"]
    assert held(export) == {"guid-one": "one", "guid-two": "two"}


# ---- a pull that fails leaves the archive as it was ----


def not_there(monkeypatch, folder):
    return folder / "gone" / "chat.db"


def not_a_database(monkeypatch, folder):
    wrong = folder / "notes.db"
    wrong.write_bytes(b"these are notes, and no database of any kind. " * 40)
    return wrong


def with_no_message(monkeypatch, folder):
    return write_database(folder / "empty" / "chat.db", [])


def closed_to_recall(monkeypatch, folder):
    hidden = lambda path=None, deep=False: DbProbe(status="fda_blocked", kind="live", path=str(path))  # noqa: E731
    monkeypatch.setattr(archive_state, "probe_database", hidden)
    return later_database(folder)


def read_stops_midway(monkeypatch, folder):
    """The reader raises after two batches of one row each: rows of this pull are written by then."""
    monkeypatch.setattr(importer, "BATCH_ROWS", 1)
    real = source.SourceReader.messages_after

    def two_batches_then_stop(reader, rowid, size):
        for count, batch in enumerate(real(reader, rowid, size)):
            if count == 2:
                raise sqlite3.OperationalError(f"disk I/O error in {QUERY}")
            yield batch

    monkeypatch.setattr(source.SourceReader, "messages_after", two_batches_then_stop)
    return later_database(folder)


def disk_fills_during_the_writes(monkeypatch, folder):
    """The second batch finds the disk full: the first batch is written by then."""
    monkeypatch.setattr(importer, "BATCH_ROWS", 1)
    real = ledger.write_messages
    calls = []

    def full_at_the_second(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise sqlite3.OperationalError("database or disk is full")
        return real(*args, **kwargs)

    monkeypatch.setattr(ledger, "write_messages", full_at_the_second)
    return later_database(folder)


def no_room_for_the_work_file(monkeypatch, folder):
    def no_room(*_args, **_more):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(source, "copy_source", no_room)
    return later_database(folder)


def last_step_fails(monkeypatch, folder):
    """Every row is written; the step that ends the run fails."""

    def fail(*_args, **_more):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(runs, "finish_run", fail)
    return later_database(folder)


def a_fault_that_is_not_of_a_database(monkeypatch, folder):
    def fail(*_args, **_more):
        raise RuntimeError("anything else")

    monkeypatch.setattr(directory, "settle", fail)
    return later_database(folder)


FAILURES = [
    not_there,
    not_a_database,
    with_no_message,
    closed_to_recall,
    read_stops_midway,
    disk_fills_during_the_writes,
    no_room_for_the_work_file,
    last_step_fails,
    a_fault_that_is_not_of_a_database,
]


@pytest.mark.parametrize("failure", FAILURES, ids=lambda failure: failure.__name__)
def test_an_update_that_fails_leaves_the_archive_as_it_was(filled, tmp_path, monkeypatch, failure):
    before = picture(filled)

    job = import_from(failure(monkeypatch, tmp_path), filled)

    assert job["status"] == "failed"
    assert job["error"].endswith(KEPT)
    assert unchanged(filled, before)
    assert held(filled) == FIRST_HELD
    # the handle that the server holds still reads it
    assert ui_server.what_the_archive_holds(filled)["messages"] == len(FIRST)
    assert beside(filled) == []


@pytest.mark.parametrize("failure", FAILURES, ids=lambda failure: failure.__name__)
def test_a_first_import_that_fails_leaves_no_archive_file(tmp_path, monkeypatch, failure):
    target = tmp_path / "new" / "archive.sqlite3"

    job = import_from(failure(monkeypatch, tmp_path), target)

    assert job["status"] == "failed"
    assert not target.exists()
    assert not target.parent.exists() or list(target.parent.iterdir()) == []
    # there was no archive, so nothing is said of one
    assert "The archive" not in job["error"]
    assert ui_server.holds_messages(target) is False


@pytest.mark.parametrize("target", ["messages.csv", "archive.sqlite3"])
def test_a_first_import_that_fails_leaves_no_archive_beside_an_export_with_no_message(tmp_path, monkeypatch, target):
    export = write_export(tmp_path / "new" / "messages.csv", [])
    kept = export.read_bytes()

    job = import_from(read_stops_midway(monkeypatch, tmp_path), tmp_path / "new" / target)

    assert job["status"] == "failed"
    assert [path.name for path in export.parent.iterdir()] == ["messages.csv"]
    assert export.read_bytes() == kept


def test_a_pull_that_is_cut_off_leaves_the_archive_as_it_was(filled, tmp_path, monkeypatch):
    # the thread of the job ends in a way that no code catches
    before = picture(filled)
    monkeypatch.setattr(importer, "BATCH_ROWS", 1)
    real = source.SourceReader.messages_after

    def one_batch_then_leave(reader, rowid, size):
        yield next(real(reader, rowid, size))
        raise SystemExit("the process is ending")

    monkeypatch.setattr(source.SourceReader, "messages_after", one_batch_then_leave)

    with pytest.raises(SystemExit):
        import_from(later_database(tmp_path), filled)

    assert unchanged(filled, before)
    assert beside(filled) == []


def test_a_first_pull_that_is_cut_off_leaves_no_archive_file(tmp_path, monkeypatch):
    target = tmp_path / "new" / "archive.sqlite3"
    monkeypatch.setattr(importer, "BATCH_ROWS", 1)
    real = source.SourceReader.messages_after

    def one_batch_then_leave(reader, rowid, size):
        yield next(real(reader, rowid, size))
        raise SystemExit("the process is ending")

    monkeypatch.setattr(source.SourceReader, "messages_after", one_batch_then_leave)

    with pytest.raises(SystemExit):
        import_from(later_database(tmp_path), target)

    assert not target.parent.exists() or list(target.parent.iterdir()) == []


def test_the_update_after_one_that_failed_runs(filled, tmp_path):
    # a failure keeps no lock and leaves no work file in the way
    with pytest.MonkeyPatch.context() as patch:
        failed = import_from(read_stops_midway(patch, tmp_path), filled)

    job = import_from(later_database(tmp_path), filled)

    assert failed["status"] == "failed"
    assert job["status"] == "completed", job["error"]
    assert held(filled) == dict(LATER)
    assert beside(filled) == []


def test_a_first_import_after_one_that_failed_runs(tmp_path):
    target = tmp_path / "new" / "archive.sqlite3"
    with pytest.MonkeyPatch.context() as patch:
        failed = import_from(read_stops_midway(patch, tmp_path), target)

    job = import_from(later_database(tmp_path), target)

    assert failed["status"] == "failed"
    assert job["status"] == "completed", job["error"]
    assert held(target) == dict(LATER)


def test_a_file_that_is_no_archive_is_not_written_over(tmp_path):
    put_in_messages(write_database(tmp_path / "chat.db", ["one"]))
    target = tmp_path / "saves" / "archive.sqlite3"
    target.parent.mkdir()
    target.write_bytes(b"not an archive, and long enough to be read as a page. " * 40)
    kept = target.read_bytes()

    job = run("setup_import", {"messagesPath": str(target), "includeContacts": False})

    assert job["status"] == "failed"
    assert job["error"] == f"The update failed. {KEPT}"
    assert target.read_bytes() == kept
    assert [path.name for path in target.parent.iterdir()] == ["archive.sqlite3"]


# ---- a database with no message to keep ----


def rows_and_no_message(folder):
    """A database whose rows are all events of a group: the check finds rows, and none is a message."""
    from tests.recall_services.state_data import write_database_of_events

    (folder / "events").mkdir()
    return write_database_of_events(folder / "events" / "chat.db")


def test_an_update_from_a_database_with_no_message_to_keep_leaves_the_archive(filled, tmp_path):
    before = picture(filled)

    job = import_from(rows_and_no_message(tmp_path), filled)

    assert job["status"] == "failed"
    assert job["error"].endswith(KEPT)
    assert unchanged(filled, before)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "product fault: a first import of a database whose rows hold no message to keep ends as completed "
        "with 0 messages and leaves an archive file with nothing in it; CONTRACTS.md says that a database "
        "with no message to keep is not imported"
    ),
)
def test_a_first_import_of_a_database_with_no_message_to_keep_makes_no_archive(tmp_path):
    target = tmp_path / "new" / "archive.sqlite3"

    job = import_from(rows_and_no_message(tmp_path), target)

    assert job["status"] == "failed"
    assert "holds no messages" in job["error"]
    assert not target.exists()


# ---- the export of an earlier build ----


EXPORT_FILES = {
    "messages.csv": None,
    "messages_attachments.csv": "message_id,filename\n1,kept.jpg\n",
    "messages_reactions.csv": "reaction_id,reaction_type\n1,kept\n",
}


@pytest.fixture
def earlier_export(tmp_path):
    """The three files that an earlier build wrote, as bytes by name. The archive is not made yet."""
    folder = tmp_path / "earlier"
    write_export(folder / "messages.csv", ["one", "two", "three"])
    for name, text in EXPORT_FILES.items():
        if text is not None:
            (folder / name).write_text(text, encoding="utf-8")
    return {folder / name: (folder / name).read_bytes() for name in EXPORT_FILES}


def as_it_was(export_files):
    return all(path.read_bytes() == kept for path, kept in export_files.items())


def stamps(export_files):
    return [path.stat().st_mtime_ns for path in export_files]


def test_a_load_of_the_export_leaves_its_three_files_as_they_were(earlier_export):
    export = next(iter(earlier_export))
    written = stamps(earlier_export)

    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "completed", job["error"]
    assert as_it_was(earlier_export) and stamps(earlier_export) == written
    assert beside(export.with_suffix(".sqlite3")) == sorted(EXPORT_FILES)


def test_a_pull_leaves_the_three_files_of_the_export_as_they_were(earlier_export, tmp_path):
    export = next(iter(earlier_export))
    written = stamps(earlier_export)
    run("bridge", {"messagesPath": str(export)})

    first = import_from(write_database(tmp_path / "chat.db", ["one", "two", "three", "four"]), export)
    second = import_from(write_database(tmp_path / "chat.db", ["one", "two", "three", "four", "five"]), export)

    assert (first["status"], second["status"]) == ("completed", "completed"), (first["error"], second["error"])
    assert len(held(export)) == 5
    assert as_it_was(earlier_export) and stamps(earlier_export) == written
    assert beside(export.with_suffix(".sqlite3")) == sorted(EXPORT_FILES)


def test_a_first_pull_beside_an_export_that_was_never_loaded_leaves_its_files(earlier_export, tmp_path):
    export = next(iter(earlier_export))
    put_in_messages(write_database(tmp_path / "chat.db", ["from messages"]))

    job = run("setup_import", {"messagesPath": str(export), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert held(export) == {"guid-from messages": "from messages"}
    assert as_it_was(earlier_export)


def test_a_pull_that_fails_leaves_the_three_files_of_the_export_as_they_were(earlier_export, tmp_path, monkeypatch):
    export = next(iter(earlier_export))
    run("bridge", {"messagesPath": str(export)})

    job = import_from(read_stops_midway(monkeypatch, tmp_path), export)

    assert job["status"] == "failed"
    assert as_it_was(earlier_export)
    assert beside(export.with_suffix(".sqlite3")) == sorted(EXPORT_FILES)
