"""What an import says when it stops.

The reason is a plain sentence. No text of a query, no trace and no name of a
Python error is in it, and none is in the lines that the job wrote: what the
reader raised goes to the log of the server. The sentence says that the
archive was not changed only when there was an archive.
"""
import sqlite3

import pytest

import ui_server
from archive import importer, run_control, source
from setup_status import DbProbe
from tests.archive.builders import ChatDbBuilder
from tests.archive_kit import (
    KEPT,
    PERSON,
    held,
    import_from,
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    picture,
    pulled,
    put_in_messages,
    run,
    said,
    unchanged,
    write_database,
)
from tests.recall_services.state_data import START

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives", "messages_home")

COPY = "/copies/chat.db"
QUERY = "SELECT message.ROWID AS message_id FROM message"
# words that a reader of the app must never be shown
NOT_PLAIN = ("SELECT", "Traceback", "Error", "Exception", "sqlite3", ".py")
FDA = "Give it Full Disk Access"


def probe(status, kind="live", detail=""):
    return DbProbe(status=status, kind=kind, path=COPY, detail=detail)


def raising(error):
    def reader(*_args, **_more):
        raise error

    return reader


@pytest.fixture
def filled(tmp_path):
    """An archive that a pull filled with three messages."""
    return pulled(tmp_path / "saves" / "archive.sqlite3", ["one", "two", "three"])


def plain(job, raised=None):
    """True when nothing that the job tells the reader holds the words of code."""
    told = said(job)
    hidden = [*NOT_PLAIN, *([str(raised), type(raised).__name__] if raised is not None else [])]
    return not any(word in told for word in hidden)


# ---- what the check of the database says ----


@pytest.mark.parametrize("status", ["readable", "locked"])
def test_a_database_that_can_be_tried_stops_nothing(status):
    assert ui_server.why_unreadable(probe(status), note=f" {KEPT}") is None


@pytest.mark.parametrize("status, kind, words", [
    ("fda_blocked", "live", "Give it Full Disk Access"),
    ("fda_blocked", "copy", f"Recall cannot read the copy at {COPY}"),
    ("missing", "live", "There is no Messages database on this Mac."),
    ("missing", "copy", f"There is no file at {COPY}."),
    ("empty", "live", "Messages holds no messages on this Mac."),
    ("empty", "copy", f"The copy at {COPY} holds no messages."),
    ("invalid", "live", "The file that Messages keeps on this Mac is not a Messages database."),
    ("invalid", "copy", f"The copy at {COPY} is not a Messages database."),
    ("error", "live", "The Messages database on this Mac could not be read."),
    ("error", "copy", f"The copy at {COPY} could not be read."),
])
def test_each_reason_has_its_sentence(status, kind, words):
    stopped = ui_server.why_unreadable(probe(status, kind), note=f" {KEPT}")

    assert isinstance(stopped, ui_server.ImportStopped)
    assert words in str(stopped) and str(stopped).endswith(KEPT)


@pytest.mark.parametrize("kind", ["live", "copy"])
def test_a_file_that_could_not_be_read_is_not_said_to_be_another_kind_of_file(kind):
    stopped = ui_server.why_unreadable(probe("error", kind, detail="Input/output error"), note=f" {KEPT}")

    assert "is not a Messages database" not in str(stopped)
    assert "could not be read (Input/output error)." in str(stopped)


def test_what_the_check_found_is_told_with_the_reason():
    stopped = ui_server.why_unreadable(probe("invalid", "copy", detail="not a SQLite database"), note=f" {KEPT}")

    assert str(stopped) == f"The copy at {COPY} is not a Messages database (not a SQLite database). {KEPT}"


@pytest.mark.parametrize("status", ["fda_blocked", "missing", "empty", "invalid", "error"])
@pytest.mark.parametrize("kind", ["live", "copy"])
def test_the_archive_is_spoken_of_only_when_there_is_one(status, kind):
    with_archive = str(ui_server.why_unreadable(probe(status, kind), note=f" {KEPT}"))
    without = str(ui_server.why_unreadable(probe(status, kind), note=""))

    assert with_archive == f"{without} {KEPT}"
    assert "archive" not in without


def test_a_probe_error_of_the_system_is_told_without_its_number(tmp_path, monkeypatch):
    import builtins

    import setup_status

    real_open = builtins.open

    def unreadable(path, *args, **kwargs):
        if str(path).endswith("chat.db"):
            raise OSError(5, "Input/output error", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unreadable)

    found = setup_status.probe_database(tmp_path / "chat.db")

    assert (found.status, found.detail) == ("error", "Input/output error")


# ---- a database that the check stops ----


def test_a_first_import_that_cannot_start_says_nothing_of_an_archive(tmp_path):
    job = import_from(tmp_path / "gone.db", tmp_path / "new" / "archive.sqlite3")

    assert job["status"] == "failed"
    assert job["error"] == f"There is no file at {(tmp_path / 'gone.db').resolve()}."
    assert job["code"] is None


def test_an_update_that_cannot_start_says_that_the_archive_was_not_changed(filled, tmp_path):
    job = import_from(tmp_path / "gone.db", filled)

    assert job["status"] == "failed"
    assert job["error"] == f"There is no Messages database on this Mac. {KEPT}"
    assert job["code"] is None


def not_a_database(folder):
    wrong = folder / "notes.db"
    wrong.write_bytes(b"these are notes, and no database of any kind. " * 40)
    return wrong


def test_a_file_that_is_no_database_is_told_in_a_plain_sentence(filled, tmp_path):
    update = import_from(not_a_database(tmp_path), filled)
    first = import_from(not_a_database(tmp_path), tmp_path / "new" / "archive.sqlite3")

    live = "The file that Messages keeps on this Mac is not a Messages database (not a SQLite database)."
    copy = f"The copy at {(tmp_path / 'notes.db').resolve()} is not a Messages database (not a SQLite database)."
    assert (update["status"], update["error"]) == ("failed", f"{live} {KEPT}")
    assert (first["status"], first["error"]) == ("failed", copy)


def test_a_database_with_no_message_is_told_in_a_plain_sentence(filled, tmp_path):
    update = import_from(write_database(tmp_path / "empty.db", []), filled)
    first = import_from(write_database(tmp_path / "empty.db", []), tmp_path / "new" / "archive.sqlite3")

    assert (update["status"], update["error"]) == ("failed", f"Messages holds no messages on this Mac. {KEPT}")
    assert (first["status"], first["error"]) == ("failed", f"The copy at {(tmp_path / 'empty.db').resolve()} holds no messages.")


# ---- a read that fails ----


def cannot_open(monkeypatch, raised):
    """The importer cannot bring the database to its work file: sqlite3 raised `raised`."""

    def fail(path, *_more):
        raise source.SourceError(f"cannot open {path}: {raised}") from raised

    monkeypatch.setattr(source, "copy_source", fail)


def stops_midway(monkeypatch, raised):
    """The reader raises `raised` after one batch of one row."""
    monkeypatch.setattr(importer, "BATCH_ROWS", 1)
    real = source.SourceReader.messages_after

    def one_batch_then_stop(reader, rowid, size):
        yield next(real(reader, rowid, size))
        raise raised

    monkeypatch.setattr(source.SourceReader, "messages_after", one_batch_then_stop)


AT_THE_OPEN = [
    (sqlite3.OperationalError("unable to open database file"), ("Recall cannot read Messages.", FDA)),
    (sqlite3.OperationalError("authorization denied"), ("Recall cannot read Messages.", FDA)),
    (sqlite3.OperationalError("no such table: handle"), ("The file that Messages keeps", "is not a Messages database.")),
    (sqlite3.OperationalError("no such column: message.attributedBody"), ("The file that Messages keeps", "is not a Messages database.")),
    (sqlite3.DatabaseError("file is not a database"), ("The file that Messages keeps", "is not a Messages database.")),
    (sqlite3.DatabaseError("database disk image is malformed"), ("The Messages database", "could not be read.")),
    (sqlite3.OperationalError("disk I/O error"), ("The Messages database", "could not be read.")),
    (sqlite3.OperationalError("database is locked"), ("The Messages database", "is busy. Quit Messages or wait a moment, then try again.")),
    (sqlite3.OperationalError(f"near FROM: syntax error in {QUERY}"), ("The Messages database", "could not be read.")),
]

IN_THE_MIDDLE = [
    sqlite3.OperationalError("disk I/O error"),
    sqlite3.OperationalError("database or disk is full"),
    sqlite3.DatabaseError("database disk image is malformed"),
    sqlite3.OperationalError(f"near FROM: syntax error in {QUERY}"),
    RuntimeError(f"anything else, even with {QUERY} in it"),
    OSError(28, "No space left on device"),
    KeyError("handle_id"),
]


@pytest.mark.parametrize("raised, words", AT_THE_OPEN, ids=lambda value: str(value) if isinstance(value, Exception) else "")
def test_a_database_that_cannot_be_opened_is_told_in_a_plain_sentence(filled, tmp_path, monkeypatch, capfd, raised, words):
    before = picture(filled)
    cannot_open(monkeypatch, raised)

    job = import_from(write_database(tmp_path / "chat.db", ["one", "two", "three", "four"]), filled)

    starts, says = words
    assert job["status"] == "failed"
    assert job["error"].startswith(starts) and says in job["error"] and job["error"].endswith(KEPT)
    # neither the reason nor a line of the job holds what the reader raised
    assert plain(job, raised)
    assert job["code"] is None
    assert unchanged(filled, before)
    # the log of the server holds it
    assert str(raised) in capfd.readouterr().err


@pytest.mark.parametrize("raised", IN_THE_MIDDLE, ids=str)
def test_a_read_that_fails_in_the_middle_is_told_in_a_plain_sentence(filled, tmp_path, monkeypatch, capfd, raised):
    before = picture(filled)
    stops_midway(monkeypatch, raised)

    job = import_from(write_database(tmp_path / "chat.db", ["one", "two", "three", "four"]), filled)

    assert job["status"] == "failed"
    assert job["error"] == f"The update failed. {KEPT}"
    assert plain(job, raised)
    assert unchanged(filled, before)
    assert str(raised) in capfd.readouterr().err


@pytest.mark.parametrize("raised, says", [
    (sqlite3.OperationalError("unable to open database file"), "macOS keeps that folder closed to it"),
    (sqlite3.OperationalError("no such table: handle"), "is not a Messages database."),
    (sqlite3.OperationalError("disk I/O error"), "could not be read."),
    (sqlite3.OperationalError("database is locked"), "is busy. Close the app that has it open, then try again."),
], ids=lambda value: str(value) if isinstance(value, Exception) else "")
def test_a_copy_that_cannot_be_opened_in_a_first_setup_is_named_and_no_archive_is_spoken_of(tmp_path, monkeypatch, raised, says):
    copy = write_database(tmp_path / "copy.db", ["one"])
    cannot_open(monkeypatch, raised)

    job = import_from(copy, tmp_path / "new" / "archive.sqlite3")

    assert job["status"] == "failed"
    assert str(copy.resolve()) in job["error"] and says in job["error"]
    assert "The archive" not in job["error"]
    assert plain(job, raised)


@pytest.mark.parametrize("raised", IN_THE_MIDDLE, ids=str)
def test_a_first_setup_that_fails_in_the_middle_speaks_of_no_archive(tmp_path, monkeypatch, raised):
    copy = write_database(tmp_path / "copy.db", ["one", "two"])
    stops_midway(monkeypatch, raised)

    job = import_from(copy, tmp_path / "new" / "archive.sqlite3")

    assert job["status"] == "failed"
    assert job["error"] == "The update failed."
    assert plain(job, raised)


def test_messages_itself_that_macos_closes_during_the_read_asks_for_full_disk_access(messages_home, filled, monkeypatch):
    write_database(messages_home / "chat.db", ["one", "two", "three"])
    cannot_open(monkeypatch, sqlite3.OperationalError("unable to open database file"))

    job = run("export", {"messagesPath": str(filled)})

    assert job["status"] == "failed"
    assert "Full Disk Access" in job["error"] and job["error"].endswith(KEPT)


def test_a_database_that_lacks_a_table_is_told_with_no_text_of_a_query(filled, tmp_path):
    # the check passes: the messages are there. A table that the importer reads is not.
    later = write_database(tmp_path / "chat.db", ["one", "two", "three"])
    conn = sqlite3.connect(str(later))
    conn.execute("DROP TABLE handle")
    conn.commit()
    conn.close()
    before = picture(filled)

    job = import_from(later, filled)

    assert job["status"] == "failed"
    assert job["error"] == f"The file that Messages keeps on this Mac is not a Messages database. {KEPT}"
    assert plain(job) and "FROM message" not in said(job)
    assert unchanged(filled, before)


def test_another_program_that_writes_the_archive_is_an_import_that_runs(messages_home, filled, monkeypatch):
    write_database(messages_home / "chat.db", ["one", "two", "three", "four"])
    monkeypatch.setattr(run_control, "LOCK_WAIT_SECONDS", 0.05)
    writer = sqlite3.connect(filled, isolation_level=None)
    try:
        writer.execute("BEGIN IMMEDIATE")
        job = run("setup_import", {"messagesPath": str(filled), "includeContacts": False})
    finally:
        writer.execute("ROLLBACK")
        writer.close()

    assert (job["status"], job["error"]) == ("failed", "An import is already running. Wait until it ends.")
    assert len(held(filled)) == 3


# ---- an import that is written ----


WRITTEN_NOT_COUNTED = "The archive is written. Recall could not read it back to count it."


def test_an_import_whose_archive_is_written_ends_as_completed_with_a_warning(filled, tmp_path, monkeypatch, capfd):
    # the archive is written; what fails comes after it, so the import did its work
    later = write_database(tmp_path / "chat.db", ["one", "two", "three", "four"])
    monkeypatch.setattr(ui_server, "archive_summary", raising(RuntimeError(f"a failure with {QUERY} in it")))

    job = import_from(later, filled)

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 4, "conversations": 1, "contactsSkipped": False, "warning": WRITTEN_NOT_COUNTED}
    assert WRITTEN_NOT_COUNTED in said(job)
    assert plain(job)
    assert len(held(filled)) == 4
    assert QUERY in capfd.readouterr().err


def test_a_first_import_that_cannot_be_counted_is_completed_too(tmp_path, monkeypatch):
    target = tmp_path / "new" / "archive.sqlite3"
    monkeypatch.setattr(ui_server, "what_the_archive_holds", raising(sqlite3.OperationalError("disk I/O error")))

    job = import_from(write_database(tmp_path / "copy.db", ["one", "two"]), target)

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 2, "conversations": 1, "contactsSkipped": False, "warning": WRITTEN_NOT_COUNTED}
    assert sorted(held(target).values()) == ["one", "two"]


def test_an_import_counts_messages_the_way_the_overview_does(tmp_path):
    # one message is in no conversation: the app shows it nowhere, so it is not in the count
    builder = ChatDbBuilder(tmp_path / "copy.db")
    person = builder.handle(PERSON)
    thread = builder.chat(PERSON, members=[person])
    for text in ("one", "two"):
        builder.message(thread, when=START, text=text, handle=person)
    builder.message(None, when=START, text="in no conversation", handle=person)
    builder.close()
    target = tmp_path / "new" / "archive.sqlite3"

    job = import_from(tmp_path / "copy.db", target)

    assert job["status"] == "completed", job["error"]
    assert len(held(target)) == 3
    assert job["result"]["messages"] == ui_server.archive_summary(target)["messages"] == 2
    assert job["result"]["conversations"] == 1
    assert "warning" not in job["result"]


def test_previews_that_cannot_be_dropped_do_not_fail_an_import_that_is_written(filled, tmp_path, monkeypatch):
    later = write_database(tmp_path / "chat.db", ["one", "two", "three", "four"])
    monkeypatch.setattr(ui_server, "clear_cache_namespace", raising(RuntimeError("the store is closed")))

    job = import_from(later, filled)

    assert job["status"] == "completed", job["error"]
    assert "warning" not in job["result"]
    assert len(held(filled)) == 4


def test_contacts_that_cannot_be_read_are_told_in_a_plain_line(filled, tmp_path, monkeypatch, capfd):
    later = write_database(tmp_path / "chat.db", ["one", "two", "three", "four"])
    monkeypatch.setattr(ui_server, "export_contact_names", raising(RuntimeError(f"a failure with {QUERY} in it")))

    job = import_from(later, filled, contacts=True)

    assert job["status"] == "completed", job["error"]
    assert job["result"]["contactsSkipped"] is True
    assert "Contacts could not be read. Some people show as numbers." in said(job)
    assert plain(job)
    assert len(held(filled)) == 4
    assert QUERY in capfd.readouterr().err


def test_contacts_that_are_read_are_not_said_to_be_skipped(filled, tmp_path, monkeypatch):
    later = write_database(tmp_path / "chat.db", ["one", "two", "three", "four"])
    monkeypatch.setattr(ui_server, "export_contact_names", lambda: {"count": 0})

    job = import_from(later, filled, contacts=True)

    assert job["status"] == "completed", job["error"]
    assert job["result"]["contactsSkipped"] is False
    assert "Contacts could not be read" not in said(job)


JOB_FAILED = "The job failed. The log of the server says why."


@pytest.mark.parametrize("action", ["setup_import", "export"])
def test_a_fault_that_nobody_planned_for_is_told_without_its_words(filled, tmp_path, monkeypatch, capfd, action):
    # this is not a reason that an import gives: the reader is pointed to the log, and the trace is there
    put_in_messages(write_database(tmp_path / "chat.db", ["one", "two", "three", "four"]))
    raised = RuntimeError(f"a fault with {QUERY} in it")
    monkeypatch.setattr(ui_server, "pull_into_archive", raising(raised))

    job = run(action, {"messagesPath": str(filled), "includeContacts": False})

    assert (job["status"], job["error"], job["code"]) == ("failed", JOB_FAILED, "job_failed")
    assert plain(job, raised)
    logged = capfd.readouterr().err
    assert "Traceback" in logged and QUERY in logged
    assert len(held(filled)) == 3


def test_a_report_with_no_conversation_is_told_in_a_plain_sentence(filled):
    job = run("analyze", {"messagesPath": str(filled)})

    assert (job["status"], job["error"]) == ("failed", "Select a conversation before you start a report.")
    assert plain(job)


def test_a_copy_for_an_archive_that_holds_messages_is_refused_in_a_plain_sentence(filled, tmp_path):
    copy = write_database(tmp_path / "copy.db", ["from a copy"])
    put_in_messages(write_database(tmp_path / "chat.db", ["one", "two", "three"]))

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(filled), "includeContacts": False})

    assert job["status"] == "failed"
    assert job["error"] == f"An update reads Messages on this Mac. It does not read a copy. {KEPT}"
    assert plain(job)
