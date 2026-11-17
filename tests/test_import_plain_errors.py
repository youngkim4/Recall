"""What an import says when it stops.

The reason is a plain sentence. No text of a query is in it, and none is in the
lines the job wrote. It says that the archive was not changed only when there
was an archive. A busy database is tried once more.
"""
import sqlite3

import pandas as pd
import pytest

import ui_server
from setup_status import DbProbe
from tests.archive_kit import left_behind, messages_csv, run, said, unchanged
from tests.invented_db import messages_db

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_saved_names", "messages_home")

COPY = "/copies/chat.db"
KEPT = "The archive was not changed."
QUERY = "SELECT message.ROWID AS message_id FROM message"


def probe(status, kind="live", detail=""):
    return DbProbe(status=status, kind=kind, path=COPY, detail=detail)


def raising(error):
    def reader(_path, *_more):
        raise error

    return reader


def import_from(copy, archive_file):
    return run("setup_import", {"dbPath": str(copy), "messagesPath": str(archive_file), "includeContacts": False})


# ---- what the check of the database says ----


@pytest.mark.parametrize("status", ["readable", "locked"])
def test_a_database_that_can_be_tried_stops_nothing(status):
    assert ui_server.why_unreadable(probe(status), archive_there=True) is None


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
    stopped = ui_server.why_unreadable(probe(status, kind), archive_there=True)

    assert isinstance(stopped, ui_server.ImportStopped)
    assert words in str(stopped) and str(stopped).endswith(KEPT)


@pytest.mark.parametrize("kind", ["live", "copy"])
def test_a_file_that_could_not_be_read_is_not_said_to_be_another_kind_of_file(kind):
    stopped = ui_server.why_unreadable(probe("error", kind, detail="Input/output error"), archive_there=True)

    assert "is not a Messages database" not in str(stopped)
    assert "could not be read (Input/output error)." in str(stopped)


def test_what_the_check_found_is_told_with_the_reason():
    stopped = ui_server.why_unreadable(probe("invalid", "copy", detail="not a SQLite database"), archive_there=True)

    assert str(stopped) == f"The copy at {COPY} is not a Messages database (not a SQLite database). {KEPT}"


@pytest.mark.parametrize("status", ["fda_blocked", "missing", "empty", "invalid", "error"])
@pytest.mark.parametrize("kind", ["live", "copy"])
def test_the_archive_is_spoken_of_only_when_there_is_one(status, kind):
    with_archive = str(ui_server.why_unreadable(probe(status, kind), archive_there=True))
    without = str(ui_server.why_unreadable(probe(status, kind), archive_there=False))

    assert with_archive == f"{without} {KEPT}"
    assert "archive" not in without


def test_a_first_import_that_cannot_start_says_nothing_of_an_archive(tmp_path):
    job = import_from(tmp_path / "gone.db", tmp_path / "new" / "messages.csv")

    assert job["status"] == "failed"
    assert job["error"] == f"There is no file at {(tmp_path / 'gone.db').resolve()}."


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


# ---- a read that fails ----


@pytest.mark.parametrize("raised, words", [
    (sqlite3.OperationalError("unable to open database file"), "macOS keeps that folder closed to it"),
    (sqlite3.OperationalError("authorization denied"), "macOS keeps that folder closed to it"),
    (sqlite3.OperationalError("no such table: handle"), "is not a Messages database."),
    (sqlite3.OperationalError("no such column: message.attributedBody"), "is not a Messages database."),
    (sqlite3.DatabaseError("file is not a database"), "is not a Messages database."),
    (sqlite3.DatabaseError("database disk image is malformed"), "could not be read."),
    (sqlite3.OperationalError("disk I/O error"), "could not be read."),
    (pd.errors.DatabaseError(f"Execution failed on sql '{QUERY}': near FROM"), "could not be read."),
    (RuntimeError(f"anything else, even with {QUERY} in it"), "could not be read."),
])
@pytest.mark.parametrize("reader", ["extract_messages", "extract_attachments", "extract_reactions"])
def test_a_read_that_fails_is_told_in_a_plain_sentence(archive, tmp_path, monkeypatch, raised, words, reader):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    monkeypatch.setattr(ui_server, reader, raising(raised))

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert job["error"].startswith(("Recall cannot read the copy at", "The copy at"))
    assert words in job["error"] and job["error"].endswith(KEPT)
    # neither the reason nor a line the job wrote holds what the reader raised
    assert str(raised) not in said(job) and "SELECT" not in said(job) and "Traceback" not in said(job)
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_messages_itself_that_macos_closes_during_the_read_asks_for_full_disk_access(messages_home, archive, monkeypatch):
    messages_db(messages_home / "chat.db", ["one"])
    monkeypatch.setattr(ui_server, "extract_messages", raising(sqlite3.OperationalError("unable to open database file")))

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert "Full Disk Access" in job["error"] and job["error"].endswith(KEPT)


def test_a_database_that_lacks_a_table_is_told_with_no_text_of_a_query(archive, tmp_path):
    # the check passes: the messages are there. A table that the readers join is not.
    copy = messages_db(tmp_path / "copy.db", ["one"])
    conn = sqlite3.connect(str(copy))
    conn.execute("DROP TABLE handle")
    conn.commit()
    conn.close()

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert job["error"] == f"The copy at {copy.resolve()} is not a Messages database. {KEPT}"
    assert "SELECT" not in said(job) and "FROM message" not in said(job)
    assert unchanged(archive)


def test_an_archive_that_cannot_be_read_back_is_not_said_to_be_as_it_was(archive, tmp_path, monkeypatch):
    # the three files are written; what fails comes after them
    copy = messages_db(tmp_path / "copy.db", ["one"])
    monkeypatch.setattr(ui_server, "cached_messages_bundle", raising(RuntimeError(f"a failure with {QUERY} in it")))

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert job["error"] == "The archive was written, but Recall could not read it back. Try again."
    assert "SELECT" not in said(job) and "Traceback" not in said(job)
    assert not unchanged(archive)


def test_contacts_that_cannot_be_read_are_told_in_a_plain_line(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    monkeypatch.setattr(ui_server, "export_contact_names", raising(RuntimeError(f"a failure with {QUERY} in it")))

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(messages_csv(archive)), "includeContacts": True})

    assert job["status"] == "completed", job["error"]
    assert job["result"]["contactsSkipped"] is True
    assert "Contacts could not be read. Some people show as numbers." in said(job)
    assert "SELECT" not in said(job) and "Traceback" not in said(job)


def test_a_job_of_another_kind_keeps_its_trace(archive):
    # this is not a reason that an import gives, so the job says where it came from
    job = run("analyze", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert job["error"] == "Select a contact before starting analysis"
    assert "Traceback" in said(job)


# ---- a busy database ----


@pytest.fixture
def no_wait(monkeypatch):
    """The wait before the second try, as the list of waits that were asked for."""
    waits = []
    monkeypatch.setattr(ui_server.time, "sleep", waits.append)
    return waits


def busy_for(monkeypatch, times):
    """Make the first `times` reads of the messages find the database busy. Gives the list of reads."""
    real = ui_server.extract_messages
    reads = []

    def reader(path):
        reads.append(path)
        if len(reads) <= times:
            raise sqlite3.OperationalError("database is locked")
        return real(path)

    monkeypatch.setattr(ui_server, "extract_messages", reader)
    return reads


def busy_lines(job):
    return [entry["message"] for entry in job["logs"] if "busy" in entry["message"]]


def test_a_busy_database_is_tried_once_more(messages_home, archive, monkeypatch, no_wait):
    messages_db(messages_home / "chat.db", ["one"])
    reads = busy_for(monkeypatch, times=1)

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "completed", job["error"]
    assert (len(reads), no_wait) == (2, [ui_server.BUSY_WAIT_SECONDS])
    assert busy_lines(job) == ["The database is busy. Trying again in a moment"]


def test_a_database_that_is_busy_twice_stops_the_import(messages_home, archive, monkeypatch, no_wait):
    messages_db(messages_home / "chat.db", ["one"])
    reads = busy_for(monkeypatch, times=5)

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert (len(reads), no_wait) == (2, [ui_server.BUSY_WAIT_SECONDS])
    assert job["error"] == f"The Messages database on this Mac is busy. Quit Messages or wait a moment, then try again. {KEPT}"
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_a_database_that_is_not_busy_is_read_once(messages_home, archive, monkeypatch, no_wait):
    messages_db(messages_home / "chat.db", ["one"])
    reads = busy_for(monkeypatch, times=0)

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "completed", job["error"]
    assert (len(reads), no_wait, busy_lines(job)) == (1, [], [])


def test_a_failure_that_is_not_busy_is_not_tried_again(messages_home, archive, monkeypatch, no_wait):
    messages_db(messages_home / "chat.db", ["one"])
    reads = []

    def reader(path):
        reads.append(path)
        raise sqlite3.OperationalError("no such table: handle")

    monkeypatch.setattr(ui_server, "extract_messages", reader)

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert (len(reads), no_wait, busy_lines(job)) == (1, [], [])


def test_export_when_free_gives_what_the_second_try_gives(messages_home, tmp_path, monkeypatch, no_wait):
    live = messages_db(messages_home / "chat.db", ["one", "two"])
    busy_for(monkeypatch, times=1)

    result = ui_server.export_when_free("no-such-job", live, tmp_path / "messages.csv")

    assert (result["messages"], result["messagesPath"]) == (2, str(tmp_path / "messages.csv"))


def test_a_busy_copy_is_named_in_what_went_wrong(archive, tmp_path, monkeypatch, no_wait):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    busy_for(monkeypatch, times=5)

    job = import_from(copy, messages_csv(archive))

    assert job["error"] == f"The copy at {copy.resolve()} is busy. Close the app that has it open, then try again. {KEPT}"
