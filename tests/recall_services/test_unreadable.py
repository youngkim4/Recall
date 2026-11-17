"""unreadable.py: one plain sentence for a Messages database that can not be read."""

from __future__ import annotations

import sqlite3

import pytest

from archive import SourceError
from recall_services import unreadable
from recall_services.unreadable import ARCHIVE_KEPT, DatabaseBusy, ImportStopped
from setup_status import DbProbe

LIVE = "/Users/someone/Library/Messages/chat.db"
COPY = "/Users/someone/Desktop/chat.db"


def probe(status: str, kind: str = "live", detail: str = "") -> DbProbe:
    return DbProbe(status=status, kind=kind, path=LIVE if kind == "live" else COPY, detail=detail)


@pytest.mark.parametrize("status", ["readable", "locked"])
def test_a_database_that_can_be_tried_stops_nothing(status: str) -> None:
    assert unreadable.why_unreadable(probe(status), note="") is None


@pytest.mark.parametrize(
    ("status", "kind", "sentence"),
    [
        ("missing", "live", "There is no Messages database on this Mac."),
        ("missing", "copy", f"There is no file at {COPY}."),
        ("empty", "live", "Messages holds no messages on this Mac."),
        ("empty", "copy", f"The copy at {COPY} holds no messages."),
        ("invalid", "live", "The file that Messages keeps on this Mac is not a Messages database."),
        ("invalid", "copy", f"The copy at {COPY} is not a Messages database."),
        ("error", "live", "The Messages database on this Mac could not be read."),
        ("error", "copy", f"The copy at {COPY} could not be read."),
    ],
)
def test_each_reason_is_one_sentence_that_names_the_archive_only_when_there_is_one(
    status: str, kind: str, sentence: str
) -> None:
    with_none = unreadable.why_unreadable(probe(status, kind), note=unreadable.archive_note(False))
    with_one = unreadable.why_unreadable(probe(status, kind), note=unreadable.archive_note(True))
    assert isinstance(with_none, ImportStopped) and str(with_none) == sentence
    assert str(with_one) == f"{sentence} {ARCHIVE_KEPT}"
    assert with_one.code == ""


@pytest.mark.parametrize("kind", ["live", "copy"])
def test_a_database_that_macos_keeps_closed_says_what_to_do(kind: str) -> None:
    said = str(unreadable.why_unreadable(probe("fda_blocked", kind), note=""))
    assert said == unreadable.closed_to_recall(probe("fda_blocked", kind))
    assert ("Full Disk Access" in said) is (kind == "live")
    assert ("Move the copy to your home folder" in said) is (kind == "copy")


def test_what_a_check_found_is_said_in_brackets() -> None:
    said = str(unreadable.why_unreadable(probe("invalid", "copy", detail="not a SQLite database"), note=""))
    assert said == f"The copy at {COPY} is not a Messages database (not a SQLite database)."


@pytest.mark.parametrize(
    ("raised", "kind_of_stop", "start"),
    [
        (sqlite3.OperationalError("database is locked"), DatabaseBusy, "The Messages database on this Mac is busy."),
        (sqlite3.OperationalError("unable to open database file"), ImportStopped, "Recall cannot read Messages."),
        (sqlite3.DatabaseError("file is not a database"), ImportStopped, "The file that Messages keeps on this Mac is not"),
        (sqlite3.OperationalError("no such table: message"), ImportStopped, "The file that Messages keeps on this Mac is not"),
        (sqlite3.OperationalError("disk I/O error"), ImportStopped, "The Messages database on this Mac could not be read."),
        (RuntimeError("database is locked"), ImportStopped, "The Messages database on this Mac could not be read."),
    ],
)
def test_what_a_reader_raised_is_told_by_its_kind_and_never_quoted(
    raised: Exception, kind_of_stop: type, start: str
) -> None:
    stopped = unreadable.reading_stopped(raised, probe("readable"), note=f" {ARCHIVE_KEPT}")
    assert type(stopped) is kind_of_stop
    assert str(stopped).startswith(start) and str(stopped).endswith(ARCHIVE_KEPT)
    assert str(raised) not in str(stopped)


def test_an_error_that_sqlite3_caused_is_read_through_its_cause() -> None:
    """The importer wraps what sqlite3 raised. The sentence still says that the database is busy."""
    try:
        try:
            raise sqlite3.OperationalError("database is locked")
        except sqlite3.OperationalError as exc:
            raise SourceError("cannot copy the source: SELECT text FROM message") from exc
    except SourceError as wrapped:
        stopped = unreadable.reading_stopped(wrapped, probe("readable", "copy"), note="")
    assert isinstance(stopped, DatabaseBusy)
    assert str(stopped) == f"The copy at {COPY} is busy. Close the app that has it open, then try again."
    assert "SELECT" not in str(stopped)


def test_an_import_that_stops_can_carry_a_code_for_the_client() -> None:
    assert ImportStopped("One sentence.", code="removal_limit").code == "removal_limit"
    assert ImportStopped("One sentence.").code == ""
