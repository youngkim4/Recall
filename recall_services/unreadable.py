"""Why a Messages database can not be read, in one plain sentence for the reader.

The one place for these words: the server and the archive state both import
them, so the guide, a job and a route say the same thing.

The check tells a database that macOS keeps closed from one that is not
there: without Full Disk Access macOS can show the file as missing.
`setup_status.probe_database` holds that rule.

A sentence ends with what is true of the archive (`archive_note`): nothing
when there is no archive, or that the archive was not changed.
"""

from __future__ import annotations

import sqlite3

from setup_status import DbProbe

ARCHIVE_KEPT = "The archive was not changed."
# a busy database is tried: Messages may only be in the middle of a write
TRIED = ("readable", "locked")
# what sqlite3 says, in small letters, when a database is held, closed to this app, or not one of Messages
BUSY_SIGNS = ("locked", "busy")
CLOSED_SIGNS = ("unable to open", "permission", "not permitted", "authoriz")
NOT_MESSAGES_SIGNS = ("no such table", "no such column", "not a database", "file is encrypted")


class ImportStopped(Exception):
    """An import that stops for a reason told to the reader in one plain sentence.

    `code` is a fixed word for the client, or "" when the sentence says it all.
    """

    def __init__(self, sentence: str, code: str = "") -> None:
        super().__init__(sentence)
        self.code = code


class DatabaseBusy(ImportStopped):
    """Another program holds the database."""


def archive_note(archive_there: bool) -> str:
    """What a failed import says of the archive: nothing when there is none, else that it was not changed."""
    return f" {ARCHIVE_KEPT}" if archive_there else ""


def closed_to_recall(probe: DbProbe) -> str:
    """What to do about a database that macOS does not let this app open."""
    if probe.kind == "live":
        return (
            "Recall cannot read Messages. Give it Full Disk Access in System Settings > "
            "Privacy & Security > Full Disk Access, then open Recall again."
        )
    return (
        f"Recall cannot read the copy at {probe.path}: macOS keeps that folder closed to it. "
        "Move the copy to your home folder and choose it again."
    )


def not_there(probe: DbProbe) -> str:
    """What to say where no database is."""
    if probe.kind == "live":
        return "There is no Messages database on this Mac."
    return f"There is no file at {probe.path}."


def no_messages(probe: DbProbe) -> str:
    """What to say of a database that holds no message to keep."""
    if probe.kind == "live":
        return "Messages holds no messages on this Mac."
    return f"The copy at {probe.path} holds no messages."


def not_messages(probe: DbProbe, said: str = "") -> str:
    """What to say of a file that is not a database of Messages. `said` is what a check found, in brackets."""
    what = "The file that Messages keeps on this Mac" if probe.kind == "live" else f"The copy at {probe.path}"
    return f"{what} is not a Messages database{said}."


def not_read(probe: DbProbe, said: str = "") -> str:
    """What to say of a database that could not be read, when nothing more is known."""
    what = "The Messages database on this Mac" if probe.kind == "live" else f"The copy at {probe.path}"
    return f"{what} could not be read{said}."


def held(probe: DbProbe) -> str:
    """What to say of a database that another program holds."""
    if probe.kind == "live":
        return "The Messages database on this Mac is busy. Quit Messages or wait a moment, then try again."
    return f"The copy at {probe.path} is busy. Close the app that has it open, then try again."


def _what_stops(probe: DbProbe) -> str:
    said = f" ({probe.detail})" if probe.detail else ""
    if probe.status == "fda_blocked":
        return closed_to_recall(probe)
    if probe.status == "missing":
        return not_there(probe)
    if probe.status == "empty":
        return no_messages(probe)
    if probe.status == "invalid":
        return not_messages(probe, said)
    # the file could not be looked at: it is not said to be another kind of file
    return not_read(probe, said)


def why_unreadable(probe: DbProbe, note: str) -> ImportStopped | None:
    """What stops an import before it starts, or None when the database can be tried.

    `note` is what the reason says of the archive (`archive_note`).
    """
    if probe.status in TRIED:
        return None
    return ImportStopped(f"{_what_stops(probe)}{note}")


def reading_stopped(exc: BaseException, probe: DbProbe, note: str) -> ImportStopped:
    """The plain sentence for what a reader raised.

    What the reader raised is not put into the sentence: it can hold the text
    of a query. Only an error of sqlite3, or one that such an error caused,
    is read for signs of its kind.
    """
    signs = _sqlite_words(exc)
    if any(sign in signs for sign in BUSY_SIGNS):
        return DatabaseBusy(f"{held(probe)}{note}")
    if any(sign in signs for sign in CLOSED_SIGNS):
        return ImportStopped(f"{closed_to_recall(probe)}{note}")
    if any(sign in signs for sign in NOT_MESSAGES_SIGNS):
        return ImportStopped(f"{not_messages(probe)}{note}")
    return ImportStopped(f"{not_read(probe)}{note}")


def _sqlite_words(exc: BaseException) -> str:
    """What sqlite3 said, in small letters, or "" when the error did not come from sqlite3."""
    seen: BaseException | None = exc
    for _ in range(4):
        if seen is None:
            return ""
        if isinstance(seen, sqlite3.Error):
            return str(seen).lower()
        seen = seen.__cause__
    return ""
