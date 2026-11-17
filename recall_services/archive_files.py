"""The files of the archive: what tells one export from another, and the steps that move a database file.

Used by `archive_state`. Nothing here holds a lock of the state.
"""

from __future__ import annotations

import itertools
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from archive import Archive, ArchiveError, ImportBusyError
from recall_services import files
from recall_services.messages import LEGACY_PREFIX

logger = logging.getLogger(__name__)

SWAP_WAIT_SECONDS = 2.0  # how long a load waits for a writer or a reader of another process
_BUSY_CODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


@dataclass(frozen=True)
class ExportSignature:
    """What tells one state of an export file from another: its path, its write time and its size."""

    path: str
    mtime_ns: int
    size: int

    @classmethod
    def of(cls, export: str | os.PathLike[str]) -> ExportSignature | None:
        """The signature of the file, or None when there is no such file."""
        try:
            resolved = Path(export).expanduser().resolve()
            found = resolved.stat()
        except (OSError, ValueError, RuntimeError):  # also a text that is no path, and a home that is not there
            return None
        return cls(str(resolved), found.st_mtime_ns, found.st_size) if resolved.is_file() else None

    @classmethod
    def from_note(cls, note: object) -> ExportSignature | None:
        held = note.get("export") if isinstance(note, dict) and note.get("source") == "export" else None
        if not isinstance(held, dict):
            return None
        path, mtime_ns, size = held.get("path"), held.get("mtime_ns"), held.get("size")
        if isinstance(path, str) and type(mtime_ns) is int and type(size) is int:
            return cls(path, mtime_ns, size)
        return None


def holds_pull(archive: Archive) -> bool:
    """True when the archive holds a message that came from a Messages database, not from the legacy bridge."""
    newest = archive.conversations(limit=1, order="recent")
    if newest:
        found = archive.messages(newest[0]["id"], limit=1)["messages"]
    else:
        found = list(itertools.islice(archive.iter_messages(None, no_conversation=True), 1))
    return bool(found) and not str(found[0]["id"]).startswith(LEGACY_PREFIX)


def stat_of(path: Path) -> tuple[int, int] | None:
    try:
        found = path.stat()
    except OSError:
        return None
    return (found.st_mtime_ns, found.st_size)


def place(named: object) -> Path | None:
    """The path that a caller named, or None when the text is no path."""
    try:
        return Path(str(named)).expanduser()
    except (ValueError, RuntimeError):
        return None


def remove_database(database: Path) -> None:
    files.remove_files([database, *files.sidecars(database)])


def remove_quietly(database: Path) -> None:
    """Remove a work file after the answer is known. A file that stays is said in the log only."""
    try:
        remove_database(database)
    except OSError as exc:
        logger.warning("a work file of the archive could not be removed: %s", type(exc).__name__)


def holds_nothing(database: Path) -> bool:
    """True when the file is an archive with no row, or no archive that can be opened."""
    try:
        with Archive.open(database) as held:
            summary = held.summary()
    except ArchiveError:
        return True
    return not summary["messages"] and not summary["conversations"]


def gather(database: Path) -> None:
    """Bring an old archive into its one file, and make sure that nobody is in a write or a read of it.

    SQLite keeps the last writes in `-wal` and finds that file by name: one
    left beside the new archive would be read into it. So the old one is
    emptied into its database first. A move that fails after this leaves the
    old archive whole. Raises ImportBusyError when another connection is in
    the way.
    """
    if not database.exists():
        files.remove_files(files.sidecars(database))
        return
    conn = sqlite3.connect(database, timeout=SWAP_WAIT_SECONDS, isolation_level=None)
    try:
        blocked = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:
        if exc.sqlite_errorcode & 0xFF in _BUSY_CODES:
            raise ImportBusyError("another connection writes the archive") from exc
        raise
    finally:
        conn.close()
    if blocked:
        raise ImportBusyError("another connection reads the archive")
    files.remove_files(files.sidecars(database))


def settle(database: Path) -> None:
    """Bring every page of a new database into its one file, so that the file can be moved alone."""
    conn = sqlite3.connect(database, isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    for sidecar in files.sidecars(database):
        if sidecar.exists() and sidecar.stat().st_size > 0 and not sidecar.name.endswith("-shm"):
            raise ArchiveError(f"the new archive is not in one file: {sidecar.name} is not empty")
    files.remove_files(files.sidecars(database))
