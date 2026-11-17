"""Where the archive is, one read handle on it, and the two ways to fill it.

    state = archive_state.shared()
    state.ensure_from_export(messages_csv)      # the bridge: run on today's export
    state.pull_from_messages(db_path, contacts) # an update: mirror a Messages database
    state.archive().search("ferry")

The archive is at `archive.default_archive_path()` unless a path is given.

The read handle is one `Archive` for every thread. `reopen()` closes it, and
the next `archive()` opens the file that is there then. `generation` goes up
each time the content of the archive changed: a cache made from the archive
is good for one generation.

`ensure_from_export` loads an export into a new file and moves that file into
place in one step. A reader that still holds the old handle gets
`ArchiveError` from its next query and asks for `archive()` again.

`pull_from_messages` keeps the rules of "Import and update" in CONTRACTS.md:
- The caller names the database. Nothing here chooses one, and nothing calls
  a pull by itself.
- A database that can not be read gives the reason (`setup_status.probe_database`).
- A failure leaves the archive as it was.
- Nothing of a second source is merged. An export never replaces an archive
  that a pull filled.
- No path of a database is kept: the note beside the archive says only
  where its content came from.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from archive import (
    Archive,
    ArchiveError,
    ArchiveNotFoundError,
    ImportBusyError,
    ImportResult,
    RemovalLimitError,
    SourceError,
    SourceFormatError,
    default_archive_path,
    import_chat_db,
    import_legacy_csv,
)
from recall_services import files, unreadable
from recall_services.messages import LEGACY_PREFIX
from recall_services.outcomes import (
    ARCHIVE_KEPT,
    BUSY,
    EnsureResult,
    PullBusy,
    PullCancelled,
    PullDone,
    PullFailed,
    PullRefused,
    PullResult,
)
from setup_status import probe_database

NOTE_SUFFIX = ".source.json"
BUILD_SUFFIX = ".building"
WORK_SUFFIX = ".import-work"
DEFAULT_MAX_REMOVED_SHARE = 0.2
KEPT_PULL = "The archive holds what Messages holds. An export does not replace it."
Contacts = Sequence[Mapping[str, object]] | None


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
        except OSError:
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


def _counts(result: ImportResult) -> dict[str, int]:
    return {name: int(value) for name, value in result.counts.items()}


def _detail(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


def _holds_pull(archive: Archive) -> bool:
    """True when the archive holds a message that came from a Messages database, not from the legacy bridge."""
    newest = archive.conversations(limit=1, order="recent")
    if newest:
        found = archive.messages(newest[0]["id"], limit=1)["messages"]
    else:
        found = list(itertools.islice(archive.iter_messages(None, no_conversation=True), 1))
    return bool(found) and not str(found[0]["id"]).startswith(LEGACY_PREFIX)


def _remove_database(database: Path) -> None:
    files.remove_files([database, *files.sidecars(database)])


def _settle(database: Path) -> None:
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


class ArchiveState:
    """The archive file, its shared read handle, and the two ways to fill it. Safe to use from several threads."""

    def __init__(self, archive_path: str | os.PathLike[str] | None = None) -> None:
        self._given = None if archive_path is None else Path(archive_path).expanduser()
        self._lock = threading.RLock()  # the handle, and the moment a new file takes the place of the old one
        self._importing = threading.Lock()  # one import or one load at a time
        self._archive: Archive | None = None
        self._generation = 0

    # -- where it is ---------------------------------------------------------

    @property
    def path(self) -> Path:
        """The archive file. With no path given, the default place, read at each call."""
        return self._given if self._given is not None else default_archive_path()

    @property
    def source_path(self) -> Path:
        """The note beside the archive that says where its content came from."""
        return self.path.with_name(self.path.name + NOTE_SUFFIX)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def exists(self) -> bool:
        return self.path.is_file()

    # -- the read handle -----------------------------------------------------

    def archive(self) -> Archive:
        """The shared read handle. Raises ArchiveNotFoundError when there is no archive yet."""
        with self._lock:
            wanted = self.path.resolve()
            if self._archive is not None and self._archive.path != wanted:
                self._drop()
            if self._archive is None:
                self._archive = Archive.open(wanted)
            return self._archive

    def reopen(self) -> None:
        """Close the handle. The next `archive()` opens the file that is there then."""
        with self._lock:
            self._drop()
            self._generation += 1

    def close(self) -> None:
        with self._lock:
            self._drop()

    def _drop(self) -> None:
        held, self._archive = self._archive, None
        if held is not None:
            held.close()

    # -- the note ------------------------------------------------------------

    def _note(self) -> object:
        try:
            return json.loads(self.source_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_note(self, note: Mapping[str, Any]) -> None:
        files.write_text_atomic(self.source_path, json.dumps(note, indent=1) + "\n")

    # -- the bridge from an export -------------------------------------------

    def ensure_from_export(
        self,
        messages_csv: str | os.PathLike[str],
        *,
        contacts: Contacts = None,
        progress: Any = None,
        cancel: object = None,
    ) -> EnsureResult:
        """Make the archive hold the export, when it does not hold it yet.

        The export is loaded when there is no archive, or when the signature of
        the export (path, write time in nanoseconds, size) differs from the one
        recorded beside the archive. It is loaded into a new file, which then
        takes the place of the archive in one step.

        An archive that a pull from Messages filled is kept as it is.
        """
        export = Path(messages_csv).expanduser()
        signature = ExportSignature.of(export)
        if signature is None:
            return EnsureResult("no_export", reason=f"There is no export at {export}.")
        if not self._importing.acquire(blocking=False):
            return EnsureResult("busy", reason=BUSY)
        try:
            standing = self._standing(signature)
            return standing if standing is not None else self._load(export, signature, contacts, progress, cancel)
        finally:
            self._importing.release()

    def _standing(self, signature: ExportSignature) -> EnsureResult | None:
        """The answer when nothing is to be loaded, or None when the export must be loaded."""
        try:
            archive = self.archive()
        except ArchiveNotFoundError:
            return None
        except ArchiveError:
            return EnsureResult(
                "failed", reason=f"The file {self.path.name} is not an archive that Recall can open. It was not changed."
            )
        note = self._note()
        if (isinstance(note, dict) and note.get("source") == "messages") or _holds_pull(archive):
            return EnsureResult("kept", reason=KEPT_PULL)
        return EnsureResult("current") if ExportSignature.from_note(note) == signature else None

    def _load(
        self, export: Path, signature: ExportSignature, contacts: Contacts, progress: Any, cancel: object
    ) -> EnsureResult:
        building = self.path.with_name(f".{self.path.name}{BUILD_SUFFIX}")
        _remove_database(building)
        try:
            result = import_legacy_csv(export, building, contacts=contacts, progress=progress, cancel=cancel)
            if result.status != "ok":
                return EnsureResult("cancelled", reason=f"The load was stopped. {ARCHIVE_KEPT}")
            _settle(building)
            self._move_into_place(building)
        except SourceFormatError:
            return EnsureResult(
                "failed",
                reason=f"The file {export.name} is not an export of messages that Recall can read. {ARCHIVE_KEPT}",
            )
        except (ArchiveError, OSError, sqlite3.Error):
            return EnsureResult("failed", reason=f"The export could not be loaded. {ARCHIVE_KEPT}")
        finally:
            _remove_database(building)
        self._write_note({"source": "export", "export": asdict(signature)})
        return EnsureResult("built", counts=_counts(result))

    def _move_into_place(self, building: Path) -> None:
        """Put a finished file in the place of the archive. No reader can open the archive in between."""
        with self._lock:
            self._drop()
            # SQLite finds these by name: one left by the old archive would be read into the new file
            files.remove_files(files.sidecars(self.path))
            os.replace(building, self.path)
            self._generation += 1
        files.fsync_folder(self.path.parent)

    # -- the pull from Messages ----------------------------------------------

    def pull_from_messages(
        self,
        db_path: str | os.PathLike[str],
        contacts: Contacts,
        *,
        max_removed_share: float = DEFAULT_MAX_REMOVED_SHARE,
        progress: Any = None,
        cancel: object = None,
    ) -> PullResult:
        """A full pull: after it the archive holds what the database holds.

        `db_path` is the database to read. The caller names it; `database_for`
        in the server decides which one a job reads. `contacts` are the records
        of the Contacts cache, or None to keep the people that the archive has.

        Returns PullDone, PullRefused (the database lacks more than
        `max_removed_share` of the stored messages), PullBusy, PullCancelled or
        PullFailed. Only PullDone changed the archive.
        """
        named = "" if db_path is None else str(db_path).strip()
        if not named:
            raise ValueError("The caller must name the database to read.")
        source = Path(named).expanduser()
        stopped = unreadable.why(probe_database(source))
        if stopped is not None:
            return PullFailed(reason=stopped)
        if not self._importing.acquire(blocking=False):
            return PullBusy()
        try:
            was_there = self.exists()
            outcome = self._pull(source, contacts, max_removed_share, progress, cancel)
            if not was_there and not isinstance(outcome, PullDone):
                self._forget_new_archive()
            return outcome
        finally:
            self._importing.release()

    def _pull(
        self, source: Path, contacts: Contacts, max_removed_share: float, progress: Any, cancel: object
    ) -> PullResult:
        try:
            result = import_chat_db(
                source,
                self.path,
                contacts=contacts,
                full=True,
                max_removed_share=max_removed_share,
                progress=progress,
                cancel=cancel,
            )
        except RemovalLimitError as exc:
            return PullRefused(would_remove=exc.would_remove, stored=exc.stored, max_share=exc.max_share)
        except ImportBusyError:
            return PullBusy()
        except SourceFormatError as exc:
            return PullFailed(
                reason=f"This is not a Messages database that Recall can read. {ARCHIVE_KEPT}", detail=_detail(exc)
            )
        except SourceError as exc:
            return PullFailed(reason=f"The Messages database could not be read. {ARCHIVE_KEPT}", detail=_detail(exc))
        except ArchiveError as exc:
            return PullFailed(reason=f"The update failed. {ARCHIVE_KEPT}", detail=_detail(exc))
        if result.status != "ok":
            return PullCancelled()
        self._write_note({"source": "messages"})
        with self._lock:
            self._generation += 1
        summary = self.archive().summary()
        return PullDone(counts=_counts(result), messages=summary["messages"], conversations=summary["conversations"])

    def _forget_new_archive(self) -> None:
        """Remove the file that a first pull made and did not fill: there was no archive, and there is none."""
        with self._lock:
            self._drop()
            _remove_database(self.path)
        shutil.rmtree(self.path.with_name(f".{self.path.name}{WORK_SUFFIX}"), ignore_errors=True)


_shared_lock = threading.Lock()
_shared: ArchiveState | None = None


def shared() -> ArchiveState:
    """The one state of the server: the archive at the default place."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ArchiveState()
        return _shared


def archive() -> Archive:
    return shared().archive()


def reopen() -> None:
    shared().reopen()


def ensure_from_export(messages_csv: str | os.PathLike[str], **more: Any) -> EnsureResult:
    return shared().ensure_from_export(messages_csv, **more)


def pull_from_messages(db_path: str | os.PathLike[str], contacts: Contacts, **more: Any) -> PullResult:
    return shared().pull_from_messages(db_path, contacts, **more)
