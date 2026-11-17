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

Another process can write the archive too (the command line importer). The
lock of this module holds inside one process only, so:
- A load takes the write lock of the old archive before the new file takes its
  place. When a connection of another process is in a write or in a read, the
  answer is `busy` and nothing changed. The lock is let go just before the
  move: a writer that starts in that instant loses its write.
- A first pull that did not end removes the archive only when the archive
  holds nothing. `PullBusy` removes nothing: the file is then another writer's.

Do not call the state from inside a query of its own handle: `close()` of the
handle waits for that query, and the move holds the lock of the state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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
from recall_services import archive_files, files, unreadable
from recall_services.archive_files import ExportSignature
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

logger = logging.getLogger(__name__)

NOTE_SUFFIX = ".source.json"
BUILD_SUFFIX = ".building"
WORK_SUFFIX = ".import-work"
DEFAULT_MAX_REMOVED_SHARE = 0.2
KEPT_PULL = "The archive holds what Messages holds. An export does not replace it."
IN_USE = f"Another program is using the archive. {ARCHIVE_KEPT}"
NO_SUCH_PLACE = "There is no file at that place."
Contacts = Sequence[Mapping[str, object]] | None


def _counts(result: ImportResult) -> dict[str, int]:
    return {name: int(value) for name, value in result.counts.items()}


def _detail(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]


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

    def file_mark(self) -> tuple[Any, ...]:
        """What tells one content of the archive file from another, also after a start of the server.

        The place, and the write time and size of the database and of the file
        in which SQLite keeps the newest writes. The mark can change while the
        content is the same; it does not stay the same when the content changed.
        """
        return (str(self.path), archive_files.stat_of(self.path), archive_files.stat_of(self.path.with_name(self.path.name + "-wal")))

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
        """Record where the content came from. The archive is right without the note: a failure is logged only."""
        try:
            files.write_text_atomic(self.source_path, json.dumps(note, indent=1) + "\n")
        except OSError as exc:
            logger.warning("the note beside the archive could not be written: %s", type(exc).__name__)

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
        export = archive_files.place(messages_csv)
        signature = None if export is None else ExportSignature.of(export)
        if export is None or signature is None:
            return EnsureResult("no_export", reason=NO_SUCH_PLACE if export is None else f"There is no export at {export}.")
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
        if (isinstance(note, dict) and note.get("source") == "messages") or archive_files.holds_pull(archive):
            return EnsureResult("kept", reason=KEPT_PULL)
        return EnsureResult("current") if ExportSignature.from_note(note) == signature else None

    def _load(
        self, export: Path, signature: ExportSignature, contacts: Contacts, progress: Any, cancel: object
    ) -> EnsureResult:
        building = self.path.with_name(f".{self.path.name}{BUILD_SUFFIX}")
        try:
            archive_files.remove_database(building)
            result = import_legacy_csv(export, building, contacts=contacts, progress=progress, cancel=cancel)
            if result.status != "ok":
                return EnsureResult("cancelled", reason=f"The load was stopped. {ARCHIVE_KEPT}")
            archive_files.settle(building)
            self._move_into_place(building)
        except SourceFormatError:
            return EnsureResult(
                "failed",
                reason=f"The file {export.name} is not an export of messages that Recall can read. {ARCHIVE_KEPT}",
            )
        except ImportBusyError:
            return EnsureResult("busy", reason=IN_USE)
        except (ArchiveError, OSError, sqlite3.Error):
            return EnsureResult("failed", reason=f"The export could not be loaded. {ARCHIVE_KEPT}")
        finally:
            archive_files.remove_quietly(building)
        self._write_note({"source": "export", "export": asdict(signature)})
        return EnsureResult("built", counts=_counts(result))

    def _move_into_place(self, building: Path) -> None:
        """Put a finished file in the place of the archive. No reader of this process can open it in between.

        Raises ImportBusyError when another process is in a write or a read of the old archive.
        """
        with self._lock:
            self._drop()
            archive_files.gather(self.path)
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
        source = archive_files.place(named)
        try:
            stopped = NO_SUCH_PLACE if source is None else unreadable.why(probe_database(source))
        except ValueError:  # a text that is no path
            stopped = NO_SUCH_PLACE
        if source is None or stopped is not None:
            return PullFailed(reason=stopped or NO_SUCH_PLACE)
        if not self._importing.acquire(blocking=False):
            return PullBusy()
        try:
            was_there = self.exists()
            try:
                outcome = self._pull(source, contacts, max_removed_share, progress, cancel)
            except BaseException:
                if not was_there:
                    self._forget_empty_archive()
                raise
            # PullBusy removes nothing: another writer has the file
            if not was_there and isinstance(outcome, (PullFailed, PullRefused, PullCancelled)):
                self._forget_empty_archive()
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
        with self._lock:
            self._generation += 1
        self._write_note({"source": "messages"})
        summary = self.archive().summary()
        return PullDone(counts=_counts(result), messages=summary["messages"], conversations=summary["conversations"])

    def _forget_empty_archive(self) -> None:
        """Remove the file that a first pull made and did not fill: there was no archive, and there is none.

        An archive that holds a row is not the file of this pull: another process filled it, and it stays.
        """
        with self._lock:
            self._drop()
            if not archive_files.holds_nothing(self.path):
                return
            archive_files.remove_quietly(self.path)
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
