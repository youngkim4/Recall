"""Control of one import run: the write lock, cancel, the run log and the result.

One run is one transaction. It either commits everything or leaves the
archive data as it was. Both importers (chat.db and the legacy export
bridge) run through `execute`.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from archive import connection, runs
from archive.errors import ArchiveError, ImportBusyError, ImportFailed
from archive.records import ReactionRow

LOCK_WAIT_SECONDS = 2.0
_BUSY_CODES = (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)

Progress = Callable[[dict[str, Any]], None]
COUNT_NAMES = (
    "rows_seen",
    "messages_seen",
    "reactions_seen",
    "system_seen",
    "messages_added",
    "messages_updated",
    "messages_removed",
    "reactions_added",
    "reactions_updated",
    "reactions_removed",
    "system_added",
    "system_updated",
    "system_removed",
    "attachments_added",
    "attachments_removed",
    "conversations_added",
    "handles_added",
    "blob_texts",
    "decode_failures",
    "oversize_texts",
)


class Cancelled(Exception):
    """The caller asked to stop."""


@dataclass(frozen=True)
class ImportResult:
    """What one import run did. `status` is `ok` or `cancelled`.

    `run_id` is None only for a cancelled run that could not be written to the run log.
    """

    run_id: int | None
    status: str
    mode: str  # full | incremental | legacy_csv: what the run read, not what was asked
    source_path: str
    archive_path: str
    started_at: int
    finished_at: int
    rows_seen: int = 0
    messages_seen: int = 0
    reactions_seen: int = 0
    system_seen: int = 0
    messages_added: int = 0
    messages_updated: int = 0
    messages_removed: int = 0
    reactions_added: int = 0
    reactions_updated: int = 0
    reactions_removed: int = 0
    system_added: int = 0
    system_updated: int = 0
    system_removed: int = 0
    attachments_added: int = 0
    attachments_removed: int = 0
    conversations_added: int = 0
    handles_added: int = 0
    blob_texts: int = 0
    decode_failures: int = 0
    oversize_texts: int = 0  # texts longer than INDEXED_TEXT_CHARS that the run read
    skipped: Mapping[str, int] = field(default_factory=dict)  # reason -> rows not stored
    counts: Mapping[str, int] = field(default_factory=dict)  # every counter of the run

    def as_dict(self) -> dict[str, Any]:
        """A plain dict that json.dumps accepts."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return {**values, "skipped": dict(self.skipped), "counts": dict(self.counts)}

    def __reduce__(self) -> tuple[Callable[..., ImportResult], tuple[dict[str, Any]]]:
        return (_result_from_dict, (self.as_dict(),))  # a mappingproxy cannot be pickled or deep-copied


def _result_from_dict(values: dict[str, Any]) -> ImportResult:
    read_only = {name: MappingProxyType(dict(values[name])) for name in ("skipped", "counts")}
    return ImportResult(**{**values, **read_only})


@dataclass
class Run:
    """Working state of one run."""

    source: Path
    archive: Path
    started_at: int
    mode: str
    report: Progress | None
    should_stop: Callable[[], bool]
    run_id: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    touched: set[int] = field(default_factory=set)  # conversations whose totals can have changed
    waiting: list[ReactionRow] = field(default_factory=list)  # tapbacks whose target is not stored yet

    def tell(self, phase: str, done: int, total: int) -> None:
        if self.report is not None:
            self.report({"phase": phase, "done": min(done, total), "total": total})

    def check_stop(self) -> None:
        if self.should_stop():
            raise Cancelled


def new_run(
    source: str | os.PathLike[str],
    archive: str | os.PathLike[str],
    mode: str,
    progress: Progress | None,
    cancel: object,
) -> Run:
    return Run(
        source=Path(source).expanduser(),
        archive=Path(archive).expanduser(),
        started_at=int(time.time()),
        mode=mode,
        report=progress,
        should_stop=_stop_check(cancel),
    )


def _stop_check(cancel: object) -> Callable[[], bool]:
    if cancel is None:
        return lambda: False
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return lambda: bool(is_set())
    if callable(cancel):
        return lambda: bool(cancel())
    raise TypeError("cancel must be a callable or an object with is_set()")


def _take_write_lock(conn: sqlite3.Connection) -> None:
    """Start the import transaction. Only one import can write to an archive at a time."""
    conn.execute(f"PRAGMA busy_timeout={int(LOCK_WAIT_SECONDS * 1000)}")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        if exc.sqlite_errorcode & 0xFF in _BUSY_CODES:
            raise ImportBusyError("another import is writing to this archive") from exc
        raise ArchiveError(f"cannot start an import: {exc}") from exc
    finally:
        conn.execute(f"PRAGMA busy_timeout={int(connection.BUSY_SECONDS * 1000)}")


def _rollback(conn: sqlite3.Connection) -> None:
    """Undo the run. An error here must not hide the error that ended the run."""
    try:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
    except sqlite3.Error:
        return


def _record_end(conn: sqlite3.Connection, run: Run, status: str, error: str | None) -> int | None:
    """Log a run that did not commit. A failure to log must not hide the first error."""
    try:
        return runs.record_ended_run(
            conn,
            source_path=str(run.source),
            mode=run.mode,
            status=status,
            error=error,
            started_at=run.started_at,
            finished_at=int(time.time()),
        )
    except sqlite3.Error:
        return None


def _after_commit(conn: sqlite3.Connection) -> None:
    """Housekeeping. The run is already safe: an error here is not a failed import."""
    try:
        conn.execute("PRAGMA optimize")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        return


def _result(run: Run, status: str, run_id: int | None) -> ImportResult:
    counts = dict(run.counts)
    skipped = {
        name[len("skipped_") :]: value for name, value in counts.items() if name.startswith("skipped_") and value
    }
    return ImportResult(
        run_id=run_id,
        status=status,
        mode=run.mode,
        source_path=str(run.source),
        archive_path=str(run.archive),
        started_at=run.started_at,
        finished_at=int(time.time()),
        skipped=MappingProxyType(skipped),
        counts=MappingProxyType(counts),
        **{name: int(counts.get(name, 0)) for name in COUNT_NAMES},
    )


def _undo_and_log(conn: sqlite3.Connection, run: Run, cause: BaseException) -> int | None:
    """Undo a run that failed and write it to the run log. Return the id of the log row."""
    _rollback(conn)
    return _record_end(conn, run, "failed", f"{type(cause).__name__}: {cause}"[:2000])


def _import_failed(cause: Exception, failed_run: int | None) -> ImportFailed:
    return ImportFailed(f"import failed: {type(cause).__name__}: {cause}", run_id=failed_run)


def execute(run: Run, work: Callable[[sqlite3.Connection, Run], None]) -> ImportResult:
    """Run `work` inside one write transaction of the archive and log the run.

    Returns the result of a run that committed or was cancelled. A failure
    raises an ArchiveError after the failed run is recorded. The counters of
    a cancelled run say how far it got; none of that work is kept.
    """
    conn = connection.open_writer(run.archive)
    try:
        _take_write_lock(conn)
        try:
            work(conn, run)
            run.check_stop()  # the last moment a cancel can still undo the run
        except Cancelled:
            _rollback(conn)
            return _result(run, "cancelled", _record_end(conn, run, "cancelled", None))
        except Exception as exc:
            failed_run = _undo_and_log(conn, run, exc)
            if isinstance(exc, ArchiveError):
                raise
            raise _import_failed(exc, failed_run) from exc
        except BaseException as exc:  # Ctrl-C and the like: undo, record, and let it pass
            _undo_and_log(conn, run, exc)
            raise
        try:
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            raise _import_failed(exc, _undo_and_log(conn, run, exc)) from exc
        _after_commit(conn)
        return _result(run, "ok", run.run_id)
    finally:
        conn.close()
