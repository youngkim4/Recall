"""The log of import runs, and the mark that lets the next run read only new rows.

The mark is a source ROWID. ROWIDs change when Messages rebuilds its
database, so each run also stores the guids of its highest ROWIDs. A later
run trusts a mark only when the source still has the same guid at that ROWID.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence

RUNS_CHECKED_FOR_MARK = 10


def start_run(conn: sqlite3.Connection, source_path: str, mode: str, started_at: int) -> int:
    """Add a run row inside the import transaction. A rollback removes it."""
    cursor = conn.execute(
        "INSERT INTO import_runs(started_at, source_path, mode, status) VALUES (?, ?, ?, 'running')",
        (started_at, source_path, mode),
    )
    return int(cursor.lastrowid or 0)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    mode: str,
    counts: Mapping[str, object],
    checkpoints: Sequence[tuple[int, str]],
    finished_at: int,
) -> None:
    conn.execute(
        "UPDATE import_runs SET status = 'ok', mode = ?, finished_at = ?, counts = ?, checkpoints = ? WHERE id = ?",
        (
            mode,
            finished_at,
            json.dumps(counts, sort_keys=True),
            json.dumps([list(pair) for pair in checkpoints]),
            run_id,
        ),
    )


def record_ended_run(
    conn: sqlite3.Connection,
    *,
    source_path: str,
    mode: str,
    status: str,
    error: str | None,
    started_at: int,
    finished_at: int,
) -> int:
    """Record a run that failed or was cancelled, after its work was rolled back."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            "INSERT INTO import_runs(started_at, finished_at, source_path, mode, status, error)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (started_at, finished_at, source_path, mode, status, error),
        )
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return int(cursor.lastrowid or 0)


def _checkpoints(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    # A full pull makes the archive equal to its source, so a mark of an earlier run can name
    # rows that the pull removed. Only the last full pull and the runs after it are trusted.
    rows = conn.execute(
        "SELECT checkpoints FROM import_runs WHERE status = 'ok' AND id >= COALESCE("
        "(SELECT MAX(id) FROM import_runs WHERE status = 'ok' AND mode = 'full'), 0)"
        " ORDER BY id DESC LIMIT ?",
        (RUNS_CHECKED_FOR_MARK,),
    ).fetchall()
    pairs: set[tuple[int, str]] = set()
    for (text,) in rows:
        try:
            pairs.update((int(rowid), str(guid)) for rowid, guid in json.loads(text))
        except (TypeError, ValueError):
            continue
    return sorted(pairs, reverse=True)


def verified_mark(conn: sqlite3.Connection, guid_at: Callable[[int], str | None]) -> int:
    """The highest ROWID that an earlier run reached in this same database, or 0.

    `guid_at` gives the guid the source has at a ROWID now.
    """
    for rowid, guid in _checkpoints(conn):
        if guid_at(rowid) == guid:
            return rowid
    return 0
