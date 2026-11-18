"""The work file of a build: the vectors that the build paid for, by content hash.

A build writes each answer of the embedder here before it asks for the next
one. A build that fails or is cancelled leaves the file. The next build reads
it and does not pay for these vectors again. A build that ends well deletes it.

A read opens the file to read only and changes nothing. A read that can not
be done is a StoreError: the file is never taken for empty because it could
not be read. Only `repair`, which a build calls with the lock held, opens the
file to write: SQLite then takes a write that died out of the file, and every
vector that was saved before stays.

The content hash holds the model and the dimensions, so a vector is never
given to another model.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np

from meaning_index.errors import StoreError
from meaning_index.store_files import PRIVATE_FILE_MODE, VECTOR_TYPE, remove

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_S = 10.0
QUERY_CHUNK = 500
PROBE_KEY = "probe"  # no content hash: a hash is 64 characters
NOT_A_DATABASE = frozenset({sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_CORRUPT})

_CREATE = """
CREATE TABLE IF NOT EXISTS vectors (
    content_hash TEXT PRIMARY KEY,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL
)
"""
_HASHES = "SELECT content_hash FROM vectors WHERE dimensions = ? AND length(vector) = ?"


def _has_table(conn: sqlite3.Connection) -> bool:
    """False for a file that a build made and did not write to yet."""
    found = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vectors'").fetchone()
    return found is not None


def _is_no_database(error: sqlite3.Error) -> bool:
    """True when SQLite says that the file is not a database, or that it is malformed."""
    return (getattr(error, "sqlite_errorcode", 0) & 0xFF) in NOT_A_DATABASE


class WorkFile:
    """Opens the file for each operation. A read of a file that is not there makes no file."""

    def __init__(self, path: Path, dimensions: int) -> None:
        self._path = path
        self._dimensions = dimensions
        self._bytes = dimensions * VECTOR_TYPE.itemsize

    @contextmanager
    def _open(self, *, to_write: bool = False) -> Iterator[sqlite3.Connection]:
        """One connection for one operation. A read opens the file to read only: it changes nothing."""
        if to_write:
            conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S)
        else:
            conn = sqlite3.connect(f"file:{quote(str(self._path))}?mode=ro", uri=True, timeout=BUSY_TIMEOUT_S)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _read_hashes(self, *, to_write: bool = False) -> frozenset[str]:
        if not self._path.exists():
            return frozenset()
        with self._open(to_write=to_write) as conn:
            if not _has_table(conn):
                return frozenset()
            return frozenset(row[0] for row in conn.execute(_HASHES, (self._dimensions, self._bytes)))

    def hashes(self) -> frozenset[str]:
        """The content hashes that have a vector of this size. A file that can not be read is a StoreError."""
        try:
            return self._read_hashes()
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"the work file of the index can not be read ({type(exc).__name__})") from exc

    def repair(self) -> None:
        """Make the file readable again. Call it with the build lock held.

        The file is opened to write, so SQLite takes a write that died out of
        it. Only a file that is no database, or a malformed one, is removed.
        Every other failure is a StoreError, and the file stays.
        """
        try:
            self._read_hashes(to_write=True)
        except sqlite3.Error as exc:
            if not _is_no_database(exc):
                raise StoreError(f"the work file of the index can not be opened ({type(exc).__name__})") from exc
            logger.warning("the work file of the index is no database (%s); it is removed", type(exc).__name__)
            self.clear()

    def _make_private(self) -> None:
        if not self._path.exists():
            os.close(os.open(self._path, os.O_WRONLY | os.O_CREAT, PRIVATE_FILE_MODE))

    def check_writable(self) -> None:
        """Write one row and take it out again. A build calls this before it pays for a vector."""
        try:
            self._make_private()
            with self._open(to_write=True) as conn:
                conn.execute(_CREATE)
                conn.execute("INSERT OR REPLACE INTO vectors VALUES (?, 0, x'')", (PROBE_KEY,))
            with self._open(to_write=True) as conn:
                conn.execute("DELETE FROM vectors WHERE content_hash = ?", (PROBE_KEY,))
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"the work file of the index can not be written ({type(exc).__name__})") from exc

    def put(self, hashes: Sequence[str], rows: np.ndarray) -> None:
        """Keep one row for each content hash. A hash that is there is replaced."""
        if rows.shape != (len(hashes), self._dimensions) or rows.dtype != VECTOR_TYPE:
            raise StoreError(f"the work file takes one float16 row of {self._dimensions} dimensions for each hash")
        if not np.isfinite(rows).all():
            raise StoreError("a row for the work file holds a value that is no number")
        try:
            self._make_private()
            with self._open(to_write=True) as conn:
                conn.execute(_CREATE)
                conn.executemany(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                    ((key, self._dimensions, row.tobytes()) for key, row in zip(hashes, rows, strict=True)),
                )
            os.chmod(self._path, PRIVATE_FILE_MODE)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"the work file of the index can not be written ({type(exc).__name__})") from exc

    def get_many(self, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        """The rows of the hashes that have one of this size."""
        if not hashes or not self._path.exists():
            return {}
        found: dict[str, np.ndarray] = {}
        try:
            with self._open() as conn:
                for start in range(0, len(hashes) if _has_table(conn) else 0, QUERY_CHUNK):
                    chunk = list(hashes[start : start + QUERY_CHUNK])
                    marks = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT content_hash, vector FROM vectors WHERE content_hash IN ({marks})"
                        " AND dimensions = ? AND length(vector) = ?",
                        (*chunk, self._dimensions, self._bytes),
                    )
                    found.update((key, np.frombuffer(blob, dtype=VECTOR_TYPE)) for key, blob in rows)
        except sqlite3.Error as exc:
            raise StoreError(f"the work file of the index can not be read ({type(exc).__name__})") from exc
        return found

    def clear(self) -> None:
        """Delete the file and what SQLite keeps beside it. A file that stays does no harm: a hash names its vector."""
        try:
            for ending in ("", "-journal", "-wal", "-shm"):
                remove(self._path.with_name(self._path.name + ending))
        except OSError as exc:
            logger.warning("the work file of the index could not be removed (%s)", type(exc).__name__)
