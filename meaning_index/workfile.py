"""The work file of a build: the vectors that the build paid for, by content hash.

A build writes each answer of the embedder here before it asks for the next
one. A build that fails or is cancelled leaves the file. The next build reads
it and does not pay for these vectors again. A build that ends well deletes it.

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

import numpy as np

from meaning_index.errors import StoreError
from meaning_index.store_files import PRIVATE_FILE_MODE, VECTOR_TYPE, remove

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_S = 10.0
QUERY_CHUNK = 500

_CREATE = """
CREATE TABLE IF NOT EXISTS vectors (
    content_hash TEXT PRIMARY KEY,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL
)
"""


class WorkFile:
    """Opens the file for each operation. A read of a file that is not there makes no file."""

    def __init__(self, path: Path, dimensions: int) -> None:
        self._path = path
        self._dimensions = dimensions
        self._bytes = dimensions * VECTOR_TYPE.itemsize

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _read_hashes(self) -> frozenset[str]:
        if not self._path.exists():
            return frozenset()
        with self._open() as conn:
            conn.execute(_CREATE)
            found = conn.execute(
                "SELECT content_hash FROM vectors WHERE dimensions = ? AND length(vector) = ?",
                (self._dimensions, self._bytes),
            ).fetchall()
        return frozenset(row[0] for row in found)

    def hashes(self) -> frozenset[str]:
        """The content hashes that have a vector of this size. A file that can not be read holds none."""
        try:
            return self._read_hashes()
        except sqlite3.Error as exc:
            logger.warning("the work file of the index can not be read (%s); it is read as empty", type(exc).__name__)
            return frozenset()

    def repair(self) -> None:
        """Remove a work file that can not be read. Call it with the build lock held."""
        try:
            self._read_hashes()
        except sqlite3.Error as exc:
            logger.warning("the work file of the index can not be read (%s); it is removed", type(exc).__name__)
            self.clear()

    def put(self, hashes: Sequence[str], rows: np.ndarray) -> None:
        """Keep one row for each content hash. A hash that is there is replaced."""
        if rows.shape != (len(hashes), self._dimensions) or rows.dtype != VECTOR_TYPE:
            raise StoreError(f"the work file takes one float16 row of {self._dimensions} dimensions for each hash")
        try:
            if not self._path.exists():
                os.close(os.open(self._path, os.O_WRONLY | os.O_CREAT, PRIVATE_FILE_MODE))
            with self._open() as conn:
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
                for start in range(0, len(hashes), QUERY_CHUNK):
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
