"""Open SQLite connections to the archive file.

Every connection is in autocommit mode: transactions are explicit.
The archive uses WAL, so readers never wait for an import.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from archive import migrate
from archive.errors import ArchiveError, ArchiveNotFoundError, SqliteFeatureError
from archive.schema import APPLICATION_ID

MIN_SQLITE: tuple[int, int, int] = (3, 44, 0)  # ordered aggregates, STRICT tables, RETURNING
BUSY_SECONDS = 30.0
WRITER_CACHE_KIB = 32_000
READER_CACHE_KIB = 8_000
READER_MMAP_BYTES = 268_435_456


def _memory_connection() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def _probe_fts5(factory: Callable[[], Any]) -> bool:
    conn = factory()
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.recall_fts5_probe USING fts5(x)")
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    return True


@lru_cache(maxsize=1)
def _this_sqlite_has_fts5() -> bool:
    return _probe_fts5(_memory_connection)


def fts5_available(factory: Callable[[], Any] | None = None) -> bool:
    """True when the SQLite library can create an FTS5 table."""
    return _this_sqlite_has_fts5() if factory is None else _probe_fts5(factory)


def require_sqlite_features(version: tuple[int, ...] | None = None, has_fts5: bool | None = None) -> None:
    """Raise SqliteFeatureError when this interpreter's SQLite cannot run the archive."""
    found = tuple(version) if version is not None else tuple(sqlite3.sqlite_version_info)
    if found < MIN_SQLITE:
        wanted = ".".join(str(part) for part in MIN_SQLITE)
        have = ".".join(str(part) for part in found)
        raise SqliteFeatureError(f"The archive needs SQLite {wanted} or newer. This Python has SQLite {have}.")
    if not (fts5_available() if has_fts5 is None else has_fts5):
        raise SqliteFeatureError(
            "The archive needs the SQLite FTS5 module for search. This Python's SQLite was built without FTS5."
        )


def file_uri(path: Path, **options: str) -> str:
    """A `file:` URI for sqlite3.connect(uri=True)."""
    query = "&".join(f"{key}={value}" for key, value in options.items())
    return f"file:{quote(str(Path(path).resolve()))}" + (f"?{query}" if query else "")


def _connect(target: str, *, uri: bool, same_thread: bool = True) -> sqlite3.Connection:
    return sqlite3.connect(target, uri=uri, isolation_level=None, timeout=BUSY_SECONDS, check_same_thread=same_thread)


def _create_private_file(path: Path) -> None:
    """The archive holds private messages: only the owner can read the file, and a folder made here."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        return


def _check_identity(conn: sqlite3.Connection, path: Path) -> bool:
    """Raise when the file is not a Recall archive. Return True for a file with nothing in it yet."""
    try:
        application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
        version = migrate.schema_version(conn)
        objects = int(conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0])
    except sqlite3.DatabaseError as exc:
        raise ArchiveError(f"{path} is not a Recall archive: {exc}") from exc
    is_new_file = application_id == 0 and version == 0 and objects == 0
    if application_id != APPLICATION_ID and not is_new_file:
        raise ArchiveError(f"{path} is not a Recall archive")
    return is_new_file


def open_writer(path: Path) -> sqlite3.Connection:
    """Open the archive for writing. Create it and apply migrations when needed.

    Every failure is an ArchiveError.
    """
    require_sqlite_features()
    path = Path(path)
    try:
        return _open_writer(path)
    except ArchiveError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise ArchiveError(f"cannot open the archive at {path}: {exc}") from exc


def _open_writer(path: Path) -> sqlite3.Connection:
    _create_private_file(path)
    conn = _connect(str(path), uri=False)
    try:
        _check_identity(conn, path)
        mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if mode != "wal":
            raise ArchiveError(f"{path} cannot use WAL mode (journal mode is {mode})")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA cache_size=-{WRITER_CACHE_KIB}")
        migrate.migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def connect_reader(path: Path) -> sqlite3.Connection:
    """A read-only connection. No checks: use open_reader for the first one."""
    conn = _connect(file_uri(path, mode="ro"), uri=True, same_thread=False)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute(f"PRAGMA cache_size=-{READER_CACHE_KIB}")
        conn.execute(f"PRAGMA mmap_size={READER_MMAP_BYTES}")
    except BaseException:
        conn.close()
        raise
    return conn


def open_reader(path: Path) -> sqlite3.Connection:
    """Open an existing archive read-only. Migrate it first when it is older than this code."""
    require_sqlite_features()
    path = Path(path)
    if not path.is_file():
        raise ArchiveNotFoundError(f"no archive at {path}")
    try:
        return _open_reader(path)
    except ArchiveError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise ArchiveError(f"cannot open the archive at {path}: {exc}") from exc


def _open_reader(path: Path) -> sqlite3.Connection:
    conn = connect_reader(path)
    try:
        if _check_identity(conn, path):
            raise ArchiveNotFoundError(f"{path} is an empty file, not an archive")  # a reader never creates one
        needs_migration = migrate.schema_version(conn) != len(migrate.MIGRATIONS)
    except BaseException:
        conn.close()
        raise
    if not needs_migration:
        return conn
    conn.close()
    open_writer(path).close()  # raises ArchiveVersionError for a newer archive
    return connect_reader(path)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """One write transaction. Roll back when the body raises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
