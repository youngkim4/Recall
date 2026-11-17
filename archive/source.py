"""Read a Messages database (chat.db) without changing it.

The source is copied first, then only the copy is read:
1. Open the source read-only and start a read transaction. That fixes one
   consistent state, even while Messages keeps writing.
2. Copy that state into a private temp file with SQLite's backup API.
3. Read the copy in batches. Delete it when done.

Columns differ between macOS versions. A column that the source does not
have is read as NULL.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from archive.connection import file_uri
from archive.errors import SourceError, SourceFormatError
from archive.records import SourceAttachment, SourceChat, SourceMessage

COPY_PAGES_PER_STEP = 2048
SQLITE_MAGIC = b"SQLite format 3\x00"
WAL_FORMAT = 2  # header byte 18: 1 is a rollback journal, 2 is WAL
COPY_PREFIX = "source-"

REQUIRED_TABLES = ("message", "chat", "handle", "chat_message_join")
REQUIRED_COLUMNS = {
    "message": ("guid", "text", "handle_id", "is_from_me", "date"),
    "chat": ("chat_identifier",),
    "handle": ("id",),
}
EDIT_MARK_COLUMNS = ("date_edited", "date_retracted")

# SourceMessage field -> column of the message table, in field order
MESSAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("guid", "guid"),
    ("text", "text"),
    ("body", "attributedBody"),
    ("handle_rowid", "handle_id"),
    ("is_from_me", "is_from_me"),
    ("date", "date"),
    ("date_delivered", "date_delivered"),
    ("date_read", "date_read"),
    ("service", "service"),
    ("item_type", "item_type"),
    ("associated_guid", "associated_message_guid"),
    ("associated_type", "associated_message_type"),
    ("associated_emoji", "associated_message_emoji"),
    ("thread_guid", "thread_originator_guid"),
    ("date_edited", "date_edited"),
    ("date_retracted", "date_retracted"),
    ("group_title", "group_title"),
    ("group_action_type", "group_action_type"),
    ("other_handle", "other_handle"),
    ("balloon_bundle_id", "balloon_bundle_id"),
    ("expressive_style", "expressive_send_style_id"),
    ("subject", "subject"),
    ("is_audio", "is_audio_message"),
)
MESSAGE_TYPES: tuple[type, ...] = (
    int, str, str, bytes, int, int, int, int, int, str, int, str, int, str, str,
    int, int, str, int, int, str, str, str, int, int,
)  # fmt: skip
ATTACHMENT_COLUMNS = ("guid", "mime_type", "transfer_name", "filename", "total_bytes")

CopyStep = Callable[[int, int], None]


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def _coerce(value: object, kind: type) -> object:
    """Bring a value of an unexpected SQLite type to the type the importer works with."""
    if value is None or type(value) is kind:
        return value
    if kind is int:
        try:
            return int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError, OverflowError):
            return None
    if kind is str:
        return _decode_text(value) if isinstance(value, bytes) else str(value)
    return None  # a blob stored as text or number cannot be a typedstream


def _message(values: tuple[object, ...]) -> SourceMessage:
    for value, kind in zip(values, MESSAGE_TYPES, strict=True):
        if value is not None and type(value) is not kind:
            return SourceMessage(*[_coerce(v, k) for v, k in zip(values, MESSAGE_TYPES, strict=True)])  # type: ignore[arg-type]
    return SourceMessage._make(values)


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _header(path: Path) -> bytes:
    try:
        with open(path, "rb") as handle:
            head = handle.read(100)
    except FileNotFoundError as exc:
        raise SourceError(f"no Messages database at {path} (or no permission to read it)") from exc
    except OSError as exc:
        raise SourceError(f"cannot read {path}: {exc}") from exc
    if not head.startswith(SQLITE_MAGIC):
        raise SourceError(f"{path} is not a SQLite database")
    return head


def _wal_path(path: Path) -> Path:
    return path.with_name(path.name + "-wal")


def _is_quiet_wal_file(path: Path, header: bytes) -> bool:
    """WAL format, but no WAL file: nothing has the database open.

    A read-only open would make SQLite create `-wal` and `-shm` next to it.
    An immutable open creates nothing.
    """
    return header[18] == WAL_FORMAT and not _wal_path(path).exists()


def _signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def _backup(path: Path, target: Path, *, immutable: bool, on_step: CopyStep | None) -> None:
    options = {"immutable": "1"} if immutable else {"mode": "ro"}
    try:
        source = sqlite3.connect(file_uri(path, **options), uri=True, isolation_level=None, timeout=30.0)
    except sqlite3.Error as exc:
        raise SourceError(f"cannot open {path}: {exc}") from exc
    try:
        source.execute("BEGIN")
        source.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()  # fixes the state that is copied
        os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        copy = sqlite3.connect(str(target))
        try:
            progress = (
                None if on_step is None else (lambda _status, remaining, total: on_step(total - remaining, total))
            )
            source.backup(copy, pages=COPY_PAGES_PER_STEP, progress=progress)
        finally:
            copy.close()
    except sqlite3.Error as exc:
        raise SourceError(f"cannot copy {path}: {exc}") from exc
    finally:
        source.close()


def copy_source(path: Path, target: Path, on_step: CopyStep | None = None) -> None:
    """Copy one consistent state of the source database to `target`."""
    header = _header(path)
    if _is_quiet_wal_file(path, header):
        before = _signature(path)
        _backup(path, target, immutable=True, on_step=on_step)
        if _signature(path) == before and not _wal_path(path).exists():
            return
        target.unlink(missing_ok=True)  # a writer started during the copy: read again with locks
    _backup(path, target, immutable=False, on_step=on_step)


def _first_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """The leading column of each index of a table."""
    found: set[str] = set()
    for row in conn.execute(f"PRAGMA index_list({_quoted(table)})").fetchall():
        first = conn.execute(f"PRAGMA index_info({_quoted(str(row[1]))})").fetchone()
        if first is not None and first[2] is not None:
            found.add(str(first[2]))
    return found


def _open_copy(target: Path, tables_with_lookup: Sequence[str]) -> sqlite3.Connection:
    """Open the private copy. Add the lookup indexes that Apple's schema has, if they are missing."""
    conn = sqlite3.connect(str(target), isolation_level=None)
    try:
        conn.text_factory = _decode_text
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table in tables_with_lookup:
            if table in tables and "message_id" not in _first_columns(conn, table):
                conn.execute(f"CREATE INDEX recall_{table}_by_message ON {table}(message_id)")
        conn.execute("PRAGMA query_only=ON")
    except sqlite3.Error as exc:
        conn.close()
        raise SourceError(f"cannot read the copy of the source: {exc}") from exc
    except BaseException:
        conn.close()
        raise
    return conn


def clear_work_dir(work_dir: Path) -> None:
    """Delete source copies, also those left by a run that was killed."""
    if not work_dir.is_dir():
        return
    for path in work_dir.glob(COPY_PREFIX + "*"):
        if path.is_file():
            path.unlink(missing_ok=True)
    try:
        work_dir.rmdir()
    except OSError:
        return  # something that is not ours is in the folder


class SourceReader:
    """Batched reads from the copy of a Messages database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = [name for name in REQUIRED_TABLES if name not in tables]
        if missing:
            raise SourceFormatError(f"not a Messages database: no {', '.join(missing)} table")
        self._columns = {name: self._table_columns(name) for name in ("message", "chat", "handle")}
        for table, required in REQUIRED_COLUMNS.items():
            absent = [name for name in required if name not in self._columns[table]]
            if absent:
                raise SourceFormatError(f"not a Messages database: the {table} table has no {', '.join(absent)} column")
        self.has_members = "chat_handle_join" in tables
        self.has_attachments = {"attachment", "message_attachment_join"} <= tables
        self._attachment_columns = self._table_columns("attachment") if self.has_attachments else frozenset()
        self._select = self._message_select()

    def _table_columns(self, table: str) -> frozenset[str]:
        return frozenset(str(row[1]) for row in self._conn.execute(f"PRAGMA table_info({table})"))

    def _column(self, table: str, alias: str, name: str) -> str:
        return f"{alias}.{name}" if name in self._columns[table] else "NULL"

    def _message_select(self) -> str:
        parts = ["m.ROWID"]
        parts.extend(self._column("message", "m", column) for _, column in MESSAGE_COLUMNS)
        parts.append("(SELECT MIN(j.chat_id) FROM chat_message_join j WHERE j.message_id = m.ROWID)")
        return f"SELECT {', '.join(parts)} FROM message m"

    def _batches(self, cursor: sqlite3.Cursor, size: int) -> Iterator[list[SourceMessage]]:
        while True:
            rows = cursor.fetchmany(size)
            if not rows:
                return
            yield [_message(row) for row in rows]

    def handles(self) -> list[tuple[int, str]]:
        """(ROWID, id) of every handle. One id can have several rows, one per service."""
        rows = self._conn.execute("SELECT ROWID, id FROM handle WHERE id IS NOT NULL ORDER BY ROWID").fetchall()
        return [(int(rowid), str(text)) for rowid, text in rows]

    def chats(self) -> list[SourceChat]:
        title = self._column("chat", "c", "display_name")
        service = self._column("chat", "c", "service_name")
        rows = self._conn.execute(f"SELECT c.ROWID, c.chat_identifier, {title}, {service} FROM chat c ORDER BY c.ROWID")
        return [
            SourceChat(int(row[0]), _as_text(row[1]), _as_text(row[2]), _as_text(row[3])) for row in rows.fetchall()
        ]

    def members(self) -> list[tuple[int, int]]:
        """(chat ROWID, handle ROWID) from chat_handle_join."""
        if not self.has_members:
            return []
        rows = self._conn.execute(
            "SELECT chat_id, handle_id FROM chat_handle_join WHERE chat_id IS NOT NULL AND handle_id IS NOT NULL"
        ).fetchall()
        return [(int(chat), int(handle)) for chat, handle in rows]

    def count_after(self, rowid: int) -> int:
        """Number of rows with a ROWID above `rowid`."""
        if rowid <= 0:
            # with no condition SQLite counts on its smallest index and does not read the table
            return int(self._conn.execute("SELECT COUNT(*) FROM message").fetchone()[0])
        return int(self._conn.execute("SELECT COUNT(*) FROM message WHERE ROWID > ?", (rowid,)).fetchone()[0])

    def top_rows(self, count: int) -> list[tuple[int, str]]:
        """The highest ROWIDs with their guids. They identify this database on the next run."""
        rows = self._conn.execute(
            "SELECT ROWID, guid FROM message WHERE guid IS NOT NULL ORDER BY ROWID DESC LIMIT ?", (count,)
        ).fetchall()
        return [(int(rowid), str(guid)) for rowid, guid in rows]

    def guid_at(self, rowid: int) -> str | None:
        row = self._conn.execute("SELECT guid FROM message WHERE ROWID = ?", (rowid,)).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def messages_after(self, rowid: int, batch_size: int) -> Iterator[list[SourceMessage]]:
        """Rows with a ROWID above `rowid`, lowest first."""
        cursor = self._conn.execute(f"{self._select} WHERE m.ROWID > ? ORDER BY m.ROWID", (rowid,))
        return self._batches(cursor, batch_size)

    def changed_messages(self, upto_rowid: int, batch_size: int) -> Iterator[list[SourceMessage]]:
        """Rows at or below `upto_rowid` that carry an edit or unsend mark."""
        marks = [name for name in EDIT_MARK_COLUMNS if name in self._columns["message"]]
        if not marks or upto_rowid <= 0:
            return iter(())
        marked = " OR ".join(f"m.{name} > 0" for name in marks)
        cursor = self._conn.execute(f"{self._select} WHERE m.ROWID <= ? AND ({marked}) ORDER BY m.ROWID", (upto_rowid,))
        return self._batches(cursor, batch_size)

    def attachments_for(self, rowids: Sequence[int]) -> dict[int, list[SourceAttachment]]:
        """Attachment metadata of the given messages, keyed by message ROWID."""
        if not self.has_attachments or not rowids:
            return {}
        columns = ", ".join(f"a.{name}" if name in self._attachment_columns else "NULL" for name in ATTACHMENT_COLUMNS)
        marks = ", ".join("?" for _ in rowids)
        rows = self._conn.execute(
            f"SELECT j.message_id, a.ROWID, {columns} FROM message_attachment_join j"
            f" JOIN attachment a ON a.ROWID = j.attachment_id WHERE j.message_id IN ({marks})"
            " ORDER BY j.message_id, a.ROWID",
            tuple(rowids),
        ).fetchall()
        found: dict[int, list[SourceAttachment]] = {}
        for row in rows:
            item = SourceAttachment(
                int(row[0]), int(row[1]), _as_text(row[2]), _as_text(row[3]), _as_text(row[4]), _as_text(row[5]),
                row[6] if type(row[6]) is int else None,
            )  # fmt: skip
            found.setdefault(item.message_rowid, []).append(item)
        return found


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(_coerce(value, str))


@contextmanager
def open_snapshot(source_path: Path, work_dir: Path, on_step: CopyStep | None = None) -> Iterator[SourceReader]:
    """Copy the source into `work_dir`, give a reader for the copy, then delete the copy."""
    clear_work_dir(work_dir)
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = work_dir / f"{COPY_PREFIX}{os.getpid()}-{time.time_ns()}.sqlite3"
    conn: sqlite3.Connection | None = None
    try:
        copy_source(source_path, target, on_step)
        conn = _open_copy(target, ("chat_message_join", "message_attachment_join"))
        try:
            reader = SourceReader(conn)
        except sqlite3.Error as exc:
            raise SourceError(f"cannot read {source_path}: {exc}") from exc
        yield reader
    finally:
        if conn is not None:
            conn.close()
        clear_work_dir(work_dir)
