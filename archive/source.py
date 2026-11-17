"""Read a Messages database (chat.db) without changing it.

The source is copied first, then only the copy is read:
1. Copy the database file and its `-wal` and `-journal` files, byte for
   byte, into a private folder. No SQLite connection opens the source, so
   nothing is written or made in the source folder.
2. Compare inode, size and write time of those files before and after the
   copy. Equal values mean that the copy is one state of the source. SQLite
   applies the copied `-wal` when it opens the copy.
3. A source that changes during three such copies is read through a
   read-only connection inside a read transaction. That is done only when
   it cannot make a file in the source folder.
4. Read the copy in batches. Delete it when done.

Columns differ between macOS versions. A column that the source does not
have is read as NULL.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from archive.connection import file_uri
from archive.errors import SourceError, SourceFormatError
from archive.records import SourceAttachment, SourceChat, SourceMessage

COPY_CHUNK_BYTES = 1 << 20
COPY_PASSES = 3
COPY_PAGES_PER_STEP = 2048
HEADER_BYTES = 100
SQLITE_MAGIC = b"SQLite format 3\x00"
WAL_FORMAT = 2  # header bytes 18 and 19: 1 is a rollback journal, 2 is WAL
COPY_PREFIX = "source-"
STATE_FILES = ("", "-wal", "-journal")  # name endings of the files that hold the state of a database
COPY_FILES = (*STATE_FILES, "-shm")  # name endings of the files that can be next to the private copy
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
KEEPS_CHANGING = "the Messages database keeps changing: try again in a moment"

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
FileState = tuple[int, int, int] | None  # inode, size, write time. None: there is no such file


class _PassFailed(Exception):
    """One copy without locks did not give a state of the source."""


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


def _coerce(value: object, kind: type) -> object:
    """Bring a value of an unexpected SQLite type to the type the importer works with.

    An integer field keeps only what SQLite can store: inf, nan and a value outside 64 bits become None.
    """
    if value is None:
        return None
    if kind is int:
        try:
            number = int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError, OverflowError):
            return None
        return number if INT64_MIN <= number <= INT64_MAX else None
    if type(value) is kind:
        return value
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


def _beside(path: Path, ending: str) -> Path:
    return path.with_name(path.name + ending)


def _real_source(path: Path) -> tuple[Path, bytes]:
    """The real path of the source and its header. SQLite looks for the side files next to the real file."""
    try:
        real = path.resolve(strict=True)  # the path behind a link
        with open(real, "rb") as handle:
            head = handle.read(HEADER_BYTES)
    except FileNotFoundError as exc:
        raise SourceError(f"no Messages database at {path} (or no permission to read it)") from exc
    except (OSError, RuntimeError) as exc:
        raise SourceError(f"cannot read {path}: {exc}") from exc
    if len(head) < HEADER_BYTES or not head.startswith(SQLITE_MAGIC):
        raise SourceError(f"{path} is not a SQLite database")
    return real, head


def _signature(path: Path) -> tuple[FileState, ...]:
    """The state of the database file and of its `-wal` and `-journal`. A write to one of them changes it."""
    found: list[FileState] = []
    for ending in STATE_FILES:
        try:
            info = os.lstat(_beside(path, ending))
        except FileNotFoundError:
            found.append(None)
        else:
            found.append((info.st_ino, info.st_size, info.st_mtime_ns))
    return tuple(found)


def _open_no_link(path: str, flags: int) -> int:
    """Opener for open(): never through a link (SQLite does the same), and a new file is private."""
    return os.open(path, flags | os.O_NOFOLLOW, 0o600)


def _copy_pass(path: Path, target: Path) -> Generator[tuple[int, int], None, None]:
    """Copy the state files with no lock and no SQLite connection. Yields (bytes done, bytes in all).

    Raises _PassFailed when a file changed during the copy or cannot be copied.
    """
    buffer = bytearray(COPY_CHUNK_BYTES)  # one buffer for every chunk: memory use does not grow with the source
    view = memoryview(buffer)
    try:
        before = _signature(path)
        total = sum(state[1] for state in before if state is not None)
        done = 0
        for ending, state in zip(STATE_FILES, before, strict=True):
            if state is None and ending:
                continue  # no such side file
            with (
                open(_beside(path, ending), "rb", opener=_open_no_link) as reader,
                open(_beside(target, ending), "xb", opener=_open_no_link) as writer,
            ):
                while count := reader.readinto(buffer):
                    writer.write(view[:count])
                    writer.flush()  # nothing waits in a buffer when a step stops the copy
                    done += count
                    yield min(done, total), total
        changed = done != total or _signature(path) != before
    except FileNotFoundError as exc:
        raise _PassFailed(KEEPS_CHANGING) from exc  # a file that was there is gone
    except OSError as exc:
        raise _PassFailed(f"cannot copy {path}: {exc}") from exc
    if changed:
        raise _PassFailed(KEEPS_CHANGING)


def _copy_unlocked(path: Path, target: Path, on_step: CopyStep | None) -> _PassFailed | None:
    """Copy with no lock, COPY_PASSES times at most. None: the copy is one state of the source."""
    failure: _PassFailed | None = None
    for _ in range(COPY_PASSES):
        steps = _copy_pass(path, target)
        try:
            for done, total in steps:
                if on_step is not None:
                    on_step(done, total)
            return None
        except _PassFailed as exc:
            failure = exc
            _remove_copy(target)
        finally:
            steps.close()
    return failure


def _remove_copy(target: Path) -> None:
    for ending in COPY_FILES:
        _beside(target, ending).unlink(missing_ok=True)


def _locked_read_makes_no_file(path: Path, header: bytes) -> bool:
    """A read-only connection to a WAL database makes `-wal` and `-shm` when one of them is missing."""
    wal, shm = (os.path.lexists(_beside(path, ending)) for ending in ("-wal", "-shm"))
    if wal or WAL_FORMAT in header[18:20]:
        return wal and shm  # both exist while a writer has the database open
    return True


def _backup(path: Path, target: Path, on_step: CopyStep | None) -> None:
    """Copy through a read-only connection. Its read transaction fixes the state that is copied."""
    try:
        source = sqlite3.connect(file_uri(path, mode="ro"), uri=True, isolation_level=None, timeout=30.0)
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
    """Copy one consistent state of the source database to `target`.

    `on_step(done, total)` reports progress. An exception that it raises stops the copy.
    A copy that fails leaves no file.
    """
    real, header = _real_source(path)
    for ending in COPY_FILES:
        if os.path.lexists(_beside(target, ending)):
            raise FileExistsError(f"{_beside(target, ending)} is in the way of the copy")
    try:
        failure = _copy_unlocked(real, target, on_step)
        if failure is not None:
            if not _locked_read_makes_no_file(real, header):
                raise SourceError(str(failure)) from failure.__cause__
            _backup(real, target, on_step)
    except BaseException:
        _remove_copy(target)
        raise


def _first_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """The leading column of each index of a table."""
    found: set[str] = set()
    for row in conn.execute(f"PRAGMA index_list({_quoted(table)})").fetchall():
        first = conn.execute(f"PRAGMA index_info({_quoted(str(row[1]))})").fetchone()
        if first is not None and first[2] is not None:
            found.add(str(first[2]))
    return found


def _open_copy(target: Path, tables_with_lookup: Sequence[str]) -> sqlite3.Connection:
    """Open the private copy. Add the lookup indexes that Apple's schema has, if they are missing.

    SQLite first applies a `-wal` or `-journal` that was copied with the database, then deletes it.
    """
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


def _is_own_folder(path: Path) -> bool:
    """True for a real directory of this user. A link to a directory is not one."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()


def _make_work_dir(work_dir: Path) -> None:
    """Make the private folder for the copy. Refuse a link and a folder of another user."""
    try:
        work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if _is_own_folder(work_dir):
            os.chmod(work_dir, 0o700)
            return
    except OSError as exc:
        raise SourceError(f"cannot make the work folder {work_dir}: {exc}") from exc
    raise SourceError(f"the work folder {work_dir} is not a folder of this user")


def clear_work_dir(work_dir: Path) -> None:
    """Delete source copies, also those left by a run that was killed. Remove the folder when it is empty.

    Only files named `source-*` in a folder of this user are deleted. A link is never followed.
    """
    if not _is_own_folder(work_dir):
        return
    with os.scandir(work_dir) as entries:
        copies = [e.path for e in entries if e.name.startswith(COPY_PREFIX) and e.is_file(follow_symlinks=False)]
    for path in copies:
        Path(path).unlink(missing_ok=True)
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
        """(chat ROWID, handle ROWID) from chat_handle_join. A row with a value that is no integer is left out."""
        if not self.has_members:
            return []
        rows = self._conn.execute(
            "SELECT chat_id, handle_id FROM chat_handle_join"
            " WHERE typeof(chat_id) = 'integer' AND typeof(handle_id) = 'integer'"
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

    def guids(self, chunk: int = 10_000) -> Iterator[list[str]]:
        """Every guid of the message table, in lists of `chunk` items at most.

        A value is the SourceMessage.guid of its row with no outer white space: the id that the
        importer stores. A row with no guid is left out.
        """
        if chunk < 1:
            raise ValueError("chunk must be 1 or more")
        cursor = self._conn.execute("SELECT guid FROM message WHERE guid IS NOT NULL")
        return self._guid_chunks(cursor, chunk)

    def _guid_chunks(self, cursor: sqlite3.Cursor, chunk: int) -> Iterator[list[str]]:
        while rows := cursor.fetchmany(chunk):
            found = [guid for guid in ((_as_text(row[0]) or "").strip() for row in rows) if guid]
            if found:
                yield found

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
    _make_work_dir(work_dir)
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
