"""The query API of the archive. Other packages code against this: it is a contract.

    with Archive.open(path) as archive:
        archive.summary()
        archive.search("ferry tickets", limit=20)

Every return value is made of plain dicts, lists, strings, numbers, booleans
and None. Time stamps are local time, "YYYY-MM-DD HH:MM:SS". `since` and
`until` are local days, "YYYY-MM-DD", and a range includes all of both days.
Group events (kind `system`) are left out unless `include_system=True`.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

from archive import connection, figures, filters, paging, rows
from archive import search as search_module
from archive.errors import ArchiveError
from archive.filters import (
    as_id_list,
    check_choice,
    check_int,
    conversation_keys,
    day_bounds,
)
from archive.paths import default_archive_path
from archive.rows import Conversation, Message, Person

CONVERSATION_ORDERS = {
    "count": "c.message_count DESC, c.last_sent_ns DESC, c.chat_identifier",
    "recent": "c.last_sent_ns DESC NULLS LAST, c.chat_identifier",
}
ID_CHUNK = 500


class _Slot:
    """One thread's connection. The slot dies with its thread, and the connection is then closed."""

    __slots__ = ("__weakref__", "conn")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn


class Archive:
    """Read access to one archive file. Safe to use from several threads: each thread gets its own connection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: set[sqlite3.Connection] = set()
        self._closed = False

    @classmethod
    def open(cls, path: str | os.PathLike[str] | None = None, *, readonly: bool = True) -> Archive:
        """Open the archive at `path`, or at the default path when `path` is None.

        `readonly=True` needs an existing file (ArchiveNotFoundError otherwise).
        `readonly=False` creates an empty archive when the file is missing.
        An archive made by older code is migrated. Queries never write.
        """
        target = Path(path).expanduser() if path is not None else default_archive_path()
        if not readonly:
            connection.open_writer(target).close()
        connection.open_reader(target).close()  # checks: exists, is a Recall archive, version, FTS5
        return cls(target)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        """Close every connection. Call it when no query is running."""
        with self._lock:
            self._closed = True
            connections, self._connections = self._connections, set()
        for conn in connections:
            conn.close()

    def __enter__(self) -> Archive:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def open_connections(self) -> int:
        """Number of live connections: one per thread that has queried and is still alive."""
        with self._lock:
            return len(self._connections)

    def _release(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            self._connections.discard(conn)
        conn.close()

    def _connection(self) -> sqlite3.Connection:
        """The read-only connection of the calling thread."""
        if self._closed:
            raise ArchiveError("this Archive is closed")
        slot: _Slot | None = getattr(self._local, "slot", None)
        if slot is None:
            conn = connection.connect_reader(self._path)
            with self._lock:
                if self._closed:
                    conn.close()
                    raise ArchiveError("this Archive is closed")
                self._connections.add(conn)
            slot = _Slot(conn)
            weakref.finalize(slot, self._release, conn)
            self._local.slot = slot
        return slot.conn

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        """One read transaction: every query inside sees the same state of the archive."""
        conn = self._connection()
        if conn.in_transaction:
            yield conn
            return
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.execute("ROLLBACK")

    # -- overview ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """`{"messages", "conversations", "first_timestamp", "last_timestamp"}` for the whole archive."""
        total, with_messages, first, last = (
            self._connection()
            .execute(
                "SELECT COALESCE(SUM(message_count), 0), COALESCE(SUM(message_count > 0), 0),"
                " MIN(first_sent_ns), MAX(last_sent_ns) FROM conversations"
            )
            .fetchone()
        )
        return {
            "messages": int(total),
            "conversations": int(with_messages),
            "first_timestamp": rows.timestamp_from_ns(first),
            "last_timestamp": rows.timestamp_from_ns(last),
        }

    def conversations(
        self, *, limit: int | None = None, offset: int = 0, order: str = "count", include_empty: bool = False
    ) -> list[Conversation]:
        """Conversations, by message count or by last message. Empty ones only with `include_empty`."""
        order_sql = CONVERSATION_ORDERS[check_choice("order", order, CONVERSATION_ORDERS)]
        size = -1 if limit is None else check_int("limit", limit, 0)
        check_int("offset", offset, 0)
        where = "" if include_empty else "WHERE c.message_count > 0"
        found = self._connection().execute(
            f"{rows.CONVERSATION_SELECT} {where} ORDER BY {order_sql} LIMIT ? OFFSET ?", (size, offset)
        )
        return [rows.conversation(row) for row in found.fetchall()]

    def conversation(self, conversation_id: str) -> Conversation | None:
        found = (
            self._connection()
            .execute(f"{rows.CONVERSATION_SELECT} WHERE c.chat_identifier = ?", (conversation_id,))
            .fetchone()
        )
        return None if found is None else rows.conversation(found)

    def people(self) -> list[Person]:
        """Every person: named people first, by name."""
        found = self._connection().execute(f"{rows.PERSON_SELECT} {rows.PERSON_ORDER}")
        return [rows.person(row) for row in found.fetchall()]

    def person(self, person_id: int) -> Person | None:
        found = self._connection().execute(f"{rows.PERSON_SELECT} WHERE p.id = ?", (person_id,)).fetchone()
        return None if found is None else rows.person(found)

    # -- messages ---------------------------------------------------------

    def iter_messages(
        self,
        conversation_id: str | Iterable[str],
        *,
        since: str | None = None,
        until: str | None = None,
        include_system: bool = False,
    ) -> Iterator[Message]:
        """Every message of a conversation, oldest first, as a stream.

        A list of ids gives one stream in time order (one person's phone and email threads).
        """
        lower, upper = day_bounds(since, until)
        keys = conversation_keys(self._connection(), conversation_id)
        return paging.stream(self._connection, keys, lower, upper, include_system)

    def messages(
        self,
        conversation_id: str | Iterable[str],
        *,
        around: str | None = None,
        before: str | None = None,
        after: str | None = None,
        limit: int = 50,
        include_system: bool = False,
    ) -> dict[str, Any]:
        """One page: `{"messages": [...oldest first], "has_older", "has_newer"}`.

        `around`, `before` and `after` are message ids. With none, the page is the newest one.
        An unknown conversation or anchor gives an empty page.
        """
        with self._reading() as conn:
            keys = conversation_keys(conn, conversation_id)
            return paging.page(
                conn, keys, around=around, before=before, after=after, limit=limit, include_system=include_system
            )

    def message(self, message_id: str) -> Message | None:
        """One message by id."""
        found = self._connection().execute(f"{rows.MESSAGE_SELECT} WHERE m.guid = ?", (message_id,)).fetchone()
        return None if found is None else rows.message(found)

    def context(self, message_id: str, *, radius: int = 6, include_system: bool = False) -> list[Message]:
        """The message with up to `radius` neighbours on each side, in the same conversation."""
        with self._reading() as conn:
            return paging.context(conn, message_id, radius=radius, include_system=include_system)

    def reactions(self, message_ids: str | Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Current tapbacks per message id: `[{"kind", "is_from_me", "sender"}]`, oldest first.

        Every requested id is a key. A person has at most one tapback on a message: the last one.
        A removed tapback is not listed. Kind `emoji` has one more key, `emoji`.
        """
        ids = as_id_list("message ids", message_ids)
        latest: dict[str, dict[object, tuple[Any, ...]]] = {message_id: {} for message_id in ids}
        with self._reading() as conn:
            for start in range(0, len(ids), ID_CHUNK):
                chunk = ids[start : start + ID_CHUNK]
                marks = ", ".join("?" for _ in chunk)
                events = conn.execute(
                    "SELECT r.target_guid, r.is_from_me, r.sender_handle_id, r.kind, r.emoji, r.removed,"
                    " COALESCE(h.handle, '') FROM reactions r LEFT JOIN handles h ON h.id = r.sender_handle_id"
                    f" WHERE r.target_guid IN ({marks}) ORDER BY r.sent_ns, r.id",
                    chunk,
                )
                for event in events.fetchall():
                    who = "me" if event[1] else event[2]
                    latest[event[0]].pop(who, None)  # a later event moves the person to the end
                    latest[event[0]][who] = event
        return {
            message_id: [_reaction(event) for event in by_person.values() if not event[5]]
            for message_id, by_person in latest.items()
        }

    # -- search and figures ----------------------------------------------

    def search(
        self,
        query: str,
        *,
        conversations: str | Iterable[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        from_me: bool | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "newest",
        include_system: bool = False,
    ) -> dict[str, Any]:
        """`{"total": every match, "hits": [{"message", "ranges"}]}`.

        Words are AND-ed, "quoted text" is a phrase, a trailing * is a prefix.
        Text typed by a user never raises. `ranges` are [start, end) code point offsets in the text.
        """
        with self._reading() as conn:
            return search_module.search(
                conn,
                query,
                conversations=conversations,
                since=since,
                until=until,
                from_me=from_me,
                limit=limit,
                offset=offset,
                order=order,
                include_system=include_system,
            )

    def count(
        self,
        *,
        conversations: str | Iterable[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        from_me: bool | None = None,
        text: str | None = None,
        group_by: str | None = None,
        include_system: bool = False,
    ) -> int | list[dict[str, Any]]:
        """Number of messages, or `[{"key", "total", "sent", "received"}]` per bucket.

        `group_by` is day, month, year, weekday_hour, conversation or sender.
        `text` follows the rules of search().
        """
        if group_by is not None:
            check_choice("group_by", group_by, figures.GROUPS)
        parsed = None if text is None else search_module.parse_query(text)
        lower, upper = day_bounds(since, until)
        direction = filters.check_flag("from_me", from_me)
        from_totals = parsed is None and lower is None and upper is None and not include_system
        with self._reading() as conn:
            if from_totals and group_by in (None, "conversation"):
                keys = None if conversations is None else conversation_keys(conn, conversations)
                return figures.count_from_totals(conn, keys, direction, group_by == "conversation")
            where = filters.scope(
                conn,
                conversations=conversations,
                since=since,
                until=until,
                from_me=from_me,
                text=parsed,
                include_system=include_system,
            )
            return figures.count(conn, where, group_by, lower, upper)

    def stats(
        self,
        conversation_id: str | Iterable[str],
        *,
        since: str | None = None,
        until: str | None = None,
        include_system: bool = False,
    ) -> dict[str, Any]:
        """The figures of compute_stats in conversation.py, with its field names."""
        lower, upper = day_bounds(since, until)
        label: object = conversation_id if isinstance(conversation_id, str) else as_id_list("ids", conversation_id)
        with self._reading() as conn:
            keys = conversation_keys(conn, conversation_id)
            return figures.stats(conn, label, keys, lower, upper, include_system)


def _reaction(event: tuple[Any, ...]) -> dict[str, Any]:
    found: dict[str, Any] = {"kind": event[3], "is_from_me": bool(event[1]), "sender": event[6]}
    if event[3] == "emoji":
        found["emoji"] = event[4]
    return found
