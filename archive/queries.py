"""The query API of the archive. Other packages code against this: it is a contract.

    with Archive.open(path) as archive:
        archive.summary()
        archive.search("ferry tickets", limit=20)

Every return value is made of plain dicts, lists, strings, numbers, booleans
and None. Time stamps are local time, "YYYY-MM-DD HH:MM:SS". `since` and
`until` are local days, "YYYY-MM-DD", and a range includes all of both days.
Group events (kind `system`) are left out unless `include_system=True`.
A conversation id is the whole chat identifier: it matches exactly, never in part.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import weakref
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from archive import connection, days, figures, filters, paging, rows
from archive import search as search_module
from archive.errors import ArchiveError
from archive.filters import (
    as_id_list,
    check_bool,
    check_choice,
    check_id,
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


class _Slot:
    """One thread's connection, and the lock that is held while the connection is in use.

    The slot dies with its thread, and the connection is then closed.
    """

    __slots__ = ("__weakref__", "conn", "in_use")

    def __init__(self, conn: sqlite3.Connection, in_use: threading.RLock) -> None:
        self.conn = conn
        self.in_use = in_use


def _release(
    connections: dict[sqlite3.Connection, threading.RLock], lock: threading.Lock, conn: sqlite3.Connection
) -> None:
    """Close the connection of a thread that ended. It holds no reference to the Archive."""
    with lock:
        connections.pop(conn, None)
    conn.close()


class Archive:
    """Read access to one archive file. Safe to use from several threads: each thread gets its own connection.

    close() can be called at any time. It waits until a query that runs in
    another thread ends. Every later query raises ArchiveError.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: dict[sqlite3.Connection, threading.RLock] = {}
        self._closed = False

    @classmethod
    def open(cls, path: str | os.PathLike[str] | None = None, *, readonly: bool = True) -> Archive:
        """Open the archive at `path`, or at the default path when `path` is None.

        `readonly=True` needs an existing file (ArchiveNotFoundError otherwise).
        `readonly=False` creates an empty archive when the file is missing.
        An archive made by older code is migrated. Queries never write.
        """
        target = (Path(path).expanduser() if path is not None else default_archive_path()).resolve()
        if not readonly:
            connection.open_writer(target).close()
        connection.open_reader(target).close()  # checks: exists, is a Recall archive, version, FTS5
        return cls(target)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        """Close every connection. A connection that another thread is using is closed when that use ends."""
        with self._lock:
            self._closed = True
            held = list(self._connections.items())
            self._connections.clear()
        for conn, in_use in held:
            with in_use:  # closing a connection in the middle of a query of another thread crashes SQLite
                conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def open_connections(self) -> int:
        """Number of live connections: one per thread that has queried and is still alive."""
        with self._lock:
            return len(self._connections)

    def _slot(self) -> _Slot:
        """The connection of the calling thread. It is opened at the first query of the thread."""
        if self._closed:
            raise ArchiveError("this Archive is closed")
        slot: _Slot | None = getattr(self._local, "slot", None)
        if slot is None:
            conn = connection.connect_reader(self._path)
            in_use = threading.RLock()
            with self._lock:
                if self._closed:
                    conn.close()
                    raise ArchiveError("this Archive is closed")
                self._connections[conn] = in_use
            slot = _Slot(conn, in_use)
            weakref.finalize(slot, _release, self._connections, self._lock, conn)
            self._local.slot = slot
        return slot

    @contextmanager
    def _using(self) -> Iterator[sqlite3.Connection]:
        """The connection of the calling thread. close() waits until the block ends."""
        slot = self._slot()
        with slot.in_use:
            if self._closed:
                raise ArchiveError("this Archive is closed")
            yield slot.conn

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        """One read transaction: every query inside sees the same state of the archive."""
        with self._using() as conn:
            if conn.in_transaction:
                yield conn
                return
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass  # the caller closed the archive inside the block: the read is over either way

    def _fetch_all(self, sql: str, params: Sequence[object]) -> list[tuple[Any, ...]]:
        """Run one query to its end. The stream of iter_messages reads each of its chunks with this."""
        with self._using() as conn:
            return conn.execute(sql, params).fetchall()

    # -- overview ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Figures for the whole archive.

        `{"messages", "conversations", "first_timestamp", "last_timestamp", "messages_without_conversation"}`.
        A message with no conversation is not in `messages`: it has its own figure.
        """
        with self._reading() as conn:
            total, with_messages, first, last = conn.execute(
                "SELECT COALESCE(SUM(message_count), 0), COALESCE(SUM(message_count > 0), 0),"
                " MIN(first_sent_ns), MAX(last_sent_ns) FROM conversations"
            ).fetchone()
            unfiled = conn.execute(
                f"SELECT COUNT(*) FROM messages m INDEXED BY {rows.CONVERSATION_INDEX}"
                f" WHERE {rows.UNFILED} AND {rows.VISIBLE}"
            ).fetchone()[0]
        return {
            "messages": int(total),
            "conversations": int(with_messages),
            "first_timestamp": rows.timestamp_from_ns(first),
            "last_timestamp": rows.timestamp_from_ns(last),
            "messages_without_conversation": int(unfiled),
        }

    def conversations(
        self, *, limit: int | None = None, offset: int = 0, order: str = "count", include_empty: bool = False
    ) -> list[Conversation]:
        """Conversations, by message count or by last message. Empty ones only with `include_empty`."""
        order_sql = CONVERSATION_ORDERS[check_choice("order", order, CONVERSATION_ORDERS)]
        size = -1 if limit is None else check_int("limit", limit, 0)
        check_int("offset", offset, 0)
        where = "" if check_bool("include_empty", include_empty) else "WHERE c.message_count > 0"
        found = self._fetch_all(
            f"{rows.CONVERSATION_SELECT} {where} ORDER BY {order_sql} LIMIT ? OFFSET ?", (size, offset)
        )
        return [rows.conversation(row) for row in found]

    def conversation(self, conversation_id: str) -> Conversation | None:
        """One conversation by id, also one that holds no message. None for an unknown id."""
        key = check_id("conversation id", conversation_id)
        found = self._fetch_all(f"{rows.CONVERSATION_SELECT} WHERE c.chat_identifier = ?", (key,))
        return rows.conversation(found[0]) if found else None

    def people(self) -> list[Person]:
        """Every person: named people first, by name."""
        return [rows.person(row) for row in self._fetch_all(f"{rows.PERSON_SELECT} {rows.PERSON_ORDER}", ())]

    def person(self, person_id: int) -> Person | None:
        key = check_int("person id", person_id, 0)
        found = self._fetch_all(f"{rows.PERSON_SELECT} WHERE p.id = ?", (key,))
        return rows.person(found[0]) if found else None

    # -- messages ---------------------------------------------------------

    def iter_messages(
        self,
        conversation_id: str | Iterable[str] | None,
        *,
        since: str | None = None,
        until: str | None = None,
        include_system: bool = False,
        no_conversation: bool = False,
    ) -> Iterator[Message]:
        """Every message of a conversation, oldest first, as a stream.

        `iter_messages(None, no_conversation=True)` gives the messages that belong to no conversation.

        A list of ids gives one stream in time order (one person's phone and email threads).
        The stream reads in chunks and holds no read transaction between them: a pull that
        commits while the stream is read can add, change or remove later rows of it.
        """
        check_bool("include_system", include_system)
        lower, upper = day_bounds(since, until)
        if check_bool("no_conversation", no_conversation) != (conversation_id is None):
            raise ValueError("give a conversation id, or None with no_conversation=True")
        keys: tuple[int | None, ...] = (None,)
        if not no_conversation:
            with self._using() as conn:
                keys = conversation_keys(conn, conversation_id)
        return paging.stream(self._fetch_all, keys, lower, upper, include_system)

    def messages(
        self,
        conversation_id: str | Iterable[str],
        *,
        around: str | None = None,
        before: str | None = None,
        after: str | None = None,
        oldest: bool = False,
        limit: int = 50,
        include_system: bool = False,
    ) -> dict[str, Any]:
        """One page: `{"messages": [...oldest first], "has_older", "has_newer"}`.

        `around`, `before` and `after` are message ids. With none, the page is the newest one,
        or the oldest one with `oldest=True`. An unknown conversation or anchor gives an empty page.
        The page is read into memory: use iter_messages() for a whole conversation.
        """
        check_bool("include_system", include_system)
        with self._reading() as conn:
            keys = conversation_keys(conn, conversation_id)
            return paging.page(
                conn,
                keys,
                around=around,
                before=before,
                after=after,
                oldest=check_bool("oldest", oldest),
                limit=limit,
                include_system=include_system,
            )

    def message(self, message_id: str) -> Message | None:
        """One message by id."""
        key = check_id("message id", message_id)
        found = self._fetch_all(f"{rows.MESSAGE_SELECT} WHERE m.guid = ?", (key,))
        return rows.message(found[0]) if found else None

    def message_by_source_rowid(self, rowid: int) -> Message | None:
        """The message that had this ROWID in the Messages database of the last pull.

        Saved chats of the app hold ROWIDs. The answer is right only while the Messages
        database is the one the ROWID came from: a rebuilt database numbers its rows again.
        The ROWID of a tapback or of a row that is not stored gives None.
        """
        key = check_int("rowid", rowid, 0)
        found = self._fetch_all(f"{rows.MESSAGE_SELECT} WHERE m.source_rowid = ? ORDER BY m.id LIMIT 1", (key,))
        return rows.message(found[0]) if found else None

    def day(
        self,
        conversation_id: str | Iterable[str],
        day: str,
        *,
        limit: int = days.DAY_LIMIT,
        offset: int | None = None,
        around: str | None = None,
        include_system: bool = False,
    ) -> dict[str, Any]:
        """One local day: `{"day", "total", "offset", "messages", "previous_day", "next_day"}`.

        `messages` are oldest first. `total` is the number of messages in the day and `offset`
        is the place of the first one given. A day longer than `limit` is given in part: from
        `offset` when one is named, else around the local time `around` ("YYYY-MM-DD HH:MM:SS"),
        else from its start. `previous_day` and `next_day` are the nearest days that hold
        messages, or None.
        """
        check_bool("include_system", include_system)
        with self._reading() as conn:
            keys = conversation_keys(conn, conversation_id)
            return days.day(conn, keys, day, limit=limit, offset=offset, around=around, include_system=include_system)

    def context(self, message_id: str, *, radius: int = 6, include_system: bool = False) -> list[Message]:
        """The message with up to `radius` neighbours on each side, in the same conversation."""
        check_bool("include_system", include_system)
        with self._reading() as conn:
            return paging.context(
                conn, check_id("message id", message_id), radius=radius, include_system=include_system
            )

    def reactions(self, message_ids: str | Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        """Current tapbacks per message id: `[{"kind", "is_from_me", "sender"}]`, oldest first.

        Every requested id is a key. A person has at most one tapback on a message: the last one.
        A removed tapback is not listed. Kind `emoji` has one more key, `emoji`.
        """
        ids = as_id_list("message ids", message_ids)
        latest: dict[str, dict[object, tuple[Any, ...]]] = {message_id: {} for message_id in ids}
        with self._reading() as conn:
            for start in range(0, len(ids), filters.ID_CHUNK):
                chunk = ids[start : start + filters.ID_CHUNK]
                marks = ", ".join("?" for _ in chunk)
                events = conn.execute(
                    "SELECT r.target_guid, r.is_from_me, r.sender_handle_id, r.kind, r.emoji, r.removed,"
                    " COALESCE(h.handle, '') FROM reactions r INDEXED BY reactions_by_target"
                    " LEFT JOIN handles h ON h.id = r.sender_handle_id"
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
        no_conversation: bool = False,
    ) -> dict[str, Any]:
        """`{"total": every match, "hits": [{"message", "ranges"}]}`.

        Words are AND-ed, "quoted text" is a phrase, a trailing * is a prefix.
        Text typed by a user never raises. `ranges` are [start, end) code point offsets in the text.
        `no_conversation=True` searches only the messages that belong to no conversation.
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
                no_conversation=no_conversation,
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
        no_conversation: bool = False,
    ) -> int | list[dict[str, Any]]:
        """Number of messages, or `[{"key", "total", "sent", "received"}]` per bucket.

        `group_by` is day, month, year, weekday_hour, conversation or sender.
        `text` follows the rules of search().
        `no_conversation=True` counts only the messages that belong to no conversation.
        """
        if group_by is not None:
            check_choice("group_by", group_by, figures.GROUPS)
        check_bool("include_system", include_system)
        parsed = None if text is None else search_module.parse_query(text)
        lower, upper = day_bounds(since, until)
        direction = filters.check_flag("from_me", from_me)
        from_totals = parsed is None and lower is None and upper is None and not include_system and not no_conversation
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
                no_conversation=no_conversation,
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
        """The figures of compute_stats in conversation.py. The README lists the keys."""
        check_bool("include_system", include_system)
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
