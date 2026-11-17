"""Build a synthetic chat.db for tests. Every name, number and text is invented.

The tables copy the part of Apple's Messages schema that the importer reads.
`without` drops optional columns, to act like an older macOS version.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

APPLE_EPOCH_UNIX = 978_307_200

TABLES: dict[str, tuple[str, ...]] = {
    "handle": (
        "ROWID INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE",
        "id TEXT NOT NULL",
        "country TEXT",
        "service TEXT NOT NULL",
        "uncanonicalized_id TEXT",
    ),
    "chat": (
        "ROWID INTEGER PRIMARY KEY AUTOINCREMENT",
        "guid TEXT UNIQUE NOT NULL",
        "style INTEGER",
        "chat_identifier TEXT",
        "service_name TEXT",
        "display_name TEXT",
    ),
    "message": (
        "ROWID INTEGER PRIMARY KEY AUTOINCREMENT",
        "guid TEXT UNIQUE NOT NULL",
        "text TEXT",
        "handle_id INTEGER DEFAULT 0",
        "subject TEXT",
        "attributedBody BLOB",
        "service TEXT",
        "date INTEGER",
        "date_read INTEGER",
        "date_delivered INTEGER",
        "is_from_me INTEGER DEFAULT 0",
        "is_audio_message INTEGER DEFAULT 0",
        "item_type INTEGER DEFAULT 0",
        "other_handle INTEGER DEFAULT 0",
        "group_title TEXT",
        "group_action_type INTEGER DEFAULT 0",
        "associated_message_guid TEXT",
        "associated_message_type INTEGER DEFAULT 0",
        "balloon_bundle_id TEXT",
        "expressive_send_style_id TEXT",
        "thread_originator_guid TEXT",
        "date_retracted INTEGER",
        "date_edited INTEGER",
        "associated_message_emoji TEXT",
    ),
    "attachment": (
        "ROWID INTEGER PRIMARY KEY AUTOINCREMENT",
        "guid TEXT UNIQUE NOT NULL",
        "filename TEXT",
        "uti TEXT",
        "mime_type TEXT",
        "transfer_name TEXT",
        "total_bytes INTEGER DEFAULT 0",
        "is_sticker INTEGER DEFAULT 0",
    ),
    "chat_handle_join": (
        "chat_id INTEGER REFERENCES chat (ROWID) ON DELETE CASCADE",
        "handle_id INTEGER REFERENCES handle (ROWID) ON DELETE CASCADE",
        "UNIQUE(chat_id, handle_id)",
    ),
    "chat_message_join": (
        "chat_id INTEGER REFERENCES chat (ROWID) ON DELETE CASCADE",
        "message_id INTEGER REFERENCES message (ROWID) ON DELETE CASCADE",
        "message_date INTEGER DEFAULT 0",
        "PRIMARY KEY (chat_id, message_id)",
    ),
    "message_attachment_join": (
        "message_id INTEGER REFERENCES message (ROWID) ON DELETE CASCADE",
        "attachment_id INTEGER REFERENCES attachment (ROWID) ON DELETE CASCADE",
        "UNIQUE(message_id, attachment_id)",
    ),
}

INDEXES: tuple[str, ...] = (
    "CREATE INDEX chat_message_join_idx_message_id_only ON chat_message_join(message_id)",
    "CREATE INDEX message_attachment_join_idx_message_id ON message_attachment_join(message_id)",
    "CREATE INDEX chat_handle_join_idx_handle_id ON chat_handle_join(handle_id)",
)

OLD_MACOS_MISSING = frozenset(
    {"display_name", "thread_originator_guid", "date_edited", "date_retracted", "associated_message_emoji"}
)


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def unix_seconds(moment: datetime) -> int:
    return int(moment.timestamp())


def apple_time(moment: datetime | int | None, unit: str = "ns") -> int | None:
    """Apple time for a UTC moment: nanoseconds or seconds since 2001-01-01."""
    if moment is None:
        return None
    if isinstance(moment, int):
        return moment
    seconds = moment.timestamp() - APPLE_EPOCH_UNIX
    if unit == "s":
        return int(seconds)
    return int(round(seconds * 1_000_000_000))


def make_guid(kind: str, number: int) -> str:
    prefix = {"message": "A", "chat": "C", "attachment": "F"}[kind]
    return f"{prefix}{number:07X}-0000-4000-8000-{number:012X}"


class ChatDbBuilder:
    """Write rows into a new synthetic chat.db."""

    def __init__(
        self,
        path: Path,
        *,
        without: Iterable[str] = (),
        date_unit: str = "ns",
        wal: bool = False,
        indexes: bool = True,
        skip_tables: Iterable[str] = (),
    ) -> None:
        self.path = Path(path)
        self.date_unit = date_unit
        self._without = frozenset(without)
        self._skipped = frozenset(skip_tables)
        self._counter = 0
        self._closed = False
        self.raw = sqlite3.connect(str(self.path))
        if wal:
            self.raw.execute("PRAGMA journal_mode=WAL")
        self._columns: dict[str, frozenset[str]] = {}
        self._create(indexes)

    def _create(self, indexes: bool) -> None:
        for table, definitions in TABLES.items():
            if table in self._skipped:
                continue
            kept = [d for d in definitions if d.split()[0] not in self._without]
            self.raw.execute(f"CREATE TABLE {table} ({', '.join(kept)})")
            self._columns[table] = frozenset(row[1] for row in self.raw.execute(f"PRAGMA table_info({table})"))
        if indexes:
            for statement in INDEXES:
                if statement.split(" ON ")[1].split("(")[0] not in self._skipped:
                    self.raw.execute(statement)
        self.raw.commit()

    def _insert(self, table: str, values: Mapping[str, Any]) -> int:
        usable = {key: value for key, value in values.items() if key in self._columns[table] or key == "ROWID"}
        names = ", ".join(usable)
        marks = ", ".join("?" for _ in usable)
        cursor = self.raw.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", tuple(usable.values()))
        return int(cursor.lastrowid or 0)

    def _next(self) -> int:
        self._counter += 1
        return self._counter

    def handle(self, handle_id: str, service: str = "iMessage") -> int:
        return self._insert("handle", {"id": handle_id, "service": service})

    def chat(
        self,
        identifier: str | None,
        *,
        display_name: str | None = None,
        service: str = "iMessage",
        members: Sequence[int] = (),
        guid: str | None = None,
    ) -> int:
        number = self._next()
        rowid = self._insert(
            "chat",
            {
                "guid": guid or f"{service};-;{identifier};{number}",
                "style": 43 if str(identifier or "").startswith("chat") else 45,
                "chat_identifier": identifier,
                "service_name": service,
                "display_name": display_name,
            },
        )
        for handle_rowid in members:
            self.member(rowid, handle_rowid)
        return rowid

    def member(self, chat_rowid: int, handle_rowid: int) -> None:
        self._insert("chat_handle_join", {"chat_id": chat_rowid, "handle_id": handle_rowid})

    def message(
        self,
        chat: int | None,
        *,
        when: datetime | int | None,
        text: str | None = None,
        body: bytes | None = None,
        from_me: bool = False,
        handle: int = 0,
        guid: str | None = None,
        service: str | None = "iMessage",
        item_type: int = 0,
        associated_guid: str | None = None,
        associated_type: int = 0,
        associated_emoji: str | None = None,
        reply_to: str | None = None,
        edited: datetime | int | None = None,
        retracted: datetime | int | None = None,
        group_title: str | None = None,
        group_action_type: int = 0,
        other_handle: int = 0,
        extra: Mapping[str, Any] | None = None,
        rowid: int | None = None,
    ) -> int:
        """Add one message row and its chat link. Returns the ROWID."""
        number = self._next()
        values: dict[str, Any] = {
            "guid": guid if guid is not None else make_guid("message", number),
            "text": text,
            "attributedBody": body,
            "handle_id": handle,
            "is_from_me": int(from_me),
            "date": apple_time(when, self.date_unit),
            "service": service,
            "item_type": item_type,
            "associated_message_guid": associated_guid,
            "associated_message_type": associated_type,
            "associated_message_emoji": associated_emoji,
            "thread_originator_guid": reply_to,
            "date_edited": apple_time(edited, self.date_unit) or 0,
            "date_retracted": apple_time(retracted, self.date_unit) or 0,
            "group_title": group_title,
            "group_action_type": group_action_type,
            "other_handle": other_handle,
        }
        values.update(extra or {})
        if rowid is not None:
            values["ROWID"] = rowid
        message_rowid = self._insert("message", values)
        if chat is not None:
            self.link(chat, message_rowid)
        return message_rowid

    def link(self, chat_rowid: int, message_rowid: int) -> None:
        self._insert("chat_message_join", {"chat_id": chat_rowid, "message_id": message_rowid})

    def tapback(
        self,
        chat: int,
        target_guid: str,
        code: int,
        *,
        when: datetime | int,
        from_me: bool = False,
        handle: int = 0,
        prefix: str = "p:0/",
        emoji: str | None = None,
        text: str | None = "Loved an invented message",
    ) -> int:
        return self.message(
            chat,
            when=when,
            text=text,
            from_me=from_me,
            handle=handle,
            associated_guid=f"{prefix}{target_guid}",
            associated_type=code,
            associated_emoji=emoji,
        )

    def attachment(
        self,
        message_rowid: int,
        *,
        mime_type: str | None = "image/jpeg",
        transfer_name: str | None = "IMG_0001.jpeg",
        filename: str | None = None,
        total_bytes: int | None = 1024,
        guid: str | None = None,
    ) -> int:
        if "attachment" in self._skipped:
            raise AssertionError("this synthetic database has no attachment table")
        number = self._next()
        path = filename if filename is not None else f"~/Library/Messages/Attachments/00/{number:02d}/{transfer_name}"
        rowid = self._insert(
            "attachment",
            {
                "guid": guid or make_guid("attachment", number),
                "filename": path,
                "mime_type": mime_type,
                "transfer_name": transfer_name,
                "total_bytes": total_bytes,
            },
        )
        self._insert("message_attachment_join", {"message_id": message_rowid, "attachment_id": rowid})
        return rowid

    def guid_of(self, message_rowid: int) -> str:
        row = self.raw.execute("SELECT guid FROM message WHERE ROWID = ?", (message_rowid,)).fetchone()
        if row is None:
            raise KeyError(message_rowid)
        return str(row[0])

    def commit(self) -> Path:
        self.raw.commit()
        return self.path

    def close(self) -> None:
        """Commit and close. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        self.raw.commit()
        self.raw.close()
