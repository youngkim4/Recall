"""Read an archive file with plain SQL, for importer tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# everything a user can see; import bookkeeping is left out
CONTENT_QUERIES: dict[str, str] = {
    "people": "SELECT id, name FROM people ORDER BY id",
    "handles": "SELECT handle, kind, match_key, person_id FROM handles ORDER BY handle",
    "conversations": (
        "SELECT chat_identifier, is_group, title, service, person_id, message_count, sent_count,"
        " received_count, first_sent_ns, last_sent_ns FROM conversations ORDER BY chat_identifier"
    ),
    "members": (
        "SELECT c.chat_identifier, h.handle FROM conversation_members cm"
        " JOIN conversations c ON c.id = cm.conversation_id JOIN handles h ON h.id = cm.handle_id"
        " ORDER BY 1, 2"
    ),
    "messages": (
        "SELECT m.guid, COALESCE(c.chat_identifier, ''), COALESCE(h.handle, ''), m.is_from_me, m.sent_ns, m.text,"
        " m.kind, m.service, m.reply_to_guid, m.edited_at, m.unsent, m.item_type, m.text_source, m.extra"
        " FROM messages m LEFT JOIN conversations c ON c.id = m.conversation_id"
        " LEFT JOIN handles h ON h.id = m.sender_handle_id ORDER BY m.guid"
    ),
    "attachments": (
        "SELECT a.guid, m.guid, a.position, a.mime_type, a.file_name, a.path, a.total_bytes"
        " FROM attachments a JOIN messages m ON m.id = a.message_id ORDER BY a.guid"
    ),
    "reactions": (
        "SELECT r.guid, r.target_guid, r.target_part, c.chat_identifier, COALESCE(h.handle, ''), r.is_from_me,"
        " r.kind, r.emoji, r.removed, r.sent_ns FROM reactions r"
        " LEFT JOIN conversations c ON c.id = r.conversation_id"
        " LEFT JOIN handles h ON h.id = r.sender_handle_id ORDER BY r.guid"
    ),
}


def query(archive_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
    try:
        return [tuple(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def scalar(archive_path: Path, sql: str, params: tuple[Any, ...] = ()) -> Any:
    return query(archive_path, sql, params)[0][0]


def content(archive_path: Path) -> dict[str, list[tuple[Any, ...]]]:
    """Every user-visible row of the archive."""
    return {name: query(archive_path, sql) for name, sql in CONTENT_QUERIES.items()}


def texts(archive_path: Path, kind: str | None = None) -> list[str]:
    where = "" if kind is None else "WHERE kind = ?"
    params = () if kind is None else (kind,)
    return [
        row[0]
        for row in query(archive_path, f"SELECT text FROM messages {where} ORDER BY sent_ns, source_rowid, id", params)
    ]


def runs(archive_path: Path) -> list[tuple[Any, ...]]:
    return query(archive_path, "SELECT id, mode, status, error FROM import_runs ORDER BY id")


def fts_is_consistent(archive_path: Path) -> bool:
    conn = sqlite3.connect(str(archive_path))
    try:
        conn.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES ('integrity-check', 1)")
        conn.rollback()
        indexed = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        stored = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return bool(indexed == stored)
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def totals_match_messages(archive_path: Path) -> bool:
    """The totals kept on each conversation equal a fresh count of its messages."""
    stale = query(
        archive_path,
        "SELECT c.chat_identifier FROM conversations c WHERE"
        " c.message_count != (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id AND m.kind != 'system')"
        " OR c.sent_count != (SELECT COALESCE(SUM(is_from_me), 0) FROM messages m"
        "    WHERE m.conversation_id = c.id AND m.kind != 'system')"
        " OR c.received_count != c.message_count - c.sent_count"
        " OR c.first_sent_ns IS NOT (SELECT MIN(sent_ns) FROM messages m"
        "    WHERE m.conversation_id = c.id AND m.kind != 'system')"
        " OR c.last_sent_ns IS NOT (SELECT MAX(sent_ns) FROM messages m"
        "    WHERE m.conversation_id = c.id AND m.kind != 'system')",
    )
    return not stale
