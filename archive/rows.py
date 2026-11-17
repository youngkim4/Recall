"""The dict shapes of the query contract, and the SQL that reads them.

Message, Conversation and Person are plain dicts that json.dumps accepts.
The key order is the order of the contract.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from archive.timeconv import local_timestamp, sql_local_timestamp

LIST_SEPARATOR = "\x1f"
UNKNOWN_MIME_TYPE = "application/octet-stream"
VISIBLE = "m.kind != 'system'"  # group events are shown only on request
FILED = "m.conversation_id IS NOT NULL"  # a message with no conversation is given only on request
UNFILED = "m.conversation_id IS NULL"

Message = dict[str, Any]
Conversation = dict[str, Any]
Person = dict[str, Any]

# (conversation_id, sent_ns, source_rowid, id, is_from_me, kind, sender_handle_id)
CONVERSATION_INDEX = "messages_by_conversation_time"
# The place of a message in time. Messages of one moment are in the order of the source.
ORDER_KEY = "(m.sent_ns, m.source_rowid, m.id)"
ORDER_OLDEST = "m.sent_ns, m.source_rowid, m.id"
ORDER_NEWEST = "m.sent_ns DESC, m.source_rowid DESC, m.id DESC"
TIME_INDEX = "messages_counted_by_time"  # (sent_ns, conversation_id, is_from_me) where kind is not system


def _message_select(messages_table: str) -> str:
    """Column 0 (sent_ns), the last column (source_rowid) and column 1 (id) are the sort key; the rest is the Message."""
    return f"""
SELECT m.sent_ns, m.id, m.guid, COALESCE(c.chat_identifier, ''), {sql_local_timestamp("m.sent_at")}, m.sent_at,
       m.is_from_me, COALESCE(h.handle, ''), m.text, m.kind,
       (SELECT group_concat(COALESCE(NULLIF(a.mime_type, ''), '{UNKNOWN_MIME_TYPE}'), char(31)
                            ORDER BY a.position, a.id)
          FROM attachments a WHERE a.message_id = m.id),
       m.reply_to_guid, m.edited_at, m.unsent, m.source_rowid
  FROM {messages_table}
  LEFT JOIN conversations c ON c.id = m.conversation_id
  LEFT JOIN handles h ON h.id = m.sender_handle_id
"""


MESSAGE_SELECT = _message_select("messages m")
# For reads of one conversation in time order. The index is named so that the cost of a page
# never depends on what the query planner guesses about a young or unanalysed archive.
PAGED_MESSAGE_SELECT = _message_select(f"messages m INDEXED BY {CONVERSATION_INDEX}")

CONVERSATION_SELECT = """
SELECT c.chat_identifier, c.is_group, c.title, c.message_count, c.sent_count, c.received_count,
       c.first_sent_ns, c.last_sent_ns,
       (SELECT group_concat(h.handle, char(31) ORDER BY cm.message_count DESC, h.handle)
          FROM conversation_members cm JOIN handles h ON h.id = cm.handle_id
         WHERE cm.conversation_id = c.id),
       c.person_id
  FROM conversations c
"""

PERSON_SELECT = """
SELECT p.id, p.name,
       (SELECT group_concat(h.handle, char(31) ORDER BY h.handle) FROM handles h WHERE h.person_id = p.id),
       (SELECT group_concat(c.chat_identifier, char(31) ORDER BY c.message_count DESC, c.chat_identifier)
          FROM conversations c WHERE c.person_id = p.id AND c.is_group = 0 AND c.message_count > 0)
  FROM people p
"""
PERSON_ORDER = "ORDER BY p.name = '', p.name COLLATE NOCASE, p.id"


def _split(joined: str | None) -> list[str]:
    return joined.split(LIST_SEPARATOR) if joined else []


def timestamp_from_ns(unix_ns: int | None) -> str | None:
    return None if unix_ns is None else local_timestamp(unix_ns // 1_000_000_000)


def sort_key(row: sqlite3.Row | tuple[Any, ...]) -> tuple[int, int, int]:
    """Position of a message row in time: (sent_ns, source_rowid, id), the columns of ORDER_KEY."""
    return row[0], row[14], row[1]


def message(row: sqlite3.Row | tuple[Any, ...]) -> Message:
    return {
        "id": row[2],
        "conversation": row[3],
        "timestamp": row[4],
        "sent_at": row[5],
        "is_from_me": bool(row[6]),
        "sender": row[7],
        "text": row[8],
        "kind": row[9],
        "attachments": _split(row[10]),
        "reply_to": row[11],
        "edited": row[12] is not None,
        "unsent": bool(row[13]),
    }


def conversation(row: sqlite3.Row | tuple[Any, ...]) -> Conversation:
    return {
        "id": row[0],
        "is_group": bool(row[1]),
        "name": row[2],
        "message_count": row[3],
        "sent_count": row[4],
        "received_count": row[5],
        "first_timestamp": timestamp_from_ns(row[6]),
        "last_timestamp": timestamp_from_ns(row[7]),
        "members": _split(row[8]),
        "person_id": row[9],
    }


def person(row: sqlite3.Row | tuple[Any, ...]) -> Person:
    return {"id": row[0], "name": row[1], "handles": _split(row[2]), "conversations": _split(row[3])}
