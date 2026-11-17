"""Write messages, reactions and attachments into the archive, one batch at a time.

Every row is keyed by Apple's guid. A row the archive has is never deleted,
and what the archive knows is never replaced by less:
- an older copy of a message does not undo an edit or an unsend
- text that the source lost does not blank the stored text
- an unknown value does not replace a known one
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import NamedTuple

from archive.records import AttachmentRow, MessageRow, ReactionRow

MESSAGE_FIELDS = ", ".join(MessageRow._fields)
REACTION_FIELDS = ", ".join(ReactionRow._fields)
SOURCE_ROWID = MessageRow._fields.index("source_rowid")


class MessageCounts(NamedTuple):
    messages_added: int
    system_added: int
    messages_updated: int
    conversations: frozenset[int]  # conversations whose totals can have changed


def _marks(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _is_older_copy(stored: MessageRow, incoming: MessageRow) -> bool:
    """True when `incoming` shows the message before an edit or unsend that the archive has."""
    if stored.unsent and not incoming.unsent:
        return True
    if stored.edited_at is None:
        return False
    return incoming.edited_at is None or incoming.edited_at < stored.edited_at


def merge_message(stored: MessageRow, incoming: MessageRow) -> MessageRow:
    """The row the archive must hold after an import sees the message again."""
    if _is_older_copy(stored, incoming):
        return stored._replace(source_rowid=incoming.source_rowid)
    merged = incoming._replace(
        sender_handle_id=incoming.sender_handle_id if incoming.sender_handle_id is not None else stored.sender_handle_id,
        service=incoming.service or stored.service,
        reply_to_guid=incoming.reply_to_guid or stored.reply_to_guid,
        extra=incoming.extra if incoming.extra is not None else stored.extra,
    )
    if merged.text == "" and stored.text != "" and not merged.unsent:
        return merged._replace(text=stored.text, kind=stored.kind, text_source=stored.text_source)
    return merged


def _stored_messages(conn: sqlite3.Connection, guids: Sequence[str]) -> dict[str, tuple[int, MessageRow]]:
    rows = conn.execute(f"SELECT id, {MESSAGE_FIELDS} FROM messages WHERE guid IN ({_marks(len(guids))})", guids)
    return {row[1]: (int(row[0]), MessageRow._make(row[1:])) for row in rows.fetchall()}


def write_messages(conn: sqlite3.Connection, run_id: int, rows: Sequence[MessageRow]) -> MessageCounts:
    """Insert new messages, update changed ones, and mark the others as seen by this run."""
    unique = list({row.guid: row for row in rows}.values())
    if not unique:
        return MessageCounts(0, 0, 0, frozenset())
    stored = _stored_messages(conn, [row.guid for row in unique])
    inserts: list[tuple[object, ...]] = []
    updates: list[tuple[object, ...]] = []
    seen: list[tuple[int, int, int]] = []
    touched: set[int] = set()
    for row in unique:
        found = stored.get(row.guid)
        if found is None:
            inserts.append((*row, run_id, run_id))
            touched.add(row.conversation_id)
            continue
        message_id, old = found
        merged = merge_message(old, row)
        if merged[:SOURCE_ROWID] == old[:SOURCE_ROWID]:
            seen.append((run_id, merged.source_rowid, message_id))
        else:
            updates.append((*merged[1:], run_id, message_id))
            touched.update((old.conversation_id, merged.conversation_id))
    conn.executemany(
        f"INSERT INTO messages({MESSAGE_FIELDS}, first_import_id, last_import_id) VALUES ({_marks(len(MessageRow._fields) + 2)})",
        inserts,
    )
    assignments = ", ".join(f"{name} = ?" for name in MessageRow._fields[1:])
    conn.executemany(f"UPDATE messages SET {assignments}, last_import_id = ? WHERE id = ?", updates)
    conn.executemany("UPDATE messages SET last_import_id = ?, source_rowid = ? WHERE id = ?", seen)
    system_added = sum(1 for values in inserts if values[MessageRow._fields.index("kind")] == "system")
    return MessageCounts(len(inserts) - system_added, system_added, len(updates), frozenset(touched))


def write_reactions(conn: sqlite3.Connection, run_id: int, rows: Sequence[ReactionRow]) -> int:
    """Insert new tapbacks and refresh known ones. Return the number added."""
    unique = list({row.guid: row for row in rows}.values())
    if not unique:
        return 0
    guids = [row.guid for row in unique]
    found = conn.execute(f"SELECT id, {REACTION_FIELDS} FROM reactions WHERE guid IN ({_marks(len(guids))})", guids)
    stored = {row[1]: (int(row[0]), ReactionRow._make(row[1:])) for row in found.fetchall()}
    inserts = [(*row, run_id, run_id) for row in unique if row.guid not in stored]
    updates: list[tuple[object, ...]] = []
    for row in unique:
        if row.guid not in stored:
            continue
        reaction_id, old = stored[row.guid]
        merged = row._replace(
            conversation_id=row.conversation_id if row.conversation_id is not None else old.conversation_id,
            sender_handle_id=row.sender_handle_id if row.sender_handle_id is not None else old.sender_handle_id,
            emoji=row.emoji or old.emoji,
        )
        updates.append((*merged[1:], run_id, reaction_id))
    conn.executemany(
        f"INSERT INTO reactions({REACTION_FIELDS}, first_import_id, last_import_id) VALUES ({_marks(len(ReactionRow._fields) + 2)})",
        inserts,
    )
    assignments = ", ".join(f"{name} = ?" for name in ReactionRow._fields[1:])
    conn.executemany(f"UPDATE reactions SET {assignments}, last_import_id = ? WHERE id = ?", updates)
    return len(inserts)


def write_attachments(conn: sqlite3.Connection, rows: Sequence[AttachmentRow]) -> int:
    """Insert new attachment records and fill in metadata that was missing. Return the number added."""
    unique = list({row.guid: row for row in rows}.values())
    if not unique:
        return 0
    message_guids = sorted({row.message_guid for row in unique})
    message_ids = dict(
        conn.execute(f"SELECT guid, id FROM messages WHERE guid IN ({_marks(len(message_guids))})", message_guids).fetchall()
    )
    guids = [row.guid for row in unique]
    stored = {
        row[0]: row[1:]
        for row in conn.execute(
            f"SELECT guid, id, mime_type, file_name, path, total_bytes FROM attachments WHERE guid IN ({_marks(len(guids))})",
            guids,
        ).fetchall()
    }
    inserts: list[tuple[object, ...]] = []
    updates: list[tuple[object, ...]] = []
    for row in unique:
        message_id = message_ids.get(row.message_guid)
        if message_id is None:
            continue
        old = stored.get(row.guid)
        if old is None:
            inserts.append((row.guid, message_id, row.position, row.mime_type, row.file_name, row.path, row.total_bytes))
            continue
        attachment_id, *known = old
        merged = [new if new is not None else kept for new, kept in zip(row[3:], known)]
        if merged != known:
            updates.append((*merged, attachment_id))
    conn.executemany(
        "INSERT INTO attachments(guid, message_id, position, mime_type, file_name, path, total_bytes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        inserts,
    )
    conn.executemany(
        "UPDATE attachments SET mime_type = ?, file_name = ?, path = ?, total_bytes = ? WHERE id = ?", updates
    )
    return len(inserts)
