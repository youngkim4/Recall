"""Write messages, reactions and attachments into the archive, one batch at a time.

Every row is keyed by Apple's guid, so a row that the source still holds
keeps its id across pulls. The source is the truth: a stored row takes every
value the source has now. A row that did not change is not written again.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import NamedTuple

from archive.records import AttachmentRow, MessageRow, ReactionRow

MESSAGE_FIELDS = ", ".join(MessageRow._fields)
REACTION_FIELDS = ", ".join(ReactionRow._fields)
KIND = MessageRow._fields.index("kind")
ROWS_PER_INSERT = 400  # 17 values each: far below SQLite's limit of 32,766 bound values


class MessageCounts(NamedTuple):
    messages_added: int
    system_added: int
    messages_updated: int
    system_updated: int
    conversations: frozenset[int]  # conversations whose totals can have changed


class ReactionCounts(NamedTuple):
    added: int
    updated: int
    waiting: tuple[ReactionRow, ...]  # tapbacks on a message that the archive does not hold (yet)


class KindChanges(NamedTuple):
    """Stored rows that were removed because their guid is now a row of the other kind."""

    messages_removed: int
    reactions_removed: int
    conversations: frozenset[int]  # conversations whose totals can have changed


class AttachmentCounts(NamedTuple):
    added: int
    removed: int


def _marks(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _insert_messages(conn: sqlite3.Connection, rows: Sequence[tuple[object, ...]]) -> None:
    """Insert message rows, many rows in each statement.

    The insert trigger writes each row to the full-text index. FTS5 flushes
    its pending data at every statement that fires a trigger, so one statement
    for each row would write one index segment for each message. That is
    about nine times slower.
    """
    one_row = f"({_marks(len(MessageRow._fields) + 2)})"
    head = f"INSERT INTO messages({MESSAGE_FIELDS}, first_import_id, last_import_id) VALUES "
    for start in range(0, len(rows), ROWS_PER_INSERT):
        chunk = rows[start : start + ROWS_PER_INSERT]
        conn.execute(head + ", ".join([one_row] * len(chunk)), [value for row in chunk for value in row])


def _stored_messages(conn: sqlite3.Connection, guids: Sequence[str]) -> dict[str, tuple[int, MessageRow]]:
    rows = conn.execute(f"SELECT id, {MESSAGE_FIELDS} FROM messages WHERE guid IN ({_marks(len(guids))})", guids)
    return {row[1]: (int(row[0]), MessageRow._make(row[1:])) for row in rows.fetchall()}


def drop_other_kind(
    conn: sqlite3.Connection, message_guids: Sequence[str], reaction_guids: Sequence[str]
) -> KindChanges:
    """Remove a stored tapback whose guid is now a message, and a stored message whose guid is now a tapback.

    One source row is one stored row. Without this step a row that changes its
    kind would stay in the table of its old kind for ever: its guid is not gone.
    """
    reactions_removed = 0
    if message_guids:
        marks = _marks(len(message_guids))
        reactions_removed = conn.execute(f"DELETE FROM reactions WHERE guid IN ({marks})", list(message_guids)).rowcount
    stale: list[tuple[int, int | None]] = []
    if reaction_guids:
        marks = _marks(len(reaction_guids))
        stale = conn.execute(
            f"SELECT id, conversation_id FROM messages WHERE guid IN ({marks})", list(reaction_guids)
        ).fetchall()
    if stale:
        ids = [(message_id,) for message_id, _ in stale]
        conn.executemany("DELETE FROM attachments WHERE message_id = ?", ids)
        conn.executemany("DELETE FROM messages WHERE id = ?", ids)
    touched = frozenset(key for _, key in stale if key is not None)
    return KindChanges(len(stale), int(reactions_removed), touched)


def write_messages(conn: sqlite3.Connection, run_id: int, rows: Sequence[MessageRow]) -> MessageCounts:
    """Insert new messages and update the ones whose source row changed."""
    unique = list({row.guid: row for row in rows}.values())  # one guid twice in a batch: the last row wins
    if not unique:
        return MessageCounts(0, 0, 0, 0, frozenset())
    stored = _stored_messages(conn, [row.guid for row in unique])
    inserts: list[tuple[object, ...]] = []
    updates: list[tuple[object, ...]] = []
    system_updated = 0
    touched: set[int] = set()
    for row in unique:
        found = stored.get(row.guid)
        if found is None:
            inserts.append((*row, run_id, run_id))
            changed: tuple[int | None, ...] = (row.conversation_id,)
        elif found[1] != row:
            updates.append((*row[1:], run_id, found[0]))
            system_updated += int(row.kind == "system")
            changed = (found[1].conversation_id, row.conversation_id)
        else:
            continue
        touched.update(key for key in changed if key is not None)  # a message can have no conversation
    _insert_messages(conn, inserts)
    assignments = ", ".join(f"{name} = ?" for name in MessageRow._fields[1:])
    conn.executemany(f"UPDATE messages SET {assignments}, last_import_id = ? WHERE id = ?", updates)
    system_added = sum(1 for values in inserts if values[KIND] == "system")
    return MessageCounts(
        len(inserts) - system_added, system_added, len(updates) - system_updated, system_updated, frozenset(touched)
    )


def write_reactions(conn: sqlite3.Connection, run_id: int, rows: Sequence[ReactionRow]) -> ReactionCounts:
    """Insert new tapbacks and update changed ones.

    A tapback is stored only when the archive holds its target message.
    The others come back as `waiting`: the target can be in a later batch.
    """
    unique = list({row.guid: row for row in rows}.values())
    if not unique:
        return ReactionCounts(0, 0, ())
    targets = sorted({row.target_guid for row in unique})
    held = {
        guid for (guid,) in conn.execute(f"SELECT guid FROM messages WHERE guid IN ({_marks(len(targets))})", targets)
    }
    ready = [row for row in unique if row.target_guid in held]
    guids = [row.guid for row in ready]
    found = conn.execute(f"SELECT id, {REACTION_FIELDS} FROM reactions WHERE guid IN ({_marks(len(guids))})", guids)
    stored = {row[1]: (int(row[0]), ReactionRow._make(row[1:])) for row in found.fetchall()}
    inserts = [(*row, run_id, run_id) for row in ready if row.guid not in stored]
    updates = [
        (*row[1:], run_id, stored[row.guid][0]) for row in ready if row.guid in stored and stored[row.guid][1] != row
    ]
    conn.executemany(
        f"INSERT INTO reactions({REACTION_FIELDS}, first_import_id, last_import_id)"
        f" VALUES ({_marks(len(ReactionRow._fields) + 2)})",
        inserts,
    )
    assignments = ", ".join(f"{name} = ?" for name in ReactionRow._fields[1:])
    conn.executemany(f"UPDATE reactions SET {assignments}, last_import_id = ? WHERE id = ?", updates)
    return ReactionCounts(len(inserts), len(updates), tuple(row for row in unique if row.target_guid not in held))


def write_attachments(
    conn: sqlite3.Connection, message_guids: Sequence[str], rows: Sequence[AttachmentRow]
) -> AttachmentCounts:
    """Make the attachment records of the given messages equal to `rows`.

    `message_guids` names every message of the batch, also one that has no
    attachment now: a record that the source dropped is removed.
    """
    if not message_guids:
        return AttachmentCounts(0, 0)
    message_ids = {
        str(guid): int(message_id)
        for guid, message_id in conn.execute(
            f"SELECT guid, id FROM messages WHERE guid IN ({_marks(len(message_guids))})", list(message_guids)
        )
    }
    ids = sorted(message_ids.values())
    stored = {
        (int(row[1]), str(row[2])): (int(row[0]), *row[3:])
        for row in conn.execute(
            "SELECT id, message_id, guid, position, mime_type, file_name, path, total_bytes FROM attachments"
            f" WHERE message_id IN ({_marks(len(ids))})",
            ids,
        )
    }
    wanted = {
        (message_ids[row.message_guid], row.guid): (
            row.position,
            row.mime_type,
            row.file_name,
            row.path,
            row.total_bytes,
        )
        for row in rows
        if row.message_guid in message_ids
    }
    inserts = [(*key, *values) for key, values in wanted.items() if key not in stored]
    updates = [
        (*values, stored[key][0]) for key, values in wanted.items() if key in stored and stored[key][1:] != values
    ]
    gone = [(found[0],) for key, found in stored.items() if key not in wanted]
    conn.executemany(
        "INSERT INTO attachments(message_id, guid, position, mime_type, file_name, path, total_bytes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        inserts,
    )
    conn.executemany(
        "UPDATE attachments SET position = ?, mime_type = ?, file_name = ?, path = ?, total_bytes = ? WHERE id = ?",
        updates,
    )
    conn.executemany("DELETE FROM attachments WHERE id = ?", gone)
    return AttachmentCounts(len(inserts), len(gone))
