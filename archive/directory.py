"""Write the small tables of the archive: handles, people, conversations, members.

These tables are bounded by the number of contacts and chats, not by the
number of messages, so each is read whole. The source is the truth: a title,
a service or a member list that changed in the source changes here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Iterable, Sequence

from archive.handles import handle_kind, handle_match_key, is_group_identifier
from archive.people import ContactGroup, HandleRow, plan_people
from archive.records import SourceChat

# when one chat_identifier has several chat rows, the conversation shows this service first
SERVICE_ORDER = ("iMessage", "RCS", "SMS")


def service_rank(service: str) -> tuple[int, str]:
    return (SERVICE_ORDER.index(service) if service in SERVICE_ORDER else len(SERVICE_ORDER), service)


def add_handle(conn: sqlite3.Connection, text: str) -> int:
    """Add one handle. A new handle starts as its own unnamed person."""
    person_id = conn.execute("INSERT INTO people(name) VALUES ('')").lastrowid
    cursor = conn.execute(
        "INSERT INTO handles(handle, kind, match_key, person_id) VALUES (?, ?, ?, ?)",
        (text, handle_kind(text), handle_match_key(text), person_id),
    )
    return int(cursor.lastrowid or 0)


def sync_handles(conn: sqlite3.Connection, source_handles: Iterable[tuple[int, str]]) -> tuple[dict[int, int], int]:
    """Add the handles the archive does not have, and bring the kind and the match key of the
    stored ones up to the rules of this version. Return (source ROWID -> handle id, number added).
    """
    known = {
        str(text): (int(handle_id), str(kind), str(match_key))
        for text, handle_id, kind, match_key in conn.execute("SELECT handle, id, kind, match_key FROM handles")
    }
    ids: dict[int, int] = {}
    added = 0
    for rowid, raw in source_handles:
        text = raw.strip()
        if not text:
            continue
        found = known.get(text)
        rules = (handle_kind(text), handle_match_key(text))
        if found is None:
            found = (add_handle(conn, text), *rules)
            known[text] = found
            added += 1
        elif found[1:] != rules:
            conn.execute("UPDATE handles SET kind = ?, match_key = ? WHERE id = ?", (*rules, found[0]))
            known[text] = (found[0], *rules)
        ids[rowid] = found[0]
    return ids, added


def _merged_chats(chats: Sequence[SourceChat]) -> dict[str, tuple[str, str]]:
    """chat_identifier -> (title, service). Several chat rows can share one identifier."""
    merged: dict[str, tuple[str, str]] = {}
    for chat in chats:
        identifier = (chat.identifier or "").strip()
        if not identifier:
            continue
        title, service = (chat.title or "").strip(), (chat.service or "").strip()
        old_title, old_service = merged.get(identifier, ("", ""))
        services = [name for name in (old_service, service) if name]
        merged[identifier] = (title or old_title, min(services, key=service_rank) if services else "")
    return merged


def add_conversation(conn: sqlite3.Connection, identifier: str, title: str = "", service: str = "") -> int:
    cursor = conn.execute(
        "INSERT INTO conversations(chat_identifier, is_group, title, service) VALUES (?, ?, ?, ?)",
        (identifier, int(is_group_identifier(identifier)), title, service),
    )
    return int(cursor.lastrowid or 0)


def set_details(conn: sqlite3.Connection, details: Iterable[tuple[str, str, int]]) -> None:
    """Set the title and the service of conversations: (title, service, conversation id)."""
    conn.executemany("UPDATE conversations SET title = ?, service = ? WHERE id = ?", list(details))


def sync_conversations(conn: sqlite3.Connection, chats: Sequence[SourceChat]) -> tuple[dict[int, int], int]:
    """Add or update conversations. Return (source chat ROWID -> conversation id, number added)."""
    stored = {
        str(identifier): (int(conversation_id), str(title), str(service))
        for identifier, conversation_id, title, service in conn.execute(
            "SELECT chat_identifier, id, title, service FROM conversations"
        )
    }
    ids_by_identifier: dict[str, int] = {}
    added = 0
    for identifier, (title, service) in _merged_chats(chats).items():
        found = stored.get(identifier)
        if found is None:
            ids_by_identifier[identifier] = add_conversation(conn, identifier, title, service)
            added += 1
            continue
        conversation_id, old_title, old_service = found
        if (title, service) != (old_title, old_service):
            conn.execute(
                "UPDATE conversations SET title = ?, service = ? WHERE id = ?", (title, service, conversation_id)
            )
        ids_by_identifier[identifier] = conversation_id
    ids = {
        chat.rowid: ids_by_identifier[(chat.identifier or "").strip()]
        for chat in chats
        if (chat.identifier or "").strip() in ids_by_identifier
    }
    return ids, added


def sync_members(conn: sqlite3.Connection, pairs: Iterable[tuple[int, int]], *, replace: bool) -> None:
    """Add (conversation id, handle id) pairs. With `replace`, a pair that is not given is removed."""
    wanted = set(pairs)
    stored = {
        (int(conversation_id), int(handle_id))
        for conversation_id, handle_id in conn.execute("SELECT conversation_id, handle_id FROM conversation_members")
    }
    conn.executemany(
        "INSERT INTO conversation_members(conversation_id, handle_id) VALUES (?, ?)", sorted(wanted - stored)
    )
    if replace:
        conn.executemany(
            "DELETE FROM conversation_members WHERE conversation_id = ? AND handle_id = ?", sorted(stored - wanted)
        )


def _used_ids(conn: sqlite3.Connection, columns: Sequence[tuple[str, str]]) -> set[int]:
    """Every id that the given (table, column) pairs hold. One pass over each table."""
    return {
        int(row[0])
        for table, column in columns
        for row in conn.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL")
    }


def drop_gone(conn: sqlite3.Connection, identifiers: Collection[str], handles: Collection[str]) -> None:
    """Remove the conversations and handles that the source no longer has and that no stored row uses.

    Call it at the end of a full pull, after the messages that are gone were removed.
    """
    gone_chats = [
        int(conversation_id)
        for conversation_id, identifier in conn.execute("SELECT id, chat_identifier FROM conversations")
        if str(identifier) not in identifiers
    ]
    if gone_chats:
        # one pass over each table, only when a conversation is a candidate
        held = _used_ids(conn, (("messages", "conversation_id"), ("reactions", "conversation_id")))
        free = [(conversation_id,) for conversation_id in gone_chats if conversation_id not in held]
        conn.executemany("DELETE FROM conversation_members WHERE conversation_id = ?", free)
        conn.executemany("DELETE FROM conversations WHERE id = ?", free)
    gone_handles = [
        int(handle_id) for handle_id, text in conn.execute("SELECT id, handle FROM handles") if str(text) not in handles
    ]
    if not gone_handles:
        return
    used = _used_ids(
        conn,
        (("messages", "sender_handle_id"), ("reactions", "sender_handle_id"), ("conversation_members", "handle_id")),
    )
    conn.executemany(
        "DELETE FROM handles WHERE id = ?", [(handle_id,) for handle_id in gone_handles if handle_id not in used]
    )


def apply_people(conn: sqlite3.Connection, groups: Sequence[ContactGroup] | None) -> None:
    """Put every handle under the person that the contact list gives it.

    `groups` is None when the import has no contact list: nothing changes.
    """
    if groups is None:
        return
    rows = [
        HandleRow(int(handle_id), str(match_key), int(person_id))
        for handle_id, match_key, person_id in conn.execute("SELECT id, match_key, person_id FROM handles ORDER BY id")
    ]
    owner = {row.id: row.person_id for row in rows}
    names = {int(person_id): str(name) for person_id, name in conn.execute("SELECT id, name FROM people")}
    for plan in plan_people(rows, groups):
        name = plan.name or ""
        person_id = plan.person_id
        if person_id is None:
            person_id = int(conn.execute("INSERT INTO people(name) VALUES (?)", (name,)).lastrowid or 0)
        elif names.get(person_id) != name:
            conn.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
        moved = [(person_id, handle_id) for handle_id in plan.handle_ids if owner.get(handle_id) != person_id]
        conn.executemany("UPDATE handles SET person_id = ? WHERE id = ?", moved)


def link_conversations(conn: sqlite3.Connection) -> None:
    """Set the person of each one-to-one conversation.

    The person is the owner of the handle that equals the conversation id.
    When no handle does, it is the only person among the members.
    """
    conn.execute(
        """
        UPDATE conversations SET person_id = COALESCE(
            (SELECT h.person_id FROM handles h WHERE h.handle = conversations.chat_identifier),
            (SELECT MIN(h.person_id) FROM conversation_members cm JOIN handles h ON h.id = cm.handle_id
              WHERE cm.conversation_id = conversations.id
             HAVING COUNT(DISTINCT h.person_id) = 1)
        ) WHERE is_group = 0
        """
    )


def drop_unused_people(conn: sqlite3.Connection) -> None:
    """Remove people that no handle points to. Call after link_conversations."""
    conn.execute("DELETE FROM people WHERE id NOT IN (SELECT person_id FROM handles)")


def settle(conn: sqlite3.Connection, groups: Sequence[ContactGroup] | None, touched: Iterable[int]) -> None:
    """The last steps of every import: people, the totals of the touched conversations, links."""
    apply_people(conn, groups)
    refresh_totals(conn, touched)
    link_conversations(conn)
    drop_unused_people(conn)


def conversation_ids(conn: sqlite3.Connection) -> list[int]:
    return [int(row[0]) for row in conn.execute("SELECT id FROM conversations")]


def refresh_totals(conn: sqlite3.Connection, conversation_ids: Iterable[int]) -> None:
    """Count the messages of the given conversations again. Group events are not messages."""
    conn.executemany(
        """
        UPDATE conversations SET (message_count, sent_count, received_count, first_sent_ns, last_sent_ns) = (
            SELECT COUNT(*), COALESCE(SUM(is_from_me), 0), COALESCE(SUM(1 - is_from_me), 0),
                   MIN(sent_ns), MAX(sent_ns)
              FROM messages INDEXED BY messages_by_conversation_time
             WHERE conversation_id = ?1 AND kind != 'system'
        ) WHERE id = ?1
        """,
        [(conversation_id,) for conversation_id in sorted(set(conversation_ids))],
    )
