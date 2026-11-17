"""Read messages of one or more conversations in time order.

The position of a message is (sent_ns, id). Every read here walks the index
on (conversation_id, sent_ns, id), so a page costs the same anywhere in a
long conversation. Several conversations are read one by one and merged.
"""

from __future__ import annotations

import heapq
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from operator import itemgetter
from typing import Any

from archive import rows
from archive.filters import check_int, time_clauses

STREAM_CHUNK = 2000
Key = tuple[int, int]
Row = tuple[Any, ...]


def _empty_page() -> dict[str, Any]:
    return {"messages": [], "has_older": False, "has_newer": False}


def _side(
    conn: sqlite3.Connection,
    conversation_keys: Sequence[int],
    key: Key | None,
    *,
    older: bool,
    limit: int,
    include_system: bool,
) -> list[Row]:
    """Up to `limit` rows before (older) or after `key`, nearest first. `key` None means from the newest end."""
    if limit <= 0:
        return []
    compare, direction = ("<", "DESC") if older else (">", "ASC")
    clauses = ["m.conversation_id = ?"]
    if key is not None:
        clauses.append(f"(m.sent_ns, m.id) {compare} (?, ?)")
    if not include_system:
        clauses.append(rows.VISIBLE)
    sql = f"{rows.PAGED_MESSAGE_SELECT} WHERE {' AND '.join(clauses)} ORDER BY m.sent_ns {direction}, m.id {direction} LIMIT ?"
    parts = [conn.execute(sql, (conversation, *(key or ()), limit)).fetchall() for conversation in conversation_keys]
    if len(parts) == 1:
        return parts[0]
    return list(heapq.merge(*parts, key=rows.sort_key, reverse=older))[:limit]


def _anchor(conn: sqlite3.Connection, conversation_keys: Sequence[int], guid: str) -> Row | None:
    marks = ", ".join("?" for _ in conversation_keys)
    found: Row | None = conn.execute(
        f"{rows.MESSAGE_SELECT} WHERE m.guid = ? AND m.conversation_id IN ({marks})", (guid, *conversation_keys)
    ).fetchone()
    return found


def _is_visible(row: Row, include_system: bool) -> bool:
    return include_system or row[9] != "system"


def _around(older: list[Row], middle: list[Row], newer: list[Row], limit: int) -> list[Row]:
    """A window of `limit` rows with the anchor near the middle. A short side is filled from the other."""
    room = limit - len(middle)
    before = min(len(older), (limit - 1) // 2)
    after = min(len(newer), room - before)
    before = min(len(older), room - after)
    return [*reversed(older[:before]), *middle, *newer[:after]]


def page(
    conn: sqlite3.Connection,
    conversation_keys: Sequence[int],
    *,
    around: str | None,
    before: str | None,
    after: str | None,
    limit: int,
    include_system: bool,
) -> dict[str, Any]:
    """One page of messages, oldest first, with has_older and has_newer."""
    check_int("limit", limit, 1)
    anchors = [guid for guid in (around, before, after) if guid is not None]
    if len(anchors) > 1:
        raise ValueError("give at most one of around, before and after")
    if not conversation_keys:
        return _empty_page()

    def side(key: Key | None, count: int, *, older: bool) -> list[Row]:
        return _side(conn, conversation_keys, key, older=older, limit=count, include_system=include_system)

    anchor_visible = False
    anchor_key: Key | None = None
    if not anchors:
        found = list(reversed(side(None, limit, older=True)))
    else:
        anchor = _anchor(conn, conversation_keys, anchors[0])
        if anchor is None:
            return _empty_page()
        anchor_key, anchor_visible = rows.sort_key(anchor), _is_visible(anchor, include_system)
        if before is not None:
            found = list(reversed(side(anchor_key, limit, older=True)))
        elif after is not None:
            found = side(anchor_key, limit, older=False)
        else:
            middle = [anchor] if anchor_visible else []
            found = _around(side(anchor_key, limit, older=True), middle, side(anchor_key, limit, older=False), limit)

    if found:
        has_older = bool(side(rows.sort_key(found[0]), 1, older=True))
        has_newer = bool(side(rows.sort_key(found[-1]), 1, older=False))
    else:
        # nothing on the asked side: the anchor itself, or what lies beyond it, is on the other side
        has_older = after is not None and (anchor_visible or bool(side(anchor_key, 1, older=True)))
        has_newer = before is not None and (anchor_visible or bool(side(anchor_key, 1, older=False)))
    return {"messages": [rows.message(row) for row in found], "has_older": has_older, "has_newer": has_newer}


def context(conn: sqlite3.Connection, guid: str, *, radius: int, include_system: bool) -> list[rows.Message]:
    """The message with up to `radius` neighbours on each side, in its own conversation."""
    check_int("radius", radius, 0)
    home = conn.execute("SELECT conversation_id FROM messages WHERE guid = ?", (guid,)).fetchone()
    anchor = None if home is None else _anchor(conn, (home[0],), guid)
    if home is None or anchor is None:
        return []
    conversation = int(home[0])
    key = rows.sort_key(anchor)
    older = _side(conn, (conversation,), key, older=True, limit=radius, include_system=include_system)
    newer = _side(conn, (conversation,), key, older=False, limit=radius, include_system=include_system)
    middle = [anchor] if _is_visible(anchor, include_system) else []
    return [rows.message(row) for row in (*reversed(older), *middle, *newer)]


def _stream_one(
    connect: Callable[[], sqlite3.Connection],
    conversation: int,
    lower: int | None,
    upper: int | None,
    include_system: bool,
) -> Iterator[tuple[Key, rows.Message]]:
    """Every message of one conversation, oldest first, read in chunks.

    No cursor stays open between chunks, so a slow consumer does not hold a
    read transaction, and the stream can move to another thread.
    """
    time_sql, time_params = time_clauses(lower, upper)
    fixed = ["m.conversation_id = ?", *time_sql, *([] if include_system else [rows.VISIBLE])]
    key: Key | None = None
    while True:
        clauses = fixed if key is None else [*fixed, "(m.sent_ns, m.id) > (?, ?)"]
        sql = f"{rows.PAGED_MESSAGE_SELECT} WHERE {' AND '.join(clauses)} ORDER BY m.sent_ns, m.id LIMIT ?"
        chunk = connect().execute(sql, (conversation, *time_params, *(key or ()), STREAM_CHUNK)).fetchall()
        for row in chunk:
            yield rows.sort_key(row), rows.message(row)
        if len(chunk) < STREAM_CHUNK:
            return
        key = rows.sort_key(chunk[-1])


def stream(
    connect: Callable[[], sqlite3.Connection],
    conversation_keys: Sequence[int],
    lower: int | None,
    upper: int | None,
    include_system: bool,
) -> Iterator[rows.Message]:
    """Every message of the conversations, oldest first. Memory use does not grow with the count."""
    parts = [_stream_one(connect, conversation, lower, upper, include_system) for conversation in conversation_keys]
    merged = parts[0] if len(parts) == 1 else heapq.merge(*parts, key=itemgetter(0))
    for _, found in merged:
        yield found
