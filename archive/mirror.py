"""Keep the archive equal to the source database after a full pull.

The source is the truth. A message or tapback whose guid is gone from the
source is removed, with its attachments and the tapbacks on it. One safety
valve: when a pull would remove more than a set share of the stored
messages, it stops before it changes anything. That protects the archive
from a database that is only half synced.

Rows that the legacy export bridge loaded (guid `legacy:<ROWID>`) are not
part of that count: the first pull from the database replaces all of them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from archive.errors import RemovalLimitError

DEFAULT_MAX_REMOVED_SHARE = 0.2
LEGACY_PREFIX = "legacy:"
# every guid that starts with the prefix sorts before this text
_IS_LEGACY = f"guid >= '{LEGACY_PREFIX}' AND guid < '{LEGACY_PREFIX[:-1]};'"
_GONE = "guid NOT IN (SELECT guid FROM temp.source_guids)"


class Removed(NamedTuple):
    messages: int
    system: int
    reactions: int
    attachments: int


def legacy_guid(source_rowid: int) -> str:
    """The id of a row that the legacy export bridge loaded."""
    return f"{LEGACY_PREFIX}{source_rowid}"


def check_share(max_removed_share: object) -> float:
    """A share from 0 to 1. Anything else is a caller error."""
    if isinstance(max_removed_share, bool) or not isinstance(max_removed_share, (int, float)):
        raise TypeError("max_removed_share must be a number from 0 to 1")
    if not 0 <= max_removed_share <= 1:
        raise ValueError("max_removed_share must be a number from 0 to 1")
    return float(max_removed_share)


def has_bridge_rows(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(f"SELECT EXISTS (SELECT 1 FROM messages WHERE {_IS_LEGACY})").fetchone()[0])


def note_source_guids(conn: sqlite3.Connection, chunks: Iterable[Sequence[str]]) -> None:
    """Write every guid of the source into a temporary table of this connection."""
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS source_guids (guid TEXT PRIMARY KEY) WITHOUT ROWID")
    conn.execute("DELETE FROM temp.source_guids")
    for chunk in chunks:
        conn.executemany("INSERT OR IGNORE INTO temp.source_guids(guid) VALUES (?)", ((guid,) for guid in chunk))


def check_removal_limit(conn: sqlite3.Connection, max_removed_share: float) -> None:
    """Raise RemovalLimitError when the source lacks more than the given share of the stored messages.

    Call it after note_source_guids and before the pull changes anything.
    An archive with no messages never trips it.
    """
    stored = int(conn.execute(f"SELECT COUNT(*) FROM messages WHERE NOT ({_IS_LEGACY})").fetchone()[0])
    if stored == 0:
        return
    gone = int(conn.execute(f"SELECT COUNT(*) FROM messages WHERE {_GONE} AND NOT ({_IS_LEGACY})").fetchone()[0])
    if gone > max_removed_share * stored:
        raise RemovalLimitError(would_remove=gone, stored=stored, max_share=max_removed_share)


def clear_bridge_rows(conn: sqlite3.Connection) -> int:
    """Remove the messages that the legacy export bridge loaded. Return how many."""
    conn.execute(f"DELETE FROM attachments WHERE message_id IN (SELECT id FROM messages WHERE {_IS_LEGACY})")
    conn.execute(f"DELETE FROM reactions WHERE {_IS_LEGACY}")
    return int(conn.execute(f"DELETE FROM messages WHERE {_IS_LEGACY}").rowcount)


def drop_orphan_tapbacks(conn: sqlite3.Connection) -> int:
    """Remove the tapbacks whose target message the archive does not hold. Return how many."""
    return int(conn.execute("DELETE FROM reactions WHERE target_guid NOT IN (SELECT guid FROM messages)").rowcount)


def remove_gone(conn: sqlite3.Connection) -> Removed:
    """Remove every stored row whose guid the source no longer has.

    A tapback goes with its target message, also when the source still
    holds the tapback row. Call it before the rows of the source are read:
    a tapback on a message that is gone then finds no target and is not stored.
    """
    by_kind = dict(conn.execute(f"SELECT kind = 'system', COUNT(*) FROM messages WHERE {_GONE} GROUP BY 1").fetchall())
    attachments = conn.execute(
        f"DELETE FROM attachments WHERE message_id IN (SELECT id FROM messages WHERE {_GONE})"
    ).rowcount
    conn.execute(f"DELETE FROM messages WHERE {_GONE}")
    reactions = conn.execute(f"DELETE FROM reactions WHERE {_GONE}").rowcount
    reactions += drop_orphan_tapbacks(conn)
    return Removed(int(by_kind.get(0, 0)), int(by_kind.get(1, 0)), int(reactions), int(attachments))
