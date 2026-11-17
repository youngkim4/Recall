"""Schema versions of the archive file.

`PRAGMA user_version` holds the version. Each migration takes the file from
version N to N + 1. Opening an archive applies every missing migration in one
transaction: either all of them are applied or none.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from archive.errors import ArchiveVersionError
from archive.schema import SCHEMA_V1

Migration = Callable[[sqlite3.Connection], None]


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    for statement in SCHEMA_V1:
        conn.execute(statement)


# Append only. Never edit or reorder a migration that has shipped.
MIGRATIONS: tuple[Migration, ...] = (_create_schema_v1,)


# What only the final layout of version 1 has. Version 1 changed while this package was built,
# and some archives of those builds can exist. They hold nothing that a pull cannot make again.
_LAYOUT_PROBES = (
    "SELECT search_text, source_rowid FROM messages LIMIT 0",
    "SELECT message_count FROM conversation_members LIMIT 0",
    "SELECT 1 FROM pragma_index_info('messages_by_conversation_time') WHERE name = 'sender_handle_id'",
)


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def check_layout(conn: sqlite3.Connection) -> None:
    """Raise ArchiveVersionError for an archive of version 1 that an earlier build of this code made."""
    if schema_version(conn) != 1:
        return
    try:
        complete = conn.execute(_LAYOUT_PROBES[0]).fetchall() == [] and conn.execute(_LAYOUT_PROBES[1]).fetchall() == []
        complete = complete and conn.execute(_LAYOUT_PROBES[2]).fetchone() is not None
    except sqlite3.OperationalError:
        complete = False
    if not complete:
        raise ArchiveVersionError(
            "this archive was made by an earlier build of Recall. Delete the file and import again."
        )


def _refuse_newer(found: int, known: int) -> None:
    if found > known:
        raise ArchiveVersionError(
            f"this archive has schema version {found}, which is newer than version {known} that this code knows. "
            "Update Recall to open it."
        )


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> int:
    """Apply the missing migrations. Return the version after the call.

    `conn` must be in autocommit mode (isolation_level=None).
    """
    known = len(migrations)
    found = schema_version(conn)
    _refuse_newer(found, known)
    if found == known:
        return found
    conn.execute("BEGIN IMMEDIATE")
    try:
        found = schema_version(conn)  # another process can have migrated while we waited
        _refuse_newer(found, known)
        for step in migrations[found:]:
            step(conn)
        conn.execute(f"PRAGMA user_version = {int(known)}")
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return known
