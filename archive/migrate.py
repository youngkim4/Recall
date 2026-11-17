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


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


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
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return known
