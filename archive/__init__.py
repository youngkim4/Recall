"""Recall's own archive of the user's iMessage history.

One SQLite file holds every message that an import has seen. The archive
only grows: a message stays when the source database no longer has it.

See archive/README.md for the query contract.
"""

from __future__ import annotations

from archive.errors import (
    ArchiveError,
    ArchiveNotFoundError,
    ArchiveVersionError,
    ImportBusyError,
    ImportFailed,
    SourceError,
    SourceFormatError,
    SqliteFeatureError,
    TypedStreamError,
)
from archive.importer import ImportResult, import_chat_db

__all__ = [
    "ImportResult",
    "import_chat_db",
    "ArchiveError",
    "ArchiveNotFoundError",
    "ArchiveVersionError",
    "ImportBusyError",
    "ImportFailed",
    "SourceError",
    "SourceFormatError",
    "SqliteFeatureError",
    "TypedStreamError",
]
