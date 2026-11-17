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
from archive.paths import default_archive_path
from archive.queries import Archive

__all__ = [
    "Archive",
    "ArchiveError",
    "ArchiveNotFoundError",
    "ArchiveVersionError",
    "ImportBusyError",
    "ImportFailed",
    "ImportResult",
    "SourceError",
    "SourceFormatError",
    "SqliteFeatureError",
    "TypedStreamError",
    "default_archive_path",
    "import_chat_db",
]
