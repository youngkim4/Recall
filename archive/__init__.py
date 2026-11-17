"""Recall's own archive of the user's iMessage history.

One SQLite file mirrors the Messages database. A full pull makes the
archive hold exactly the messages of the source: new rows are added, changed
rows are updated, and rows that are gone from the source are removed.

See archive/README.md for the query contract.
"""

from __future__ import annotations

from archive.errors import (
    ArchiveError,
    ArchiveNotEmptyError,
    ArchiveNotFoundError,
    ArchiveVersionError,
    ImportBusyError,
    ImportFailed,
    RemovalLimitError,
    SourceError,
    SourceFormatError,
    SqliteFeatureError,
    TypedStreamError,
)
from archive.importer import ImportResult, import_chat_db
from archive.legacy_csv import import_legacy_csv
from archive.paths import default_archive_path
from archive.queries import Archive
from archive.schema import INDEXED_TEXT_CHARS

__all__ = [
    "INDEXED_TEXT_CHARS",
    "Archive",
    "ArchiveError",
    "ArchiveNotEmptyError",
    "ArchiveNotFoundError",
    "ArchiveVersionError",
    "ImportBusyError",
    "ImportFailed",
    "ImportResult",
    "RemovalLimitError",
    "SourceError",
    "SourceFormatError",
    "SqliteFeatureError",
    "TypedStreamError",
    "default_archive_path",
    "import_chat_db",
    "import_legacy_csv",
]
