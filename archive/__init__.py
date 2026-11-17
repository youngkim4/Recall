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

__all__ = [
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
