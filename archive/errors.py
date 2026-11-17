"""Errors raised by the archive package."""

from __future__ import annotations


class ArchiveError(Exception):
    """Base class for every error this package raises on purpose."""


class SqliteFeatureError(ArchiveError):
    """The SQLite library of this interpreter is too old or has no FTS5."""


class ArchiveNotFoundError(ArchiveError, FileNotFoundError):
    """The archive file does not exist and the caller asked for read-only access."""


class ArchiveVersionError(ArchiveError):
    """The archive was written by newer code than this."""


class SourceError(ArchiveError):
    """The source database cannot be opened or read."""


class SourceFormatError(SourceError):
    """The source file is not a Messages database this importer understands."""


class ImportBusyError(ArchiveError):
    """Another import is writing to the same archive."""


class ImportFailed(ArchiveError):
    """An import run failed. The archive data is as it was before the run."""

    def __init__(self, message: str, *, run_id: int | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id


class TypedStreamError(ArchiveError, ValueError):
    """A blob is not a typedstream archive of an attributed string."""
