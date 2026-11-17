"""ReportSource over the archive package's query API.

The archive package is not part of this branch. This adapter is written
against its published contract and is checked only with a stand-in:

    Archive.iter_messages(conversation_id, *, since=None, until=None)

yields dicts with `id`, `timestamp`, `is_from_me`, `sender`, `text`, `kind`.
Rows whose `kind` is `system` are skipped.

The archive takes `since` and `until` as days. Whether its `until` day is
included is not relied on: the adapter asks for one day more and applies the
exact bounds itself.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from typing import Any

from report_engine.dates import DAY_LENGTH, shift_day
from report_engine.errors import SourceError
from report_engine.ports import SourceMessage
from report_engine.scope import in_bounds, lower_bound, upper_bound

SYSTEM_KIND = "system"
ARCHIVE_MODULES = ("archive.queries", "archive")


def _archive_class() -> Any:
    """Import the archive package on first use, not when this module loads."""
    failure: ImportError | None = None
    for name in ARCHIVE_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            failure = failure or exc
            continue
        found = getattr(module, "Archive", None)
        if found is not None:
            return found
    raise SourceError(
        f"the archive package can not be imported: {failure or 'it has no Archive class'}"
    )


def _message(row: Mapping[str, Any]) -> SourceMessage:
    try:
        return {
            "id": str(row["id"]),
            "timestamp": str(row["timestamp"]),
            "is_from_me": bool(row["is_from_me"]),
            "sender": str(row.get("sender") or ""),
            "text": str(row.get("text") or ""),
        }
    except KeyError as exc:
        raise SourceError(f"an archive row has no {exc.args[0]!r} field") from exc


class ArchiveSource:
    def __init__(self, archive: Any) -> None:
        self._archive = archive

    @classmethod
    def open(cls, path: str | os.PathLike[str] | None = None) -> ArchiveSource:
        """Open the archive file read-only. `None` means the archive's default path."""
        return cls(_archive_class().open(path, readonly=True))

    @property
    def archive(self) -> Any:
        return self._archive

    def close(self) -> None:
        close = getattr(self._archive, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> ArchiveSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        low, high = lower_bound(since), upper_bound(until)
        rows = self._archive.iter_messages(
            conversation,
            since=since[:DAY_LENGTH] if since else None,
            until=shift_day(until[:DAY_LENGTH], 1) if until else None,
        )
        for row in rows:
            if row.get("kind") == SYSTEM_KIND:
                continue
            message = _message(row)
            if in_bounds(message["timestamp"], low, high):
                yield message
