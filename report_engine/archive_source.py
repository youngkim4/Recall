"""ReportSource over the archive package's query API.

The tests of this package check this adapter with a stand-in.
`tests/integration/test_report_on_archive.py` runs a whole report through it
on a real archive. It reads the archive with one call:

    Archive.iter_messages(conversation_id, *, since=None, until=None)

yields dicts with `id`, `timestamp`, `is_from_me`, `sender`, `text`, `kind`.
Rows whose `kind` is `system` are skipped.

The archive takes `since` and `until` as days. Whether its `until` day is
included is not relied on: the adapter asks for one day more and applies the
exact bounds itself. A row that breaks the contract is a SourceError, not a
guess.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from typing import Any

from report_engine.dates import DAY_LENGTH, has_stamp_shape, lower_bound, shift_day, upper_bound
from report_engine.errors import SourceError
from report_engine.ports import SourceMessage
from report_engine.scope import in_bounds

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


def _flag(value: object, row_id: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise SourceError(f"archive row {row_id!r} has an is_from_me value that is not true or false")


def _message(row: Mapping[str, Any]) -> SourceMessage:
    try:
        row_id, stamp, own = row["id"], row["timestamp"], row["is_from_me"]
    except KeyError as exc:
        raise SourceError(f"an archive row has no {exc.args[0]!r} field") from exc
    if row_id is None:
        raise SourceError("an archive row has no id")
    if not has_stamp_shape(stamp):
        raise SourceError(f"archive row {row_id!r} has a timestamp that is not YYYY-MM-DD HH:MM:SS")
    return {
        "id": str(row_id),
        "timestamp": stamp,
        "is_from_me": _flag(own, row_id),
        "sender": str(row.get("sender") or ""),
        "text": str(row.get("text") or ""),
    }


def _day_after(until: str | None) -> str | None:
    """The day after `until`, to ask the archive for. None at the end of the calendar."""
    if until is None:
        return None
    try:
        return shift_day(until[:DAY_LENGTH], 1)
    except ValueError:
        return None


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
            until=_day_after(until),
        )
        for row in rows:
            if row.get("kind") == SYSTEM_KIND:
                continue
            message = _message(row)
            if in_bounds(message["timestamp"], low, high):
                yield message
