"""MessageSource over the query API of the package `archive`.

    with Archive.open(path) as archive:
        source = ArchiveSource(archive)

It reads the archive with two calls: `conversations()` and
`iter_messages(conversation_id)`. It leaves out group events (kind `system`)
and every message whose text is empty or only white space: an attachment with
no caption, a message that was unsent. A message in no conversation is in no
conversation of the archive, so it is never read.

A row that breaks the contract of the archive is a SourceError. The error
holds no text and no id.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from meaning_index.errors import SourceError
from meaning_index.ports import SourceMessage

SYSTEM_KIND = "system"


def _flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise SourceError("an archive row has an is_from_me value that is not true or false")


def _message(row: Mapping[str, Any]) -> SourceMessage:
    try:
        row_id, stamp, own = row["id"], row["timestamp"], row["is_from_me"]
    except KeyError as exc:
        raise SourceError(f"an archive row has no {exc.args[0]!r} field") from None
    if row_id is None:
        raise SourceError("an archive row has no id")
    return {"id": str(row_id), "timestamp": stamp, "text": str(row.get("text") or ""), "is_from_me": _flag(own)}


class ArchiveSource:
    """Adapts an open `archive.Archive`. The caller opens and closes the archive."""

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    def conversation_ids(self) -> Sequence[str]:
        """The conversations that hold a message, in the order of the archive: most messages first."""
        try:
            return [str(conversation["id"]) for conversation in self._archive.conversations()]
        except (KeyError, TypeError):
            raise SourceError("the archive gave a conversation with no id") from None

    def messages(self, conversation_id: str) -> Iterator[SourceMessage]:
        """The messages of one conversation that have text, oldest first."""
        for row in self._archive.iter_messages(conversation_id):
            if not isinstance(row, Mapping):
                raise SourceError("the archive gave a row that is not a message")
            if row.get("kind") == SYSTEM_KIND or not str(row.get("text") or "").strip():
                continue
            yield _message(row)
