"""The real archive behind the `ChatArchive` port.

The tests of this package use a stand-in for the archive.
`tests/integration/test_chat_on_archive.py` runs whole questions through this
adapter on a real archive.

The `archive` package is imported only inside `open_archive`, so importing
`chat_engine` never needs it. Every call passes its keywords by name: a
default that changes in the archive cannot silently change what chat asks.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Sequence, Union

from .errors import ArchiveError
from .ports import ArchiveSummary, Conversation, CountRow, Message, MessagePage, SearchResult


class ArchiveAdapter:
    def __init__(self, archive: Any) -> None:
        self._archive = archive

    def __enter__(self) -> ArchiveAdapter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self._archive, "close", None)
        if callable(close):
            close()

    def summary(self) -> ArchiveSummary:
        return self._archive.summary()

    def conversations(
        self, *, limit: Optional[int] = None, offset: int = 0, order: str = "count"
    ) -> list[Conversation]:
        return self._archive.conversations(limit=limit, offset=offset, order=order)

    def conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._archive.conversation(conversation_id)

    def messages(
        self,
        conversation_id: str,
        *,
        around: Optional[str] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 50,
    ) -> MessagePage:
        cursors = {"around": around, "before": before, "after": after}
        given = {name: value for name, value in cursors.items() if value is not None}
        return self._archive.messages(conversation_id, limit=limit, include_system=False, **given)

    @property
    def iter_messages(self) -> Optional[Callable[..., Iterator[Message]]]:
        """The reader of the archive: the messages of one conversation, oldest first.

        Not in the port. It is None when the archive has no reader: the
        engine then walks pages.
        """
        inner = getattr(self._archive, "iter_messages", None)
        if not callable(inner):
            return None

        def read(
            conversation_id: str, *, since: Optional[str] = None, until: Optional[str] = None
        ) -> Iterator[Message]:
            return iter(inner(conversation_id, since=since, until=until, include_system=False))

        return read

    def context(self, message_id: str, *, radius: int = 6) -> list[Message]:
        return self._archive.context(message_id, radius=radius, include_system=False)

    def message(self, message_id: str) -> Optional[Message]:
        """One message by its id. A system row is no message here, as in every other call."""
        found = self._archive.message(message_id)
        return None if found is None or found.get("kind") == "system" else found

    def search(
        self,
        query: str,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "newest",
    ) -> SearchResult:
        return self._archive.search(
            query,
            conversations=None if conversations is None else list(conversations),
            since=since,
            until=until,
            from_me=from_me,
            limit=limit,
            offset=offset,
            order=order,
            include_system=False,
        )

    def count(
        self,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        text: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> Union[int, list[CountRow]]:
        return self._archive.count(
            conversations=None if conversations is None else list(conversations),
            since=since,
            until=until,
            from_me=from_me,
            text=text,
            group_by=group_by,
            include_system=False,
        )

    def stats(
        self, conversation_id: str, *, since: Optional[str] = None, until: Optional[str] = None
    ) -> dict[str, Any]:
        return self._archive.stats(conversation_id, since=since, until=until, include_system=False)


def _archive_class() -> Any:
    try:
        from archive import Archive  # noqa: PLC0415 - the package is built in another branch

        return Archive
    except ImportError:
        pass
    try:
        from archive.queries import Archive  # noqa: PLC0415

        return Archive
    except ImportError as error:
        raise ArchiveError(detail=f"the archive package cannot be imported: {error}") from error


def open_archive(path: Any = None) -> ArchiveAdapter:
    """Open the archive read-only. `path=None` opens the default archive."""
    archive_class = _archive_class()
    try:
        return ArchiveAdapter(archive_class.open(path, readonly=True))
    except Exception as error:
        raise ArchiveError(detail=f"{type(error).__name__}: {error}") from error
