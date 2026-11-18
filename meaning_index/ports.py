"""What meaning_index needs from the outside: messages, an embedder, a token counter."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, TypedDict


class SourceMessage(TypedDict):
    """One message of the archive, as this package needs it."""

    id: str
    timestamp: str  # "YYYY-MM-DD HH:MM:SS", local time
    text: str
    is_from_me: bool


class MessageSource(Protocol):
    def conversation_ids(self) -> Sequence[str]: ...

    def messages(self, conversation_id: str) -> Iterator[SourceMessage]:
        """The messages of one conversation, oldest first."""
        ...


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """One vector for each text, in the same order."""
        ...


CountTokens = Callable[[str], int]
Sleep = Callable[[float], None]
