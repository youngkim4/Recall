"""Invented data and stand-ins for the tests of meaning_index."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta

from meaning_index.ports import SourceMessage

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
START = datetime(2024, 6, 3, 9, 0, 0)


def stamp(minutes: float = 0.0, start: datetime = START) -> str:
    """The local time `minutes` after the start, as the archive writes it."""
    return (start + timedelta(minutes=minutes)).strftime(TIME_FORMAT)


def msg(message_id: str, timestamp: str, text: str, is_from_me: bool = False) -> SourceMessage:
    return {"id": message_id, "timestamp": timestamp, "text": text, "is_from_me": is_from_me}


def run(count: int, *, prefix: str = "m", text: str = "a line", step: float = 1.0, start: float = 0.0) -> list[SourceMessage]:
    """`count` messages, `step` minutes apart. Each second one is an own message."""
    return [msg(f"{prefix}{n}", stamp(start + n * step), f"{text} {n}", n % 2 == 1) for n in range(count)]


def words(text: str) -> int:
    """The fixed token counter of the tests: one token for each word."""
    return len(text.split())


class ListSource:
    """A MessageSource over lists. `reads` counts the conversations that were read."""

    def __init__(self, conversations: Mapping[str, Sequence[SourceMessage]]) -> None:
        self._conversations = {key: list(value) for key, value in conversations.items()}
        self.reads = 0

    def conversation_ids(self) -> Sequence[str]:
        return list(self._conversations)

    def messages(self, conversation_id: str) -> Iterator[SourceMessage]:
        self.reads += 1
        yield from (dict(message) for message in self._conversations[conversation_id])  # type: ignore[misc]
