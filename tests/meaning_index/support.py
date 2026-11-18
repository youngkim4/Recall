"""Invented data and stand-ins for the tests of meaning_index."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta

import numpy as np

from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.ports import SourceMessage
from meaning_index.segments import cut_segments
from meaning_index.shapes import Segment

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
START = datetime(2024, 6, 3, 9, 0, 0)


def stamp(minutes: float = 0.0, start: datetime = START) -> str:
    """The local time `minutes` after the start, as the archive writes it."""
    return (start + timedelta(minutes=minutes)).strftime(TIME_FORMAT)


def msg(message_id: str, timestamp: str, text: str, is_from_me: bool = False) -> SourceMessage:
    return {"id": message_id, "timestamp": timestamp, "text": text, "is_from_me": is_from_me}


def run(
    count: int, *, prefix: str = "m", text: str = "a line", step: float = 1.0, start: float = 0.0
) -> list[SourceMessage]:
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


class ScriptedEmbedder:
    """A fixed embedder that fails as a script says.

    `script` maps the number of a call (from 1) to an error to raise, or it is
    a function of (call number, texts) that returns an error or None.
    `sizes` holds the number of texts of each call, failed ones too.
    """

    def __init__(self, script: object = None, *, dimensions: int = 16, model: str = "scripted-1") -> None:
        self._inner = FakeEmbedder(dimensions=dimensions, model=model)
        self._script = script or {}
        self.model = model
        self.dimensions = dimensions
        self.sizes: list[int] = []
        self.answered: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.sizes.append(len(texts))
        number = len(self.sizes)
        script = self._script
        error = script(number, texts) if callable(script) else script.get(number)  # type: ignore[attr-defined]
        if error is not None:
            raise error
        self.answered.append(tuple(texts))
        return self._inner.embed(texts)

    @property
    def texts(self) -> list[str]:
        """Every text that got a vector, in order."""
        return [text for call in self.answered for text in call]


def kept(
    number: int, *, conversation: str = "chat-a", day: str = "2024-06-03", content: str | None = None
) -> Segment:
    """An invented stored segment. Its ids follow `number`."""
    return Segment(
        segment_id=f"{number:064x}",
        conversation_id=conversation,
        first_message_id=f"first-{number}",
        last_message_id=f"last-{number}",
        first_timestamp=f"{day} 09:00:00",
        last_timestamp=f"{day} 09:30:00",
        message_count=3,
        content_hash=content if content is not None else f"{number + 1000:064x}",
    )


def unit(count: int, dimensions: int, seed: int = 7) -> np.ndarray:
    """`count` invented rows of length 1, as the store keeps them."""
    rows = np.random.default_rng(seed).standard_normal((count, dimensions))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype("<f2")


def notes(count: int, *, prefix: str = "n", word: str = "note") -> list[SourceMessage]:
    """`count` messages three hours apart: each one is a segment of its own."""
    return [msg(f"{prefix}{n}", stamp(n * 180), f"{word} {n}", n % 2 == 1) for n in range(count)]


def cut_texts(source: ListSource) -> list[str]:
    """The text of every segment of a source, in the order of a build."""
    return [
        segment.text
        for conversation in source.conversation_ids()
        for segment in cut_segments(conversation, source.messages(conversation))
    ]
