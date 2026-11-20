"""What the insight engine reads and what it gives back.

A `Thread` is everything one person and the owner said to each other, oldest
first. One person can have more than one conversation (a phone and an
email); the caller joins them into one thread. Groups are not threads.

The engine opens no file and reads no clock. The caller gives `now`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Kind = Literal["drift", "balance", "unanswered", "birthday", "plan", "promise", "news"]
KINDS: tuple[Kind, ...] = ("drift", "balance", "unanswered", "birthday", "plan", "promise", "news")


@dataclass(frozen=True)
class Note:
    """One message. `when` is local time with no zone."""

    id: str
    when: datetime
    from_me: bool
    text: str


@dataclass(frozen=True)
class Thread:
    """One person and every message with them, oldest first. `key` is stable; `name` is what the owner reads."""

    key: str
    name: str
    notes: tuple[Note, ...]

    def __post_init__(self) -> None:
        if any(later.when < earlier.when for earlier, later in zip(self.notes, self.notes[1:])):
            raise ValueError("the notes of a thread are not oldest first")

    def between(self, start: datetime, end: datetime) -> tuple[Note, ...]:
        """The notes from `start` up to, but not including, `end`."""
        return tuple(note for note in self.notes if start <= note.when < end)


@dataclass(frozen=True)
class Insight:
    """One thing worth knowing about one person.

    `summary` is one sentence to the owner. `message_ids` are the messages
    that show it; a rhythm insight cites the last message. `figures` holds the
    numbers behind a rhythm insight, for the screen to show.
    """

    kind: Kind
    thread: str
    name: str
    summary: str
    when: datetime
    message_ids: tuple[str, ...] = ()
    figures: tuple[tuple[str, float], ...] = field(default=())

    def figure(self, key: str) -> float | None:
        return dict(self.figures).get(key)
