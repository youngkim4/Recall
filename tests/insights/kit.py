"""Small threads for the insight tests. Every name and line is invented."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from insights.thread import Note, Thread

NOW = datetime(2025, 12, 20, 12, 0)


def at(stamp: str) -> datetime:
    """A "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" local time."""
    return datetime.fromisoformat(stamp if " " in stamp else f"{stamp} 12:00")


def note(stamp: str, who: str, text: str, ref: str = "") -> Note:
    when = at(stamp)
    return Note(ref or f"{who}-{when:%Y%m%d%H%M}-{abs(hash(text)) % 997}", when, who == "me", text)


def thread(key: str, notes: Iterable[Note], name: str = "") -> Thread:
    return Thread(key, name or key.title(), tuple(sorted(notes, key=lambda item: item.when)))


def every(start: str, end: str, days: int, *, starter: str = "me", reply: bool = True) -> list[Note]:
    """A short exchange each `days` days from `start` through `end`: `starter` writes, the other answers."""
    other = "them" if starter == "me" else "me"
    notes: list[Note] = []
    when = at(start)
    while when <= at(end):
        notes.append(Note(f"{starter}-{when:%Y%m%d}", when, starter == "me", "hey how are you"))
        if reply:
            later = when + timedelta(minutes=20)
            notes.append(Note(f"{other}-{when:%Y%m%d}", later, other == "me", "good, you?"))
        when += timedelta(days=days)
    return notes
