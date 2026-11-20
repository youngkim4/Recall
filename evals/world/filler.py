"""Filler days: everyday exchanges that make the history look lived in.

Each window gives a chat a chance of one exchange on each day between two
days. The chance draws the shape that the pattern questions ask about:
Wren texts less after the move, Juniper fades in 2024, Theo texts less after
Denver, and the family chat is busiest in 2024.

The draw uses `random.Random` with a fixed text seed, so every run gives the
same days, times and guids. A day that holds a scene of the same chat, or a
quiet day, gets no filler.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta

from evals.world import filler_lines as pools
from evals.world.people import CHAT_BY_KEY
from evals.world.script import Item, Line, guid_for, local

THEM = "them"
FIRST_HOUR, LAST_HOUR = 8, 21
LONGEST_GAP_MINUTES = 6


@dataclass(frozen=True)
class Window:
    """A chance of one exchange a day in `chat`, from `first` to `last` (days, both included)."""

    chat: str
    first: str
    last: str
    rate: float
    pool: tuple[pools.Exchange, ...]
    weekdays_only: bool = False


WINDOWS: tuple[Window, ...] = (
    Window("dash", "2023-01-15", "2023-01-31", 0.55, pools.TALKING),
    Window("dash", "2023-02-01", "2023-03-05", 0.12, pools.TALKING),
    Window("wren", "2023-02-27", "2025-02-27", 0.32, pools.PARTNER),
    Window("wren", "2025-03-03", "2025-12-18", 0.07, pools.PARTNER),
    Window("juni", "2023-01-02", "2023-12-31", 0.28, pools.CLOSE_FRIEND),
    Window("juni", "2024-01-01", "2024-10-31", 0.03, pools.CLOSE_FRIEND),
    Window("theo", "2023-01-02", "2023-08-18", 0.30, pools.FRIEND),
    Window("theo", "2023-08-23", "2025-12-15", 0.07, pools.FRIEND),
    Window("mari", "2023-01-10", "2024-09-26", 0.12, pools.WORK, weekdays_only=True),
    Window("family", "2023-01-03", "2023-12-31", 0.05, pools.FAMILY),
    Window("family", "2024-01-01", "2024-12-31", 0.10, pools.FAMILY),
    Window("family", "2025-01-01", "2025-12-15", 0.05, pools.FAMILY),
)


def _days(first: str, last: str) -> Iterator[date]:
    day, end = date.fromisoformat(first), date.fromisoformat(last)
    while day <= end:
        yield day
        day += timedelta(days=1)


def _speaker(chat: str, speaker: str) -> str:
    return CHAT_BY_KEY[chat].members[0] if speaker == THEM else speaker


def _exchange(window: Window, day: date, draw: random.Random) -> tuple[Line, ...]:
    exchange = draw.choice(window.pool)
    clock = local(f"{day.isoformat()} {draw.randint(FIRST_HOUR, LAST_HOUR):02d}:{draw.randint(0, 59):02d}")
    lines = []
    for place, (speaker, text) in enumerate(exchange):
        if place:
            clock = clock + timedelta(minutes=draw.randint(1, LONGEST_GAP_MINUTES))
        guid = guid_for(window.chat, "filler", day.isoformat(), place)
        lines.append(Line(window.chat, clock, _speaker(window.chat, speaker), text, guid))
    return tuple(lines)


def _window_lines(window: Window, busy: frozenset[str]) -> Iterator[Line]:
    draw = random.Random(f"filler|{window.chat}|{window.first}")
    for day in _days(window.first, window.last):
        # draw for every day, so a new scene does not move the filler of later days
        hit = draw.random() < window.rate
        exchange = _exchange(window, day, draw)
        if not hit or day.isoformat() in busy or (window.weekdays_only and day.weekday() >= 5):
            continue
        yield from exchange


def filler(scenes: Iterable[Item], quiet: dict[str, frozenset[str]]) -> tuple[Line, ...]:
    """The filler lines of every window. `quiet` gives more days with no filler for a chat."""
    scene_days: dict[str, set[str]] = {}
    for item in scenes:
        scene_days.setdefault(item.conversation, set()).add(item.when.strftime("%Y-%m-%d"))
    made: list[Line] = []
    for window in WINDOWS:
        busy = frozenset(scene_days.get(window.chat, set()) | set(quiet.get(window.chat, frozenset())))
        made.extend(_window_lines(window, busy))
    return tuple(made)
