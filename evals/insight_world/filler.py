"""Plain talk that fills each thread to its rhythm. It holds no plan, promise, news, question or wish.

Each exchange is three lines: the starter, the other side, the starter
again. No line has a question mark, so filler never leaves a question open.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from evals.insight_world.cast import RHYTHMS, Rhythm
from evals.world.script import Line, guid_for, local

OPENERS = (
    "hey how's your week going",
    "lol that video you sent earlier",
    "this weather is wild today",
    "work is so slow today",
    "just got home finally",
    "ok this song is stuck in my head",
    "made pasta again, very proud of myself",
    "my train was late again",
)
REPLIES = (
    "pretty good, just busy",
    "haha it's so dumb",
    "same honestly",
    "it's been a long one",
    "not bad, tired though",
    "lmao",
    "yeah for real",
)
CLOSERS = ("ok talk later", "haha fair", "love that", "ok ok", "nice")
HOURS = (9, 12, 18, 20, 21)


def starter(pattern: str, place: int) -> str:
    """Who writes first in exchange `place`: "me", "them", "even", or a pattern like "me*5,them"."""
    if pattern in ("me", "them"):
        return pattern
    if pattern == "even":
        return "me" if place % 2 == 0 else "them"
    cycle: list[str] = []
    for part in pattern.split(","):
        who, _, count = part.partition("*")
        cycle.extend([who] * int(count or 1))
    return cycle[place % len(cycle)]


def _exchange(rhythm: Rhythm, place: int, when: datetime) -> tuple[Line, ...]:
    first = starter(rhythm.starts, place)
    second = "me" if first == "them" else "them"
    texts = (
        (first, OPENERS[place % len(OPENERS)]),
        (second, REPLIES[place % len(REPLIES)]),
        (first, CLOSERS[place % len(CLOSERS)]),
    )
    return tuple(
        Line(
            rhythm.key,
            when + timedelta(minutes=4 * step),
            rhythm.key if who == "them" else "me",
            text,
            guid_for(rhythm.key, "filler", place, step),
        )
        for step, (who, text) in enumerate(texts)
    )


def thread_filler(rhythm: Rhythm) -> tuple[Line, ...]:
    lines: list[Line] = []
    day = local(f"{rhythm.first} 00:00")
    end = local(f"{rhythm.last} 23:59")
    place = 0
    while day <= end:
        when = day.replace(hour=HOURS[place % len(HOURS)], minute=(place * 7) % 50)
        lines.extend(_exchange(rhythm, place, when))
        day += timedelta(days=rhythm.every)
        place += 1
    return tuple(lines)


def filler() -> tuple[Line, ...]:
    return tuple(line for rhythm in RHYTHMS for line in thread_filler(rhythm))
