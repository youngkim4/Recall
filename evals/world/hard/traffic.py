"""The everyday traffic of the hard world: bursts, gaps, one-word replies, links, tapbacks and codes.

A `Profile` gives a chat its pool of templates and its periods: from a day,
to a day, a chance that the chat is active on a day. An active day has a
burst of exchanges. A group exchange has more lines and more speakers. Days
outside every period are gaps. Tapbacks land on some group messages.

Each chat draws from `random.Random` with its own text seed, so a run always
gives the same lines, times and guids, and a change of one chat moves no
other chat.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from evals.world.hard import talk
from evals.world.hard.cast import ACQUAINTANCES, GROUPS, acquaintance_key
from evals.world.script import TAPBACKS, Item, Line, Tapback, guid_for, local

Period = tuple[str, str, float]
FIRST_HOUR, LAST_HOUR = 7, 23
TAPBACK_RATE = 0.04
ME_SHARE = 0.25  # the share of group lines that Sam writes


@dataclass(frozen=True)
class Profile:
    chat: str
    pool: str  # "group", "friend", "work", or a service
    periods: tuple[Period, ...]
    burst: float = 1.0  # the mean number of extra exchanges on an active day
    members: tuple[str, ...] = ()
    weekdays_only: bool = False


def _group(key: str) -> tuple[str, ...]:
    return next(chat.members for chat in GROUPS if chat.key == key)


ALL_YEARS = ("2020-01-01", "2025-12-19")

PROFILES: tuple[Profile, ...] = (
    Profile("crew", "group", (("2020-01-01", "2021-12-31", 0.4), ("2022-01-01", "2022-12-31", 0.25),
                              ("2023-01-01", "2024-12-31", 0.15), ("2025-01-01", "2025-12-19", 0.35)),
            2.0, _group("crew")),
    Profile("bwdesign", "work", (("2020-06-01", "2024-09-27", 0.4),), 1.5, _group("bwdesign"), True),
    Profile("kestrel", "work", (("2024-10-07", "2025-12-19", 0.75),), 2.0, _group("kestrel"), True),
    Profile("building", "group", (("2022-09-17", "2025-02-27", 0.12),), 0.5, _group("building")),
    Profile("climb", "group", (("2021-01-01", "2025-12-19", 0.22),), 1.0, _group("climb")),
    Profile("books", "group", (("2021-02-01", "2025-12-19", 0.1),), 1.0, _group("books")),
    Profile("cousins", "group", (ALL_YEARS + (0.12,),), 1.0, _group("cousins")),
    Profile("trivia", "group", (("2021-01-01", "2024-12-31", 0.15),), 1.0, _group("trivia")),
    Profile("hoops", "group", (ALL_YEARS + (0.14,),), 1.0, _group("hoops")),
    Profile("plants", "group", (("2022-03-01", "2025-12-19", 0.06),), 0.5, _group("plants")),
    Profile("raff", "friend", (("2020-01-01", "2022-11-11", 0.5), ("2022-11-13", "2025-04-04", 0.06),
                               ("2025-04-06", "2025-12-19", 0.45)), 1.0),
    Profile("petra", "friend", (("2020-01-01", "2022-09-16", 0.45), ("2022-09-18", "2025-12-19", 0.05)), 0.5),
    Profile("maya", "friend", (ALL_YEARS + (0.2,),), 0.5),
    Profile("mayao", "work", (("2024-10-07", "2025-12-19", 0.25),), 0.5, weekdays_only=True),
    Profile("odalys", "friend", (ALL_YEARS + (0.2,),), 0.5),
    Profile("kip", "friend", (("2020-01-01", "2022-12-31", 0.15), ("2024-01-21", "2025-12-19", 0.12)), 0.5),
    Profile("benedikt", "work", (("2020-06-02", "2024-09-27", 0.12), ("2024-09-28", "2025-12-19", 0.02)), 0.3),
)  # fmt: skip


def _acquaintance_profiles() -> tuple[Profile, ...]:
    """Each acquaintance has one stretch of years and a low rate, from a seeded draw."""
    made = []
    for place in range(len(ACQUAINTANCES)):
        draw = random.Random(f"acquaintance|{place}")
        first = date(2020, 1, 1) + timedelta(days=draw.randrange(0, 1200))
        last = min(first + timedelta(days=draw.randrange(300, 1800)), date(2025, 12, 19))
        rate = draw.choice((0.02, 0.04, 0.06, 0.09, 0.12))
        made.append(Profile(acquaintance_key(place), "friend", ((first.isoformat(), last.isoformat(), rate),), 0.3))
    return tuple(made)


ALL_PROFILES: tuple[Profile, ...] = (*PROFILES, *_acquaintance_profiles())


def _days(first: str, last: str) -> Iterator[date]:
    day, end = date.fromisoformat(first), date.fromisoformat(last)
    while day <= end:
        yield day
        day += timedelta(days=1)


def _speaker(profile: Profile, draw: random.Random, before: str) -> str:
    if not profile.members:
        return "me" if before != "me" else profile.chat
    if before != "me" and draw.random() < ME_SHARE:
        return "me"
    others = [member for member in profile.members if member != before]
    return draw.choice(others)


def _exchange(profile: Profile, clock: datetime, draw: random.Random, day: str, start: int) -> list[Item]:
    group = bool(profile.members)
    count = draw.randint(2, 9) if group else draw.randint(1, 5)
    speaker = draw.choice((*profile.members, "me")) if group else draw.choice(("me", profile.chat))
    items: list[Item] = []
    for place in range(count):
        text = talk.opener(profile.pool, draw) if place == 0 else talk.reply(profile.pool, draw)
        guid = guid_for("hard", profile.chat, day, start + place)
        line = Line(profile.chat, clock, speaker, text, guid)
        items.append(line)
        if group and draw.random() < TAPBACK_RATE:
            fan = draw.choice([member for member in (*profile.members, "me") if member != speaker])
            kind = draw.choice(tuple(TAPBACKS))
            items.append(Tapback(profile.chat, clock + timedelta(seconds=30), fan, guid, kind, guid_for(guid, "tap")))
        clock = clock + timedelta(minutes=draw.choice((0, 1, 1, 2, 3, 5, 12)), seconds=draw.randint(1, 50))
        speaker = _speaker(profile, draw, speaker)
    return items


def _active_day(profile: Profile, day: date, draw: random.Random) -> list[Item]:
    clock = local(f"{day.isoformat()} {draw.randint(FIRST_HOUR, LAST_HOUR - 2):02d}:{draw.randint(0, 59):02d}")
    exchanges = 1
    while draw.random() < profile.burst / (profile.burst + 1):
        exchanges += 1
    items: list[Item] = []
    for _ in range(exchanges):
        items.extend(_exchange(profile, clock, draw, day.isoformat(), len(items)))
        clock = clock + timedelta(minutes=draw.randint(20, 180))
    return [item for item in items if item.when.date() == day]  # a burst stops at midnight


def traffic(profile: Profile) -> tuple[Item, ...]:
    """Every generated line and tapback of one chat."""
    draw = random.Random(f"hard|{profile.chat}")
    items: list[Item] = []
    for first, last, rate in profile.periods:
        for day in _days(first, last):
            if draw.random() >= rate or (profile.weekdays_only and day.weekday() >= 5):
                continue
            items.extend(_active_day(profile, day, draw))
    return tuple(items)
