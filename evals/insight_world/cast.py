"""The people of the insight world and the rhythm of each thread. Every name and number is invented.

Numbers are +1 707 555-01xx; no real line uses them. "fernbank" is a sender
of codes that nobody saved and nobody answers.

Each `Rhythm` makes the filler of one thread: a short, plain exchange each
`every` days from `first` through `last`. `starts` says who writes first, by
place: "me", "them", "even" (they take turns), or a pattern such as
"me*5,them" (five of mine, then one of theirs, again and again).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from evals.world.people import Chat, Person

NOW = datetime(2025, 12, 20, 12, 0)


@dataclass(frozen=True)
class Rhythm:
    key: str
    first: str
    last: str
    every: int
    starts: str = "even"


@dataclass(frozen=True)
class Friend:
    key: str
    name: str  # "" when the handle is not saved
    number: int  # the last two digits of the handle


FRIENDS: tuple[Friend, ...] = (
    Friend("felix", "Felix Arnaud", 1),
    Friend("greta", "Greta Lindahl", 2),
    Friend("hugo", "Hugo Brennan", 3),
    Friend("iris", "Iris Calder", 4),
    Friend("jonah", "Jonah Whitlow", 5),
    Friend("kaya", "Kaya Morrow", 6),
    Friend("leo", "Leo Santangelo", 7),
    Friend("mira", "Mira Okonkwo", 8),
    Friend("nora", "Nora Bexley", 9),
    Friend("otto", "Otto Reinholt", 10),
    Friend("pia", "Pia Strand", 11),
    Friend("quinn", "Quinn Albers", 12),
    Friend("rosa", "Rosa Delgado", 13),
    Friend("sol", "Sol Hartigan", 14),
    Friend("tess", "Tess Ambler", 15),
    Friend("fernbank", "", 30),
)

RHYTHMS: tuple[Rhythm, ...] = (
    Rhythm("felix", "2024-01-03", "2025-12-17", 7),
    Rhythm("greta", "2023-03-01", "2025-11-05", 60),
    Rhythm("hugo", "2024-02-01", "2025-12-15", 5, "me*5,them"),
    Rhythm("iris", "2023-06-10", "2025-12-10", 30),
    Rhythm("jonah", "2024-01-05", "2025-12-12", 7),
    Rhythm("kaya", "2024-01-02", "2025-12-16", 6),
    Rhythm("leo", "2024-01-04", "2025-12-14", 7),
    Rhythm("mira", "2024-01-06", "2025-12-15", 7),
    Rhythm("nora", "2024-01-07", "2025-07-15", 7),
    Rhythm("otto", "2024-01-08", "2025-12-12", 8),
    Rhythm("pia", "2020-03-01", "2022-05-01", 7),
    Rhythm("quinn", "2025-01-03", "2025-12-18", 5, "them*9,me"),
    Rhythm("rosa", "2024-01-09", "2025-12-13", 7),
    Rhythm("sol", "2024-01-10", "2025-12-17", 4),
    Rhythm("tess", "2024-01-11", "2025-12-16", 7),
)


def handle(friend: Friend) -> str:
    return f"+17075550{100 + friend.number:03d}"


PEOPLE: tuple[Person, ...] = tuple(Person(friend.key, handle(friend), friend.name) for friend in FRIENDS)
CHATS: tuple[Chat, ...] = tuple(Chat(friend.key, handle(friend), (friend.key,)) for friend in FRIENDS)
CONTACTS: tuple[dict[str, object], ...] = tuple(
    {"name": friend.name, "phones": [handle(friend)], "emails": []} for friend in FRIENDS if friend.name
)
NAME_OF: dict[str, str] = {friend.key: friend.name for friend in FRIENDS}
