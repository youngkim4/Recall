"""The hard world: the small world inside a busy six-year phone.

It holds every message of the small world unchanged, so the 60 questions of
the small world keep their answers. Around it: about 50,000 more messages
in 56 more conversations from 2020 to 2025. There are busy group chats, code
and delivery senders, one-word replies, links, tapbacks, bursts and long
gaps. Distractors make search hard: two people named Maya, two Chicago
trips, two job offers, three homes, facts said in slang or with a typo, and
facts that need the message before them.

`hard_world()` gives it; `evals.world.build_world(folder, hard_world())`
writes it.
"""

from __future__ import annotations

from functools import lru_cache

from evals.world.build import World, world
from evals.world.hard.cast import HARD_CHATS, HARD_CONTACTS, HARD_PEOPLE
from evals.world.hard.scenes_groups import group_scenes
from evals.world.hard.scenes_life import life_scenes
from evals.world.hard.services import service_lines
from evals.world.hard.traffic import ALL_PROFILES, traffic
from evals.world.script import Item, Line, Tapback


def hard_items() -> tuple[Item, ...]:
    """The written scenes and the generated traffic that the hard world adds."""
    generated = [item for profile in ALL_PROFILES for item in traffic(profile)]
    return (*life_scenes(), *group_scenes(), *generated, *service_lines())


@lru_cache(maxsize=1)
def hard_world() -> World:
    """The hard world, built once for each process. It is immutable."""
    small = world()
    ordered = sorted([*small.lines, *small.tapbacks, *hard_items()], key=lambda item: item.when)
    return World(
        lines=tuple(item for item in ordered if isinstance(item, Line)),
        tapbacks=tuple(item for item in ordered if isinstance(item, Tapback)),
        name="hard",
        people=HARD_PEOPLE,
        chats=HARD_CHATS,
        contacts=HARD_CONTACTS,
    )


__all__ = ["hard_items", "hard_world"]
