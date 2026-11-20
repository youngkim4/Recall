"""The insight world: sixteen invented threads, each with an insight to find or a trap.

`insight_world()` gives it as an `evals.world.World`, so
`evals.world.build_world(folder, insight_world())` writes it into a real
archive the way the app pulls a history. "Today" is `cast.NOW`.
"""

from __future__ import annotations

from functools import lru_cache

from evals.insight_world.cast import CHATS, CONTACTS, NOW, PEOPLE
from evals.insight_world.filler import filler
from evals.insight_world.scenes import scenes
from evals.world.build import World
from evals.world.script import Line, Tapback


@lru_cache(maxsize=1)
def insight_world() -> World:
    ordered = sorted([*scenes(), *filler()], key=lambda item: item.when)
    return World(
        lines=tuple(item for item in ordered if isinstance(item, Line)),
        tapbacks=tuple(item for item in ordered if isinstance(item, Tapback)),
        name="insights",
        people=PEOPLE,
        chats=CHATS,
        contacts=CONTACTS,
    )


__all__ = ["NOW", "insight_world"]
