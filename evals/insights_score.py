"""Score the insights of one run against the gold list of the insight world.

An insight belongs to a person by the name of its thread. Labels turn into
message ids through the world: a message id of the archive is the guid of
the line.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evals.insight_world.cast import NAME_OF
from evals.insight_world.gold import TRAPS, WANTS, Trap, Want
from evals.world.build import World
from insights.thread import KINDS, Insight

_PERSON_OF = {name: key for key, name in NAME_OF.items() if name}


@dataclass(frozen=True)
class Score:
    found: tuple[Want, ...]
    missed: tuple[Want, ...]
    sprung: tuple[Trap, ...]
    extras: tuple[Insight, ...]

    @property
    def recall(self) -> float:
        total = len(self.found) + len(self.missed)
        return len(self.found) / total if total else 1.0


def person(insight: Insight) -> str:
    return _PERSON_OF.get(insight.name, "?")


def _matches(insight: Insight, kind: str, who: str, ids: set[str]) -> bool:
    if insight.kind != kind or person(insight) != who:
        return False
    return not ids or bool(ids & set(insight.message_ids))


def score(world: World, insights: Sequence[Insight]) -> Score:
    def ids(labels: Iterable[str]) -> set[str]:
        return set(world.guids(tuple(labels)))

    found = tuple(want for want in WANTS if any(_matches(item, want.kind, want.person, ids(want.labels)) for item in insights))
    missed = tuple(want for want in WANTS if want not in found)
    sprung = tuple(trap for trap in TRAPS if any(_matches(item, trap.kind, trap.person, ids(trap.labels)) for item in insights))
    extras = tuple(
        item
        for item in insights
        if not any(_matches(item, want.kind, want.person, ids(want.labels)) for want in WANTS)
        and not any(_matches(item, trap.kind, trap.person, ids(trap.labels)) for trap in TRAPS)
    )
    return Score(found, missed, sprung, extras)


def by_kind(result: Score) -> list[tuple[str, int, int]]:
    """(kind, found, wanted) for each kind with a want."""
    rows = []
    for kind in KINDS:
        wanted = [want for want in (*result.found, *result.missed) if want.kind == kind]
        if wanted:
            rows.append((kind, sum(1 for want in result.found if want.kind == kind), len(wanted)))
    return rows
