"""Reciprocal rank fusion: join several rankings of the same things into one.

Each ranking gives an item `1 / (k + rank)`; the rank of the best item is 1.
The scores of an item are added. An item that only one ranking found still
gets its one score. The function reads ranks only: the scores of two search
methods have different units and are never compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, Optional, Sequence, TypeVar

Key = TypeVar("Key", bound=Hashable)
RRF_K = 60  # the usual constant: it keeps one first place from deciding everything


@dataclass(frozen=True)
class Fused(Generic[Key]):
    key: Key
    score: float
    ranks: tuple[Optional[int], ...]  # the rank in each ranking, from 1; None when that ranking did not find it


def fuse(rankings: Sequence[Sequence[Key]], *, k: int = RRF_K) -> list[Fused[Key]]:
    """The items of all rankings, the best first.

    A key that comes twice in one ranking counts at its first place. On equal
    scores the item with the better single rank wins, then the item that the
    earlier ranking holds, then the item that comes earlier there.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    ranks: dict[Key, list[Optional[int]]] = {}
    for index, ranking in enumerate(rankings):
        for place, key in enumerate(dict.fromkeys(ranking), start=1):
            ranks.setdefault(key, [None] * len(rankings))[index] = place
    fused = [
        Fused(key, sum(1.0 / (k + place) for place in places if place is not None), tuple(places))
        for key, places in ranks.items()
    ]
    return sorted(fused, key=_order)


def _order(item: Fused[Key]) -> tuple[float, int, tuple[int, ...]]:
    found = [place for place in item.ranks if place is not None]
    where = tuple(len(item.ranks) if place is None else 0 for place in item.ranks)
    return -item.score, min(found), where
