"""Helpers for the tests of the store: a store with a fixed clock, and invented indexes in it."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from meaning_index.store import IndexStore
from tests.meaning_index.support import kept, unit

NOW = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
FILES = ("segments.sqlite3", "vectors.npy", "meta.json")


def make(folder: Path) -> IndexStore:
    return IndexStore(folder, now=lambda: NOW)


def fill(store: IndexStore, count: int = 5, dimensions: int = 8, seed: int = 7, model: str = "model-a") -> tuple:
    segments = [kept(n, day=f"2024-06-{n + 1:02d}") for n in range(count)]
    rows = unit(count, dimensions, seed)
    with store.building():
        store.replace(model, dimensions, segments, [rows[:2], rows[2:]])
    return segments, rows


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def content(folder: Path) -> dict[str, bytes]:
    return {name: (folder / name).read_bytes() for name in FILES}


def torn(folder: Path, store: IndexStore, replaced: int) -> dict[str, bytes]:
    """Leave the folder as a process that died in a swap leaves it: `replaced` files are new."""
    fill(store)
    before = content(folder)
    other = folder.parent / "other"
    fill(make(other), count=4, seed=11, model="model-b")
    for name in FILES[:replaced]:
        os.link(folder / name, folder / f"{name}.old")
        os.replace(other / name, folder / name)
    return before


Build = tuple[list, np.ndarray]


def two_builds() -> tuple[Build, Build]:
    """Two invented builds of one shape: the same segments in another order, and other vectors."""
    first = [kept(n) for n in range(4)]
    return (first, unit(4, 8, seed=1)), (list(reversed(first)), unit(4, 8, seed=2))


def put(store: IndexStore, build: Build) -> None:
    with store.building():
        store.replace("model-a", 8, build[0], [build[1]])


def side(store: IndexStore, a: Build, b: Build) -> str:
    """Which build the store gives: "a", "b" or "none". Segments of one with vectors of the other fail."""
    found = store.read()
    status = store.status()
    if found is None:
        assert status.state == "none"
        return "none"
    assert status.state == "ready"
    for name, (segments, rows) in (("a", a), ("b", b)):
        if list(found.segments) == segments and np.array_equal(np.asarray(found.vectors), rows):
            return name
    raise AssertionError("the store pairs the segments of one build with the vectors of another")


def torn_between(folder: Path, a: Build, b: Build, replaced: int) -> IndexStore:
    """Build `a`, then leave the folder as a process that died in the swap to `b` leaves it."""
    store = make(folder)
    put(store, a)
    other = folder.parent / "other-build"
    put(make(other), b)
    for name in FILES[:replaced]:
        os.link(folder / name, folder / f"{name}.old")
        os.replace(other / name, folder / name)
    return store


class Faults:
    """Stands in for os.replace: the calls with the numbers in `at` raise `error`."""

    def __init__(self, at: set[int], error: BaseException | None = None) -> None:
        self.at = at
        self.error = error if error is not None else OSError("the disk said no")
        self.calls = 0
        self._real = os.replace

    def __call__(self, source: object, target: object) -> None:
        self.calls += 1
        if self.calls in self.at:
            raise self.error
        self._real(source, target)  # type: ignore[arg-type]
