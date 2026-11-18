"""Helpers for the tests of the store: a store with a fixed clock, and invented indexes in it."""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from pathlib import Path

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
