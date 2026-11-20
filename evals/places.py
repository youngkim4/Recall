"""Where a run writes: its results file, and the saves of the world.

The results go to `<data folder>/evals/`. In a checkout the data folder is
the repository itself, so the results then go to the data folder of the
packaged app instead: never into the repository tree.

While a run lasts, the saves folder of Recall (the index of meanings, the
people index, the notes of digests) points into the temp folder of the
world. So a run never reads or writes the saves of the owner, and the owner's
index can not answer a question about the invented world.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import recall_paths
from recall_services import memory

RESULTS_LEAF = "evals"
APP_DATA = Path.home() / "Library" / "Application Support" / "Recall"


def _inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def results_folder() -> Path:
    """`<data folder>/evals`, outside the repository tree."""
    data = Path(recall_paths.DATA_DIR)
    if _inside(data, Path(recall_paths.ROOT)):
        data = APP_DATA
    return data / RESULTS_LEAF


def default_out(now: datetime) -> Path:
    return results_folder() / f"chat-eval-{now:%Y%m%d-%H%M%S}.json"


@contextmanager
def isolated_saves(folder: Path) -> Iterator[Path]:
    """Point the saves and the out folder of Recall into `folder` for the length of the block."""
    before = (recall_paths.SAVES_DIR, recall_paths.OUT_DIR)
    saves = Path(folder) / "saves"
    saves.mkdir(parents=True, exist_ok=True)
    recall_paths.SAVES_DIR, recall_paths.OUT_DIR = saves, Path(folder) / "out"
    memory.forget()  # an index that a port kept from before
    try:
        yield saves
    finally:
        memory.forget()
        recall_paths.SAVES_DIR, recall_paths.OUT_DIR = before
