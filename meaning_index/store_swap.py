"""Read the three files as one index, swap a new set for the old one, and repair a swap that stopped.

A swap keeps each old file as a hard link with the ending `.old`, and puts the
new files in place in this order: segments, vectors, meta. A put-back and a
repair go the other way: meta, vectors, segments. Each file names its build,
and three files are an index only when they name one build. So at every step,
and after a fault at any step, a reader gets one whole build or none.

While `.old` files are there, a reader that finds no whole build in the main
files reads the `.old` files in their place. A folder that only needs the
put-back thus still gives the build before it. A reader changes no file.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meaning_index import store_files as files
from meaning_index.errors import IndexFormatError, StoreError
from meaning_index.shapes import IndexMeta, Segment
from meaning_index.store_files import META_FILE, SEGMENTS_FILE, VECTORS_FILE

logger = logging.getLogger(__name__)

SWAP_ORDER = (SEGMENTS_FILE, VECTORS_FILE, META_FILE)
NEW_ENDING = ".new"
OLD_ENDING = ".old"
READ_TRIES = 2

_View = Mapping[str, Path]


@dataclass(frozen=True)
class Whole:
    """Three files of one build. `segments` and `vectors` are None when only the facts were asked for."""

    meta: IndexMeta
    segments: tuple[Segment, ...] | None
    vectors: np.ndarray | None


def _old(folder: Path, name: str) -> Path:
    return folder / (name + OLD_ENDING)


def _views(folder: Path) -> list[_View]:
    """The sets of files that can be an index: the main files, then the `.old` files in their place."""
    main = {name: folder / name for name in SWAP_ORDER}
    kept = {name: _old(folder, name) for name in SWAP_ORDER if _old(folder, name).exists()}
    return [main, {**main, **kept}] if kept else [main]


def _read_view(view: _View, rows: bool) -> tuple[Whole | None, str]:
    """One whole build from these files, or None and the reason in fixed words."""
    vectors = files.open_vectors(view[VECTORS_FILE], rows=rows)
    segments = files.read_segments(view[SEGMENTS_FILE], rows=rows)
    meta = files.read_meta(view[META_FILE])
    if meta is None:
        return None, "the meta file can not be read"
    if vectors is None or segments is None:
        return None, "the vectors file or the segments file can not be read"
    if not meta.build_id == segments.build_id == vectors.build_id:
        return None, "the files name more than one build"
    if vectors.shape != (meta.segments, meta.dimensions) or segments.count != meta.segments:
        return None, "the files do not hold what the meta file names"
    if files.read_meta(view[META_FILE]) != meta:
        return None, "the meta file changed during the read"
    return Whole(meta=meta, segments=segments.segments, vectors=vectors.rows), ""


def read_whole(folder: Path, *, rows: bool) -> Whole | None:
    """The index of the folder as one whole build, or None. A read during a swap is tried one more time."""
    reason = ""
    for _ in range(READ_TRIES):
        views = _views(folder)
        for place, view in enumerate(views):
            found, why = _read_view(view, rows)
            if found is not None:
                return found
            reason = reason if place else why
    if any((folder / name).exists() for name in SWAP_ORDER):
        logger.warning("the files of the index are not one whole build (%s); the index is read as none", reason)
    return None


def drop_leftovers(folder: Path) -> None:
    for name in SWAP_ORDER:
        files.remove(_old(folder, name))
        files.remove(folder / (name + NEW_ENDING))


def _put_back(folder: Path, replaced: Sequence[str]) -> None:
    """Undo a swap that stopped, last file first. A fault here leaves the `.old` files for the repair."""
    try:
        for name in reversed(replaced):
            if _old(folder, name).exists():
                os.replace(_old(folder, name), folder / name)
            else:
                files.remove(folder / name)  # a first build: there was no file before
        drop_leftovers(folder)
    except BaseException:  # noqa: BLE001 - the failure of the swap is the one to raise
        logger.error("the old index files could not be put back; they are read as they are and the next build repairs")


def swap(folder: Path, staged: Mapping[str, Path]) -> None:
    """Put the staged files in place. Whatever stops the swap, the old files are put back first."""
    replaced: list[str] = []
    try:
        for name in SWAP_ORDER:
            final = folder / name
            if final.exists():
                files.remove(_old(folder, name))
                os.link(final, _old(folder, name))
            os.replace(staged[name], final)
            replaced.append(name)
        files.sync_folder(folder)
    except OSError as exc:
        _put_back(folder, replaced)
        raise StoreError(f"the new index files could not be put in place ({type(exc).__name__})") from exc
    except BaseException:
        _put_back(folder, replaced)
        raise
    drop_leftovers(folder)


def recover(folder: Path) -> None:
    """Repair a folder that a build left in the middle of a swap. Call it with the build lock held."""
    backups = [name for name in SWAP_ORDER if _old(folder, name).exists()]
    if backups:
        try:
            whole = _read_view({name: folder / name for name in SWAP_ORDER}, rows=False)[0] is not None
        except IndexFormatError:
            whole = True  # a newer build wrote these files: they are left alone
        if not whole:
            for name in reversed(backups):
                os.replace(_old(folder, name), folder / name)
            logger.warning("a swap of the index files did not end; the files of the build before it are back")
    drop_leftovers(folder)
