"""The folder of the index: its status, its content, and the one build that can write it.

    store = IndexStore(saves_dir / "meaning")
    store.status()        # IndexStatus: "none" or "ready"
    store.read()          # the segments and the vectors, or None

A build holds the folder with `building()`. It writes a new set of files
beside the old one and swaps them at the end (`replace`). A build that fails
or is cancelled leaves the old files as they were. A reader makes no file.
"""

from __future__ import annotations

import fcntl
import logging
import os
import secrets
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from meaning_index import store_files as files
from meaning_index import store_swap as swapping
from meaning_index.errors import IndexBusyError, StoreError
from meaning_index.recipe import DEFAULT_NAME
from meaning_index.shapes import IndexMeta, IndexStatus, Segment
from meaning_index.store_files import FORMAT_VERSION, META_FILE, SEGMENTS_FILE, VECTORS_FILE
from meaning_index.workfile import WorkFile

logger = logging.getLogger(__name__)

__all__ = ["FORMAT_VERSION", "IndexStore", "StoredIndex"]

WORK_FILE = "work.sqlite3"
LOCK_FILE = "build.lock"
PRIVATE_FOLDER_MODE = 0o700
BUILD_ID_BYTES = files.BUILD_ID_CHARS // 2


@dataclass(frozen=True)
class StoredIndex:
    """One whole index. `segments[n]` belongs to row `n` of `vectors`. No value of `vectors` is NaN or infinite."""

    meta: IndexMeta
    segments: tuple[Segment, ...]
    vectors: np.ndarray


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _remove_quietly(path: Path) -> None:
    """For a `finally`: a file that stays must not hide the failure that is on its way up."""
    try:
        files.remove(path)
    except OSError as exc:
        logger.warning("a new index file could not be removed (%s); the next build removes it", type(exc).__name__)


class IndexStore:
    def __init__(self, folder: str | os.PathLike[str], now: Callable[[], datetime] = _utc_now) -> None:
        self._folder = Path(folder)
        self._now = now
        self._lock: int | None = None

    @property
    def folder(self) -> Path:
        return self._folder

    def status(self) -> IndexStatus:
        """What the folder holds now. Files that are not one whole build are state "none".

        The status does not read the rows. A folder that only needs the
        put-back of a swap that stopped still gives the build before it.
        """
        whole = swapping.read_whole(self._folder, rows=False)
        if whole is None:
            return IndexStatus(state="none")
        return IndexStatus(
            state="ready",
            segments=whole.meta.segments,
            model=whole.meta.model,
            dimensions=whole.meta.dimensions,
            built_at=whole.meta.built_at,
            covered_until=whole.meta.covered_until,
        )

    def meta(self) -> IndexMeta | None:
        """The facts of the build on disk, with its `build_id`. None when the folder holds no whole build.

        It reads no row, so a caller can ask it for each question to learn
        whether the build that it loaded is still the build on disk.
        """
        whole = swapping.read_whole(self._folder, rows=False)
        return None if whole is None else whole.meta

    def read(self) -> StoredIndex | None:
        """The index as one whole build, or None when the folder holds none.

        Vectors with a value that is no number are a StoreError.
        """
        whole = swapping.read_whole(self._folder, rows=True)
        if whole is None or whole.segments is None or whole.vectors is None:
            return None
        if not files.all_finite(whole.vectors):
            raise StoreError("the vectors file holds a value that is no number")
        return StoredIndex(meta=whole.meta, segments=whole.segments, vectors=whole.vectors)

    def work_file(self, dimensions: int) -> WorkFile:
        """The vectors that a stopped build paid for, for vectors of this size."""
        return WorkFile(self._folder / WORK_FILE, dimensions)

    def _take_lock(self) -> int:
        if self._lock is not None:
            raise IndexBusyError("this store is inside a build already")
        try:
            self._folder.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
            descriptor = os.open(self._folder / LOCK_FILE, os.O_RDWR | os.O_CREAT, files.PRIVATE_FILE_MODE)
        except OSError as exc:
            raise StoreError(f"the folder of the index can not be used ({type(exc).__name__})") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(descriptor)
            raise IndexBusyError("another build holds the folder of the index") from None
        return descriptor

    @contextmanager
    def building(self) -> Iterator[None]:
        """Hold the folder for one build. A second build at the same time is an IndexBusyError."""
        descriptor = self._take_lock()
        try:
            try:
                swapping.recover(self._folder)
            except OSError as exc:
                raise StoreError(f"the folder of the index can not be repaired ({type(exc).__name__})") from exc
            self._lock = descriptor
            yield
        finally:
            self._lock = None
            try:
                os.close(descriptor)  # this lets go of the lock too
            except OSError as exc:
                logger.warning("the lock of the build could not be closed (%s)", type(exc).__name__)

    def _stamp(self) -> str:
        return self._now().astimezone(UTC).isoformat(timespec="seconds")

    def replace(
        self,
        model: str,
        dimensions: int,
        segments: Sequence[Segment],
        rows: Iterable[np.ndarray],
        recipe: str = DEFAULT_NAME,
    ) -> IndexMeta:
        """Write a new index and swap it for the old one. Call it inside `building()`.

        `rows` gives the vectors in the order of `segments`, in chunks of any size.
        `recipe` is the name of the recipe of the segments.
        """
        if self._lock is None:
            raise StoreError("the index is replaced only inside building()")
        meta = IndexMeta(
            format_version=FORMAT_VERSION,
            build_id=secrets.token_hex(BUILD_ID_BYTES),
            model=model,
            dimensions=dimensions,
            built_at=self._stamp(),
            segments=len(segments),
            covered_until=max((segment.last_timestamp for segment in segments), default=None),
            recipe=recipe,
        )
        staged = {name: self._folder / (name + swapping.NEW_ENDING) for name in swapping.SWAP_ORDER}
        try:
            files.write_vectors(staged[VECTORS_FILE], meta.build_id, len(segments), dimensions, rows)
            files.write_segments(staged[SEGMENTS_FILE], meta.build_id, segments)
            files.write_meta(staged[META_FILE], meta)
            swapping.swap(self._folder, staged)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"the new index files can not be written ({type(exc).__name__})") from exc
        finally:
            for path in staged.values():
                _remove_quietly(path)
        return meta
