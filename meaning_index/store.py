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
from meaning_index.errors import IndexBusyError, StoreError
from meaning_index.shapes import IndexMeta, IndexStatus, Segment
from meaning_index.store_files import FORMAT_VERSION, META_FILE, SEGMENTS_FILE, SWAP_ORDER, VECTORS_FILE
from meaning_index.workfile import WorkFile

logger = logging.getLogger(__name__)

__all__ = ["FORMAT_VERSION", "IndexStore", "StoredIndex"]

WORK_FILE = "work.sqlite3"
LOCK_FILE = "build.lock"
PRIVATE_FOLDER_MODE = 0o700


@dataclass(frozen=True)
class StoredIndex:
    """One whole index. `segments[n]` belongs to row `n` of `vectors`."""

    meta: IndexMeta
    segments: tuple[Segment, ...]
    vectors: np.ndarray


def _utc_now() -> datetime:
    return datetime.now(UTC)


class IndexStore:
    def __init__(self, folder: str | os.PathLike[str], now: Callable[[], datetime] = _utc_now) -> None:
        self._folder = Path(folder)
        self._now = now
        self._lock: int | None = None

    @property
    def folder(self) -> Path:
        return self._folder

    def status(self) -> IndexStatus:
        """What the folder holds now. Files that are not one whole build are state "none"."""
        meta = files.usable_meta(self._folder)
        if meta is None:
            return IndexStatus(state="none")
        return IndexStatus(
            state="ready",
            segments=meta.segments,
            model=meta.model,
            dimensions=meta.dimensions,
            built_at=meta.built_at,
            covered_until=meta.covered_until,
        )

    def read(self) -> StoredIndex | None:
        """The index, or None when the folder holds no whole build."""
        meta = files.usable_meta(self._folder)
        if meta is None:
            return None
        segments = files.read_segments(self._folder / SEGMENTS_FILE)
        if len(segments) != meta.segments:
            logger.warning("the segments file does not hold the segments of its build; the index is read as none")
            return None
        vectors = files.open_vectors(self._folder / VECTORS_FILE, meta.segments, meta.dimensions)
        return StoredIndex(meta=meta, segments=segments, vectors=vectors)

    def work_file(self, dimensions: int) -> WorkFile:
        """The vectors that a stopped build paid for, for vectors of this size."""
        return WorkFile(self._folder / WORK_FILE, dimensions)

    @contextmanager
    def building(self) -> Iterator[None]:
        """Hold the folder for one build. A second build at the same time is an IndexBusyError."""
        try:
            self._folder.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
            descriptor = os.open(self._folder / LOCK_FILE, os.O_RDWR | os.O_CREAT, files.PRIVATE_FILE_MODE)
        except OSError as exc:
            raise StoreError(f"the folder of the index can not be used ({type(exc).__name__})") from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise IndexBusyError("another build holds the folder of the index") from None
            try:
                files.recover(self._folder)
            except OSError as exc:
                raise StoreError(f"the folder of the index can not be repaired ({type(exc).__name__})") from exc
            self._lock = descriptor
            try:
                yield
            finally:
                self._lock = None
        finally:
            os.close(descriptor)  # this lets go of the lock too

    def _stamp(self) -> str:
        return self._now().astimezone(UTC).isoformat(timespec="seconds")

    def replace(
        self, model: str, dimensions: int, segments: Sequence[Segment], rows: Iterable[np.ndarray]
    ) -> IndexMeta:
        """Write a new index and swap it for the old one. Call it inside `building()`.

        `rows` gives the vectors in the order of `segments`, in chunks of any size.
        """
        if self._lock is None:
            raise StoreError("the index is replaced only inside building()")
        meta = IndexMeta(
            format_version=FORMAT_VERSION,
            build_id=secrets.token_hex(8),
            model=model,
            dimensions=dimensions,
            built_at=self._stamp(),
            segments=len(segments),
            covered_until=max((segment.last_timestamp for segment in segments), default=None),
        )
        staged = {name: self._folder / (name + files.NEW_ENDING) for name in SWAP_ORDER}
        try:
            files.write_vectors(staged[VECTORS_FILE], len(segments), dimensions, rows)
            files.write_segments(staged[SEGMENTS_FILE], meta.build_id, segments)
            files.write_meta(staged[META_FILE], meta)
            files.swap(self._folder, staged)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"the new index files can not be written ({type(exc).__name__})") from exc
        finally:
            for path in staged.values():
                files.remove(path)
        return meta
