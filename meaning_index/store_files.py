"""Read and write the three files of the index, and swap a new set for the old one.

- `segments.sqlite3`: one row for each segment, and the id of the build.
- `vectors.npy`: N by D, float16, each row of length 1.
- `meta.json`: the facts of the build. It is replaced last.

Each file is for its owner only. A new set is written beside the old one and
put in place file by file: segments, vectors, meta. Until the meta file is in
place, the build id of the segments file differs from the one in the meta
file, so files of two builds are never read as one index.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np

from meaning_index.errors import StoreError
from meaning_index.shapes import IndexMeta, Segment

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
SEGMENTS_FILE = "segments.sqlite3"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"
SWAP_ORDER = (SEGMENTS_FILE, VECTORS_FILE, META_FILE)
NEW_ENDING = ".new"
OLD_ENDING = ".old"
PRIVATE_FILE_MODE = 0o600
VECTOR_TYPE = np.dtype("<f2")

_CREATE = (
    """
CREATE TABLE segments (
    segment_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    first_message_id TEXT NOT NULL,
    last_message_id TEXT NOT NULL,
    first_timestamp TEXT NOT NULL,
    last_timestamp TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    "row" INTEGER NOT NULL UNIQUE
)
""",
    "CREATE TABLE build (build_id TEXT NOT NULL)",
)
_COLUMNS = (
    "segment_id, conversation_id, first_message_id, last_message_id,"
    " first_timestamp, last_timestamp, message_count, content_hash"
)
_META_TYPES: Mapping[str, tuple[type, ...]] = {
    "format_version": (int,),
    "build_id": (str,),
    "model": (str,),
    "dimensions": (int,),
    "built_at": (str,),
    "segments": (int,),
    "covered_until": (str, type(None)),
}


def private_file(path: Path) -> int:
    """Open a new, empty file that only its owner can read. Returns the descriptor."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    os.fchmod(descriptor, PRIVATE_FILE_MODE)  # the file was there, with a wider mode
    return descriptor


def remove(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# --- write -------------------------------------------------------------------


def write_meta(path: Path, meta: IndexMeta) -> None:
    with os.fdopen(private_file(path), "w", encoding="utf-8") as handle:
        json.dump(asdict(meta), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def write_segments(path: Path, build_id: str, segments: Sequence[Segment]) -> None:
    os.close(private_file(path))
    conn = sqlite3.connect(path)
    try:
        with conn:
            for statement in _CREATE:
                conn.execute(statement)
            conn.execute("INSERT INTO build VALUES (?)", (build_id,))
            conn.executemany(
                f'INSERT INTO segments ({_COLUMNS}, "row") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    (s.segment_id, s.conversation_id, s.first_message_id, s.last_message_id,
                     s.first_timestamp, s.last_timestamp, s.message_count, s.content_hash, row)
                    for row, s in enumerate(segments)
                ),
            )
    finally:
        conn.close()


def write_vectors(path: Path, count: int, dimensions: int, chunks: Iterable[np.ndarray]) -> None:
    """Write `count` rows from the chunks, in order. Rows that do not fit are a StoreError."""
    header = {"descr": np.lib.format.dtype_to_descr(VECTOR_TYPE), "fortran_order": False, "shape": (count, dimensions)}
    written = 0
    with os.fdopen(private_file(path), "wb") as handle:
        np.lib.format.write_array_header_1_0(handle, header)
        for chunk in chunks:
            rows = np.asarray(chunk)
            if rows.dtype != VECTOR_TYPE or rows.ndim != 2 or rows.shape[1] != dimensions:
                raise StoreError(f"rows of the index must be float16 with {dimensions} dimensions")
            written += rows.shape[0]
            if written > count:
                break
            handle.write(np.ascontiguousarray(rows).tobytes())
        if written != count:
            raise StoreError(f"the index has {count} segments and was given another number of rows")
        handle.flush()
        os.fsync(handle.fileno())


# --- read --------------------------------------------------------------------


def _connect_to_read(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)


def read_meta(path: Path) -> IndexMeta | None:
    """The meta file, or None when there is none or it can not be read. A newer format is a StoreError."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("the meta file of the index can not be read (%s); the index is read as none", type(exc).__name__)
        return None
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if isinstance(version, int) and version > FORMAT_VERSION:
        raise StoreError(f"the index has format version {version}; this build reads up to {FORMAT_VERSION}")
    fits = isinstance(raw, dict) and all(
        isinstance(raw.get(key), kinds) and not isinstance(raw.get(key), bool) for key, kinds in _META_TYPES.items()
    )
    if not fits:
        logger.warning("the meta file of the index has not the fields of a build; the index is read as none")
        return None
    return IndexMeta(**{key: raw[key] for key in _META_TYPES})


def read_build_id(path: Path) -> str | None:
    """The build that wrote the segments file, or None when the file can not be read."""
    try:
        conn = _connect_to_read(path)
        try:
            found = conn.execute("SELECT build_id FROM build").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return str(found[0]) if found else None


def read_segments(path: Path) -> tuple[Segment, ...]:
    """Every segment, in the order of its row in the vectors file."""
    try:
        conn = _connect_to_read(path)
        try:
            found = conn.execute(f'SELECT {_COLUMNS}, "row" FROM segments ORDER BY "row"').fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise StoreError(f"the segments file can not be read ({type(exc).__name__})") from exc
    if [row[8] for row in found] != list(range(len(found))):
        raise StoreError("the rows of the segments file are not 0 to N-1")
    return tuple(Segment(*row[:8]) for row in found)


def vectors_shape(path: Path) -> tuple[int, ...] | None:
    """The shape that the vectors file names, or None when it is no float16 array of this package."""
    try:
        with open(path, "rb") as handle:
            if np.lib.format.read_magic(handle) != (1, 0):
                return None
            shape, fortran, kind = np.lib.format.read_array_header_1_0(handle)
            body = os.fstat(handle.fileno()).st_size - handle.tell()
    except (OSError, ValueError):
        return None
    whole = len(shape) == 2 and body == shape[0] * shape[1] * VECTOR_TYPE.itemsize
    return tuple(shape) if whole and not fortran and kind == VECTOR_TYPE else None


def open_vectors(path: Path, count: int, dimensions: int) -> np.ndarray:
    """The rows as a read-only map of the file. The map keeps its file when a build replaces it."""
    if count == 0:
        return np.zeros((0, dimensions), dtype=VECTOR_TYPE)
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise StoreError(f"the vectors file can not be read ({type(exc).__name__})") from exc


def usable_meta(folder: Path) -> IndexMeta | None:
    """The meta file when the three files belong to one build. None when there is no whole index."""
    meta = read_meta(folder / META_FILE)
    if meta is None:
        return None
    one_build = read_build_id(folder / SEGMENTS_FILE) == meta.build_id
    if not one_build or vectors_shape(folder / VECTORS_FILE) != (meta.segments, meta.dimensions):
        logger.warning("the files of the index do not belong to one build; the index is read as none")
        return None
    return meta


# --- swap --------------------------------------------------------------------


def _sync_folder(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _put_back(folder: Path, replaced: Sequence[str]) -> None:
    for name in replaced:
        backup = folder / (name + OLD_ENDING)
        if backup.exists():
            os.replace(backup, folder / name)
        else:
            remove(folder / name)


def drop_leftovers(folder: Path) -> None:
    for name in SWAP_ORDER:
        remove(folder / (name + OLD_ENDING))
        remove(folder / (name + NEW_ENDING))


def swap(folder: Path, staged: Mapping[str, Path]) -> None:
    """Put the staged files in place. A failure puts the old files back and raises StoreError."""
    replaced: list[str] = []
    try:
        for name in SWAP_ORDER:
            final = folder / name
            if final.exists():
                remove(folder / (name + OLD_ENDING))
                os.link(final, folder / (name + OLD_ENDING))
            os.replace(staged[name], final)
            replaced.append(name)
        _sync_folder(folder)
    except OSError as exc:
        try:
            _put_back(folder, replaced)
            drop_leftovers(folder)
        except OSError:
            logger.error("the old index files could not be put back; the next build repairs the folder")
        raise StoreError(f"the new index files could not be put in place ({type(exc).__name__})") from exc
    drop_leftovers(folder)


def recover(folder: Path) -> None:
    """Repair a folder that a build left in the middle of a swap. Call it with the build lock held."""
    backups = [name for name in SWAP_ORDER if (folder / (name + OLD_ENDING)).exists()]
    if backups:
        try:
            whole = usable_meta(folder) is not None
        except StoreError:
            whole = True  # a newer build wrote these files: they are left alone
        if not whole:
            for name in backups:
                os.replace(folder / (name + OLD_ENDING), folder / name)
            logger.warning("a swap of the index files did not end; the files of the build before it are back")
    drop_leftovers(folder)
