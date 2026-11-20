"""Read and write each of the three files of the index.

- `segments.sqlite3`: one row for each segment, and the id of the build.
- `vectors.npy`: N by D, float16, each row of length 1. After the rows the
  file holds one line with the id of the build. numpy reads the file as it is.
- `meta.json`: the facts of the build, with its id.

Each file names its build. A reader takes each file through one open handle,
so the id that it reads belongs to the content that it reads.
Each file is for its owner only.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

from meaning_index.errors import IndexFormatError, StoreError
from meaning_index.recipe import DEFAULT_NAME
from meaning_index.shapes import IndexMeta, Segment

FORMAT_VERSION = 1
SEGMENTS_FILE = "segments.sqlite3"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"
PRIVATE_FILE_MODE = 0o600
VECTOR_TYPE = np.dtype("<f2")
FINITE_CHUNK = 8192
BUILD_MARK = b"\nRECALL-BUILD "
BUILD_ID_CHARS = 16
TAIL_BYTES = len(BUILD_MARK) + BUILD_ID_CHARS + 1

_BUILD_ID = re.compile(rf"[0-9a-f]{{{BUILD_ID_CHARS}}}")
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


@dataclass(frozen=True)
class SegmentsFile:
    """What one read of the segments file gave. `segments` is None when only the count was asked for."""

    build_id: str
    count: int
    segments: tuple[Segment, ...] | None


@dataclass(frozen=True)
class VectorsFile:
    """What one read of the vectors file gave. `rows` is None when only the shape was asked for."""

    build_id: str
    shape: tuple[int, int]
    rows: np.ndarray | None


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


def sync_folder(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def all_finite(rows: np.ndarray) -> bool:
    """True when no value of the rows is NaN or infinite. Reads the rows in chunks."""
    starts = range(0, len(rows), FINITE_CHUNK)
    return all(bool(np.isfinite(rows[start : start + FINITE_CHUNK]).all()) for start in starts)


# --- write -------------------------------------------------------------------


def write_meta(path: Path, meta: IndexMeta) -> None:
    """The meta file. The default recipe is not written: the file is then the one of an older build."""
    facts = {key: value for key, value in asdict(meta).items() if key != "recipe" or value != DEFAULT_NAME}
    with os.fdopen(private_file(path), "w", encoding="utf-8") as handle:
        json.dump(facts, handle, indent=2)
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


def write_vectors(path: Path, build_id: str, count: int, dimensions: int, chunks: Iterable[np.ndarray]) -> None:
    """Write `count` rows from the chunks, in order, and then the id of the build.

    Rows that do not fit, and a row with a value that is no number, are a StoreError.
    """
    if not _BUILD_ID.fullmatch(build_id):
        raise StoreError("the id of a build is 16 characters, 0 to 9 and a to f")
    header = {"descr": np.lib.format.dtype_to_descr(VECTOR_TYPE), "fortran_order": False, "shape": (count, dimensions)}
    written = 0
    with os.fdopen(private_file(path), "wb") as handle:
        np.lib.format.write_array_header_1_0(handle, header)
        for chunk in chunks:
            rows = np.asarray(chunk)
            if rows.dtype != VECTOR_TYPE or rows.ndim != 2 or rows.shape[1] != dimensions:
                raise StoreError(f"rows of the index must be float16 with {dimensions} dimensions")
            if not np.isfinite(rows).all():
                raise StoreError("a row of the index holds a value that is no number")
            written += rows.shape[0]
            if written > count:
                break
            handle.write(np.ascontiguousarray(rows).tobytes())
        if written != count:
            raise StoreError(f"the index has {count} segments and was given another number of rows")
        handle.write(BUILD_MARK + build_id.encode("ascii") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


# --- read --------------------------------------------------------------------


def read_meta(path: Path) -> IndexMeta | None:
    """The meta file, or None when there is none or it is not the meta file of a build.

    An index of a newer format is an IndexFormatError.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if isinstance(version, int) and version > FORMAT_VERSION:
        raise IndexFormatError(f"the index has format version {version}; this build reads up to {FORMAT_VERSION}")
    fits = isinstance(raw, dict) and all(
        isinstance(raw.get(key), kinds) and not isinstance(raw.get(key), bool) for key, kinds in _META_TYPES.items()
    )
    recipe = raw.get("recipe", DEFAULT_NAME) if isinstance(raw, dict) else None
    if not fits or not isinstance(recipe, str) or not recipe:
        return None
    return IndexMeta(**{key: raw[key] for key in _META_TYPES}, recipe=recipe)


def read_segments(path: Path, *, rows: bool) -> SegmentsFile | None:
    """The build id and the segments of the file, through one connection. None when it can not be read.

    With `rows`, every segment in the order of its row in the vectors file. The rows must be 0 to N-1.
    """
    try:
        conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
        try:
            conn.execute("BEGIN")  # the id and the rows are read from one state of the file
            build = conn.execute("SELECT build_id FROM build").fetchone()
            if not rows:
                count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
                return SegmentsFile(str(build[0]), int(count), None) if build else None
            found = conn.execute(f'SELECT {_COLUMNS}, "row" FROM segments ORDER BY "row"').fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if build is None or [row[8] for row in found] != list(range(len(found))):
        return None
    return SegmentsFile(str(build[0]), len(found), tuple(Segment(*row[:8]) for row in found))


def open_vectors(path: Path, *, rows: bool) -> VectorsFile | None:
    """The shape, the build id and the rows of the file, through one handle. None when it can not be read.

    The rows are a read-only map of the file. The map keeps its file when a build replaces it.
    """
    try:
        with open(path, "rb") as handle:
            if np.lib.format.read_magic(handle) != (1, 0):
                return None
            shape, fortran, kind = np.lib.format.read_array_header_1_0(handle)
            start, size = handle.tell(), os.fstat(handle.fileno()).st_size
            if len(shape) != 2 or fortran or kind != VECTOR_TYPE:
                return None
            data = shape[0] * shape[1] * VECTOR_TYPE.itemsize
            handle.seek(start + data)
            tail = handle.read(TAIL_BYTES + 1)  # one byte more shows a file that is too long
            build_id = tail[len(BUILD_MARK) : -1].decode("ascii", "replace")
            named = len(tail) == TAIL_BYTES and tail.startswith(BUILD_MARK) and _BUILD_ID.fullmatch(build_id)
            if size != start + data + TAIL_BYTES or not named:
                return None
            if not rows:
                return VectorsFile(build_id, (shape[0], shape[1]), None)
            if data == 0:
                return VectorsFile(build_id, (shape[0], shape[1]), np.zeros(shape, dtype=VECTOR_TYPE))
            mapped = np.memmap(handle, dtype=VECTOR_TYPE, mode="r", offset=start, shape=shape)
    except (OSError, ValueError):
        return None
    return VectorsFile(build_id, (shape[0], shape[1]), mapped)
