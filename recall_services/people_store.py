"""The file of the people index: one row for each profile, written whole.

Each archive has a store of its own: `people/<name>/people.sqlite3` in the
saves folder, where the name is the first 16 hex digits of the SHA-256 of the
resolved path of the archive file (`people_folder`), as for the index of
meanings and the marker of the memory.

- A row holds the key, the conversation ids, the label, the figures, the kind,
  the status, the summary, the turning points, the hash of the inputs of the
  model (`input_hash`), the hash of the text that was embedded with the model
  and the dimensions (`text_hash`), and the vector (float32, length 1), or no
  vector when the text could not be embedded yet.
- The table `meta` holds the format, the model of the profiles, the embedding
  model and its dimensions, the time of the build, and the time of the last
  build that completed (None until one did).
- A write makes a new file beside the old one and puts it in place with one
  rename: a reader gets the old file or the new one, never a part. A write
  that fails leaves the old file as it was.
- The folder is for its owner only (0700), and so is the file (0600).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from recall_services.people_figures import Figures
from recall_services.people_reply import TurningPoint

FORMAT_VERSION = 1
STORE_LEAF = "people"
STORE_FILE = "people.sqlite3"
NEW_ENDING = ".new"
NAME_CHARS = 16
PRIVATE_FILE_MODE = 0o600
PRIVATE_FOLDER_MODE = 0o700

_CREATE = (
    "CREATE TABLE meta (format_version INTEGER NOT NULL, model TEXT NOT NULL, embed_model TEXT NOT NULL,"
    " dimensions INTEGER NOT NULL, built_at TEXT NOT NULL, completed_at TEXT)",
    "CREATE TABLE profiles (position INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, conversations TEXT NOT NULL,"
    " label TEXT NOT NULL, is_group INTEGER NOT NULL, figures TEXT NOT NULL, kind TEXT NOT NULL,"
    " status TEXT NOT NULL, summary TEXT NOT NULL, turning_points TEXT NOT NULL, input_hash TEXT NOT NULL,"
    " text_hash TEXT NOT NULL, vector BLOB)",
)
_COLUMNS = (
    "key, conversations, label, is_group, figures, kind, status, summary, turning_points, input_hash, text_hash, vector"
)


class PeopleStoreError(Exception):
    """The store can not be read or written. The message holds no text of a profile."""


class PeopleFormatError(PeopleStoreError):
    """A newer build of the app wrote the store. It is not read and not replaced."""


@dataclass(frozen=True)
class Profile:
    key: str
    conversations: tuple[str, ...]
    label: str
    is_group: bool
    figures: Figures
    kind: str
    status: str
    summary: str
    turning_points: tuple[TurningPoint, ...]
    input_hash: str
    text_hash: str
    vector: bytes | None = None


@dataclass(frozen=True)
class StoreMeta:
    model: str
    embed_model: str
    dimensions: int
    built_at: str
    completed_at: str | None = None


@dataclass(frozen=True)
class Held:
    meta: StoreMeta
    profiles: tuple[Profile, ...]


def people_folder(archive_file: str | os.PathLike[str]) -> Path:
    """The folder of the store of one archive, read at each call."""
    import recall_paths

    resolved = str(Path(archive_file).expanduser().resolve())
    name = hashlib.sha256(resolved.encode("utf-8", "surrogatepass")).hexdigest()[:NAME_CHARS]
    return Path(recall_paths.SAVES_DIR) / STORE_LEAF / name


def profile_text(profile: Profile) -> str:
    """The text of a profile that is embedded and searched."""
    points = [f"{point.date}: {point.what}" for point in profile.turning_points]
    return "\n".join([f"Kind: {profile.kind}. Status: {profile.status}.", profile.summary, *points])


def text_hash(text: str, model: str, dimensions: int) -> str:
    material = json.dumps([text, model, dimensions], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()


def _row(position: int, profile: Profile) -> tuple[Any, ...]:
    points = [{"date": point.date, "what": point.what} for point in profile.turning_points]
    return (
        position,
        profile.key,
        json.dumps(list(profile.conversations), ensure_ascii=False),
        profile.label,
        int(profile.is_group),
        json.dumps(profile.figures.as_dict()),
        profile.kind,
        profile.status,
        profile.summary,
        json.dumps(points, ensure_ascii=False),
        profile.input_hash,
        profile.text_hash,
        profile.vector,
    )


def _profile(row: tuple[Any, ...]) -> Profile:
    points = tuple(TurningPoint(str(point["date"]), str(point["what"])) for point in json.loads(row[8]))
    return Profile(
        key=str(row[0]),
        conversations=tuple(str(one) for one in json.loads(row[1])),
        label=str(row[2]),
        is_group=bool(row[3]),
        figures=Figures(**json.loads(row[4])),
        kind=str(row[5]),
        status=str(row[6]),
        summary=str(row[7]),
        turning_points=points,
        input_hash=str(row[9]),
        text_hash=str(row[10]),
        vector=None if row[11] is None else bytes(row[11]),
    )


class PeopleStore:
    """The store in one folder."""

    def __init__(self, folder: str | os.PathLike[str]) -> None:
        self.folder = Path(folder)
        self.path = self.folder / STORE_FILE

    def signature(self) -> tuple[int, int] | None:
        """The time and the size of the file, or None with no file."""
        try:
            found = self.path.stat()
        except OSError:
            return None
        return (found.st_mtime_ns, found.st_size)

    def read(self) -> Held | None:
        """What the store holds, or None with no file. Raises PeopleStoreError."""
        if not self.path.is_file():
            return None
        try:
            with contextlib.closing(sqlite3.connect(f"file:{quote(str(self.path))}?mode=ro", uri=True)) as conn:
                meta = conn.execute(
                    "SELECT format_version, model, embed_model, dimensions, built_at, completed_at FROM meta"
                ).fetchone()
                if meta is None:
                    raise PeopleStoreError("the people index has no facts")
                if int(meta[0]) > FORMAT_VERSION:
                    raise PeopleFormatError("a newer build of the app wrote the people index")
                rows = conn.execute(f"SELECT {_COLUMNS} FROM profiles ORDER BY position").fetchall()
            return Held(StoreMeta(str(meta[1]), str(meta[2]), int(meta[3]), str(meta[4]), meta[5]), tuple(map(_profile, rows)))
        except PeopleStoreError:
            raise
        except (sqlite3.Error, ValueError, TypeError, KeyError) as error:
            raise PeopleStoreError(f"the people index can not be read ({type(error).__name__})") from None

    def write(self, held: Held) -> None:
        """Put a new store in place of the old one, whole. Raises PeopleStoreError; the old file is then kept."""
        work = self.path.with_name(STORE_FILE + NEW_ENDING)
        try:
            self.folder.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
            os.chmod(self.folder, PRIVATE_FOLDER_MODE)
            work.unlink(missing_ok=True)
            os.close(os.open(work, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE))
            self._fill(work, held)
            os.replace(work, self.path)
        except (OSError, sqlite3.Error) as error:
            with contextlib.suppress(OSError):
                work.unlink(missing_ok=True)
            raise PeopleStoreError(f"the people index could not be written ({type(error).__name__})") from None

    @staticmethod
    def _fill(work: Path, held: Held) -> None:
        meta = held.meta
        with contextlib.closing(sqlite3.connect(work)) as conn:
            for statement in _CREATE:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?)",
                (FORMAT_VERSION, meta.model, meta.embed_model, meta.dimensions, meta.built_at, meta.completed_at),
            )
            conn.executemany(
                "INSERT INTO profiles (position, key, conversations, label, is_group, figures, kind, status, summary,"
                " turning_points, input_hash, text_hash, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [_row(position, profile) for position, profile in enumerate(held.profiles)],
            )
            conn.commit()
