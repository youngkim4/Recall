"""Small file steps that more than one service needs. Every file here is private to its owner."""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterable
from pathlib import Path

PRIVATE_FILE_MODE = 0o600
PRIVATE_FOLDER_MODE = 0o700
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


def fsync_folder(folder: Path) -> None:
    """Make a rename in this folder last. Not every file system can sync a folder: then nothing is done."""
    try:
        handle = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def write_text_atomic(path: Path, text: str) -> None:
    """Write a text file in one step. A reader sees the old file or the new one, never a part."""
    path.parent.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    handle = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    fsync_folder(path.parent)


def remove_files(paths: Iterable[Path]) -> None:
    """Remove each file that is there. A file that can not be removed raises OSError."""
    for path in paths:
        path.unlink(missing_ok=True)


def sidecars(database: Path) -> list[Path]:
    """The files that SQLite keeps beside a database and finds by its name."""
    return [database.with_name(database.name + suffix) for suffix in SQLITE_SIDECARS]
