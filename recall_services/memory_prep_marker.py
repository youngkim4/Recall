"""The marker of a `memory` job that completed, one for each archive.

The server writes it when a `memory` job ends completed. A question, a report
and a `digest` job never write it. After an import the server reads the rest
of the memory by itself only when the marker of that archive is there: the
first reading stays the choice of the reader.

The marker is `memory/<name>.json` in the saves folder, read at each call.
The name is the first 16 hex digits of the SHA-256 of the resolved path of the
archive file, as for the index of meanings. The file holds the time and the
least number of messages of a conversation that the reading read
(`minMessages`), and only its owner reads it. A first run reads every
conversation (1): the reading after an import keeps that threshold. A marker
of an earlier build has no threshold (`threshold` gives None).

The threshold saved is the lower of the one there and the new one: a reading
with the threshold of the app does not take back what a first run read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MARKER_LEAF = "memory"
NAME_CHARS = 16
PRIVATE_FILE_MODE = 0o600
PRIVATE_FOLDER_MODE = 0o700


def marker_path(archive_file: str | os.PathLike[str]) -> Path:
    """Where the marker of one archive is kept."""
    import recall_paths

    resolved = str(Path(archive_file).expanduser().resolve())
    name = hashlib.sha256(resolved.encode("utf-8", "surrogatepass")).hexdigest()[:NAME_CHARS]
    return Path(recall_paths.SAVES_DIR) / MARKER_LEAF / f"{name}.json"


def completed_before(archive_file: str | os.PathLike[str]) -> bool:
    """True when a `memory` job of this archive completed before."""
    try:
        return marker_path(archive_file).is_file()
    except OSError:
        return False


def threshold(archive_file: str | os.PathLike[str]) -> int | None:
    """The least number of messages that the readings of this archive read, or None: no marker, a marker
    of an earlier build, or one that can not be read."""
    try:
        saved = json.loads(marker_path(archive_file).read_text(encoding="utf-8")).get("minMessages")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(saved, bool) or not isinstance(saved, int) or saved < 1:
        return None
    return saved


def mark_completed(archive_file: str | os.PathLike[str], min_messages: int | None = None) -> None:
    """Write the marker of this archive, with the threshold of the reading that completed (`min_messages`).

    The lower of that and the threshold already saved is kept. A marker that
    can not be written is a warning, never a failure.
    """
    path = marker_path(archive_file)
    before = threshold(archive_file) if path.is_file() else None
    kept = min((one for one in (before, min_messages) if one is not None), default=None)
    text = json.dumps({"completedAt": datetime.now(UTC).isoformat(timespec="seconds"), "minMessages": kept})
    work = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
        handle = os.open(work, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(work, path)
    except OSError as error:
        logger.warning("the marker of a completed memory job could not be written (%s)", type(error).__name__)
