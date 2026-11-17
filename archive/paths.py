"""Where the archive lives when the caller gives no path."""

from __future__ import annotations

from pathlib import Path

ARCHIVE_FILE_NAME = "archive.sqlite3"


def default_archive_path() -> Path:
    """`saves/archive.sqlite3` in Recall's data folder (see recall_paths)."""
    import recall_paths  # read at call time: the data folder can be set by RECALL_DATA_DIR

    return Path(recall_paths.SAVES_DIR) / ARCHIVE_FILE_NAME
