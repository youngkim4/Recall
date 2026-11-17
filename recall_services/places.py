"""Which archive stands for the archive file that the app names.

The app names one file in every request (`messagesPath`). The archive itself
is a SQLite file:

- Nothing named, the default export (`messages.csv` in the data folder) or the
  default archive: the archive at `archive.default_archive_path()`. Its export
  is the default export.
- A file whose name ends in `.sqlite3`: that file is the archive. It has no
  export.
- Any other file is an export of an earlier build. Its archive stands beside
  it, with the same name and the ending `.sqlite3`.

No code here lists a folder, and no file is taken for an archive because it
is found: a place is worked out from the name that the app gave.

`state_for` gives one `ArchiveState` for each archive file, so every request
on one archive shares one read handle and one import lock.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path

from archive import default_archive_path
from recall_services import archive_state
from recall_services.archive_state import ArchiveState

ARCHIVE_SUFFIX = ".sqlite3"
# how many archives beside the default one keep an open handle
KEPT_STATES = 4

_lock = threading.Lock()
_states: OrderedDict[str, ArchiveState] = OrderedDict()


def _default_export() -> Path:
    import recall_paths  # read at call time: the data folder can be set by RECALL_DATA_DIR

    return Path(recall_paths.DEFAULT_MESSAGES_CSV)


def _whole(path: Path) -> Path:
    """The path with its links followed, or as it is when that can not be done."""
    try:
        return path.expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        return path


def _named(messages_path: str | os.PathLike[str] | None) -> Path | None:
    """The file that the app named, or None for the default archive."""
    if messages_path is None or not str(messages_path).strip():
        return None
    named = _whole(Path(messages_path))
    if named in (_whole(_default_export()), _whole(default_archive_path())):
        return None
    return named


def archive_path_for(messages_path: str | os.PathLike[str] | None) -> Path:
    """The archive file that stands for the file that the app named."""
    named = _named(messages_path)
    if named is None:
        return default_archive_path()
    return named if named.suffix == ARCHIVE_SUFFIX else named.with_suffix(ARCHIVE_SUFFIX)


def export_for(messages_path: str | os.PathLike[str] | None) -> Path | None:
    """The export that the archive of this place is first loaded from, or None when the place has none."""
    named = _named(messages_path)
    if named is None:
        return _default_export()
    return None if named.suffix == ARCHIVE_SUFFIX else named


def state_for(messages_path: str | os.PathLike[str] | None) -> ArchiveState:
    """The state of the archive that stands for the file that the app named."""
    named = _named(messages_path)
    if named is None:
        return archive_state.shared()
    place = str(archive_path_for(named))
    with _lock:
        held = _states.get(place)
        if held is None:
            held = _states[place] = ArchiveState(place)
        _states.move_to_end(place)
        dropped = [_states.popitem(last=False)[1] for _ in range(max(0, len(_states) - KEPT_STATES))]
    for old in dropped:
        old.close()
    return held


def forget() -> None:
    """Close every state but the default one. For tests, and for the end of the server."""
    with _lock:
        dropped = list(_states.values())
        _states.clear()
    for old in dropped:
        old.close()
