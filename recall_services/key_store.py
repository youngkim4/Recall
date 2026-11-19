"""The key that a person gives in the app: `GET`, `POST` and `DELETE /api/key`.

`keys.py` reads the key: the environment wins, else `OPENAI_API_KEY` in the
file `.env` of the data folder. This module writes that line of the file.

- `state` says whether a key is set and where. It never gives the key or any
  part of it.
- `save` checks a key and writes it. Every other line of the file stays as it
  is, byte for byte. The file is written whole to a temporary file in the
  same folder, then put in place with one rename. Its mode is 0600.
- `remove` takes out only the line of the key.
- `same_origin` is the check of the three routes: a request that names an
  origin must come from the page of this server, not from another program on
  another local port.

No sentence, no log line and no error of this module holds the key. A fault
of the disk is told in one sentence, with the cause dropped.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from recall_services import keys

MIN_CHARS = 20
MAX_CHARS = 300
PREFIX = "sk-"
PRIVATE_FILE_MODE = 0o600
_SHAPE = re.compile(r"sk-[A-Za-z0-9_-]+")
# a line of the file that sets the key: `OPENAI_API_KEY=...`, with `export` and spaces allowed, as dotenv reads it
_KEY_LINE = re.compile(rf"\s*(?:export\s+)?{keys.KEY_NAME}\s*=")
# the file is read and written as bytes: a line that is not text is kept as it is
_ENCODING, _ERRORS = "utf-8", "surrogateescape"

NOT_TEXT = "The key must be text."
NOT_A_KEY = f"This is not an API key. A key starts with {PREFIX} and has {MIN_CHARS} to {MAX_CHARS} letters, digits, - or _."
ENVIRONMENT_WINS = (
    f"{keys.KEY_NAME} is set where the server starts, and that key is used. A key saved here would not be used."
)
NOT_WRITTEN = f"The key could not be saved in {keys.KEY_PLACE}. Check the free space on the disk, then try again."
NOT_REMOVED = f"The key could not be removed from {keys.KEY_PLACE}. Check the free space on the disk, then try again."

# one write of the file at a time
_LOCK = threading.Lock()


class KeyRefused(ValueError):
    """A key that is not saved, or a file that is not changed. `status` is the answer to the request."""

    def __init__(self, sentence: str, status: int) -> None:
        super().__init__(sentence)
        self.status = status


def _key_file() -> Path:
    import recall_paths  # read at call time: the data folder can be set by RECALL_DATA_DIR

    return Path(recall_paths.DOTENV_PATH)


def state() -> dict[str, Any]:
    """`present`, `source` (`environment`, `file` or None) and `place`, the sentence that names the file."""
    if os.environ.get(keys.KEY_NAME, "").strip():
        source: str | None = "environment"
    elif keys.api_key():
        source = "file"
    else:
        source = None
    return {"present": source is not None, "source": source, "place": keys.KEY_PLACE}


def checked(value: object) -> str:
    """The key, trimmed. Raises KeyRefused (400) for a value that is no key. The sentence never repeats it."""
    if not isinstance(value, str):
        raise KeyRefused(NOT_TEXT, 400)
    key = value.strip()
    if not MIN_CHARS <= len(key) <= MAX_CHARS or not _SHAPE.fullmatch(key):
        raise KeyRefused(NOT_A_KEY, 400)
    return key


def _lines(path: Path) -> list[str]:
    """The lines of the file, each with its newline. Only a newline ends a line: the other breaks of
    `str.splitlines` (\\x0b, \\x0c, \\x85, \\u2028 and more) stay inside their line, so no text is lost."""
    try:
        text = path.read_bytes().decode(_ENCODING, _ERRORS)
    except FileNotFoundError:
        return []
    parts = text.split("\n")
    return [f"{part}\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _is_key_line(line: str) -> bool:
    return _KEY_LINE.match(line) is not None


def _with_key(lines: list[str], key: str) -> list[str]:
    """The lines with the first line of the key replaced, any later one dropped, or the line added at the end."""
    line = f"{keys.KEY_NAME}={key}\n"
    if not any(_is_key_line(one) for one in lines):
        last = lines[-1] if lines else "\n"
        return [*lines, *([] if last.endswith("\n") else ["\n"]), line]
    first = next(place for place, one in enumerate(lines) if _is_key_line(one))
    return [line if place == first else one for place, one in enumerate(lines) if place == first or not _is_key_line(one)]


def _write(path: Path, lines: list[str]) -> None:
    """Write the file whole: a new temporary file in the same folder, then one rename.

    `mkstemp` makes the file with a random name, mode 0600 from its start, and
    never opens a file or a link that is there already.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    work = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write("".join(lines).encode(_ENCODING, _ERRORS))
            out.flush()
            os.fsync(out.fileno())
        os.chmod(work, PRIVATE_FILE_MODE)
        os.replace(work, path)
    finally:
        if work.exists():
            work.unlink()


def save(value: object) -> dict[str, Any]:
    """Check the key and write it to the file. Gives `state`. Raises KeyRefused: 400, 409 or 500."""
    key = checked(value)
    if os.environ.get(keys.KEY_NAME, "").strip():
        raise KeyRefused(ENVIRONMENT_WINS, 409)
    path = _key_file()
    try:
        with _LOCK:
            _write(path, _with_key(_lines(path), key))
    except OSError:
        # the cause can name the file and the call; it is dropped, so no trace of the request is kept
        raise KeyRefused(NOT_WRITTEN, 500) from None
    return state()


def remove() -> dict[str, Any]:
    """Take the line of the key out of the file. With no file, or no line of the key, nothing is written."""
    path = _key_file()
    try:
        with _LOCK:
            lines = _lines(path)
            if any(_is_key_line(one) for one in lines):
                _write(path, [one for one in lines if not _is_key_line(one)])
    except OSError:
        raise KeyRefused(NOT_REMOVED, 500) from None
    return state()


def same_origin(host: str | None, origin: str | None) -> bool:
    """True when a request names no origin, or names the exact origin of this server: `http://` and its `Host`.

    The Mac app and curl send no origin. A browser sends one with a request
    from a page: only the page of this server passes, not a page of another port.
    """
    named = str(origin or "").strip()
    if not named:
        return True
    seen = str(host or "").strip().lower()
    try:
        parts = urlsplit(named)
    except ValueError:
        return False
    if parts.scheme != "http" or parts.path or parts.query or parts.fragment or "@" in parts.netloc:
        return False
    return bool(seen) and parts.netloc.lower() == seen
