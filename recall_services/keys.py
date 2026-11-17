"""The key of the model provider.

The key is `OPENAI_API_KEY` in the environment of the server, or in the file
`.env` in the data folder of Recall. The Mac app starts the server with no
shell environment, so the file is where the key of the app is.

- The environment wins over the file.
- The file is read at each call: a key that was added is used by the next
  question, with no new start of the server.
- Only the key is taken from the file. Nothing is written to the environment,
  so no other line of the file can change where a request goes.
- No file above the data folder is looked for.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY_NAME = "OPENAI_API_KEY"
# where a person puts the key, as a sentence names it
KEY_PLACE = "the file .env in Recall's data folder"


def _key_file() -> Path:
    import recall_paths  # read at call time: the data folder can be set by RECALL_DATA_DIR

    return Path(recall_paths.DOTENV_PATH)


def _from_file(path: Path) -> str:
    try:
        from dotenv import dotenv_values

        found = dotenv_values(path).get(KEY_NAME) if path.is_file() else None
    except (OSError, ValueError):  # a file that can not be read, or that is not text
        return ""
    return found.strip() if isinstance(found, str) else ""


def api_key() -> str:
    """The key, or "" when none is set."""
    return os.environ.get(KEY_NAME, "").strip() or _from_file(_key_file())
