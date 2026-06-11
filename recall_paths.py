"""Single source of truth for where Recall reads and writes user data.

Dev mode (repo checkout): DATA_DIR == repo root, bit-identical to the old
layout, so tests and the founder's archive keep working untouched. Bundled
mode (packaged .app, RECALL_BUNDLED=1 or a frozen interpreter): everything
mutable lives under ~/Library/Application Support/Recall so the app bundle
stays read-only and signable. RECALL_DATA_DIR overrides both.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BUNDLED = bool(getattr(sys, "frozen", False)) or os.environ.get("RECALL_BUNDLED") == "1"


def _data_dir() -> Path:
    env = str(os.environ.get("RECALL_DATA_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if BUNDLED:
        return (Path.home() / "Library" / "Application Support" / "Recall").resolve()
    return ROOT


DATA_DIR = _data_dir()
SAVES_DIR = DATA_DIR / "saves"
OUT_DIR = DATA_DIR / "out"
DEFAULT_MESSAGES_CSV = DATA_DIR / "messages.csv"
SNAPSHOT_DB = DATA_DIR / "chat.db"
DOTENV_PATH = DATA_DIR / ".env"


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, SAVES_DIR, OUT_DIR):
        path.mkdir(parents=True, exist_ok=True)
