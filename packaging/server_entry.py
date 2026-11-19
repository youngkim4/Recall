"""PyInstaller entry point for the bundled Recall server.

Kept separate from ui_server so the frozen build has a stable, minimal
entry that never grows dev-only side effects.
"""

import multiprocessing
import os
import sys
from pathlib import Path

# the vocabularies of tiktoken that the spec ships (recall_server.spec)
VOCABULARY_FOLDER = "tiktoken_cache"


def use_bundled_vocabulary() -> None:
    """Point tiktoken at the vocabularies inside the app: a first start with no network fetches nothing.
    A folder that was set before is kept, and outside the app nothing is set."""
    root = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or not root:
        return
    folder = Path(root) / VOCABULARY_FOLDER
    if folder.is_dir():
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(folder))


def run() -> None:
    # frozen macOS builds re-exec the binary for spawned children; without
    # this a stray multiprocessing use would fork-bomb the app
    multiprocessing.freeze_support()
    use_bundled_vocabulary()
    import ui_server

    ui_server.main()


if __name__ == "__main__":
    run()
