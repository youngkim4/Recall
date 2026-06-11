"""PyInstaller entry point for the bundled Recall server.

Kept separate from ui_server so the frozen build has a stable, minimal
entry that never grows dev-only side effects.
"""

import multiprocessing

import ui_server


def run() -> None:
    # frozen macOS builds re-exec the binary for spawned children; without
    # this a stray multiprocessing use would fork-bomb the app
    multiprocessing.freeze_support()
    ui_server.main()


if __name__ == "__main__":
    run()
