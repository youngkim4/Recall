"""The steps of an index build as a job shows them: a progress line for each step, and a log line.

No line holds text of a message, a name, a handle or an id: the lines hold counts only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from meaning_index import BuildProgress

logger = logging.getLogger(__name__)

Listener = Callable[[dict[str, Any]], None]

# the three steps of a build, as a job shows them
STEPS = {
    "read": (1, "Reading your conversations"),
    "embed": (2, "Reading the meaning of your messages"),
    "write": (3, "Saving the index"),
}


class Teller:
    """Turns the steps of a build into updates for a listener. A listener that fails does not stop the build."""

    def __init__(self, listener: Listener | None) -> None:
        self._listener = listener
        self._warned = False

    def tell(self, update: dict[str, Any]) -> None:
        if self._listener is None:
            return
        try:
            self._listener(update)
        except Exception as error:  # noqa: BLE001 - the build is worth more than its progress line
            if not self._warned:
                logger.warning("the progress listener of an index build failed (%s)", type(error).__name__)
                self._warned = True

    def step(self, event: BuildProgress) -> None:
        number, label = STEPS[event.phase]
        progress = {"step": number, "of": len(STEPS), "label": label, "done": event.done, "total": event.total}
        update: dict[str, Any] = {"progress": progress}
        if event.done == 0:
            update["log"] = _started(event)
        self.tell(update)


def count(number: int, word: str) -> str:
    return f"{number:,} {word}{'' if number == 1 else 's'}"


def _started(event: BuildProgress) -> str:
    """The line of a step that starts. An exchange is a segment of the index."""
    if event.phase == "read":
        return f"Reading {count(event.total, 'conversation')}"
    if event.phase == "embed":
        return f"Reading the meaning of {count(event.total, 'exchange')}" if event.total else "No new exchange to read"
    return f"Saving {count(event.total, 'exchange')}"
