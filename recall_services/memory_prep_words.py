"""The words of the `memory` job: its failures, its lines and its progress.

A line and a progress label hold counts only: no text of a message, no name,
no handle, no id and no label of a part. A failure is one plain sentence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from recall_services import keys
from recall_services.report_failures import ReportFailure

logger = logging.getLogger(__name__)

Listener = Callable[[dict[str, Any]], None]

AGAIN = "then read the memory again."
KEPT = "What was read is kept."
# failures that each conversation would have: the job stops at the first one
FATAL = frozenset({"no_key", "key_rejected", "unknown_model", "rate_limit", "offline", "cache_unreadable"})
STEPS = {
    "look": (1, "Looking at your conversations"),
    "read": (2, "Reading your conversations"),
}
MODEL_NOT_RUN = "The model of the memory can not be run. Nothing was read."
ARCHIVE_UNREADABLE = f"The message archive could not be read. Update the archive, {AGAIN}"
STOPPED = f"The reading was stopped. {KEPT}"


class MemoryFailure(ReportFailure):
    """A `memory` job that stopped. The message is the sentence for the person; `usage` is what it spent."""


def sentence_of(code: str, provider: str, model: str) -> str:
    """One plain sentence for a failure that stops the job."""
    sentences = {
        "no_key": f"No API key is set. Add {keys.KEY_NAME} to {keys.KEY_PLACE}, {AGAIN}",
        "key_rejected": f"{provider} rejected the API key. Check {keys.KEY_NAME} in {keys.KEY_PLACE}, {AGAIN} {KEPT}",
        "unknown_model": f"{provider} does not know the model {model}. {KEPT}",
        "rate_limit": f"{provider} is limiting requests, or the account has no credit left. Wait a moment, {AGAIN} {KEPT}",
        "offline": f"{provider} could not be reached. Check the internet connection, {AGAIN} {KEPT}",
        "cache_unreadable": f"What earlier runs read could not be read or kept. {KEPT}",
        "archive_error": ARCHIVE_UNREADABLE,
        "cancelled": STOPPED,
    }
    return sentences.get(code, f"The memory could not be read. {KEPT}")


def count(number: int, word: str) -> str:
    return f"{number:,} {word}{'' if number == 1 else 's'}"


class Teller:
    """Turns the steps of the job into updates for a listener. A listener that fails does not stop the job."""

    def __init__(self, listener: Listener | None) -> None:
        self._listener = listener
        self._warned = False

    def tell(self, *, step: str | None = None, done: int = 0, total: int = 0, log: str = "") -> None:
        if self._listener is None:
            return
        update: dict[str, Any] = {}
        if step is not None:
            number, label = STEPS[step]
            update["progress"] = {"step": number, "of": len(STEPS), "label": label, "done": done, "total": total}
        if log:
            update["log"] = log
        try:
            self._listener(update)
        except Exception as error:  # noqa: BLE001 - the reading is worth more than its progress line
            if not self._warned:
                logger.warning("the progress listener of the memory job failed (%s)", type(error).__name__)
                self._warned = True


def looking(conversations: int, min_messages: int) -> str:
    return f"Looking at {count(conversations, 'conversation')} with {min_messages:,} messages or more"


def to_read(parts: int, conversations: int, read: int) -> str:
    return f"{count(parts, 'part')} of {count(conversations, 'conversation')} are not read yet, {read:,} are read"


def one_read(built: int, cached: int, failed: int, place: int, total: int) -> str:
    return (
        f"A conversation is read: {count(built, 'part')} read now, {cached:,} from earlier runs, "
        f"{failed:,} not read ({place} of {total})"
    )


def one_failed(place: int, total: int) -> str:
    return f"A conversation could not be read ({place} of {total})"


def all_read(built: int, cached: int, failed: int) -> str:
    return f"The memory is read: {count(built, 'part')} read now, {cached:,} from earlier runs, {failed:,} not read."
