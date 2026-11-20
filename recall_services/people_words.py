"""The words of the people index: its failures, the lines and the progress of the `people` job.

A line and a progress label hold counts only: no text of a message, no
name, no handle, no id and no label. A failure is one plain sentence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from recall_services import keys
from recall_services.models import SOME_PROVIDER
from report_engine import LLMError

logger = logging.getLogger(__name__)

Listener = Callable[[dict[str, Any]], None]

AGAIN = "then build the people index again."
KEPT = "The profiles that were written are kept."
# failures that each profile would have: the build stops at the first one
FATAL = frozenset({"no_key", "key_rejected", "unknown_model", "rate_limit", "offline"})
STEPS = {
    "look": (1, "Looking at the people in your conversations"),
    "write": (2, "Writing a profile of each person"),
    "save": (3, "Saving the people index"),
}
_UNAUTHORIZED, _FORBIDDEN, _NOT_FOUND, _TOO_MANY = 401, 403, 404, 429


class PeopleFailure(Exception):
    """A build, an estimate or a status that gave no result. The message is the sentence for the person.

    `code` is a fixed word for the client. `usage` is what a build spent on
    the model until it stopped, when it spent anything.
    """

    def __init__(self, code: str, sentence: str, usage: dict[str, Any] | None = None) -> None:
        super().__init__(sentence)
        self.code = code
        self.usage = usage


def sentence_of(code: str, provider: str = SOME_PROVIDER, model: str = "") -> str:
    """One plain sentence for a failure."""
    sentences = {
        "no_key": f"{keys.NO_KEY}, {AGAIN}",
        "key_rejected": f"{provider} refused the API key. {keys.CHECK_KEY}, {AGAIN} {KEPT}",
        "unknown_model": f"{provider} does not know the model that writes the profiles. {KEPT}",
        "rate_limit": (
            f"{provider} is limiting requests, or the account has no credit left. Wait a moment, {AGAIN} {KEPT}"
        ),
        "offline": f"{provider} could not be reached. Check the internet connection, {AGAIN} {KEPT}",
        "model_not_run": "The model of the people index can not be run. Nothing was written.",
        "archive_error": f"The message archive could not be read. Update the archive, {AGAIN}",
        "cache_unreadable": "What earlier runs read could not be read. Nothing was written.",
        "store_error": f"The people index could not be read or written. Check the free space on the disk, {AGAIN}",
        "store_newer": "A newer version of Recall made the people index. Update Recall, then try again.",
        "embed_failed": f"The profiles could not be made searchable. {KEPT} Build the people index again.",
        "cancelled": f"The build was stopped. {KEPT}",
    }
    return sentences.get(code, f"The people index could not be built. {KEPT}")


def failure(
    code: str, usage: dict[str, Any] | None = None, *, provider: str = SOME_PROVIDER, model: str = ""
) -> PeopleFailure:
    return PeopleFailure(code, sentence_of(code, provider, model), usage)


def code_of(error: LLMError) -> str:
    """The code of a failed call of the profile model. Only the codes in FATAL stop the build."""
    status, said = error.status_code, str(error)
    if status in (_UNAUTHORIZED, _FORBIDDEN):
        return "key_rejected"
    if status == _NOT_FOUND:
        return "unknown_model"
    if status == _TOO_MANY:
        return "rate_limit"
    if status is None and "Connection" in said:
        return "offline"
    if status is None and "Timeout" in said:
        return "timeout"
    return "model_error"


def count(number: int, word: str) -> str:
    return f"{number:,} {word}{'' if number == 1 else 's'}"


class Teller:
    """Turns the steps of a build into updates for a listener. A listener that fails does not stop the build."""

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
        except Exception as error:  # noqa: BLE001 - the build is worth more than its progress line
            if not self._warned:
                logger.warning("the progress listener of the people job failed (%s)", type(error).__name__)
                self._warned = True


def looking(people: int) -> str:
    return f"Looking at {people:,} people and groups"


def to_write(new: int, kept: int, not_read: int) -> str:
    return (
        f"{count(new, 'profile')} to write, {kept:,} kept as they are, "
        f"{not_read:,} left out because their memory is not read"
    )


def all_written(result: dict[str, Any]) -> str:
    return (
        f"The people index is saved: {count(int(result['built']), 'profile')} written now, {int(result['kept']):,} kept, "
        f"{int(result['failed']):,} failed, {int(result['notRead']):,} left out, {int(result['removed']):,} removed."
    )
