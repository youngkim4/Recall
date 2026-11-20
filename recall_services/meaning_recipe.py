"""The recipe of the segments of this process, and the words of their context line.

`from_env` reads RECALL_SEGMENTS: a comma list of the switches `context`,
`overlap` and `adaptive` (see `meaning_index/recipe.py`). With no value the
recipe is the default. A word that names no switch is left out, and one
warning names it: a typo must not stop the app at its start.

`ContextWords` gives the words after the month in the context line of a
segment, from what Recall has: no model call.

    Context: March 2023; direct chat with Mina; memory: They planned ...
    Context: June 2024; group chat of 4 people: Lake crew
    Context: June 2024; direct chat with a number ending in 0171

Rules:
- A conversation goes by the label of chat (`Names.chat_labeler`): a saved
  name or a title as it is, a mask for each handle that nobody saved. So the
  context line masks a handle as the text of the segment masks it.
- The memory is the summary of the part of the day of the segment, only when
  the memory read that part before (`meaning_notes`). It is masked the same way.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from archive import Archive
from chat_engine.handles import GROUP_MASK
from meaning_index.recipe import SWITCHES, Recipe, SegmentContext, recipe_from_words
from recall_services.meaning_notes import Memory, MemoryNotes
from recall_services.names import Names

logger = logging.getLogger(__name__)

ENV_NAME = "RECALL_SEGMENTS"
DAY_LENGTH = 10


def from_env(environ: Mapping[str, str]) -> Recipe:
    """The recipe that RECALL_SEGMENTS names. Unknown words are left out with one warning."""
    recipe, unknown = recipe_from_words(str(environ.get(ENV_NAME) or ""))
    if unknown:
        logger.warning(
            "%s: left out %s; the switches are %s", ENV_NAME, ", ".join(unknown), ", ".join(SWITCHES)
        )  # fmt: skip
    return recipe


def _who(row: Mapping[str, Any], label: str) -> str:
    """`direct chat with <label>`, or `group chat of <n> people` with the label of the group.

    The people of a group are its members and the owner.
    """
    if not row.get("is_group"):
        return f"direct chat with {label}"
    people = len({str(member) for member in row.get("members") or []}) + 1
    if label.startswith(GROUP_MASK):
        return f"group chat of {people} people{label[len(GROUP_MASK):]}"
    return f"group chat of {people} people: {label}"


class ContextWords:
    """`SegmentContext` for the archive: the kind, the name and the memory of a conversation."""

    def __init__(self, archive: Archive, names: Names, notes: MemoryNotes | None = None) -> None:
        listed = archive.conversations()
        self._labeler = names.chat_labeler(listed)
        self._rows = {str(row["id"]): row for row in listed}
        self._labels: dict[str, str] = {}
        self._memory = Memory(archive, names, self._labeler.redact, notes)

    def _row(self, conversation_id: str) -> Mapping[str, Any]:
        return self._rows.get(conversation_id) or {"id": conversation_id, "is_group": False, "members": []}

    def __call__(self, conversation_id: str, first_timestamp: str) -> str:
        row = self._row(conversation_id)
        if conversation_id not in self._labels:
            self._labels[conversation_id] = _who(row, self._labeler.conversation(row))  # type: ignore[arg-type]
        summary = self._memory.summary(row, first_timestamp[:DAY_LENGTH])
        words = [self._labels[conversation_id], *([f"memory: {summary}"] if summary else [])]
        return "; ".join(words)


def context_for(
    recipe: Recipe, archive: Archive, names: Names, notes: MemoryNotes | None = None
) -> SegmentContext | None:
    """The words of the context line when the recipe has `context`, or None."""
    return ContextWords(archive, names, notes) if recipe.context else None
