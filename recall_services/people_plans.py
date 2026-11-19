"""The plans of the conversations, kept between calls, so a look at the people index reads no message.

To know whether the notes hold every part of a conversation, the parts must
be laid out as a digest lays them out: every message is read, and every part
is counted in tokens. That is slow on a large archive. So the cache keys of
the parts of each conversation are kept in `plans.sqlite3` beside the store,
with a fingerprint of what decides them:

- the conversation as the archive lists it: its id, kind, name, members,
  counts and first and last time;
- the settings of the reading: the prompt version of a part, the model, the
  options, the token counter, and the saved name of each handle of the
  archive (the names and the masks of a transcript).

A conversation whose fingerprint did not change keeps its keys, and is not
planned again. A look then asks the notes for those keys only.

The fingerprint of a profile (`profile_fingerprint`) is what decides its
row: the prompt version of a profile, the model, the label, the counts and
days of its conversations (they decide the figures) and the keys of its
parts (they decide the notes). A row with that fingerprint is current.

The file is a cache: a fault reading it plans again, and a fault writing it
is a warning. It holds conversation ids and keys, no text. It is for its
owner only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recall_services import people_prompt

logger = logging.getLogger(__name__)

PLANS_FILE = "plans.sqlite3"
PRIVATE_FILE_MODE = 0o600
PRIVATE_FOLDER_MODE = 0o700
BUSY_TIMEOUT_S = 10.0
PARTS, NO_TEXT, TOO_LARGE = "parts", "empty", "too_large"
# what a conversation is, as the archive lists it
RECORD_FIELDS = (
    "id", "is_group", "name", "members", "message_count", "sent_count", "first_timestamp", "last_timestamp",
)  # fmt: skip
FIGURE_FIELDS = ("id", "message_count", "sent_count", "first_timestamp", "last_timestamp")
# an upper bound of the text around the notes of a request: the figures, the label line, the frames
AROUND_NOTES_CHARS = 600
PART_HEAD_CHARS = 80
MOMENT_HEAD_CHARS = 20
# fewer characters for one token than English text has (about 4), so the input is not counted short
INPUT_CHARS_PER_TOKEN = 3

_CREATE = (
    "CREATE TABLE IF NOT EXISTS plans (conversation TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,"
    " state TEXT NOT NULL, keys TEXT NOT NULL)"
)


@dataclass(frozen=True)
class ConversationPlan:
    fingerprint: str
    state: str
    keys: tuple[str, ...] = ()


def _hash(material: Any) -> str:
    text = json.dumps(material, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def reading_fingerprint(map_version: str, model: str, options: object, counter: str, saved: Iterable[tuple[str, str]]) -> str:
    """What decides the parts of every conversation, apart from the conversation itself."""
    return _hash([map_version, model, repr(options), counter, sorted(saved)])


def conversation_fingerprint(record: Mapping[str, Any], reading: str) -> str:
    return _hash([reading, [record.get(name) for name in RECORD_FIELDS]])


def profile_fingerprint(model: str, label: str, is_group: bool, records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    """The hash of the inputs of a profile, with no message read."""
    shown = [[record.get(name) for name in FIGURE_FIELDS] for record in records]
    return _hash([people_prompt.PROMPT_VERSION, model, label, is_group, shown, list(keys)])


def input_bound(label: str, parts: Sequence[Any]) -> int:
    """An upper bound of the input tokens of a request, from the lengths of its parts. `parts` are the
    parsed notes (`PeriodDigest`); a moment that the check drops still counts."""
    chars = len(people_prompt.PROFILE_SYSTEM) + len(people_prompt.PROFILE_TASK) + len(people_prompt.NOTES_INTRO)
    chars += AROUND_NOTES_CHARS + len(label)
    for part in parts:
        chars += PART_HEAD_CHARS + len(part.summary)
        chars += sum(MOMENT_HEAD_CHARS + len(moment.title) + len(moment.detail) for moment in part.moments)
    return math.ceil(chars / INPUT_CHARS_PER_TOKEN)


class PlanStore:
    """The plans of one archive, in the folder of its store. It never raises."""

    def __init__(self, folder: str | os.PathLike[str]) -> None:
        self.path = Path(folder) / PLANS_FILE

    def read(self) -> dict[str, ConversationPlan]:
        if not self.path.is_file():
            return {}
        try:
            with contextlib.closing(sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S)) as conn:
                rows = conn.execute("SELECT conversation, fingerprint, state, keys FROM plans").fetchall()
            return {str(row[0]): ConversationPlan(str(row[1]), str(row[2]), tuple(json.loads(row[3]))) for row in rows}
        except (sqlite3.Error, ValueError, TypeError) as error:
            logger.warning("the plans of the people index can not be read (%s); they are made again", type(error).__name__)
            return {}

    def write(self, plans: Mapping[str, ConversationPlan]) -> None:
        if not plans:
            return
        try:
            self.path.parent.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
            os.chmod(self.path.parent, PRIVATE_FOLDER_MODE)
            if not self.path.exists():
                os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, PRIVATE_FILE_MODE))
            with contextlib.closing(sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S)) as conn, conn:
                conn.execute(_CREATE)
                conn.executemany(
                    "INSERT OR REPLACE INTO plans VALUES (?, ?, ?, ?)",
                    [(key, plan.fingerprint, plan.state, json.dumps(list(plan.keys))) for key, plan in plans.items()],
                )
        except (OSError, sqlite3.Error) as error:
            logger.warning("the plans of the people index could not be kept (%s)", type(error).__name__)
