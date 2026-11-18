"""An invented archive for the tests of the index of meanings and of the two chat ports.

Two people, a few short exchanges, all made up. The archive is a real one: an
invented chat.db, pulled with `import_chat_db`. A message id is Apple's guid.

The exchange about the ferry and the supper holds no word of `BOAT_QUERY`.
`FakeEmbedder` reads `boat` as `ferry` and `meal` as `supper`, so only a
search by meaning finds it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from archive import Archive, ArchiveError
from meaning_index import EmbedderFacts, FakeEmbedder, IndexStore
from recall_services import meaning
from recall_services.names import Names
from tests.integration.report_data import MINA_CONTACT, Pulled
from tests.report_engine.synth import MINA, THEO, message

# a word that no log line, no job line and no error may hold
MARK = "zebrafinch"
BOAT_QUERY = "a boat ride and a meal"
FERRY_LINE = "We took the ferry to the island for supper"
FERRY_REPLY = "Best supper of the year"
LATE_FERRY_LINE = "The ferry was late so we skipped supper"
SAME_WORDS = "Happy birthday"
FERRY_DAY = "2023-03-04"
LATE_FERRY_DAY = "2024-06-05"
LAST_TEXT = "2024-06-05 20:03:00"
SEGMENTS = 5
FAKE_FACTS = EmbedderFacts(
    model="fake-embedding-1", dimensions=64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="words"
)
# (conversation, local time, text, written by the owner)
LINES: tuple[tuple[str, str, str, bool], ...] = (
    (MINA, "2023-03-04 20:00:00", FERRY_LINE, False),
    (MINA, "2023-03-04 20:02:00", FERRY_REPLY, True),
    (MINA, "2023-03-10 09:00:00", f"The kettle broke again {MARK}", False),
    (MINA, "2023-03-10 09:01:00", "Buy a new kettle", True),
    (MINA, "2023-04-02 12:00:00", SAME_WORDS, False),
    (THEO, "2023-04-02 12:00:00", SAME_WORDS, False),
    (THEO, "2024-06-05 20:00:00", LATE_FERRY_LINE, False),
    (THEO, LAST_TEXT, "Next time we drive", True),
)


def words(text: str) -> int:
    """The token counter of these tests: one token for each word."""
    return len(text.split())


@dataclass
class Lake:
    """The archive, the source that it is pulled from, and the folder of the index."""

    pulled: Pulled
    archive: Archive
    names: Names
    store: IndexStore

    def add(self, lines: Sequence[tuple[str, str, str, bool]]) -> None:
        """Write more lines into the source and pull again. The archive is opened again."""
        write(self.pulled, lines)
        self.archive.close()
        self.pulled.pull([MINA_CONTACT])
        self.archive = self.pulled.open()

    def close(self) -> None:
        self.archive.close()
        self.pulled.close()


def write(pulled: Pulled, lines: Sequence[tuple[str, str, str, bool]]) -> None:
    for conversation, stamp, text, own in lines:
        pulled.write(conversation, [message(0, stamp, text, sender=conversation, is_from_me=own)])


def build_lake(folder: Path) -> Lake:
    pulled = Pulled(folder / "lake")
    write(pulled, LINES)
    pulled.pull([MINA_CONTACT])
    return Lake(pulled, pulled.open(), Names.from_records([MINA_CONTACT]), IndexStore(folder / "saves" / "meaning"))


def ids_of(archive: Archive, text: str) -> list[str]:
    """The ids of the messages with exactly this text, in every conversation."""
    chats = [chat["id"] for chat in archive.conversations()]
    return [row["id"] for chat in chats for row in archive.iter_messages(chat) if row["text"] == text]


class Failing:
    """An embedder that raises at each request."""

    model, dimensions = FAKE_FACTS.model, FAKE_FACTS.dimensions

    def __init__(self, error: Exception) -> None:
        self._error = error

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise self._error


class Unreadable:
    """An archive that fails when its messages are read."""

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    def conversations(self, **more: Any) -> Any:
        return self._archive.conversations(**more)

    def iter_messages(self, *_args: Any, **_more: Any) -> Any:
        raise ArchiveError(f"the text {MARK} of {MINA}")


def build(lake: Lake, embedder: Any = None, **more: Any) -> dict[str, Any]:
    """Build the index of the lake with a fixed embedder, or with the one that is given."""
    made = embedder or FakeEmbedder()
    settings = {"store": lake.store, "facts": FAKE_FACTS, "count_tokens": words, **more}
    return meaning.build(lake.archive, lake.names, embedder_factory=lambda _model, _dimensions: made, **settings)


def estimate(lake: Lake) -> dict[str, Any]:
    return meaning.estimate(lake.archive, lake.names, store=lake.store, facts=FAKE_FACTS, count_tokens=words)


def failure(lake: Lake, embedder: Any = None, **more: Any) -> meaning.MeaningFailure:
    """The failure of a build that must fail."""
    try:
        build(lake, embedder, **more)
    except meaning.MeaningFailure as told:
        return told
    raise AssertionError("the build did not fail")


def newer_format(store: IndexStore) -> None:
    """Make the folder hold an index that a newer build of the app wrote."""
    store.folder.mkdir(parents=True, exist_ok=True)
    (store.folder / "meta.json").write_text(json.dumps({"format_version": 99}), encoding="utf-8")
