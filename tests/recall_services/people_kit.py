"""An invented archive for the tests of the people index.

Mina is saved, and she writes from a phone and from an email address: one
person with two threads. Theo is not saved. The three of them have a group.
Ravi sent one photo and no text. All of it is made up.

The figures of each person are counted by hand in the comments. The times
are local times of the test zone (America/Los_Angeles); no gap crosses a
change of the clock.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from archive import Archive
from meaning_index import EmbedderFacts, FakeEmbedder
from recall_services import digests, memory_prep, reports
from recall_services.names import Names
from report_engine.fake_llm import FakeLLM
from tests.integration.report_data import Pulled, moment
from tests.recall_services.report_world import OPTIONS
from tests.report_engine.synth import WordCounter, message

MINA = "+12025550143"
MINA_MAIL = "mina@example.com"
THEO = "theo@example.com"
RAVI = "+12025550171"
GROUP = "chat000000000000000042"
GROUP_TITLE = "Harbor Walkers"
MINA_RECORD = {"name": "Mina", "phones": ["(202) 555-0143"], "emails": [MINA_MAIL]}
# a word that no log line, no job line and no error may hold
MARK = "zebrafinch"
FAKE_FACTS = EmbedderFacts(
    model="fake-embedding-1", dimensions=64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="words"
)

# (conversation, local time, text, written by the owner, sender of a received group line)
LINES: tuple[tuple[str, str, str, bool, str], ...] = (
    (MINA, "2023-01-02 09:00:00", "Morning, coffee later?", True, ""),
    (MINA, "2023-01-02 09:05:00", "Yes, at ten", False, ""),
    (MINA_MAIL, "2023-01-05 20:00:00", "Sending the photos by mail", False, ""),
    (MINA, "2023-01-05 21:00:00", "Did you see the lantern festival", False, ""),
    (MINA, "2023-01-05 21:10:00", "Not yet", True, ""),
    (MINA, "2023-02-14 08:00:00", "Happy valentines", True, ""),
    (MINA_MAIL, "2023-02-20 10:00:00", "Lunch on Friday?", False, ""),
    (MINA_MAIL, "2023-02-20 10:30:00", "Sure", True, ""),
    (GROUP, "2023-06-10 18:00:00", "Walk at six?", False, MINA),
    (GROUP, "2023-06-10 18:05:00", "In", False, THEO),
    (GROUP, "2023-06-10 18:06:00", "Me too", True, ""),
    (THEO, "2024-03-01 12:00:00", f"Is the boat still for sale {MARK}", False, ""),
    (THEO, "2024-03-01 12:30:00", "Yes", True, ""),
    (THEO, "2024-03-03 09:00:00", "Still interested?", True, ""),
)

# Mina, both threads: 8 messages, 4 sent. Days 01-02 (you first), 01-05 (Mina first,
# by mail at 20:00), 02-14 (you), 02-20 (Mina): you first on 2 of 4. January holds 5.
# The longest gap: 2023-01-05 21:10 to 2023-02-14 08:00, 39 days 10 h 50 min.
MINA_FIGURES = {
    "first_day": "2023-01-02",
    "last_day": "2023-02-20",
    "messages": 8,
    "sent": 4,
    "received": 4,
    "active_days": 4,
    "longest_gap_days": 39.45,
    "busiest_month": "2023-01",
    "wrote_first_share": 0.5,
}
# Theo: 03-01 (Theo first), 03-03 (you); the gap 03-01 12:30 to 03-03 09:00 is 1 day 20.5 h.
THEO_FIGURES = {
    "first_day": "2024-03-01",
    "last_day": "2024-03-03",
    "messages": 3,
    "sent": 2,
    "received": 1,
    "active_days": 2,
    "longest_gap_days": 1.85,
    "busiest_month": "2024-03",
    "wrote_first_share": 0.5,
}
# the group: one day, Mina first, six minutes from the first line to the last
GROUP_FIGURES = {
    "first_day": "2023-06-10",
    "last_day": "2023-06-10",
    "messages": 3,
    "sent": 1,
    "received": 2,
    "active_days": 1,
    "longest_gap_days": 0.0,
    "busiest_month": "2023-06",
    "wrote_first_share": 0.0,
}


def write(pulled: Pulled, lines: Sequence[tuple[str, str, str, bool, str]]) -> None:
    for conversation, stamp, text, own, sender in lines:
        pulled.write(conversation, [message(0, stamp, text, sender=sender or conversation, is_from_me=own)])


@dataclass
class Town:
    """The archive, its source, the names and the places of the notes and the store."""

    pulled: Pulled
    archive: Archive
    names: Names
    folder: Path

    @property
    def places(self) -> reports.ReportPlaces:
        return reports.ReportPlaces(
            out_dir=self.folder / "out",
            store_dir=self.folder / "saves" / "reports",
            cache_path=self.folder / "saves" / "cache.sqlite3",
        )

    def notes(self) -> dict[str, Any]:
        """What a reading of the notes gets in these tests: the places, the options and the counter."""
        return {"places": self.places, "options": OPTIONS, "counter": WordCounter()}

    def read_memory(self, conversations: Sequence[str] | None = None, llm: FakeLLM | None = None) -> FakeLLM:
        """Read the digests of these conversations (all with text by default) with the fake model."""
        model = llm or FakeLLM()
        chosen = conversations if conversations is not None else (MINA, MINA_MAIL, THEO, GROUP)
        for conversation in chosen:
            digests.build(
                conversation,
                model=memory_prep.model(),
                archive=self.archive,
                names=self.names,
                llm_factory=lambda _info: model,
                **self.notes(),
            )
        return model

    def add(self, lines: Sequence[tuple[str, str, str, bool, str]]) -> None:
        """Write more lines into the source and pull again. The archive is opened again."""
        write(self.pulled, lines)
        self.archive.close()
        self.pulled.pull([MINA_RECORD])
        self.archive = self.pulled.open()

    def close(self) -> None:
        self.archive.close()
        self.pulled.close()


def build_town(folder: Path) -> Town:
    pulled = Pulled(folder / "town")
    pulled.chat(GROUP, members=(MINA, THEO), title=GROUP_TITLE)
    write(pulled, LINES)
    photo = pulled.builder.message(
        pulled.chat(RAVI), when=moment("2023-08-01 10:00:00"), text=None, handle=pulled.handle(RAVI)
    )
    pulled.builder.attachment(photo)
    pulled.pull([MINA_RECORD])
    return Town(pulled, pulled.open(), Names.from_records([MINA_RECORD]), folder)


@pytest.fixture
def town(tmp_path: Path) -> Iterator[Town]:
    """The invented archive of these tests, its own for each test."""
    made = build_town(tmp_path)
    try:
        yield made
    finally:
        made.close()


def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=FAKE_FACTS.dimensions, model=FAKE_FACTS.model)


def settings(town: Town, llm: Any = None, embedder: Any = None, **more: Any) -> dict[str, Any]:
    """The keywords of a build, an estimate or a search in these tests."""
    from recall_services.people_store import PeopleStore

    model = llm if llm is not None else FakeLLM(lambda _call: profile_reply())
    made = embedder if embedder is not None else fake_embedder()
    return {
        **town.notes(),
        "llm_factory": lambda _info: model,
        "embedder_factory": lambda _model, _dimensions: made,
        "facts": FAKE_FACTS,
        "count_tokens": lambda text: len(text.split()),
        "store": PeopleStore(town.folder / "saves" / "people" / "town"),
        **more,
    }


def profile_reply(**fields: Any) -> dict[str, Any]:
    """A reply of the profile model in the right shape."""
    reply: dict[str, Any] = {
        "kind": "friend",
        "status": "ongoing",
        "summary": "You and this person kept in touch.",
        "turning_points": [],
    }
    return {**reply, **fields}


def by_key(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["key"]): row for row in rows}


@contextmanager
def opened(path: Path) -> Iterator[sqlite3.Connection]:
    """A connection to a file of a test, committed and closed at the end of the block."""
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()
