"""The `memory` job reads several conversations at once: N parts in flight, and about N times faster."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from recall_services import memory_prep, reports
from recall_services.names import Names
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.integration.report_data import Pulled
from tests.recall_services.report_world import OPTIONS
from tests.report_engine.support import runtime
from tests.report_engine.synth import WordCounter, conversation

TALKS = 8
WAIT = 10.0


def handle(number: int) -> str:
    return f"+1202555{300 + number:04d}"


class Many:
    """An archive of eight conversations of three months each, with other words in each one."""

    def __init__(self, folder: Path) -> None:
        self.pulled = Pulled(folder / "world")
        for number in range(TALKS):
            others = (handle(number),)
            self.pulled.write(handle(number), conversation("2023-01-01", days=90, per_day=6, words=8, others=others, seed=300 + number))
        self.pulled.pull([])
        self.archive = self.pulled.open()
        self.folder = folder

    def more(self, llm: Any, cache: str) -> dict[str, Any]:
        places = reports.ReportPlaces(out_dir=self.folder / "out", store_dir=self.folder / "saves", cache_path=self.folder / cache)
        return {
            "archive": self.archive, "names": Names.from_records([]), "places": places, "options": OPTIONS,
            "counter": WordCounter(), "llm_factory": lambda _info: llm, "runtime": runtime(), "min_messages": 100,
        }  # fmt: skip

    def close(self) -> None:
        self.archive.close()
        self.pulled.close()


@pytest.fixture
def many(tmp_path: Path) -> Iterator[Many]:
    made = Many(tmp_path)
    try:
        yield made
    finally:
        made.close()


class Inside:
    """A fake model that counts the calls inside it at once, and holds each one for `pause` seconds."""

    def __init__(self, pause: float = 0.0, barrier: threading.Barrier | None = None) -> None:
        self.lock = threading.Lock()
        self.inside = 0
        self.most = 0
        self.pause = pause
        self.barrier = barrier
        self.llm = FakeLLM(self._answer)

    def _answer(self, call: FakeCall) -> Any:
        with self.lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
            first = len(self.llm.calls) <= (self.barrier.parties if self.barrier else 0)
        try:
            if first and self.barrier is not None:
                self.barrier.wait()
            time.sleep(self.pause)
        finally:
            with self.lock:
                self.inside -= 1
        return None


def test_the_job_has_n_parts_in_flight_at_once(many: Many) -> None:
    """The first four calls go on only when four are inside together."""
    inside = Inside(barrier=threading.Barrier(memory_prep.PREPARE_PARALLEL, timeout=WAIT))
    found = memory_prep.run(**many.more(inside.llm, "four.sqlite"))
    assert not inside.barrier.broken  # type: ignore[union-attr]
    assert inside.most == memory_prep.PREPARE_PARALLEL
    assert (found["conversations"], found["conversationsRead"], found["periodsFailed"]) == (TALKS, TALKS, 0)
    assert found["periodsBuilt"] == len(inside.llm.calls) >= TALKS


def test_the_wall_time_of_the_job_goes_down_by_about_n(many: Many) -> None:
    """24 parts of 0.1 s: about 2.4 s one at a time. Four at a time, the parts take 0.6 s at best.
    The plan of each conversation is the same work in both runs, so the gain is a little under four."""
    pause = 0.1
    started = time.monotonic()
    alone = memory_prep.run(parallel=1, **many.more(Inside(pause).llm, "alone.sqlite"))
    one_by_one = time.monotonic() - started
    started = time.monotonic()
    four = memory_prep.run(**many.more(Inside(pause).llm, "four.sqlite"))
    at_once = time.monotonic() - started
    counts = [key for key in alone if key not in ("usage", "usd")]
    assert {key: four[key] for key in counts} == {key: alone[key] for key in counts}
    assert four["usd"] == pytest.approx(alone["usd"])
    parts = alone["periodsBuilt"]
    assert parts >= TALKS * 2 and one_by_one >= parts * pause
    assert one_by_one / at_once >= 2.5, (parts, one_by_one, at_once)
