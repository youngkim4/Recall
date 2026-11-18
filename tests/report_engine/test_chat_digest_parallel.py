"""Digests of several conversations read at the same time through one `CallPool`.

The parts of one conversation stay one at a time: the prompt of a part holds
the notes of the part before it. Conversations do not wait for each other.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import PeriodCache
from report_engine.call_pool import CallPool
from report_engine.chat_digest import ConversationDigest, DigestTooLarge
from report_engine.control import CancelToken, ProgressEvent
from report_engine.errors import ReportCancelled
from report_engine.fake_llm import FakeCall, FakeLLM
from report_engine.parallel import run_in_parallel

from .chat_digest_support import cache_rows, digest, fail_month, notes_seen
from .support import LARGE_LABELS
from .synth import conversation

WAIT = 10.0
TALKS = 6


def talk(number: int) -> list[Any]:
    """Four months of chatter, other words in each conversation: four parts of their own."""
    return conversation("2023-01-01", days=120, per_day=6, words=8, seed=100 + number)


class Inside:
    """A model port that every conversation shares. It counts the calls that are
    inside at once, for all and for each conversation, and can hold each call."""

    def __init__(self, pause: float = 0.0, barrier: threading.Barrier | None = None) -> None:
        self.lock = threading.Lock()
        self.pause = pause
        self.barrier = barrier
        self.inside = 0
        self.most = 0
        self.by_talk: dict[str, int] = {}
        self.most_by_talk: dict[str, int] = {}

    def llm(self, name: str) -> FakeLLM:
        return FakeLLM(lambda call: self._answer(name, call))

    def _enter(self, name: str) -> None:
        with self.lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
            self.by_talk[name] = self.by_talk.get(name, 0) + 1
            self.most_by_talk[name] = max(self.most_by_talk.get(name, 0), self.by_talk[name])

    def _answer(self, name: str, call: FakeCall) -> Any:
        self._enter(name)
        try:
            if self.barrier is not None and not self.barrier.broken:
                try:
                    self.barrier.wait()
                except threading.BrokenBarrierError:
                    pass
            time.sleep(self.pause)
        finally:
            with self.lock:
                self.inside -= 1
                self.by_talk[name] -= 1
        return None


def tasks(
    names: list[str], inside: Inside, cache_path: Path, pool: CallPool | None, **more: Any
) -> list[Callable[[], ConversationDigest]]:
    def one(name: str, number: int) -> Callable[[], ConversationDigest]:
        def work() -> ConversationDigest:
            kept = {"pool": pool} if pool is not None else {}
            return digest(
                talk(number), llm=inside.llm(name), cache=PeriodCache(cache_path),
                conversation_id=name, **kept, **more,
            )  # fmt: skip

        return work

    return [one(name, number) for number, name in enumerate(names)]


def names(count: int = TALKS) -> list[str]:
    return [f"talk-{number}" for number in range(count)]


def summaries(found: ConversationDigest) -> list[str]:
    return [period.summary for period in found.periods]


# --- N parts in flight, and the same digests as one at a time ------------------------


def test_n_parts_of_several_conversations_are_in_flight_at_once(cache_path: Path) -> None:
    """The barrier lets the calls go only when three are inside together."""
    inside = Inside(barrier=threading.Barrier(3, timeout=WAIT))
    pool = CallPool(max_parallel=3, tokens_per_minute=None)
    found = run_in_parallel(tasks(names(), inside, cache_path, pool), max_parallel=3)
    assert all(outcome.error is None for outcome in found)
    assert not inside.barrier.broken  # type: ignore[union-attr]
    assert inside.most == pool.most_in_flight == 3
    assert pool.sent == TALKS * 4


def test_the_parts_of_one_conversation_are_one_at_a_time(cache_path: Path) -> None:
    """Each prompt holds the notes of every part before it: a part waits for the one before."""
    inside = Inside(pause=0.01)
    pool = CallPool(max_parallel=4, tokens_per_minute=None)
    llm = inside.llm("only")
    found = digest(talk(0), llm=llm, cache=PeriodCache(cache_path), conversation_id="only", pool=pool)
    assert found.periods_built == 4 and inside.most == 1
    assert notes_seen(llm) == [
        ("2023-01", []),
        ("2023-02", ["Jan 2023"]),
        ("2023-03", ["Jan 2023", "Feb 2023"]),
        ("2023-04", ["Jan 2023", "Feb 2023", "Mar 2023"]),
    ]


def test_conversations_read_at_once_give_what_they_give_one_at_a_time(tmp_path: Path) -> None:
    """Same keys, same notes, same checks, same results, in the order of the conversations."""
    alone, together = tmp_path / "alone.sqlite", tmp_path / "together.sqlite"
    one_by_one = [work() for work in tasks(names(), Inside(), alone, None)]
    pool = CallPool(max_parallel=4, tokens_per_minute=None)
    inside = Inside(pause=0.005)
    found = run_in_parallel(tasks(names(), inside, together, pool), max_parallel=4)
    assert [outcome.value.conversation_id for outcome in found] == names()  # type: ignore[union-attr]
    for single, outcome in zip(one_by_one, found, strict=True):
        assert outcome.value is not None and outcome.value.periods == single.periods
        assert summaries(outcome.value) == summaries(single)
        assert outcome.value.usage == single.usage
    assert cache_rows(together) == cache_rows(alone)
    assert inside.most > 1 and max(inside.most_by_talk.values()) == 1


def test_the_wall_time_of_many_parts_goes_down_by_about_n(tmp_path: Path) -> None:
    """32 parts of 60 ms each: about 1.9 s one at a time, about a quarter of it four at a time."""
    pause, many = 0.06, names(8)
    started = time.monotonic()
    for work in tasks(many, Inside(pause=pause), tmp_path / "alone.sqlite", None):
        work()
    one_by_one = time.monotonic() - started
    pool = CallPool(max_parallel=4, tokens_per_minute=None)
    started = time.monotonic()
    found = run_in_parallel(tasks(many, Inside(pause=pause), tmp_path / "four.sqlite", pool), max_parallel=4)
    four = time.monotonic() - started
    assert [outcome.error for outcome in found if outcome.error] == [] and pool.most_in_flight == 4
    assert one_by_one >= len(many) * 4 * pause
    assert one_by_one / four >= 2.8, (one_by_one, four)


# --- the limit of a run, a stop and progress with a pool -----------------------------


def test_each_run_keeps_its_own_limit_of_calls_in_a_shared_pool(cache_path: Path) -> None:
    inside = Inside(pause=0.005)
    pool = CallPool(max_parallel=4, tokens_per_minute=None)
    found = run_in_parallel(tasks(names(4), inside, cache_path, pool, max_new_periods=2), max_parallel=4)
    assert all(isinstance(outcome.error, DigestTooLarge) for outcome in found)
    assert pool.sent == 0  # the check before the first call refused each one


def test_a_stop_of_the_pool_ends_every_run_before_its_next_call(cache_path: Path) -> None:
    pool = CallPool(max_parallel=2, tokens_per_minute=None)
    calls = {"count": 0}
    lock = threading.Lock()

    class Stopper(Inside):
        def _answer(self, name: str, call: FakeCall) -> Any:
            with lock:
                calls["count"] += 1
                if calls["count"] == 3:
                    pool.stop()
            return super()._answer(name, call)

    found = run_in_parallel(tasks(names(4), Stopper(pause=0.01), cache_path, pool), max_parallel=2, stop=lambda: pool.stopped)
    errors = [outcome.error for outcome in found if outcome.started]
    assert errors and all(isinstance(error, ReportCancelled) for error in errors)
    assert [outcome.started for outcome in found][2:] == [False, False]
    # the calls in flight ended and were kept: nothing that was paid for is lost
    assert pool.sent <= 4 and len(cache_rows(cache_path)) == pool.sent


def test_a_cancel_of_one_run_leaves_the_other_runs_of_the_pool(cache_path: Path) -> None:
    pool = CallPool(max_parallel=2, tokens_per_minute=None)
    token = CancelToken()
    token.cancel()
    _first, second = tasks(names(2), Inside(), cache_path, pool)
    with pytest.raises(ReportCancelled):
        digest(talk(0), llm=FakeLLM(), cache=PeriodCache(cache_path), conversation_id="talk-0", pool=pool, cancel=token)
    assert second().periods_built == 4 and not pool.stopped


def test_each_part_says_when_it_starts_ends_is_in_the_notes_or_fails(cache_path: Path) -> None:
    """The events of a digest with a pool: the label and the number of each part."""
    pool = CallPool(max_parallel=2, tokens_per_minute=None)
    digest(talk(0), until="2023-01-31", cache=PeriodCache(cache_path), conversation_id="talk-0", pool=pool)

    events: list[ProgressEvent] = []
    digest(
        talk(0), llm=FakeLLM(fail_month("2023-03", ValueError("no"))), cache=PeriodCache(cache_path),
        conversation_id="talk-0", pool=pool, on_progress=events.append,
    )  # fmt: skip
    told = [(event.state, event.period, event.index, event.total) for event in events if event.step == "map"]
    assert told == [
        ("cached", LARGE_LABELS[0], 1, 4),
        ("start", LARGE_LABELS[1], 2, 4),
        ("done", LARGE_LABELS[1], 2, 4),
        ("start", LARGE_LABELS[2], 3, 4),
        ("failed", LARGE_LABELS[2], 3, 4),
        ("start", LARGE_LABELS[3], 4, 4),
        ("done", LARGE_LABELS[3], 4, 4),
    ]
