"""digests.py with a pool of calls: several conversations read at once, each part told by its label and number."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from recall_services import digests, models
from recall_services.models import ModelChoiceError
from report_engine import CallPool, ProgressEvent, run_in_parallel
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.recall_services.report_world import MODEL, PERIOD_LABELS, Desk
from tests.report_engine.synth import MINA, THEO

WAIT = 10.0


def test_a_pool_for_digests_takes_the_rate_of_the_model_and_its_number_of_slots() -> None:
    pool = digests.call_pool(MODEL, 3)
    assert isinstance(pool, CallPool) and pool.max_parallel == 3
    assert pool.tokens_per_minute == models.model_info(MODEL).tokens_per_minute
    assert digests.call_pool("", 1).max_parallel == 1  # the default model of the reports


def test_a_pool_for_a_model_that_can_not_be_run_is_refused() -> None:
    with pytest.raises(ModelChoiceError):
        digests.call_pool("no-such-model", 2)


def test_two_conversations_built_at_once_share_the_pool(desk: Desk) -> None:
    """The barrier lets the first two calls go only when a part of each conversation is inside."""
    barrier, first = threading.Barrier(2, timeout=WAIT), {"Jan": 0}
    lock = threading.Lock()

    def together(call: FakeCall) -> Any:
        with lock:
            first["Jan"] += 1
            waits = first["Jan"] <= 2
        if waits:
            barrier.wait()
        return None

    pool = digests.call_pool(MODEL, 2)
    more = {**desk.more(FakeLLM(together)), "model": MODEL, "pool": pool}

    def one(conversation: str) -> Any:
        return lambda: digests.build(conversation, **more)

    found = run_in_parallel([one(MINA), one(THEO)], max_parallel=2)
    assert [outcome.error for outcome in found] == [None, None]
    assert [outcome.value.periods_built for outcome in found] == [4, 1]  # type: ignore[union-attr]
    assert not barrier.broken and pool.sent == 5 and pool.most_in_flight == 2


def test_each_event_of_a_part_says_its_label_its_number_and_how_many_there_are(desk: Desk) -> None:
    """What a caller needs to show the progress of each part: start, done, in the notes, failed."""
    digests.build(MINA, "2023-01-01", "2023-01-31", model=MODEL, **desk.more())

    def refuse_march(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-03"):
            raise ValueError("no")
        return None

    events: list[ProgressEvent] = []
    pool = digests.call_pool(MODEL, 4)
    found = digests.build(
        MINA, on_progress=events.append, model=MODEL, pool=pool, **desk.more(FakeLLM(refuse_march))
    )
    parts = [(event.state, event.period, event.index, event.total) for event in events if event.step == "map"]
    assert parts == [
        ("cached", PERIOD_LABELS[0], 1, 4),
        ("start", PERIOD_LABELS[1], 2, 4),
        ("done", PERIOD_LABELS[1], 2, 4),
        ("start", PERIOD_LABELS[2], 3, 4),
        ("failed", PERIOD_LABELS[2], 3, 4),
        ("start", PERIOD_LABELS[3], 4, 4),
        ("done", PERIOD_LABELS[3], 4, 4),
    ]
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (2, 1, 1)
    assert [event.step for event in events][:2] == ["prepare", "prepare"] and events[-1].step == "done"
