"""run_beside: the slow part of a lookup on a worker thread, its reports as events in order."""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.tool_context import Fetch
from chat_engine.turn_fetch import run_beside


def _event(raw: object) -> Optional[dict[str, Any]]:
    return None if raw == "drop" else {"type": "progress", "raw": raw}


def test_the_reports_come_in_order_and_a_dropped_one_is_not_given() -> None:
    seen: list[str] = []

    def work(report: Callable[[object], None]) -> None:
        seen.append(threading.current_thread().name)
        for raw in (1, "drop", 2, 3):
            report(raw)

    events = list(run_beside(Fetch(work, _event), CancelToken()))
    assert [event["raw"] for event in events] == [1, 2, 3]
    assert seen == ["chat-lookup"]


def test_what_the_work_raises_is_raised_on_the_thread_of_the_turn_after_its_reports() -> None:
    def work(report: Callable[[object], None]) -> None:
        report(1)
        raise RuntimeError("the port broke")

    cancel = CancelToken()
    given: list[Any] = []
    with pytest.raises(RuntimeError, match="the port broke"):
        for event in run_beside(Fetch(work, _event), cancel):
            given.append(event["raw"])
    assert given == [1] and not cancel.cancelled  # the work ended: nothing to stop


def test_a_turn_that_stops_reading_cancels_the_work_and_does_not_wait_for_it() -> None:
    release = threading.Event()

    def work(report: Callable[[object], None]) -> None:
        report(1)
        release.wait(5)

    cancel = CancelToken()
    events = run_beside(Fetch(work, _event), cancel)
    assert next(events)["raw"] == 1
    events.close()
    assert cancel.cancelled
    release.set()
