"""Run tasks N at a time, for a caller that reads several conversations at once.

The outcomes come back in the order of the tasks. A task that raises is its
outcome; the other tasks go on. `stop` is asked before each task starts:
when it says True no new task starts, and the tasks that run are waited for.
Give the runs of the tasks one `CallPool` (`call_pool.py`): the pool bounds
the requests in flight and shares the budget of tokens.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from report_engine.call_pool import checked_slots

T = TypeVar("T")


@dataclass(frozen=True)
class TaskOutcome(Generic[T]):
    """What one task gave. `started` is False for a task that a stop kept from starting."""

    started: bool
    value: T | None = None
    error: Exception | None = None


def _outcome(task: Callable[[], T]) -> TaskOutcome[T]:
    try:
        return TaskOutcome(started=True, value=task())
    except Exception as error:  # noqa: BLE001 - the error of a task is its outcome
        return TaskOutcome(started=True, error=error)


def run_in_parallel(
    tasks: Sequence[Callable[[], T]],
    *,
    max_parallel: int,
    stop: Callable[[], bool] | None = None,
) -> tuple[TaskOutcome[T], ...]:
    """Run every task, at most `max_parallel` at once, and wait for all of them."""
    slots = checked_slots(max_parallel)
    outcomes: list[TaskOutcome[T]] = [TaskOutcome(started=False) for _ in tasks]
    lock = threading.Lock()
    turn = iter(range(len(tasks)))

    def worker() -> None:
        while True:
            with lock:
                if stop is not None and stop():
                    return
                index = next(turn, None)
            if index is None:
                return
            outcomes[index] = _outcome(tasks[index])

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(slots, len(tasks)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return tuple(outcomes)
