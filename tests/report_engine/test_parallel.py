"""`run_in_parallel`: tasks run N at a time, their outcomes come back in the
order of the tasks, and a stop starts no new task and waits for the ones that run."""

from __future__ import annotations

import threading
import time

import pytest

from report_engine.errors import ConfigError
from report_engine.parallel import TaskOutcome, run_in_parallel

WAIT = 10.0


class Counted:
    """Tasks that count how many of them run at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inside = 0
        self.most = 0

    def task(self, value: int, pause: float = 0.02) -> object:
        def work() -> int:
            with self.lock:
                self.inside += 1
                self.most = max(self.most, self.inside)
            time.sleep(pause)
            with self.lock:
                self.inside -= 1
            return value * 10

        return work


def test_the_outcomes_come_in_the_order_of_the_tasks() -> None:
    counted = Counted()
    # the first tasks take longest: they end last
    tasks = [counted.task(number, pause=0.01 * (6 - number)) for number in range(6)]
    found = run_in_parallel(tasks, max_parallel=3)
    assert [outcome.value for outcome in found] == [0, 10, 20, 30, 40, 50]
    assert all(outcome.started and outcome.error is None for outcome in found)
    assert counted.most == 3


def test_at_most_the_number_of_slots_run_at_once() -> None:
    counted = Counted()
    run_in_parallel([counted.task(number) for number in range(9)], max_parallel=2)
    assert counted.most == 2


def test_fewer_tasks_than_slots_run_all_at_once() -> None:
    barrier = threading.Barrier(3, timeout=WAIT)
    found = run_in_parallel([barrier.wait for _ in range(3)], max_parallel=8)
    assert sorted(outcome.value for outcome in found) == [0, 1, 2]


def test_no_task_gives_nothing() -> None:
    assert run_in_parallel([], max_parallel=4) == ()


def test_a_task_that_fails_is_its_outcome_and_the_others_go_on() -> None:
    def broken() -> int:
        raise ValueError("no")

    found = run_in_parallel([lambda: 1, broken, lambda: 3], max_parallel=2)
    assert [outcome.value for outcome in found] == [1, None, 3]
    assert isinstance(found[1].error, ValueError) and found[1].started


def test_a_stop_starts_no_new_task_and_waits_for_the_tasks_that_run() -> None:
    stop, inside, go_on = threading.Event(), threading.Barrier(2, timeout=WAIT), threading.Event()
    ended: list[int] = []

    def held(number: int) -> object:
        def work() -> int:
            inside.wait()
            assert go_on.wait(WAIT)
            ended.append(number)
            return number

        return work

    def later() -> None:
        # both first tasks are inside: stop, then let them end
        stop.set()
        go_on.set()

    tasks = [held(0), held(1), *(lambda: pytest.fail("a stopped run started a task") for _ in range(4))]
    timer = threading.Timer(0.5, later)
    timer.start()
    found = run_in_parallel(tasks, max_parallel=2, stop=stop.is_set)
    timer.join()
    assert sorted(ended) == [0, 1]
    assert [outcome.started for outcome in found] == [True, True, False, False, False, False]
    assert found[2] == TaskOutcome(started=False)


@pytest.mark.parametrize("bad", [0, -2, True, 1.5])
def test_the_number_of_slots_is_a_whole_number_of_one_or_more(bad: object) -> None:
    with pytest.raises(ConfigError, match="max_parallel"):
        run_in_parallel([lambda: 1], max_parallel=bad)  # type: ignore[arg-type]
