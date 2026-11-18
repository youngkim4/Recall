"""A pool of calls that several runs share: at most N requests in flight, one
tokens-per-minute budget, an exact limit of calls, and a stop that waits for
the requests in flight."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest

from report_engine.call_pool import CallPool
from report_engine.calls import Caller
from report_engine.control import CancelToken, ProgressEvent, Runtime
from report_engine.errors import CallLimitReached, ConfigError, ReportCancelled
from report_engine.fake_llm import FakeCall, FakeLLM, completed, default_reply
from report_engine.ports import LLMUsage

from .support import FakeTime, runtime
from .synth import model_info

WAIT = 10.0
START = ProgressEvent(step="map", state="start")
WAITING = ProgressEvent(step="map", state="waiting")
WORDS = [{"role": "user", "content": "one"}]


class Gate:
    """A fake model whose calls wait inside until the test lets them go. It
    counts how many calls are inside at once."""

    def __init__(self) -> None:
        self.go_on = threading.Event()
        self.lock = threading.Lock()
        self.inside = 0
        self.most = 0
        self.entered = threading.Semaphore(0)
        self.llm = FakeLLM(self._answer)

    def _answer(self, call: FakeCall) -> Any:
        with self.lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
        self.entered.release()
        try:
            assert self.go_on.wait(WAIT)
        finally:
            with self.lock:
                self.inside -= 1
        return None

    def wait_inside(self, count: int) -> None:
        for _ in range(count):
            assert self.entered.acquire(timeout=WAIT)


def caller(llm: Any, pool: CallPool | None = None, **more: Any) -> Caller:
    return Caller(
        llm=llm,
        model_info=more.pop("info", model_info()),
        runtime=more.pop("runtime", Runtime()),
        cancel=more.pop("cancel", None),
        emit=more.pop("emit", lambda _event: None),
        reasoning=None,
        pool=pool,
        **more,
    )


def send(one: Caller, tokens: int = 1, output: int = 10) -> Any:
    return one.send(
        WORDS,  # type: ignore[arg-type]
        schema={},
        max_output_tokens=output,
        input_estimate=tokens,
        waiting=WAITING,
        starting=START,
    )


def in_threads(count: int, work: Callable[[int], Any]) -> tuple[list[threading.Thread], dict[int, Any]]:
    """Start `count` threads. Each one keeps what `work` gave, or what it raised."""
    ended: dict[int, Any] = {}

    def body(number: int) -> None:
        try:
            ended[number] = work(number)
        except Exception as error:  # noqa: BLE001 - the test looks at it
            ended[number] = error

    threads = [threading.Thread(target=body, args=(number,)) for number in range(count)]
    for thread in threads:
        thread.start()
    return threads, ended


def joined(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(WAIT)
        assert not thread.is_alive()


# --- what a pool is ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "4", None])
def test_a_pool_needs_one_slot_or_more(bad: Any) -> None:
    with pytest.raises(ConfigError, match="max_parallel"):
        CallPool(max_parallel=bad, tokens_per_minute=None)


def test_a_pool_with_a_limit_of_tokens_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError):
        CallPool(max_parallel=2, tokens_per_minute=0)


def test_a_new_pool_has_sent_nothing() -> None:
    pool = CallPool(max_parallel=3, tokens_per_minute=None)
    assert (pool.max_parallel, pool.sent, pool.most_in_flight, pool.stopped) == (3, 0, 0, False)


# --- at most N requests in flight -----------------------------------------------------


def test_runs_that_share_a_pool_send_at_most_its_number_of_requests_at_once() -> None:
    gate, pool = Gate(), CallPool(max_parallel=3, tokens_per_minute=None)
    threads, ended = in_threads(7, lambda _n: send(caller(gate.llm, pool)))
    try:
        gate.wait_inside(3)
        # three requests are inside the model; the other four wait for a slot
        assert gate.inside == 3 and len(gate.llm.calls) == 3
    finally:
        gate.go_on.set()
    joined(threads)
    assert len(ended) == 7 and not [value for value in ended.values() if isinstance(value, Exception)]
    assert (gate.most, pool.most_in_flight, pool.sent, len(gate.llm.calls)) == (3, 3, 7, 7)


def test_a_run_with_no_pool_of_its_own_sends_as_before() -> None:
    llm = FakeLLM()
    one = caller(llm)
    send(one)
    send(one)
    assert len(llm.calls) == 2 and one.usage.calls == 2


# --- the limit of calls is exact, also when calls are sent at the same time ----------


def test_threads_that_share_one_run_never_send_more_than_its_limit() -> None:
    """A call counts when it is sent. The one that would pass the limit is refused, not sent."""
    gate, pool = Gate(), CallPool(max_parallel=8, tokens_per_minute=None)
    shared = caller(gate.llm, pool, max_calls=3)
    threads, ended = in_threads(8, lambda _n: send(shared))
    gate.wait_inside(3)
    gate.go_on.set()
    joined(threads)
    refused = [value for value in ended.values() if isinstance(value, CallLimitReached)]
    assert len(gate.llm.calls) == 3 and len(refused) == 5 and gate.most == 3
    assert all(error.limit == 3 for error in refused)
    assert shared.usage.calls == 3 and pool.sent == 3


def test_a_refused_call_takes_no_room_of_the_budget() -> None:
    ticks = FakeTime()
    pool = CallPool(max_parallel=1, tokens_per_minute=1_000, clock=ticks.clock)
    one = caller(FakeLLM(), pool, max_calls=0, runtime=runtime(ticks))
    with pytest.raises(CallLimitReached):
        send(one, tokens=500)
    other = caller(FakeLLM(), pool, runtime=runtime(ticks))
    send(other, tokens=900, output=100)
    assert ticks.slept == []


# --- one budget of tokens for every run of the pool -----------------------------------


def full(call: FakeCall) -> Any:
    """A reply that used the 600 tokens that its request reserved."""
    return completed(call, default_reply(call), usage=LLMUsage(input_tokens=550, output_tokens=50))


def test_runs_that_share_a_pool_share_its_tokens_per_minute() -> None:
    """Two runs each send 600 tokens with a limit of 1,000 a minute: the second waits."""
    ticks = FakeTime()
    info = model_info(tokens_per_minute=1_000)
    pool = CallPool(max_parallel=2, tokens_per_minute=1_000, clock=ticks.clock)
    events: list[ProgressEvent] = []
    first = caller(FakeLLM(full), pool, info=info, runtime=runtime(ticks))
    second = caller(FakeLLM(), pool, info=info, runtime=runtime(ticks), emit=events.append)
    send(first, tokens=550, output=50)
    send(second, tokens=550, output=50)
    assert events[0] == WAITING and events[-1] == START
    assert sum(ticks.slept) >= 59.0


def test_runs_with_no_shared_pool_keep_a_budget_each() -> None:
    ticks = FakeTime()
    info = model_info(tokens_per_minute=1_000)
    first = caller(FakeLLM(), info=info, runtime=runtime(ticks))
    second = caller(FakeLLM(), info=info, runtime=runtime(ticks))
    send(first, tokens=550, output=50)
    send(second, tokens=550, output=50)
    assert ticks.slept == []


def test_a_request_larger_than_the_budget_of_the_pool_is_a_config_error() -> None:
    pool = CallPool(max_parallel=1, tokens_per_minute=100)
    with pytest.raises(ConfigError):
        send(caller(FakeLLM(), pool), tokens=500)
    assert pool.sent == 0


# --- a stop sends nothing new and waits for the requests in flight -------------------


def test_a_stopped_pool_sends_no_new_request_and_lets_the_ones_in_flight_end() -> None:
    gate, pool = Gate(), CallPool(max_parallel=2, tokens_per_minute=None)
    threads, ended = in_threads(5, lambda _n: send(caller(gate.llm, pool)))
    gate.wait_inside(2)
    pool.stop()
    gate.go_on.set()
    joined(threads)
    answered = [value for value in ended.values() if not isinstance(value, Exception)]
    stopped = [value for value in ended.values() if isinstance(value, ReportCancelled)]
    assert (len(answered), len(stopped), len(gate.llm.calls), pool.sent) == (2, 3, 2, 2)
    assert pool.stopped is True


def test_a_run_that_is_cancelled_while_it_waits_for_a_slot_sends_nothing() -> None:
    gate, pool = Gate(), CallPool(max_parallel=1, tokens_per_minute=None)
    token = CancelToken()
    threads, ended = in_threads(1, lambda _n: send(caller(gate.llm, pool)))
    gate.wait_inside(1)
    waiting, later = in_threads(1, lambda _n: send(caller(gate.llm, pool, cancel=token)))
    token.cancel()
    joined(waiting)
    gate.go_on.set()
    joined(threads)
    assert isinstance(later[0], ReportCancelled) and len(gate.llm.calls) == 1
    assert not isinstance(ended[0], Exception)
