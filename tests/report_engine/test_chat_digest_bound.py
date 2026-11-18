"""The limit of a digest holds whatever the cache does. No run pays for what another run read."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import CacheEntry, PeriodCache
from report_engine.calls import Caller
from report_engine.chat_digest import DigestTooLarge
from report_engine.control import ProgressEvent, Runtime
from report_engine.errors import CallLimitReached, LLMError, RunError, StoreError
from report_engine.fake_llm import FakeCall, FakeLLM, completed

from .chat_digest_support import cache_rows, digest, months
from .support import LARGE, run
from .synth import model_info

LOCKED = "period cache failed: database is locked"


class NoSecondRead(PeriodCache):
    """A cache that answers the check of the limit and then can not be read: the
    read of the map step and each look after it fail."""

    def get_many(self, keys: Sequence[str], touch: bool = False) -> dict[str, CacheEntry]:
        if touch:
            raise StoreError(LOCKED)
        return super().get_many(keys, touch)


class LosesEntries(PeriodCache):
    """A cache that loses some entries right after the check of the limit read them."""

    def __init__(self, path: Path, lose: int) -> None:
        super().__init__(path)
        self._lose = lose

    def get_many(self, keys: Sequence[str], touch: bool = False) -> dict[str, CacheEntry]:
        found = super().get_many(keys, touch)
        if not touch and self._lose:
            for key in sorted(found)[: self._lose]:
                self._delete([key])
            self._lose = 0
        return found


class NeverReads(PeriodCache):
    def get_many(self, keys: Sequence[str], touch: bool = False) -> dict[str, CacheEntry]:
        raise StoreError(LOCKED)


def filled(path: Path) -> None:
    """Every period of the long conversation is in the cache."""
    assert digest(LARGE, cache=PeriodCache(path)).periods_built == 4


# --- the limit holds whatever the cache does -------------------------------------------


def test_a_cache_that_can_not_be_read_a_second_time_does_not_lift_the_limit(
    cache_path: Path,
) -> None:
    """The check counts four periods in the cache. Then the read of the map step fails."""
    filled(cache_path)
    llm = FakeLLM()
    events: list[ProgressEvent] = []
    with pytest.raises(DigestTooLarge) as caught:
        digest(
            LARGE,
            llm=llm,
            cache=NoSecondRead(cache_path),
            max_new_periods=2,
            on_progress=events.append,
        )
    assert months(llm) == ["2023-01", "2023-02"]
    assert caught.value.limit == 2 and caught.value.new_periods == 3
    assert caught.value.usage is not None and caught.value.usage.calls == 2
    assert str(caught.value) == "the digest made 2 model calls and needs more; the limit is 2"
    assert not isinstance(caught.value, RunError)
    # the period that was refused did not start, and the run did not end as done
    started = [event.period for event in events if event.step == "map" and event.state == "start"]
    assert started == ["Jan 2023", "Feb 2023"]
    last = ProgressEvent(step="map", state="done", period="Feb 2023", index=2, total=4)
    assert events[-1] == last


@pytest.mark.parametrize(("lose", "limit", "calls"), [(4, 2, 2), (2, 1, 1), (3, 0, 0)])
def test_a_cache_that_loses_entries_after_the_check_does_not_lift_the_limit(
    cache_path: Path, lose: int, limit: int, calls: int
) -> None:
    filled(cache_path)
    llm = FakeLLM()
    with pytest.raises(DigestTooLarge) as caught:
        digest(LARGE, llm=llm, cache=LosesEntries(cache_path, lose), max_new_periods=limit)
    assert len(llm.calls) == calls
    assert caught.value.usage is not None and caught.value.usage.calls == calls
    # what was read stays in the cache
    assert len(cache_rows(cache_path)) == 4 - lose + calls


def test_a_loss_within_the_limit_is_read_again_and_the_digest_is_whole(cache_path: Path) -> None:
    filled(cache_path)
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, cache=LosesEntries(cache_path, 2), max_new_periods=2)
    assert len(llm.calls) == 2
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (2, 2, 0)


def test_a_limit_of_zero_makes_no_call_when_no_read_of_the_cache_works(cache_path: Path) -> None:
    filled(cache_path)
    for cache in (NeverReads(cache_path), NoSecondRead(cache_path)):
        llm = FakeLLM()
        with pytest.raises(DigestTooLarge) as caught:
            digest(LARGE, llm=llm, cache=cache, max_new_periods=0)
        assert llm.calls == []
        assert caught.value.usage is None or caught.value.usage.calls == 0
    with pytest.raises(DigestTooLarge):
        digest(LARGE, llm=llm, cache=NeverReads(cache_path), since="2023-03-01", max_new_periods=0)
    assert llm.calls == []


def test_a_reply_that_can_not_be_used_is_a_paid_call_and_counts(cache_path: Path) -> None:
    filled(cache_path)

    def cut_off(call: FakeCall) -> Any:
        return completed(call, None, text="cut", status="incomplete")

    llm = FakeLLM(cut_off)
    with pytest.raises(DigestTooLarge) as caught:
        digest(LARGE, llm=llm, cache=NoSecondRead(cache_path), max_new_periods=2)
    assert len(llm.calls) == 2
    assert caught.value.usage is not None and caught.value.usage.calls == 2


def test_a_call_that_the_port_fails_counts_too(cache_path: Path) -> None:
    """Nobody knows whether such a call was paid. It is counted as paid."""
    filled(cache_path)

    def fails(call: FakeCall) -> Any:
        raise LLMError("x (HTTP 500)", status_code=500)

    llm = FakeLLM(fails)
    with pytest.raises(DigestTooLarge):
        digest(LARGE, llm=llm, cache=NoSecondRead(cache_path), max_new_periods=3)
    assert len(llm.calls) == 3


def test_with_no_limit_nothing_changes(cache_path: Path) -> None:
    filled(cache_path)
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, cache=NoSecondRead(cache_path))
    assert len(llm.calls) == 4
    assert (found.periods_built, found.periods_cached) == (4, 0)


def test_the_one_place_that_sends_a_request_counts_them() -> None:
    """`Caller` refuses the request that would pass its limit, and says what the run spent."""
    llm = FakeLLM()
    sent: list[ProgressEvent] = []
    caller = Caller(
        llm=llm,
        model_info=model_info(),
        runtime=Runtime(),
        cancel=None,
        emit=sent.append,
        reasoning=None,
        max_calls=1,
    )
    event = ProgressEvent(step="map", state="start")
    request = {"schema": {}, "max_output_tokens": 10, "input_estimate": 1}
    words = [{"role": "user", "content": "one"}]
    caller.send(words, waiting=event, starting=event, **request)  # type: ignore[arg-type]
    with pytest.raises(CallLimitReached) as caught:
        caller.send(words, waiting=event, starting=event, **request)  # type: ignore[arg-type]
    assert isinstance(caught.value, RunError)
    assert (caught.value.limit, caught.value.usage.calls) == (1, 1)  # type: ignore[union-attr]
    assert len(llm.calls) == 1 and len(sent) == 1


def test_a_report_has_no_limit_of_calls(cache_path: Path) -> None:
    llm = FakeLLM()
    report = run(LARGE, llm=llm, cache=NoSecondRead(cache_path))
    assert len(llm.map_calls) == 4 and report.status == "complete"


# --- a period that another run read meanwhile is not paid again -----------------------


class Held:
    """A fake model whose first call waits until the test lets it go."""

    def __init__(self) -> None:
        self.inside = threading.Event()
        self.go_on = threading.Event()
        self.llm = FakeLLM(self._answer)

    def _answer(self, call: FakeCall) -> Any:
        if len(self.llm.calls) == 1:
            self.inside.set()
            assert self.go_on.wait(30)
        return None


def beside(first: Any, second: Any) -> tuple[Any, Any, FakeLLM, FakeLLM]:
    """Run `first` in a thread. While it is inside its first model call, run `second` to its end."""
    held, results = Held(), {}

    def work() -> None:
        results["first"] = first(held.llm)

    thread = threading.Thread(target=work)
    thread.start()
    try:
        assert held.inside.wait(30)
        other = FakeLLM()
        results["second"] = second(other)
    finally:
        held.go_on.set()
        thread.join(30)
    assert not thread.is_alive()
    return results["first"], results["second"], held.llm, other


def test_two_digests_of_one_conversation_pay_one_period_twice_at_most(cache_path: Path) -> None:
    """The second digest starts while the first is inside its first call, and reads every period."""

    def one(llm: FakeLLM) -> Any:
        return digest(LARGE, llm=llm, cache=PeriodCache(cache_path))

    first, second, held, other = beside(one, one)
    assert len(other.calls) == 4 and months(held) == ["2023-01"]
    assert len(held.calls) + len(other.calls) == 4 + 1
    assert (first.periods_built, first.periods_cached) == (1, 3)
    assert (second.periods_built, second.periods_cached) == (4, 0)
    told = [period.summary for period in second.periods]
    assert [period.summary for period in first.periods] == told
    assert len(cache_rows(cache_path)) == 4


def test_a_period_that_was_read_meanwhile_is_said_as_cached(cache_path: Path) -> None:
    events: list[ProgressEvent] = []

    def watched(llm: FakeLLM) -> Any:
        return digest(LARGE, llm=llm, cache=PeriodCache(cache_path), on_progress=events.append)

    def plain(llm: FakeLLM) -> Any:
        return digest(LARGE, llm=llm, cache=PeriodCache(cache_path))

    beside(watched, plain)
    states = [(event.period, event.state) for event in events if event.step == "map"]
    assert states == [
        ("Jan 2023", "start"),
        ("Jan 2023", "done"),
        ("Feb 2023", "cached"),
        ("Mar 2023", "cached"),
        ("Apr 2023", "cached"),
    ]


def test_a_digest_within_its_limit_beside_another_run_pays_within_its_limit(
    cache_path: Path,
) -> None:
    """The check counts four new periods and refuses. With a span of two, two calls are the most."""

    def whole(llm: FakeLLM) -> Any:
        return digest(LARGE, llm=llm, cache=PeriodCache(cache_path))

    def two(llm: FakeLLM) -> Any:
        more = {"until": "2023-02-28", "max_new_periods": 2}
        return digest(LARGE, llm=llm, cache=PeriodCache(cache_path), **more)

    first, second, held, other = beside(whole, two)
    assert len(other.calls) == 2 and second.periods_built == 2
    assert months(held) == ["2023-01", "2023-03", "2023-04"]
    assert (first.periods_built, first.periods_cached) == (3, 1)


def test_a_report_beside_a_digest_does_not_pay_for_what_the_digest_read(cache_path: Path) -> None:
    """The second look is in the map step, so a report run has it too."""

    def report(llm: FakeLLM) -> Any:
        return run(LARGE, llm=llm, cache=PeriodCache(cache_path))

    def one(llm: FakeLLM) -> Any:
        return digest(LARGE, llm=llm, cache=PeriodCache(cache_path))

    first, second, held, other = beside(report, one)
    assert months(held) == ["2023-01"] and len(held.reduce_calls) == 1
    assert [period.from_cache for period in first.periods] == [False, True, True, True]
    assert first.status == "complete" and second.periods_built == 4


def test_a_look_that_fails_just_before_a_period_is_sent_costs_the_call_and_no_more(
    cache_path: Path,
) -> None:
    llm = FakeLLM()
    found = digest(LARGE, llm=llm, cache=NeverReads(cache_path))
    assert len(llm.calls) == 4 and found.periods_built == 4
    assert len(cache_rows(cache_path)) == 4
