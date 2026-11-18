"""Progress, cancel, failed periods and the limit on new periods of a digest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from report_engine.cache import CacheEntry, PeriodCache
from report_engine.chat_digest import DigestTooLarge
from report_engine.control import CancelToken, ProgressEvent
from report_engine.errors import (
    AllPeriodsFailed,
    ConfigError,
    LLMError,
    ReportCancelled,
    ReportEngineError,
    RunError,
    StoreError,
)
from report_engine.fake_llm import FakeCall, FakeLLM, completed

from .chat_digest_support import cache_rows, digest, fail_month, months
from .support import LARGE, LARGE_LABELS, FakeTime, run
from .synth import model_info


def without_reduce(events: Sequence[ProgressEvent]) -> list[ProgressEvent]:
    return [event for event in events if event.step != "reduce"]


# --- progress ----------------------------------------------------------------------


def test_progress_events_name_the_step_and_the_period() -> None:
    events: list[ProgressEvent] = []
    digest(LARGE, on_progress=events.append)
    assert events[:2] == [
        ProgressEvent(step="prepare", state="start"),
        ProgressEvent(step="prepare", state="done", total=4),
    ]
    assert [(e.state, e.period, e.index, e.total) for e in events if e.step == "map"] == [
        (state, label, index, 4)
        for index, label in enumerate(LARGE_LABELS, start=1)
        for state in ("start", "done")
    ]
    assert events[-1] == ProgressEvent(step="done", state="done", total=4)
    assert {event.step for event in events} == {"prepare", "map", "done"}


def test_the_events_of_a_digest_are_the_events_of_a_report_without_the_reduce_step(
    tmp_path: Path,
) -> None:
    """Fresh, cached, failed and waiting: a period says the same in both."""
    info = model_info(tokens_per_minute=5_600)
    seen: dict[str, list[list[ProgressEvent]]] = {"report": [], "digest": []}
    for name, work in (("report", run), ("digest", digest)):
        cache = PeriodCache(tmp_path / f"{name}.sqlite3")
        for messages in (LARGE[: 59 * 6], LARGE):
            events: list[ProgressEvent] = []
            llm = FakeLLM(fail_month("2023-03", LLMError("no answer")))
            more = {"llm": llm, "cache": cache, "info": info, "time": FakeTime()}
            work(messages, on_progress=events.append, **more)
            seen[name].append(events)
    assert [without_reduce(events) for events in seen["report"]] == seen["digest"]
    states = {(event.period, event.state) for event in seen["digest"][1]}
    assert {("Jan 2023", "cached"), ("Mar 2023", "failed"), ("Apr 2023", "waiting")} <= states


def test_the_events_of_a_span_count_the_periods_of_the_span() -> None:
    events: list[ProgressEvent] = []
    digest(LARGE, on_progress=events.append, since="2023-02-10", until="2023-03-05")
    assert events[1] == ProgressEvent(step="prepare", state="done", total=2)
    read = [event for event in events if event.step == "map" and event.state == "done"]
    assert [(e.period, e.index, e.total) for e in read] == [("Feb 2023", 1, 2), ("Mar 2023", 2, 2)]
    assert events[-1] == ProgressEvent(step="done", state="done", total=2)


def test_a_progress_callback_that_raises_does_not_stop_the_digest() -> None:
    def broken(_event: ProgressEvent) -> None:
        raise RuntimeError("the listener is gone")

    assert digest(LARGE, on_progress=broken).periods_built == 4


# --- cancel --------------------------------------------------------------------------


def test_cancel_between_periods_stops_the_work_and_keeps_what_is_done(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    token = CancelToken()

    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            token.cancel()
        return None

    llm = FakeLLM(responder)
    with pytest.raises(ReportCancelled) as caught:
        digest(LARGE, llm=llm, cache=cache, cancel=token)
    assert months(llm) == ["2023-01", "2023-02"]
    assert caught.value.usage is not None and caught.value.usage.calls == 2
    assert [row[2] for row in sorted(cache_rows(cache_path), key=lambda row: row[2])] == [
        "Feb 2023",
        "Jan 2023",
    ]
    resumed = FakeLLM()
    found = digest(LARGE, llm=resumed, cache=cache)
    assert months(resumed) == ["2023-03", "2023-04"]
    assert (found.periods_built, found.periods_cached) == (2, 2)


def test_a_cancelled_digest_and_a_cancelled_report_leave_the_same_cache(tmp_path: Path) -> None:
    rows = []
    for name, work in (("report", run), ("digest", digest)):
        token = CancelToken()

        def responder(call: FakeCall, token: CancelToken = token) -> Any:
            if call.first_day.startswith("2023-03"):
                token.cancel()
            return None

        path = tmp_path / f"{name}.sqlite3"
        with pytest.raises(ReportCancelled):
            work(LARGE, llm=FakeLLM(responder), cache=PeriodCache(path), cancel=token)
        rows.append(cache_rows(path))
    assert rows[0] == rows[1] and len(rows[0]) == 3


def test_a_token_cancelled_before_the_digest_reads_nothing_and_makes_no_call() -> None:
    token = CancelToken()
    token.cancel()
    llm = FakeLLM()
    events: list[ProgressEvent] = []
    with pytest.raises(ReportCancelled):
        digest(LARGE, llm=llm, cancel=token, on_progress=events.append)
    assert llm.calls == [] and events == []


def test_cancel_is_seen_while_the_digest_waits_for_the_rate_limit() -> None:
    token = CancelToken()
    time = FakeTime()
    real_sleep = time.sleep

    def sleep_then_cancel(seconds: float) -> None:
        real_sleep(seconds)
        token.cancel()

    time.sleep = sleep_then_cancel  # type: ignore[method-assign]
    llm = FakeLLM()
    with pytest.raises(ReportCancelled):
        digest(LARGE, llm=llm, info=model_info(tokens_per_minute=6_000), time=time, cancel=token)
    assert len(time.slept) == 1 and len(llm.calls) == 2


# --- failed periods --------------------------------------------------------------


def test_a_failed_period_is_counted_and_the_others_are_given(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    llm = FakeLLM(fail_month("2023-02", LLMError("the model did not answer in time", attempts=3)))
    found = digest(LARGE, llm=llm, cache=cache)
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (3, 0, 1)
    assert [period.label for period in found.periods] == ["Jan 2023", "Mar 2023", "Apr 2023"]
    assert len(cache_rows(cache_path)) == 3
    retry = FakeLLM()
    again = digest(LARGE, llm=retry, cache=cache)
    assert months(retry) == ["2023-02"]
    assert (again.periods_built, again.periods_cached, again.periods_failed) == (1, 3, 0)


@pytest.mark.parametrize("status", ["incomplete", "refused", "failed"])
def test_a_reply_that_can_not_be_used_fails_the_period_but_is_paid_for(status: str) -> None:
    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-03"):
            return completed(call, None, text="cut", status=status)  # type: ignore[arg-type]
        return None

    found = digest(LARGE, llm=FakeLLM(responder))
    assert (found.periods_built, found.periods_failed) == (3, 1)
    assert found.usage.calls == 4


def test_when_every_period_fails_a_typed_error_is_raised() -> None:
    def responder(call: FakeCall) -> Any:
        raise LLMError("no service")

    llm = FakeLLM(responder)
    with pytest.raises(AllPeriodsFailed) as caught:
        digest(LARGE, llm=llm)
    assert isinstance(caught.value, RunError)
    assert len(llm.calls) == 4
    assert "no service" in str(caught.value)
    assert caught.value.usage is not None and caught.value.usage.calls == 0


def test_when_every_period_of_a_span_fails_the_error_is_raised(cache_path: Path) -> None:
    """Periods outside the span that the cache holds do not make the digest a success."""
    cache = PeriodCache(cache_path)
    run(LARGE[: 31 * 6], cache=cache)
    with pytest.raises(AllPeriodsFailed):
        llm = FakeLLM(fail_month("2023-0", LLMError("no service")))
        digest(LARGE, llm=llm, cache=cache, since="2023-02-01")


def test_usable_replies_paid_for_are_in_the_usage_of_the_error() -> None:
    def responder(call: FakeCall) -> Any:
        return completed(call, None, text="cut", status="incomplete")

    with pytest.raises(AllPeriodsFailed) as caught:
        digest(LARGE, llm=FakeLLM(responder))
    assert caught.value.usage is not None and caught.value.usage.calls == 4


class BrokenCache:
    """A cache that fails in a way that is not its own."""

    def get_many(self, keys: Sequence[str], touch: bool = False) -> Mapping[str, CacheEntry]:
        return {}

    def put(self, entry: CacheEntry, **_: str) -> None:
        raise ZeroDivisionError("a fault with a secret in it")


def test_an_unexpected_error_after_the_first_call_is_a_run_error_with_the_usage() -> None:
    with pytest.raises(RunError) as caught:
        digest(LARGE, cache=BrokenCache())
    assert type(caught.value) is RunError
    assert str(caught.value) == "the digest stopped on an unexpected error: ZeroDivisionError"
    assert caught.value.usage is not None and caught.value.usage.calls == 1
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


class FailingCache:
    def __init__(self, fail_reads: bool) -> None:
        self._fail_reads = fail_reads

    def get_many(self, keys: Sequence[str], touch: bool = False) -> Mapping[str, CacheEntry]:
        if self._fail_reads:
            raise StoreError("the disk is gone")
        return {}

    def put(self, entry: CacheEntry, **_: str) -> None:
        raise StoreError("the disk is full")


@pytest.mark.parametrize("fail_reads", [True, False])
def test_a_cache_that_fails_costs_reuse_but_not_the_digest(
    fail_reads: bool, caplog: pytest.LogCaptureFixture
) -> None:
    found = digest(LARGE, cache=FailingCache(fail_reads), since="2023-02-01")
    assert (found.periods_built, found.periods_failed) == (3, 0)
    assert "period cache" in caplog.text


# --- the limit on new periods -------------------------------------------------------


def test_a_digest_over_the_limit_is_refused_before_any_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE[: 31 * 6], cache=cache)
    llm = FakeLLM()
    events: list[ProgressEvent] = []
    with pytest.raises(DigestTooLarge) as caught:
        digest(LARGE, llm=llm, cache=cache, max_new_periods=2, on_progress=events.append)
    assert llm.calls == []
    assert isinstance(caught.value, ReportEngineError) and not isinstance(caught.value, RunError)
    assert (caught.value.periods, caught.value.periods_cached, caught.value.limit) == (4, 1, 2)
    assert caught.value.new_periods == 3
    assert str(caught.value) == "the digest needs 3 model calls; the limit is 2"
    assert [event.step for event in events] == ["prepare", "prepare"]


def test_the_limit_counts_only_the_periods_that_need_a_call(cache_path: Path) -> None:
    cache = PeriodCache(cache_path)
    run(LARGE[: 31 * 6], cache=cache)
    llm = FakeLLM()
    assert digest(LARGE, llm=llm, cache=cache, max_new_periods=3).periods_built == 3
    assert digest(LARGE, llm=llm, cache=cache, max_new_periods=0).periods_cached == 4
    assert len(llm.calls) == 3


def test_a_span_brings_a_digest_under_the_limit() -> None:
    llm = FakeLLM()
    with pytest.raises(DigestTooLarge):
        digest(LARGE, llm=llm, max_new_periods=1)
    found = digest(LARGE, llm=llm, max_new_periods=1, since="2023-04-01")
    assert found.periods_built == 1 and len(llm.calls) == 1


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
def test_a_limit_that_is_no_count_is_refused(limit: Any) -> None:
    llm = FakeLLM()
    with pytest.raises(ConfigError):
        digest(LARGE, llm=llm, max_new_periods=limit)
    assert llm.calls == []
