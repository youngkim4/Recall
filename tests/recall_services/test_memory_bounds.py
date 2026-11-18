"""memory.py: a digest inside a question is bounded, is paid for one time, and says what it cost."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from typing import Any

import pytest

from chat_engine.cancel import CancelToken
from recall_services import digests, memory, models, reports
from report_engine import PeriodCache, StoreError
from report_engine.chat_digest import DigestTooLarge
from report_engine.control import ProgressEvent
from report_engine.fake_llm import FakeCall, FakeLLM, completed
from report_engine.models import Usage
from report_engine.openai_llm import OpenAIResponsesLLM
from tests.recall_services.report_world import MODEL, Desk
from tests.recall_services.test_memory_digests import port, settings
from tests.report_engine.synth import MINA

JANUARY = {"since": "2023-01-01", "until": "2023-01-31"}
TWO_MONTHS = {"since": "2023-01-01", "until": "2023-02-28"}
ALL = {"since": None, "until": None}


def bounded(desk: Desk, llm: Any = None, **more: Any) -> memory.DigestPort:
    """The port as a question of the app gets it: with the limit of parts of `memory.py`."""
    found = memory.ports(desk.archive, desk.names, digest=settings(desk, llm, parts_limit=None), **more)["digests"]
    assert isinstance(found, memory.DigestPort)
    return found


# --- the limit of parts ------------------------------------------------------------


def test_a_digest_inside_a_question_reads_at_most_its_limit_of_parts_that_are_not_read_yet(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert memory.DIGEST_PARTS_LIMIT == 12
    monkeypatch.setattr(memory, "DIGEST_PARTS_LIMIT", 2)  # Mina has four parts
    model = FakeLLM()
    asked = bounded(desk, model)
    with pytest.raises(digests.ReportFailure) as refused:
        asked.digest(MINA, cancel=CancelToken(), **ALL)
    assert refused.value.code == "too_many_parts" and model.calls == []
    assert len(asked.digest(MINA, cancel=CancelToken(), **TWO_MONTHS)["periods"]) == 2
    # two parts are in the notes now: the other two are within the limit
    assert len(asked.digest(MINA, cancel=CancelToken(), **ALL)["periods"]) == 4
    assert len(model.calls) == 4


def test_the_limit_of_memory_reaches_the_digest_as_its_most_new_parts(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = digests.build

    def noted(*args: Any, **more: Any) -> Any:
        seen.append(more["max_new_periods"])
        return real(*args, **more)

    monkeypatch.setattr(digests, "build", noted)
    bounded(desk).digest(MINA, cancel=CancelToken(), **JANUARY)
    monkeypatch.setattr(memory, "DIGEST_PARTS_LIMIT", 1)
    bounded(desk).digest(MINA, cancel=CancelToken(), **JANUARY)
    assert seen == [12, 1]


# --- the cost ----------------------------------------------------------------------


def test_the_digest_says_what_it_cost_and_the_log_has_the_cost_with_counts_only(
    desk: Desk, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)
    asked = port(desk)
    quote = digests.estimate(MINA, model=MODEL, **desk.estimate_more())
    first = asked.digest(MINA, cancel=CancelToken(), **ALL)
    again = asked.digest(MINA, cancel=CancelToken(), **ALL)
    # the estimate counts the most that a part can answer: the run costs that much at most
    assert 0 < first["usd"] <= quote.usd and again["usd"] == 0.0
    lines = [record.getMessage() for record in caplog.records if record.name == memory.SPEND_LOG]
    assert lines == [
        f"a digest inside a question: 4 parts read now, 0 from the notes, 0 failed, 4 model calls, {first['usd']:.4f} USD",
        "a digest inside a question: 0 parts read now, 4 from the notes, 0 failed, 0 model calls, 0.0000 USD",
    ]
    assert MINA not in caplog.text and "2025550143" not in caplog.text


# --- two questions at once ---------------------------------------------------------


def slow(seconds: float) -> FakeLLM:
    def respond(_call: FakeCall) -> Any:
        time.sleep(seconds)
        return None

    return FakeLLM(respond)


def test_two_questions_at_once_pay_for_the_parts_one_time(desk: Desk) -> None:
    """One lock for the archive, the conversation and the model: the second question finds the parts read."""
    model = slow(0.2)
    found: list[Any] = []

    def one() -> None:
        found.append(bounded(desk, model).digest(MINA, cancel=CancelToken(), **TWO_MONTHS))

    askers = [threading.Thread(target=one) for _ in range(2)]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(30)
    assert len(model.map_calls) == 2
    assert sorted(len(result["periods"]) for result in found) == [2, 2]
    assert sorted(result["usd"] > 0 for result in found) == [False, True]


def test_a_question_that_is_cancelled_while_it_waits_for_another_reads_nothing(desk: Desk) -> None:
    model = slow(0.6)
    first = threading.Thread(target=lambda: bounded(desk, model).digest(MINA, cancel=CancelToken(), **JANUARY))
    first.start()
    until = time.monotonic() + 10.0
    while not model.calls and first.is_alive() and time.monotonic() < until:
        time.sleep(0.01)
    assert model.calls, "the first question never reached the model"
    token = CancelToken()
    token.cancel()
    started = time.monotonic()
    with pytest.raises(memory.DigestStopped):
        bounded(desk, model).digest(MINA, cancel=token, **JANUARY)
    assert time.monotonic() - started < 0.5
    first.join(30)
    assert len(model.calls) == 1


def test_another_conversation_and_another_model_do_not_wait(desk: Desk) -> None:
    one = memory.digest_lock(desk.archive, MINA, MODEL)
    assert memory.digest_lock(desk.archive, MINA, MODEL) is one
    assert memory.digest_lock(desk.archive, "another", MODEL) is not one
    assert memory.digest_lock(desk.archive, MINA, "gpt-6-luna") is not one


# --- the model port of a digest inside a question ----------------------------------


def test_the_model_port_inside_a_question_waits_two_minutes_and_tries_one_more_time() -> None:
    assert (memory.DIGEST_TIMEOUT_SECONDS, memory.DIGEST_RETRIES) == (120.0, 1)
    info = models.model_info(memory.DIGEST_MODEL)
    made = memory.question_llm(info)
    assert isinstance(made, OpenAIResponsesLLM)
    assert (made._timeout_s, made._max_retries, made._model) == (120.0, 1, memory.DIGEST_MODEL)
    assert made._client.timeout == 120.0 and made._client.max_retries == 0
    # the report job keeps its own: five minutes and two more tries
    of_a_report = reports.default_llm(info)
    assert (of_a_report._timeout_s, of_a_report._max_retries) == (300.0, 2)


def test_with_no_key_the_model_port_inside_a_question_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(reports.ReportFailure) as told:
        memory.question_llm(models.model_info(memory.DIGEST_MODEL))
    assert told.value.code == "no_key"


def test_a_digest_of_the_app_is_read_through_the_port_of_a_question(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeLLM()
    asked_for: list[str] = []

    def stand_in(info: Any) -> FakeLLM:
        asked_for.append(info.id)
        return model

    monkeypatch.setattr(memory, "question_llm", stand_in)
    more = desk.more()
    plain = memory.DigestSettings(places=more["places"], options=more["options"], counter=more["counter"])
    asked = memory.ports(desk.archive, desk.names, digest=plain)["digests"]
    asked.digest(MINA, cancel=CancelToken(), **JANUARY)
    assert asked_for == [memory.DIGEST_MODEL] and len(model.calls) == 1


# --- the progress of each part, and nothing started outside the question -----------

PARTS = ["Jan 2023", "Feb 2023", "Mar 2023", "Apr 2023"]


def test_the_digest_tells_each_part_as_the_report_engine_reads_it(desk: Desk) -> None:
    asked = bounded(desk)
    told: list[Any] = []
    asked.digest(MINA, cancel=CancelToken(), progress=told.append, **ALL)
    assert told == [
        {"part": number, "of": 4, "label": label, "state": state}
        for number, label in enumerate(PARTS, start=1)
        for state in ("start", "done")
    ]
    again: list[Any] = []
    asked.digest(MINA, cancel=CancelToken(), progress=again.append, **JANUARY)
    assert again == [{"part": 1, "of": 1, "label": "Jan 2023", "state": "cached"}]
    asked.digest(MINA, cancel=CancelToken(), **JANUARY)  # with no one to tell, nothing is told


def test_a_part_that_fails_is_told_as_failed(desk: Desk) -> None:
    told: list[Any] = []
    with pytest.raises(digests.ReportFailure):
        bounded(desk, FakeLLM(refused_by_the_model)).digest(MINA, cancel=CancelToken(), progress=told.append, **JANUARY)
    assert told == [
        {"part": 1, "of": 1, "label": "Jan 2023", "state": "start"},
        {"part": 1, "of": 1, "label": "Jan 2023", "state": "failed"},
    ]


@pytest.mark.parametrize(
    "event",
    [
        ProgressEvent(step="prepare", state="start"),
        ProgressEvent(step="prepare", state="done", total=4),
        ProgressEvent(step="map", state="waiting", period="Jan 2023", index=1, total=4),
        ProgressEvent(step="map", state="start", period=None, index=1, total=4),
        ProgressEvent(step="reduce", state="start"),
        ProgressEvent(step="done", state="done", total=4),
    ],
)
def test_only_a_change_of_a_part_is_told(event: ProgressEvent) -> None:
    told: list[Any] = []
    memory.progress_relay(told.append)(event)
    assert told == []


def test_no_question_starts_a_reading_outside_itself(desk: Desk) -> None:
    assert not hasattr(bounded(desk), "prepare")
    assert "start_digest" not in inspect.signature(memory.ports).parameters
    assert "start_digest" not in inspect.signature(memory.DigestPort).parameters


# --- a digest that was paid for and gave nothing -----------------------------------


def refused_by_the_model(call: FakeCall) -> Any:
    return completed(call, None, text="", status="refused")  # paid, and no result


def spend_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == memory.SPEND_LOG]


def test_a_digest_that_was_paid_for_and_failed_says_what_it_cost(desk: Desk, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)
    model = FakeLLM(refused_by_the_model)
    with pytest.raises(digests.ReportFailure) as told:
        bounded(desk, model).digest(MINA, cancel=CancelToken(), **TWO_MONTHS)
    failure = told.value
    assert failure.code == "model_error" and len(model.calls) == 2
    assert failure.usd > 0 and failure.usd == failure.usage["cost_usd"]
    assert spend_lines(caplog) == [
        f"a digest inside a question gave nothing (model_error): 2 model calls, {failure.usd:.4f} USD"
    ]
    assert MINA not in caplog.text


def test_a_digest_that_is_cancelled_after_a_part_says_what_that_part_cost(
    desk: Desk, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)
    token = CancelToken()

    def stops(_call: FakeCall) -> Any:
        token.cancel()
        return None

    model = FakeLLM(stops)
    with pytest.raises(digests.ReportFailure) as told:
        bounded(desk, model).digest(MINA, cancel=token, **TWO_MONTHS)
    assert told.value.code == "cancelled" and len(model.calls) == 1 and told.value.usd > 0
    assert spend_lines(caplog) == [
        f"a digest inside a question gave nothing (cancelled): 1 model calls, {told.value.usd:.4f} USD"
    ]


def test_a_digest_that_failed_before_any_call_cost_nothing_and_leaves_no_line(
    desk: Desk, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)
    with pytest.raises(digests.ReportFailure) as told:
        bounded(desk, FakeLLM()).digest(MINA, cancel=CancelToken(), **ALL)
    assert (told.value.code, told.value.usd) == ("too_many_parts", 0.0) and spend_lines(caplog) == []


# --- a digest that is stopped at the limit after it paid ---------------------------

STOPPED_IN_THE_MIDDLE = (
    "This conversation has more parts that are not read yet than the 2 that can be read now. "
    "What was read is kept. Choose fewer dates."
)
# the report engine counts each request of a digest and stops before the one that passes the limit
ENGINE_STOPS_IN_THE_MIDDLE = "usage" in inspect.signature(DigestTooLarge.__init__).parameters


def too_large_after(calls: int, cost_usd: float) -> DigestTooLarge:
    """The error of a digest that the report engine stopped at the limit, after it paid for some calls."""
    error = DigestTooLarge(4, 1, 2)  # after a stop, `new_periods` is the limit and one: no count of what is left
    error.usage = Usage(input_tokens=900, output_tokens=40, cost_usd=cost_usd, calls=calls)
    return error


def test_a_digest_that_stops_at_the_limit_after_paid_calls_says_what_they_cost(
    desk: Desk, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)

    def stops(**_more: Any) -> Any:
        raise too_large_after(calls=2, cost_usd=0.0123)

    monkeypatch.setattr(digests, "conversation_digest", stops)
    with pytest.raises(digests.ReportFailure) as told:
        bounded(desk).digest(MINA, cancel=CancelToken(), **ALL)
    failure = told.value
    assert (failure.code, str(failure)) == ("too_many_parts", STOPPED_IN_THE_MIDDLE)
    assert failure.usage["calls"] == 2 and failure.usd == pytest.approx(0.0123)
    assert spend_lines(caplog) == ["a digest inside a question gave nothing (too_many_parts): 2 model calls, 0.0123 USD"]


def test_the_job_keeps_the_usage_of_a_digest_that_stopped_after_paid_calls(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall_services import digest_job

    monkeypatch.setattr(digests, "conversation_digest", lambda **_more: (_ for _ in ()).throw(too_large_after(1, 0.5)))
    with pytest.raises(digest_job.DigestFailure) as told:
        digest_job.run(MINA, **desk.more())
    assert told.value.code == "too_many_parts" and told.value.usage["cost_usd"] == 0.5


@pytest.mark.skipif(not ENGINE_STOPS_IN_THE_MIDDLE, reason="waits for the report engine: a digest that stops at the limit")
def test_notes_that_fail_at_the_second_read_do_not_let_a_question_pay_for_more_than_its_limit(
    desk: Desk, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every part is in the notes, and the read of the run fails: the digest pays for two parts and stops."""
    caplog.set_level(logging.INFO, logger=memory.SPEND_LOG)
    model = FakeLLM()
    port(desk, model).digest(MINA, cancel=CancelToken(), **ALL)
    assert len(model.calls) == 4
    real = PeriodCache.get_many

    def flaky(self: Any, keys: Any, touch: bool = False) -> Any:
        if touch:  # the read of the run; the read of the limit check has touch=False
            raise StoreError("period cache failed: database is locked")
        return real(self, keys, touch)

    monkeypatch.setattr(PeriodCache, "get_many", flaky)
    monkeypatch.setattr(memory, "DIGEST_PARTS_LIMIT", 2)
    caplog.clear()
    with pytest.raises(digests.ReportFailure) as told:
        bounded(desk, model).digest(MINA, cancel=CancelToken(), **ALL)
    assert len(model.calls) - 4 <= memory.DIGEST_PARTS_LIMIT
    assert told.value.usd > 0 and len(spend_lines(caplog)) == 1
