"""digests.py when the key, the model, the archive or the notes of earlier runs fail, and what it may say."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from archive import Archive
from recall_services import digests, report_failures, reports
from recall_services.names import Names
from report_engine import LLMError, ReportOptions
from report_engine.chat_digest import ConversationDigest
from report_engine.fake_llm import FakeCall, FakeLLM, completed
from tests.integration.report_data import Pulled
from tests.recall_services.report_world import MODEL, OPTIONS, Desk
from tests.report_engine.synth import MINA, THEO, WordCounter, conversation


def failing(error: Exception) -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        raise error

    return FakeLLM(responder)


def expect_failure(desk: Desk, model: FakeLLM, code: str, *dates: Any, **more: Any) -> digests.ReportFailure:
    with pytest.raises(digests.ReportFailure) as caught:
        digests.build(MINA, *dates, model=MODEL, **{**desk.more(model), **more})
    assert caught.value.code == code
    return caught.value


# --- the key -----------------------------------------------------------------------------


def test_with_no_key_no_request_is_made(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    more = {name: value for name, value in desk.more().items() if name != "llm_factory"}
    with pytest.raises(digests.ReportFailure) as caught:
        digests.build(MINA, model=MODEL, **more)
    assert caught.value.code == "no_key"
    assert str(caught.value) == report_failures.NO_KEY
    assert not desk.places.out_dir.exists()


def test_the_estimate_needs_no_key(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    kept = ("archive", "names", "places", "options", "counter")
    more = {name: value for name, value in desk.more().items() if name in kept}
    assert digests.estimate(MINA, model=MODEL, **more).periods == 4


def test_a_key_that_is_not_real_is_said_as_a_report_says_it(desk: Desk) -> None:
    """The provider answers 401 to each part. The cache keeps nothing, and the sentence names the key."""
    model = failing(LLMError("x (HTTP 401)", status_code=401))
    failure = expect_failure(desk, model, "key_rejected")
    assert str(failure) == (
        "OpenAI rejected the API key. Check OPENAI_API_KEY in the file .env in Recall's data folder, "
        "then run the report again."
    )
    assert len(model.calls) == 4
    assert digests.estimate(MINA, model=MODEL, **desk.estimate_more()).periods_cached == 0


# --- the model ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LLMError("x (HTTP 401)", status_code=401), "key_rejected"),
        (LLMError("x (HTTP 403)", status_code=403), "key_rejected"),
        (LLMError("x (HTTP 404)", status_code=404), "unknown_model"),
        (LLMError("x (HTTP 429)", status_code=429), "rate_limit"),
        (LLMError("the model service failed (APIConnectionError)"), "offline"),
        (LLMError("the model service failed (APITimeoutError)"), "timeout"),
        (LLMError("x (HTTP 500)", status_code=500), "model_error"),
        (RuntimeError("a fault of the port"), "model_error"),
    ],
)
def test_a_digest_in_which_every_part_fails_says_what_a_report_says(desk: Desk, error: Exception, code: str) -> None:
    failure = expect_failure(desk, failing(error), code)
    with pytest.raises(reports.ReportFailure) as of_report:
        reports.run(MINA, MODEL, **desk.more(failing(error)))
    assert (failure.code, str(failure)) == (of_report.value.code, str(of_report.value))
    assert isinstance(failure, reports.ReportFailure)


def test_a_part_that_fails_is_counted_and_the_others_are_given(desk: Desk) -> None:
    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            raise LLMError("x (HTTP 500)", status_code=500)
        return None

    found = digests.build(MINA, model=MODEL, **desk.more(FakeLLM(responder)))
    assert (found.periods_built, found.periods_cached, found.periods_failed) == (3, 0, 1)
    assert [period.label for period in found.periods] == ["Jan 2023", "Mar 2023", "Apr 2023"]
    again = FakeLLM()
    retried = digests.build(MINA, model=MODEL, **desk.more(again))
    assert [call.first_day[:7] for call in again.map_calls] == ["2023-02"]
    assert (retried.periods_built, retried.periods_cached, retried.periods_failed) == (1, 3, 0)


def test_replies_that_can_not_be_used_are_paid_for_and_the_failure_says_the_cost(desk: Desk) -> None:
    def responder(call: FakeCall) -> Any:
        return completed(call, None, text="cut", status="incomplete")

    failure = expect_failure(desk, FakeLLM(responder), "model_error")
    assert failure.usage is not None and failure.usage["calls"] == 4 and failure.usage["input_tokens"] > 0


def test_a_model_port_that_can_not_be_made_is_a_failure_with_a_sentence(desk: Desk) -> None:
    def factory(_info: object) -> Any:
        raise ImportError("no module named openai")

    failure = expect_failure(desk, FakeLLM(), "run_failed", llm_factory=factory)
    assert str(failure) == "The model could not be started. No report was written."


# --- nothing to read, too much to read ---------------------------------------------------


def test_dates_with_no_message_are_a_failure_and_no_call(desk: Desk) -> None:
    model = FakeLLM()
    failure = expect_failure(desk, model, "empty", "2020-01-01", "2020-12-31")
    assert str(failure) == "This conversation has no messages with text in these dates. No report was written."
    assert model.calls == []
    with pytest.raises(digests.ReportFailure) as caught:
        digests.build("+12025550199", model=MODEL, **desk.more(model))
    assert caught.value.code == "empty" and model.calls == []


def test_a_budget_that_can_not_hold_one_message_is_said_before_any_call(desk: Desk) -> None:
    tiny = ReportOptions(period_token_budget=1, map_max_output_tokens=500, reduce_max_output_tokens=800)
    model = FakeLLM()
    failure = expect_failure(desk, model, "too_large", options=tiny)
    assert str(failure) == "The model can not read this conversation with its limits. No report was written."
    with pytest.raises(digests.ReportFailure) as caught:
        digests.estimate(MINA, model=MODEL, **{**desk.estimate_more(), "options": tiny})
    assert caught.value.code == "too_large" and model.calls == []


def test_a_digest_over_the_limit_of_new_parts_is_refused_before_any_call(desk: Desk) -> None:
    reports.run(MINA, MODEL, "2023-01-01", "2023-01-31", **desk.more())
    model = FakeLLM()
    failure = expect_failure(desk, model, "too_many_parts", max_new_periods=2)
    # a report helps only when it was run with the model of the digest: the sentence does not advise one
    assert str(failure) == "This conversation has 3 parts that are not read yet, and 2 can be read now. Choose fewer dates."
    assert model.calls == [] and failure.usage is None
    one = expect_failure(desk, model, "too_many_parts", max_new_periods=0, since="2023-02-01", until="2023-02-28")
    assert str(one).startswith("This conversation has 1 part that is not read yet, and 0 can be read now.")


def test_a_digest_that_the_engine_stopped_at_the_limit_after_paid_calls_keeps_its_usage_and_names_no_count(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a stop in the middle the error does not know how many parts are left: the sentence gives no such number."""
    from report_engine.chat_digest import DigestTooLarge
    from report_engine.models import Usage

    def stops(**_more: Any) -> Any:
        error = DigestTooLarge(4, 1, 2)
        error.usage = Usage(input_tokens=900, output_tokens=40, cost_usd=0.0123, calls=2)
        raise error

    monkeypatch.setattr(digests, "conversation_digest", stops)
    failure = expect_failure(desk, FakeLLM(), "too_many_parts", max_new_periods=2)
    assert str(failure) == (
        "This conversation has more parts that are not read yet than the 2 that can be read now. "
        "What was read is kept. Choose fewer dates."
    )
    assert failure.usage == {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 40, "cost_usd": 0.0123, "calls": 2}


def test_a_limit_of_new_parts_lets_a_digest_through_that_is_under_it(desk: Desk) -> None:
    reports.run(MINA, MODEL, "2023-01-01", "2023-01-31", **desk.more())
    model = FakeLLM()
    found = digests.build(MINA, model=MODEL, max_new_periods=3, **desk.more(model))
    assert (found.periods_built, found.periods_cached) == (3, 1)
    only_read = digests.build(MINA, model=MODEL, max_new_periods=0, **desk.more(model))
    assert only_read.periods_cached == 4 and len(model.calls) == 3


# --- the archive and the notes of earlier runs ------------------------------------------


def test_an_archive_that_can_not_be_read_is_a_failure_in_plain_words(desk: Desk) -> None:
    closed = Archive.open(desk.pulled.archive_path)
    closed.close()
    for ask in (
        lambda: digests.estimate(MINA, model=MODEL, **{**desk.estimate_more(), "archive": closed}),
        lambda: digests.build(MINA, model=MODEL, **{**desk.more(), "archive": closed}),
    ):
        with pytest.raises(digests.ReportFailure) as caught:
            ask()
        assert caught.value.code == "archive_error"
        assert str(caught.value) == "The message archive could not be read. Update the archive, then try again."


class ClosesLater:
    """An archive that answers the first questions and then fails, as a file that goes away does."""

    def __init__(self, archive: Archive) -> None:
        self._archive = archive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._archive, name)

    def iter_messages(self, *args: Any, **more: Any) -> Any:
        from archive import ArchiveError

        raise ArchiveError("the archive file is gone")


def test_messages_that_can_not_be_read_are_a_failure_in_plain_words(desk: Desk) -> None:
    gone = ClosesLater(desk.archive)
    for ask in (
        lambda: digests.estimate(MINA, model=MODEL, **{**desk.estimate_more(), "archive": gone}),
        lambda: digests.build(MINA, model=MODEL, **{**desk.more(), "archive": gone}),
    ):
        with pytest.raises(digests.ReportFailure) as caught:
            ask()
        assert caught.value.code == "archive_error"
        assert str(caught.value) == "The messages of this conversation could not be read. No report was written."


def broken_cache(desk: Desk) -> None:
    desk.places.cache_path.parent.mkdir(parents=True, exist_ok=True)
    desk.places.cache_path.write_bytes(b"this is no database " * 400)


def test_an_estimate_carries_on_when_the_notes_of_earlier_runs_can_not_be_read(desk: Desk) -> None:
    clean = digests.estimate(MINA, model=MODEL, **desk.estimate_more())
    broken_cache(desk)
    assert digests.estimate(MINA, model=MODEL, **desk.estimate_more()) == clean
    assert clean.periods_cached == 0


def test_a_digest_says_that_the_notes_of_earlier_runs_can_not_be_opened_and_makes_no_call(desk: Desk) -> None:
    broken_cache(desk)
    model = FakeLLM()
    failure = expect_failure(desk, model, "cache_unreadable")
    assert str(failure) == (
        "The file cache.sqlite3, which holds what earlier report runs read, can not be opened. No report was written."
    )
    assert model.calls == []


def test_the_estimate_and_the_digest_count_with_the_counter_of_a_report(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two counters can plan two sets of parts, and then no cache key of one is a key of the other."""
    loaded: list[str] = []
    real = reports.load_counter

    def load(family: str) -> Any:
        loaded.append(family)
        return real(family)

    monkeypatch.setattr(reports, "load_counter", load)
    reports.shared_counter.cache_clear()
    more = {name: value for name, value in desk.more().items() if name != "counter"}
    estimate_more = {name: value for name, value in desk.estimate_more().items() if name != "counter"}
    found = digests.build(THEO, model=MODEL, **more)
    assert len(loaded) == 1
    reports.shared_counter.cache_clear()
    planned = digests.estimate(THEO, model=MODEL, **estimate_more)
    assert len(loaded) == 2
    model = FakeLLM()
    reports.run(THEO, MODEL, **{**more, "llm_factory": lambda _info: model})
    assert len(loaded) == 2
    assert model.map_calls == [] and found.periods_built == planned.periods_cached >= 1
    reports.shared_counter.cache_clear()


# --- no private text in a log line or in a failure ---------------------------------------

MARK = "zebramark"
SAVED_NAME = "Quillon Zebraname"
PRIVATE = (MARK, "Quillon", "Zebraname", MINA, "2025550143", "555-0143", "555 0143")


@pytest.fixture
def marked(tmp_path: Path) -> Any:
    """An archive in which every message holds a marked word and the contact has a marked name."""
    pulled = Pulled(tmp_path / "private")
    lines = conversation("2023-01-01", days=120, per_day=6, words=8)
    pulled.write(MINA, [{**line, "text": f"{MARK} {line['text']}"} for line in lines])  # type: ignore[misc]
    contact = {"name": SAVED_NAME, "phones": ["(202) 555-0143"], "emails": []}
    pulled.pull([contact])
    archive = pulled.open()
    places = reports.ReportPlaces(tmp_path / "out", tmp_path / "saves" / "reports", tmp_path / "saves" / "cache.sqlite3")
    more = {
        "archive": archive,
        "names": Names.from_records([contact]),
        "places": places,
        "options": OPTIONS,
        "counter": WordCounter(),
    }
    try:
        yield more
    finally:
        archive.close()
        pulled.close()


def heard(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join([caplog.text, *(record.getMessage() for record in caplog.records)])


def assert_clean(*texts: str) -> None:
    for text in texts:
        assert not [part for part in PRIVATE if part in text]


def test_a_failing_digest_says_no_word_of_a_message_no_name_and_no_handle(
    marked: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    sent: list[str] = []

    def responder(call: FakeCall) -> Any:
        sent.append(call.text)
        raise RuntimeError(f"the reply was about: {call.text}")

    with pytest.raises(digests.ReportFailure) as caught:
        digests.build(MINA, model=MODEL, llm_factory=lambda _info: FakeLLM(responder), **marked)
    # the model was shown the marked text and the saved name: the test can fail
    assert sent and MARK in sent[0] and SAVED_NAME in sent[0]
    assert_clean(str(caught.value), repr(caught.value), heard(caplog))
    assert heard(caplog) != ""


def test_a_digest_with_a_failed_part_logs_no_word_of_a_message(
    marked: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            raise LLMError("x (HTTP 500)", status_code=500)
        line = call.lines[0]
        claim = {"date": "1999-01-01", "title": MARK, "detail": SAVED_NAME, "category": "milestone", "score": 0.9}
        return {
            "summary": f"{SAVED_NAME} wrote {MARK}.",
            "carry_over": f"{SAVED_NAME} {MARK}",
            "moments": [
                {**claim, "quote": f"{MARK} words that were never written", "refs": [line.ref]},
                {**claim, "quote": line.text, "refs": ["m99999", MINA]},
            ],
        }

    found = digests.build(MINA, model=MODEL, llm_factory=lambda _info: FakeLLM(responder), **marked)
    assert isinstance(found, ConversationDigest) and found.periods_failed == 1
    assert MARK in found.periods[0].summary
    digests.estimate(MINA, model=MODEL, **marked)
    digests.build(MINA, model=MODEL, llm_factory=lambda _info: FakeLLM(), **marked)
    assert_clean(heard(caplog))
    assert heard(caplog) != ""


def test_a_refused_digest_and_bad_input_say_no_handle(marked: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    said: list[str] = []
    for ask in (
        lambda: digests.build(MINA, model=MODEL, max_new_periods=0, llm_factory=lambda _info: FakeLLM(), **marked),
        lambda: digests.build(MINA, "2020-01-01", "2020-01-02", model=MODEL, llm_factory=lambda _info: FakeLLM(), **marked),
        lambda: digests.estimate(MINA, "2023-03-01", "2023-02-01", model=MODEL, **marked),
    ):
        with pytest.raises((digests.ReportFailure, ValueError)) as caught:
            ask()
        said.append(str(caught.value))
    assert len(said) == 3
    assert_clean(*said, heard(caplog))
