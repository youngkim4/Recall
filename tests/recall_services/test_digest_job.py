"""digest_job.py: one conversation read for chat outside a question, with the lines and the counts of the job."""

from __future__ import annotations

from typing import Any

import pytest

from recall_services import digest_job, digests, memory, reports
from report_engine import LLMError
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.recall_services.report_world import MODEL, PERIOD_LABELS, Desk
from tests.report_engine.synth import MINA

RESULT_KEYS = {"conversation", "periods", "periodsBuilt", "periodsCached", "periodsFailed", "usd", "usage"}


def run(desk: Desk, llm: Any = None, *dates: Any, **more: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []
    settings = {**desk.more(llm), **more}
    return digest_job.run(MINA, *dates, on_progress=updates.append, **settings), updates


def lines(updates: list[dict[str, Any]]) -> list[str]:
    return [update["log"] for update in updates if update.get("log")]


def test_the_job_reads_every_part_with_the_digest_model_and_gives_the_counts_and_the_cost(
    desk: Desk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory, "DIGEST_PARTS_LIMIT", 2)  # Mina has four parts: more than a question reads
    seen: list[str] = []
    model = FakeLLM()

    def factory(info: Any) -> FakeLLM:
        seen.append(info.id)
        return model

    found, updates = run(desk, llm_factory=factory)
    assert set(found) == RESULT_KEYS and found["conversation"] == MINA
    assert (found["periods"], found["periodsBuilt"], found["periodsCached"], found["periodsFailed"]) == (4, 4, 0, 0)
    assert found["usd"] > 0 and found["usd"] == found["usage"]["cost_usd"] and found["usage"]["calls"] == 4
    # more parts than a question reads: the job has no limit of parts
    assert len(model.map_calls) == 4 > memory.DIGEST_PARTS_LIMIT and model.reduce_calls == []
    assert seen == [memory.DIGEST_MODEL]
    assert lines(updates) == [
        "Reading the conversation",
        *(f"Reading {label} (part {place} of 4)" for place, label in enumerate(PERIOD_LABELS, start=1)),
        "The conversation is read for chat: 4 parts read now, 0 from earlier runs, 0 not read.",
    ]
    steps = [update["progress"] for update in updates if update.get("progress")]
    assert steps[0] == {"step": 1, "of": 1, "label": "Reading the conversation"}
    assert steps[-1] == {"step": 5, "of": 5, "label": "Reading part 4 of 4"}
    assert [step["step"] for step in steps] == sorted(step["step"] for step in steps)


def test_a_second_run_reads_no_part_again_and_a_question_then_reads_the_whole_conversation(desk: Desk) -> None:
    run(desk)
    model = FakeLLM()
    found, updates = run(desk, model)
    assert (found["periodsBuilt"], found["periodsCached"], found["usd"]) == (0, 4, 0.0) and model.calls == []
    assert lines(updates)[1] == "Jan 2023 was read in an earlier run (part 1 of 4)"
    # the port of a question, with its limit of two new parts, now has nothing new to read
    more = desk.more(model)
    plain = memory.DigestSettings(
        places=more["places"], options=more["options"], counter=more["counter"], llm_factory=more["llm_factory"]
    )
    port = memory.DigestPort(desk.archive, desk.names, plain)
    from chat_engine.cancel import CancelToken

    assert len(port.digest(MINA, since=None, until=None, cancel=CancelToken())["periods"]) == 4 and model.calls == []


def test_the_days_choose_the_parts_and_a_part_that_fails_is_counted(desk: Desk) -> None:
    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith("2023-02"):
            raise LLMError("x (HTTP 500)", status_code=500)
        return None

    found, updates = run(desk, FakeLLM(responder), "2023-01-01", "2023-02-28")
    assert (found["periods"], found["periodsBuilt"], found["periodsFailed"]) == (1, 1, 1)
    assert "Feb 2023 could not be read (part 2 of 2)" in lines(updates)
    assert lines(updates)[-1] == "The conversation is read for chat: 1 part read now, 0 from earlier runs, 1 not read."


def test_the_model_port_of_the_job_is_the_one_of_a_report(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeLLM()
    monkeypatch.setattr(reports, "default_llm", lambda _info: model)
    more = {name: value for name, value in desk.more().items() if name != "llm_factory"}
    found = digest_job.run(MINA, "2023-01-01", "2023-01-31", **more)
    assert found["periodsBuilt"] == 1 and len(model.calls) == 1


def failing(error: Exception) -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        raise error

    return FakeLLM(responder)


OPENAI_KEY = "OPENAI_API_KEY in the file .env in Recall's data folder"


@pytest.mark.parametrize(
    ("error", "code", "sentence"),
    [
        (LLMError("x (HTTP 401)", status_code=401), "key_rejected", f"OpenAI rejected the API key. Check {OPENAI_KEY}, then ask again."),
        (LLMError("x (HTTP 404)", status_code=404), "unknown_model", "OpenAI does not know the model gpt-6.1-sol. The conversation was not read."),
        (
            LLMError("x (HTTP 429)", status_code=429),
            "rate_limit",
            "OpenAI is limiting requests, or the account has no credit left. Wait a moment, then ask again.",
        ),
        (
            LLMError("the model service failed (APIConnectionError)"),
            "offline",
            "OpenAI could not be reached. Check the internet connection, then ask again.",
        ),
        (LLMError("the model service failed (APITimeoutError)"), "timeout", "The model took too long to answer. Ask again."),
        (LLMError("x (HTTP 500)", status_code=500), "model_error", "The model could not read the conversation. Ask again."),
    ],
)
def test_a_job_in_which_every_part_fails_says_why_in_words_of_its_own(
    desk: Desk, error: Exception, code: str, sentence: str
) -> None:
    with pytest.raises(digest_job.DigestFailure) as told:
        run(desk, failing(error), model=MODEL)
    assert (told.value.code, str(told.value)) == (code, sentence)
    # it is the failure that the server tells in one sentence, with what the run spent
    assert isinstance(told.value, digests.ReportFailure) and told.value.usage is not None


def test_days_with_no_message_and_a_conversation_that_is_not_there_say_so(desk: Desk) -> None:
    sentence = "This conversation has no messages with text in these dates."
    for conversation, dates in ((MINA, ("1999-01-01", "1999-01-02")), ("+15559990000", ())):
        with pytest.raises(digest_job.DigestFailure) as told:
            digest_job.run(conversation, *dates, **desk.more())
        assert (told.value.code, str(told.value)) == ("empty", sentence)
    with pytest.raises(ValueError):
        digest_job.run("", **desk.more())


def test_with_no_key_the_job_says_where_the_key_goes(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    more = {name: value for name, value in desk.more().items() if name != "llm_factory"}
    with pytest.raises(digest_job.DigestFailure) as told:
        digest_job.run(MINA, **more)
    assert (told.value.code, str(told.value)) == (
        "no_key",
        "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then ask again.",
    )


def test_no_sentence_of_the_job_names_a_report(desk: Desk) -> None:
    codes = [
        "no_key", "key_rejected", "unknown_model", "rate_limit", "timeout", "offline", "model_error", "empty",
        "too_large", "archive_error", "cache_unreadable", "cancelled", "run_failed", "too_many_parts", "a-new-code",
    ]  # fmt: skip
    said = {code: digest_job.sentence_of(code, "OpenAI", "gpt-6-luna") for code in codes}
    assert len(set(said.values())) == len(codes) - 1  # a code with no sentence of its own gets the last one
    assert said["a-new-code"] == said["too_many_parts"] == "The conversation could not be read for chat. Ask again."
    for sentence in said.values():
        assert "report" not in sentence.lower() and sentence.endswith(".")
    assert said["cancelled"] == "The reading was stopped. What was read is kept."
    assert said["cache_unreadable"] == "What earlier runs read could not be read or kept. The conversation was not read."
    assert said["archive_error"] == "The messages of this conversation could not be read. Update the archive, then ask again."
    assert said["too_large"] == "The model can not read this conversation with its limits. The conversation was not read."
    assert said["run_failed"] == "The reading failed. What was read is kept: ask again."


def test_a_listener_that_fails_does_not_stop_the_job(desk: Desk) -> None:
    def broken(_update: dict[str, Any]) -> None:
        raise RuntimeError("a listener that fails")

    assert digest_job.run(MINA, on_progress=broken, **desk.more())["periods"] == 4
