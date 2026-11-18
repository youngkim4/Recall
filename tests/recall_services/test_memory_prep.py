"""memory_prep.py: the memory of chat read ahead of time, and what is read of it."""

from __future__ import annotations

from typing import Any

import pytest

from recall_services import digests, memory, memory_prep
from recall_services.report_failures import ReportFailure
from report_engine import CancelToken, LLMError
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.recall_services.report_world import Desk
from tests.report_engine.synth import MINA, THEO

RESULT_KEYS = {
    "conversations", "conversationsRead", "conversationsFailed",
    "periodsBuilt", "periodsCached", "periodsFailed", "usd", "usage",
}  # fmt: skip
PAYLOAD_KEYS = {
    "state", "model", "minMessages", "conversations", "conversationsRead", "conversationsSkipped",
    "parts", "partsRead", "partsNotRead", "estimatedCost", "error", "code",
}  # fmt: skip


def estimate_more(desk: Desk) -> dict[str, Any]:
    """The settings of an estimate, without the archive and the names."""
    return {name: value for name, value in desk.estimate_more().items() if name not in ("archive", "names")}


def run(desk: Desk, llm: Any = None, **more: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []
    found = memory_prep.run(on_progress=updates.append, **{**desk.more(llm), **more})
    return found, updates


def lines(updates: list[dict[str, Any]]) -> list[str]:
    return [update["log"] for update in updates if update.get("log")]


def steps(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [update["progress"] for update in updates if update.get("progress")]


def failing_for(month: str, error: Exception) -> FakeLLM:
    def responder(call: FakeCall) -> Any:
        if call.first_day.startswith(month):
            raise error
        return None

    return FakeLLM(responder)


# --- what is read ------------------------------------------------------------------


def test_the_constants_of_the_contract() -> None:
    assert (memory_prep.PREPARE_MIN_MESSAGES, memory_prep.PREPARE_PARALLEL) == (200, 4)
    assert memory_prep.PREPARE_AUTO_USD == 0.25


def test_the_conversations_to_read_have_enough_messages_and_the_busiest_come_first(desk: Desk) -> None:
    assert memory_prep.candidates(desk.archive) == (MINA,)
    assert memory_prep.candidates(desk.archive, min_messages=10) == (MINA, THEO)
    assert memory_prep.candidates(desk.archive, min_messages=100_000) == ()


def test_with_nothing_read_the_state_is_none_and_the_cost_is_the_estimate_of_the_digest(desk: Desk) -> None:
    found = memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))
    expected = digests.estimate(MINA, model=memory.DIGEST_MODEL, **desk.estimate_more())
    assert set(found) == PAYLOAD_KEYS
    assert (found["state"], found["model"], found["minMessages"]) == ("none", memory.DIGEST_MODEL, 200)
    assert (found["conversations"], found["conversationsRead"], found["conversationsSkipped"]) == (1, 0, 0)
    assert (found["parts"], found["partsRead"], found["partsNotRead"]) == (4, 0, 4)
    assert found["estimatedCost"] == pytest.approx(expected.usd) and found["estimatedCost"] > 0
    assert (found["error"], found["code"]) == (None, None)


def test_with_some_parts_read_the_state_is_partial_and_with_all_ready(desk: Desk) -> None:
    digests.build(MINA, "2023-01-01", "2023-01-31", model=memory.DIGEST_MODEL, **desk.more())
    partial = memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))
    assert (partial["state"], partial["partsRead"], partial["partsNotRead"]) == ("partial", 1, 3)
    run(desk)
    ready = memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))
    assert (ready["state"], ready["conversationsRead"], ready["partsNotRead"], ready["estimatedCost"]) == (
        "ready", 1, 0, 0.0
    )  # fmt: skip


def test_parts_read_with_another_model_are_not_read_for_chat(desk: Desk) -> None:
    digests.build(MINA, model="gpt-6.1-sol", **desk.more())
    assert memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))["state"] == "none"


def test_a_conversation_whose_parts_can_not_be_planned_is_skipped(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    real = digests.estimate

    def too_large(conversation: str, *args: Any, **more: Any) -> Any:
        if conversation == THEO:
            raise ReportFailure("too_large", "The model can not read this conversation with its limits.")
        return real(conversation, *args, **more)

    monkeypatch.setattr(digests, "estimate", too_large)
    found = memory_prep.payload(desk.archive, desk.names, min_messages=10, **estimate_more(desk))
    assert (found["conversations"], found["conversationsSkipped"], found["parts"]) == (2, 1, 4)


def test_an_archive_with_no_conversation_to_read_says_none_and_costs_nothing(desk: Desk) -> None:
    found = memory_prep.payload(desk.archive, desk.names, min_messages=100_000, **estimate_more(desk))
    assert (found["state"], found["conversations"], found["parts"], found["estimatedCost"]) == ("none", 0, 0, 0.0)


def test_a_model_that_can_not_be_run_is_said_in_the_answer(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "DIGEST_MODEL", "no-such-model")
    found = memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))
    assert (found["state"], found["code"], found["parts"]) == ("none", "unknown_model", 0)
    assert found["error"] and "no-such-model" not in found["error"]


def test_when_the_archive_can_not_be_read_the_answer_says_so(desk: Desk) -> None:
    desk.archive.close()
    found = memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))
    assert (found["state"], found["code"]) == ("none", "archive_error") and found["error"]


# --- the server starts a reading by itself only when it is small ---------------------


@pytest.mark.parametrize(
    ("state", "cost", "not_read", "start"),
    [
        ("none", 0.01, 3, False),
        ("partial", 0.25, 3, True),
        ("ready", 0.10, 1, True),
        ("partial", 0.26, 3, False),
        ("ready", 0.0, 0, False),
        ("partial", None, 3, False),
    ],
)
def test_the_server_starts_a_reading_after_an_import_only_when_it_is_small(
    state: str, cost: Any, not_read: int, start: bool
) -> None:
    found = {"state": state, "estimatedCost": cost, "partsNotRead": not_read, "error": None}
    assert memory_prep.worth_starting(found) is start


def test_after_an_import_that_changed_every_part_read_before_the_reading_still_starts() -> None:
    """The parts that were read have new keys now: nothing is read, but the reader chose the memory before."""
    found = {"state": "none", "estimatedCost": 0.05, "partsNotRead": 2, "error": None}
    assert memory_prep.worth_starting(found) is False
    assert memory_prep.worth_starting(found, read_before=True) is True
    assert memory_prep.worth_starting({**found, "estimatedCost": 0.3}, read_before=True) is False


def test_read_before_looks_for_notes_of_the_digest_model_on_a_conversation_of_the_memory(desk: Desk) -> None:
    where = desk.places.cache_path
    assert memory_prep.read_before(desk.archive, places=desk.places) is False
    assert not where.exists()  # a look makes no file
    digests.build(MINA, model="gpt-6.1-sol", **desk.more())
    assert memory_prep.read_before(desk.archive, places=desk.places) is False
    digests.build(THEO, model=memory.DIGEST_MODEL, **desk.more())
    assert memory_prep.read_before(desk.archive, places=desk.places) is False  # Theo has too few messages
    assert memory_prep.read_before(desk.archive, places=desk.places, min_messages=10) is True
    digests.build(MINA, "2023-01-01", "2023-01-31", model=memory.DIGEST_MODEL, **desk.more())
    assert memory_prep.read_before(desk.archive, places=desk.places) is True


def test_notes_that_can_not_be_read_say_nothing_was_read(desk: Desk) -> None:
    desk.places.cache_path.parent.mkdir(parents=True, exist_ok=True)
    desk.places.cache_path.write_text("not a database", encoding="utf-8")
    assert memory_prep.read_before(desk.archive, places=desk.places) is False


def test_the_least_number_of_messages_is_read_at_each_call(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 10)
    assert memory_prep.candidates(desk.archive) == (MINA, THEO)
    assert memory_prep.payload(desk.archive, desk.names, **estimate_more(desk))["minMessages"] == 10


def test_an_answer_with_an_error_starts_nothing() -> None:
    found = {"state": "partial", "estimatedCost": 0.0, "partsNotRead": 1, "error": "x"}
    assert memory_prep.worth_starting(found) is False


# --- the job -----------------------------------------------------------------------


def test_the_job_reads_every_part_with_the_digest_model_and_gives_the_counts_and_the_cost(desk: Desk) -> None:
    seen: list[str] = []
    model = FakeLLM()

    def factory(info: Any) -> FakeLLM:
        seen.append(info.id)
        return model

    found, updates = run(desk, llm_factory=factory)
    assert set(found) == RESULT_KEYS
    assert (found["conversations"], found["conversationsRead"], found["conversationsFailed"]) == (1, 1, 0)
    assert (found["periodsBuilt"], found["periodsCached"], found["periodsFailed"]) == (4, 0, 0)
    assert found["usd"] > 0 and found["usd"] == found["usage"]["cost_usd"] and found["usage"]["calls"] == 4
    assert set(seen) == {memory.DIGEST_MODEL} and len(model.map_calls) == 4 and model.reduce_calls == []
    assert lines(updates) == [
        "Looking at 1 conversation with 200 messages or more",
        "4 parts of 1 conversation are not read yet, 0 are read",
        "A conversation is read: 4 parts read now, 0 from earlier runs, 0 not read (1 of 1)",
        "The memory is read: 4 parts read now, 0 from earlier runs, 0 not read.",
    ]
    told = steps(updates)
    assert told[0] == {"step": 1, "of": 2, "label": "Looking at your conversations", "done": 0, "total": 1}
    assert told[1] == {"step": 2, "of": 2, "label": "Reading your conversations", "done": 0, "total": 4}
    assert told[-1] == {"step": 2, "of": 2, "label": "Reading your conversations", "done": 4, "total": 4}
    assert [step["done"] for step in told[1:]] == sorted(step["done"] for step in told[1:])


def test_the_lines_and_the_progress_hold_counts_only(desk: Desk) -> None:
    _found, updates = run(desk, min_messages=10)
    told = "\n".join([*lines(updates), *map(str, steps(updates))])
    text = desk.archive.messages(MINA, limit=1)["messages"][0]["text"]
    for private in (MINA, THEO, "2025550143", "Mina", "Jan 2023", text):
        assert private not in told


def test_a_second_run_reads_no_part_again(desk: Desk) -> None:
    run(desk)
    model = FakeLLM()
    found, updates = run(desk, model)
    assert (found["periodsBuilt"], found["periodsCached"], found["usd"]) == (0, 4, 0.0) and model.calls == []
    assert lines(updates)[1] == "0 parts of 0 conversations are not read yet, 4 are read"


def test_the_busiest_conversation_is_read_first(desk: Desk) -> None:
    model = FakeLLM()
    found, _updates = run(desk, model, min_messages=10, parallel=1)
    assert found["conversations"] == 2 and found["periodsBuilt"] == 5
    assert [call.first_day[:4] for call in model.map_calls] == ["2023", "2023", "2023", "2023", "2024"]


def test_a_conversation_that_fails_is_counted_and_the_others_are_read(desk: Desk) -> None:
    found, updates = run(desk, failing_for("2024", LLMError("x (HTTP 500)", status_code=500)), min_messages=10)
    assert (found["conversationsRead"], found["conversationsFailed"]) == (1, 1)
    assert (found["periodsBuilt"], found["periodsFailed"]) == (4, 1)
    # a call that the port failed is in no usage: nobody knows what it cost
    assert found["usage"]["calls"] == 4
    # the two run at once: the short one can end first
    assert [line for line in lines(updates) if line.startswith("A conversation could not be read (")] in (
        ["A conversation could not be read (1 of 2)"], ["A conversation could not be read (2 of 2)"]
    )  # fmt: skip


def test_a_part_that_fails_is_counted_and_its_conversation_is_not_read_whole(desk: Desk) -> None:
    found, _updates = run(desk, failing_for("2023-02", LLMError("x (HTTP 500)", status_code=500)))
    assert (found["conversationsRead"], found["conversationsFailed"]) == (0, 0)
    assert (found["periodsBuilt"], found["periodsFailed"]) == (3, 1)


OPENAI_KEY = "OPENAI_API_KEY in the file .env in Recall's data folder"


@pytest.mark.parametrize(
    ("error", "code", "sentence"),
    [
        (
            LLMError("x (HTTP 401)", status_code=401),
            "key_rejected",
            f"OpenAI rejected the API key. Check {OPENAI_KEY}, then read the memory again. What was read is kept.",
        ),
        (
            LLMError("x (HTTP 429)", status_code=429),
            "rate_limit",
            "OpenAI is limiting requests, or the account has no credit left. Wait a moment, then read the memory "
            "again. What was read is kept.",
        ),
        (
            LLMError("the model service failed (APIConnectionError)"),
            "offline",
            "OpenAI could not be reached. Check the internet connection, then read the memory again. "
            "What was read is kept.",
        ),
    ],
)
def test_a_failure_that_every_conversation_would_have_stops_the_job(
    desk: Desk, error: Exception, code: str, sentence: str
) -> None:
    model = failing_for("20", error)
    with pytest.raises(memory_prep.MemoryFailure) as told:
        run(desk, model, min_messages=10, parallel=1)
    assert (told.value.code, str(told.value)) == (code, sentence)
    assert isinstance(told.value, ReportFailure) and told.value.usage is not None
    # the busiest conversation failed: the next one was not started
    assert {call.first_day[:4] for call in model.calls} == {"2023"}


def test_with_no_key_the_job_says_where_the_key_goes(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    more = {name: value for name, value in desk.more().items() if name != "llm_factory"}
    with pytest.raises(memory_prep.MemoryFailure) as told:
        memory_prep.run(**more)
    assert told.value.code == "no_key"
    assert str(told.value) == (
        "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then read the memory again."
    )


def test_a_cancelled_job_reads_nothing_more_and_says_so(desk: Desk) -> None:
    token = CancelToken()
    token.cancel()
    model = FakeLLM()
    with pytest.raises(memory_prep.MemoryFailure) as told:
        run(desk, model, cancel=token)
    assert (told.value.code, str(told.value)) == ("cancelled", "The reading was stopped. What was read is kept.")
    assert model.calls == []


def test_a_job_on_an_archive_that_can_not_be_read_fails_in_one_sentence(desk: Desk) -> None:
    desk.archive.close()
    with pytest.raises(memory_prep.MemoryFailure) as told:
        run(desk)
    assert told.value.code == "archive_error" and "archive" in str(told.value)


def test_a_listener_that_fails_does_not_stop_the_job(desk: Desk) -> None:
    def broken(_update: dict[str, Any]) -> None:
        raise RuntimeError("the screen is gone")

    found = memory_prep.run(on_progress=broken, **desk.more())
    assert found["periodsBuilt"] == 4


def test_an_error_that_nobody_planned_for_stops_the_pool_and_fails_the_job(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    pools: list[Any] = []

    def broken(*_args: Any, pool: Any = None, **_more: Any) -> Any:
        pools.append(pool)
        raise RuntimeError("a fault")

    monkeypatch.setattr(digests, "build", broken)
    with pytest.raises(RuntimeError, match="a fault"):
        run(desk)
    assert pools and pools[0].stopped
