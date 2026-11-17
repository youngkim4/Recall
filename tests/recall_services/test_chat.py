"""chat.py: a question to the chat engine, in the shapes that the app reads. The model is a script."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pytest

from archive import Archive
from chat_engine import errors as engine_errors
from chat_engine.cancel import CancelToken
from chat_engine.fake_llm import FakeLLM, Step, call, say
from chat_engine.ports import LLMRequest
from recall_services import chat, models
from recall_services.models import ChatModelConfig
from recall_services.names import Names
from tests.recall_services import world
from tests.recall_services.test_messages import RESULT_KEYS

# AskResponse in app/src/types.ts at 89ab504
ASK_KEYS = {"question", "contact", "terms", "answer", "mode", "citations"}
# what the engine knows beyond that; the app does not read these yet
MORE_KEYS = {"unverifiedQuotes", "steps", "cancelled", "truncated", "usage"}
FLAT_TIRE = "The wheelbarrow has a flat tire"
QUESTION = "Who said the wheelbarrow has a flat tire?"
NOW = datetime(2026, 11, 17, 9, 30)


def shown_ref(request: LLMRequest, words: str) -> str:
    """The reference of the line with these words, in what the model was shown."""
    for message in request.messages:
        for line in message.content.splitlines():
            found = re.match(r"(m\d+) ", line)
            if found and words in line:
                return found.group(1)
    raise AssertionError(f"no line that the model was shown holds {words!r}")


def cite(words: str, lead: str = "Someone in the group wrote it") -> Callable[[LLMRequest], list[Any]]:
    return lambda request: say(f"{lead} [{shown_ref(request, words)}].", chunk=12)


class Factory:
    """Stands in for the model port. It keeps what it was asked for."""

    def __init__(self, script: Sequence[Step]) -> None:
        self.llm = FakeLLM(script)
        self.configs: list[ChatModelConfig] = []

    def __call__(self, config: ChatModelConfig) -> FakeLLM:
        self.configs.append(config)
        return self.llm

    def sent(self) -> str:
        return "\n".join(message.content for request in self.llm.requests for message in request.messages)


def run(archive: Archive, saved: Names, factory: Factory, question: str = QUESTION, **more: Any) -> list[dict[str, Any]]:
    settings = {"contact": "", "model": "gpt-6.1-sol", "history": None, **more}
    return list(chat.stream(question, archive=archive, names=saved, llm_factory=factory, clock=lambda: NOW, **settings))


def done(events: list[dict[str, Any]]) -> dict[str, Any]:
    assert events[-1]["type"] == "done", events[-1]
    return events[-1]["payload"]


def test_the_events_have_the_shapes_the_app_reads(archive: Archive, saved: Names) -> None:
    events = run(archive, saved, Factory([cite(FLAT_TIRE)]))
    kinds = [event["type"] for event in events]
    assert set(kinds) == {"status", "delta", "done"} and kinds[-1] == "done" and kinds.count("done") == 1
    for event in events[:-1]:
        assert set(event) <= {"type", "text", "reset"} and isinstance(event["text"], str)
    payload = done(events)
    assert set(payload) == ASK_KEYS | MORE_KEYS
    assert all(set(citation) == RESULT_KEYS for citation in payload["citations"])
    json.dumps(events)


def test_the_answer_is_the_text_that_streamed(archive: Archive, saved: Names) -> None:
    events = run(archive, saved, Factory([cite(FLAT_TIRE)]))
    streamed = "".join(event["text"] for event in events if event["type"] == "delta")
    payload = done(events)
    assert payload["answer"] == streamed == "Someone in the group wrote it [1]."
    assert (payload["question"], payload["contact"], payload["mode"], payload["terms"]) == (QUESTION, "", "ai", [])
    assert (payload["cancelled"], payload["truncated"], payload["unverifiedQuotes"]) == (False, False, [])
    assert payload["usage"]["model_calls"] == 1


def test_a_citation_is_a_search_result_with_the_names_of_the_screen(
    archive: Archive, saved: Names, orchard: world.Orchard
) -> None:
    citation = done(run(archive, saved, Factory([cite(FLAT_TIRE)])))["citations"][0]
    assert citation == {
        "messageId": orchard.id("g3"),
        "chatId": world.GARDEN,
        "displayName": world.GARDEN_TITLE,
        # the writer is not saved: no name, and never the mask that the model read
        "senderName": "",
        "timestamp": "2024-04-01 10:02:00",
        "sender": world.UNSAVED,
        "text": FLAT_TIRE,
        "attachment": "",
        "isFromMe": False,
    }
    # the day view can open at this message
    held = archive.day(citation["chatId"], citation["timestamp"][:10], around=citation["timestamp"])
    assert citation["messageId"] in [message["id"] for message in held["messages"]]


def test_a_citation_of_a_saved_person_and_of_an_unsaved_conversation(archive: Archive, saved: Names) -> None:
    by_tomas = done(run(archive, saved, Factory([cite("I do, behind the shed")]), "Who has the wheelbarrow behind the shed?"))
    assert by_tomas["citations"][0]["senderName"] == "Tomas Reyes"
    script = [cite("Your delivery of mulch is on its way", "The delivery is coming")]
    unsaved = done(run(archive, saved, Factory(script), "When is the delivery of mulch on its way?"))["citations"][0]
    assert (unsaved["chatId"], unsaved["displayName"], unsaved["senderName"]) == (world.UNSAVED, "", "")


def test_the_model_reads_names_and_masks_and_no_handle(archive: Archive, saved: Names) -> None:
    factory = Factory([cite(FLAT_TIRE)])
    run(archive, saved, factory, "Who wrote about the wheelbarrow and who should call 555 017 9999?")
    sent = factory.sent()
    assert "Ines Calder" in sent and "a number ending in 9999" in sent
    for raw in (world.UNSAVED, "5550179999", "555 017 9999", world.UNSAVED_EMAIL, world.INES_PHONE, world.GARDEN):
        assert raw not in sent, raw


def test_the_model_and_its_limits_come_from_the_registry(archive: Archive, saved: Names) -> None:
    factory = Factory([cite(FLAT_TIRE)])
    run(archive, saved, factory, model="gpt-6.1-sol")
    assert factory.configs == [models.chat_config("gpt-6.1-sol")]
    request = factory.llm.requests[0]
    assert (request.model, request.max_output_tokens) == ("gpt-6.1-sol", models.chat_config("gpt-6.1-sol").max_output_tokens)


@pytest.mark.parametrize("nothing", ["", None, "   "])
def test_no_model_named_is_the_default_model(archive: Archive, saved: Names, nothing: object) -> None:
    factory = Factory([cite(FLAT_TIRE)])
    run(archive, saved, factory, model=nothing)
    assert factory.configs[0].model == models.default_id()


@pytest.mark.parametrize(
    ("model", "code", "sentence"),
    [
        ("gpt-7", "unknown_model", "Recall does not know the model gpt-7. Choose one of: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol."),
        ("gpt-6-astra", "model_not_offered", "The model gpt-6-astra is not offered. Choose one of: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol."),
    ],
)
def test_a_model_that_can_not_be_run_is_one_error_and_no_call(
    archive: Archive, saved: Names, model: str, code: str, sentence: str
) -> None:
    factory = Factory([])
    assert run(archive, saved, factory, model=model) == [{"type": "error", "code": code, "error": sentence}]
    assert factory.configs == []


def test_the_history_of_the_browser_goes_to_the_model(archive: Archive, saved: Names) -> None:
    history = [
        {"role": "user", "content": "Who has the wheelbarrow?"},
        {"role": "assistant", "content": "Tomas has it behind the shed [1]."},
        "not a row",
        {"role": "user"},
    ]
    factory = Factory([cite(FLAT_TIRE, "It has a flat tire")])
    payload = done(run(archive, saved, factory, "And what is wrong with it? wheelbarrow tire", history=history))
    sent = factory.sent()
    assert "Who has the wheelbarrow?" in sent and "Tomas has it behind the shed" in sent
    # the number of the shown answer named a message of that answer, not of this one
    assert "shed [1]" not in sent
    assert payload["answer"] == "It has a flat tire [1]."


EARLIER = [
    {"role": "user", "content": "When does the mulch come?"},
    {"role": "assistant", "content": "The delivery of mulch is on Monday [1]."},
]


def with_conversation(rows: list[dict[str, Any]], conversation: str) -> list[dict[str, Any]]:
    return [{**row, "contact": conversation} if row["role"] == "user" else row for row in rows]


def test_an_earlier_answer_of_a_wider_chat_does_not_go_to_a_question_with_a_scope(
    archive: Archive, saved: Names
) -> None:
    """The picker of the app can change inside one chat. A row that does not say its conversation was asked of everything."""
    factory = Factory([cite(FLAT_TIRE)])
    done(run(archive, saved, factory, contact=world.GARDEN, history=EARLIER))
    assert "The delivery of mulch is on Monday" not in factory.sent()


def test_an_earlier_answer_of_another_conversation_does_not_go_to_a_question_with_a_scope(
    archive: Archive, saved: Names
) -> None:
    factory = Factory([cite(FLAT_TIRE)])
    done(run(archive, saved, factory, contact=world.GARDEN, history=with_conversation(EARLIER, world.INES_PHONE)))
    assert "The delivery of mulch is on Monday" not in factory.sent()


def test_an_earlier_answer_of_the_same_conversation_goes_to_the_model(archive: Archive, saved: Names) -> None:
    factory = Factory([cite(FLAT_TIRE)])
    done(run(archive, saved, factory, contact=world.GARDEN, history=with_conversation(EARLIER, world.GARDEN)))
    assert "The delivery of mulch is on Monday" in factory.sent()


def test_every_earlier_answer_goes_to_a_question_of_everything(archive: Archive, saved: Names) -> None:
    factory = Factory([cite(FLAT_TIRE)])
    done(run(archive, saved, factory, history=with_conversation(EARLIER, world.INES_PHONE)))
    assert "The delivery of mulch is on Monday" in factory.sent()


def test_a_question_that_is_too_long_is_one_error_and_no_call(archive: Archive, saved: Names) -> None:
    factory = Factory([])
    events = run(archive, saved, factory, "why " * 5_000)
    assert events == [
        {"type": "error", "code": "question_too_long", "error": "The question is too long. Shorten it, then ask again."}
    ]
    assert factory.configs == []


def test_a_long_row_of_the_history_is_cut(archive: Archive, saved: Names) -> None:
    long_answer = "The delivery of mulch is on Monday. " + "filler " * 20_000 + "THE-END"
    history = [EARLIER[0], {"role": "assistant", "content": long_answer}]
    factory = Factory([cite(FLAT_TIRE)])
    done(run(archive, saved, factory, history=history))
    assert "The delivery of mulch is on Monday." in factory.sent() and "THE-END" not in factory.sent()


def test_a_conversation_is_the_scope_and_is_never_widened(archive: Archive, saved: Names) -> None:
    def look_outside(_request: LLMRequest) -> list[Any]:
        return [call("search_messages", "s1", query="mulch")]

    factory = Factory([look_outside, cite(FLAT_TIRE)])
    payload = done(run(archive, saved, factory, contact=world.GARDEN))
    assert payload["contact"] == world.GARDEN
    assert {citation["chatId"] for citation in payload["citations"]} == {world.GARDEN}
    sent = factory.sent()
    # "mulch" is in three other conversations and in none of this one
    assert "mulch order" not in sent and "delivery of mulch" not in sent


def test_a_scope_that_names_no_conversation_of_the_archive_finds_nothing(archive: Archive, saved: Names) -> None:
    factory = Factory([say("I found nothing about that.")])
    payload = done(run(archive, saved, factory, contact="+15550000000"))
    assert payload["citations"] == [] and FLAT_TIRE not in factory.sent()


@pytest.mark.parametrize(
    ("error", "code", "sentence"),
    [
        (engine_errors.NoApiKeyError(), "no_key", "No API key is set. Add your OpenAI API key in Settings, then ask again."),
        (engine_errors.KeyRejectedError(), "key_rejected", "OpenAI rejected the API key. Check the key in Settings, then ask again."),
        (engine_errors.UnknownModelError(), "unknown_model", "OpenAI does not know the model gpt-6.1-sol. Choose another level in Settings."),
        (engine_errors.RateLimitedError(), "rate_limit", "OpenAI is limiting requests. Wait a moment, then ask again."),
        (engine_errors.RateLimitedError(retry_after=19.2), "rate_limit", "OpenAI is limiting requests. Wait 20 seconds, then ask again."),
        (engine_errors.OfflineError(), "offline", "OpenAI could not be reached. Check the internet connection, then ask again."),
        (engine_errors.ModelTimeoutError(), "timeout", "The model took too long to answer. Ask again."),
        (engine_errors.ModelError(), "model_error", "The model call failed. Ask again."),
    ],
)
def test_each_failure_of_the_model_is_one_plain_sentence(
    archive: Archive, saved: Names, error: engine_errors.ChatEngineError, code: str, sentence: str
) -> None:
    events = run(archive, saved, Factory([[error]]))
    assert events[-1]["type"] == "error"
    assert (events[-1]["code"], events[-1]["error"]) == (code, sentence)
    # no keyword answer takes the place of the answer that failed
    assert "done" not in [event["type"] for event in events]
    if error.retry_after is not None:
        assert events[-1]["retry_after"] == 19.2


def test_a_failure_with_words_of_its_own_keeps_them(archive: Archive, saved: Names) -> None:
    no_credit = "The API account has no credit left, or its spending limit is reached."
    events = run(archive, saved, Factory([[engine_errors.RateLimitedError(no_credit)]]))
    assert (events[-1]["code"], events[-1]["error"]) == ("rate_limit", no_credit)


def test_what_the_provider_said_never_reaches_the_person(archive: Archive, saved: Names) -> None:
    secret = "sk-live-0123456789 request id req_42"
    events = run(archive, saved, Factory([[engine_errors.KeyRejectedError(detail=secret)]]))
    assert secret not in json.dumps(events)


def test_a_fault_of_the_model_port_is_a_failed_model_call(archive: Archive, saved: Names) -> None:
    """The engine gives a type to a fault of the port that has none. Its words do not go out."""
    events = run(archive, saved, Factory([[RuntimeError("an inner fault with a path /private/data")]]))
    assert events[-1] == {"type": "error", "code": "model_error", "error": "The model call failed. Ask again."}


def test_a_fault_of_the_service_is_said_in_plain_words_and_is_in_the_log(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*_args: object) -> list[dict[str, Any]]:
        raise RuntimeError("an inner fault with a path /private/data")

    monkeypatch.setattr(chat.messages, "payloads", broken)
    events = run(archive, saved, Factory([cite(FLAT_TIRE)]))
    assert events[-1] == {"type": "error", "code": "chat_error", "error": "The question could not be answered."}
    assert "done" not in [event["type"] for event in events]
    # it is not lost: the log has it
    assert "RuntimeError" in caplog.text and "a chat question failed" in caplog.text


def test_a_failure_of_the_factory_is_an_error_event(archive: Archive, saved: Names) -> None:
    def no_key(_config: ChatModelConfig) -> FakeLLM:
        raise engine_errors.NoApiKeyError()

    events = list(chat.stream(QUESTION, "", "gpt-6.1-sol", None, archive=archive, names=saved, llm_factory=no_key))
    assert [(event["type"], event.get("code")) for event in events] == [("error", "no_key")]


def test_an_archive_that_can_not_be_read_is_an_error_event(orchard: world.Orchard, saved: Names) -> None:
    closed = Archive.open(orchard.archive_path)
    closed.close()
    events = run(closed, saved, Factory([cite(FLAT_TIRE)]))
    assert events[-1] == {
        "type": "error",
        "code": "archive_error",
        "error": "The message archive could not be read. Update the archive, then ask again.",
    }


@pytest.mark.parametrize("question", ["", "   ", None])
def test_an_empty_question_is_refused_in_plain_words(archive: Archive, saved: Names, question: object) -> None:
    factory = Factory([])
    assert run(archive, saved, factory, question) == [  # type: ignore[arg-type]
        {"type": "error", "code": "empty_question", "error": "Type a question first."}
    ]
    assert factory.configs == []


def test_a_cancelled_question_ends_with_what_was_written(archive: Archive, saved: Names) -> None:
    token = CancelToken()

    def write_then_stop(_request: LLMRequest) -> list[Any]:
        token.cancel()
        return say("This is the start of an answer")

    payload = done(run(archive, saved, Factory([write_then_stop]), cancel=token))
    assert payload["cancelled"] is True and payload["mode"] == "ai"


def test_closing_the_stream_stops_the_model_call(archive: Archive, saved: Names) -> None:
    factory = Factory([say("one two three four five six seven eight", chunk=4)])
    events = chat.stream(QUESTION, "", "gpt-6.1-sol", None, archive=archive, names=saved, llm_factory=factory)
    for event in events:
        if event["type"] == "delta":
            break
    events.close()
    assert factory.llm.closed_early == 1


# --- an answer without AI, only when the caller asks for it --------------------


def test_an_answer_without_ai_calls_no_model(archive: Archive, saved: Names) -> None:
    factory = Factory([])
    events = run(archive, saved, factory, "wheelbarrow tire", use_ai=False)
    payload = done(events)
    assert factory.configs == [] and [event["type"] for event in events] == ["done"]
    assert set(payload) == ASK_KEYS | MORE_KEYS
    assert payload["mode"] == "local"
    assert payload["answer"].startswith("No AI model was used.")
    assert FLAT_TIRE in [citation["text"] for citation in payload["citations"]]
    assert all(set(citation) == RESULT_KEYS for citation in payload["citations"])


def test_an_answer_without_ai_keeps_the_scope(archive: Archive, saved: Names) -> None:
    payload = done(run(archive, saved, Factory([]), "mulch", contact=world.GARDEN, use_ai=False))
    assert payload["citations"] == [] and payload["contact"] == world.GARDEN


def test_an_answer_without_ai_needs_no_model_name(archive: Archive, saved: Names) -> None:
    assert done(run(archive, saved, Factory([]), "wheelbarrow", model="gpt-7", use_ai=False))["mode"] == "local"


# --- one answer, with no stream -------------------------------------------------


def test_ask_gives_the_payload_of_the_last_event(archive: Archive, saved: Names) -> None:
    payload = chat.ask(
        QUESTION, "", "gpt-6.1-sol", None, archive=archive, names=saved, llm_factory=Factory([cite(FLAT_TIRE)])
    )
    assert set(payload) == ASK_KEYS | MORE_KEYS and payload["answer"].endswith("[1].")


def test_ask_raises_the_failure_with_its_code(archive: Archive, saved: Names) -> None:
    with pytest.raises(chat.ChatFailure) as caught:
        chat.ask(QUESTION, "", "gpt-7", None, archive=archive, names=saved, llm_factory=Factory([]))
    assert caught.value.code == "unknown_model"
    assert str(caught.value).startswith("Recall does not know the model gpt-7.")


# --- the model port that the app uses -------------------------------------------


def test_with_no_key_no_request_is_made(archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    events = list(chat.stream(QUESTION, "", "gpt-6.1-sol", None, archive=archive, names=saved))
    assert events == [
        {"type": "error", "code": "no_key", "error": "No API key is set. Add your OpenAI API key in Settings, then ask again."}
    ]


def test_the_model_port_of_the_app_goes_to_the_address_of_the_provider_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/elsewhere")
    llm = chat.default_llm(models.chat_config("gpt-6.1-sol"))
    client = llm._client
    assert str(client.base_url).startswith("https://api.openai.com/")
    assert client.max_retries == 0
    assert llm._reasoning_effort == "low"
    assert chat.default_llm(models.chat_config("gpt-6-luna"))._reasoning_effort is None


def test_with_no_way_out_the_real_port_says_offline(archive: Archive, saved: Names) -> None:
    """The real adapter and the real SDK, with every socket blocked: the failure is the one for no connection."""

    def no_retries(config: ChatModelConfig) -> Any:
        llm = chat.default_llm(config)
        llm._max_retries = 0
        return llm

    events = list(chat.stream(QUESTION, "", "gpt-6.1-sol", None, archive=archive, names=saved, llm_factory=no_retries))
    assert (events[-1]["type"], events[-1]["code"]) == ("error", "offline")
