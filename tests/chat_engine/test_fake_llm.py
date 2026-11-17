import json

import pytest

from chat_engine.errors import RateLimitedError
from chat_engine.fake_llm import FakeLLM, call, done, say
from chat_engine.ports import ChatMessage, Completed, LLMRequest, TextDelta, ToolCall


def _request() -> LLMRequest:
    return LLMRequest(model="test-model", messages=(ChatMessage("user", "hi"),), tools=(), max_output_tokens=100)


def test_static_step_plays_its_events_and_adds_a_completion() -> None:
    fake = FakeLLM([say("hello there", chunk=4)])
    events = list(fake.stream(_request()))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["hell", "o th", "ere"]
    assert isinstance(events[-1], Completed)
    assert fake.finished == 1 and fake.closed_early == 0
    assert fake.requests[0].model == "test-model"


def test_function_step_sees_the_request() -> None:
    fake = FakeLLM([lambda request: say(request.messages[-1].content.upper()) + [done(10, 5)]])
    events = list(fake.stream(_request()))
    assert events[0].text == "HI"
    assert events[-1].usage.input_tokens == 10


def test_call_builds_a_tool_call_with_json_arguments() -> None:
    made = call("search_messages", query="lease", limit=3)
    assert isinstance(made, ToolCall)
    assert json.loads(made.arguments) == {"query": "lease", "limit": 3}


def test_exception_in_the_script_is_raised_in_the_stream() -> None:
    fake = FakeLLM([[*say("partial"), RateLimitedError()]])
    stream = fake.stream(_request())
    assert next(stream).text == "partial"
    with pytest.raises(RateLimitedError):
        next(stream)


def test_closing_the_stream_early_is_counted() -> None:
    fake = FakeLLM([say("one two three", chunk=4)])
    stream = fake.stream(_request())
    next(stream)
    stream.close()
    assert fake.closed_early == 1 and fake.finished == 0


def test_running_out_of_script_fails_loudly() -> None:
    fake = FakeLLM([])
    with pytest.raises(AssertionError):
        fake.stream(_request())
