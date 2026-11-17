"""The OpenAI adapter, against a fake client. No request leaves the machine."""

from types import SimpleNamespace as NS
from typing import Any

import httpx
import openai
import pytest

from chat_engine.cancel import CancelToken
from chat_engine.errors import (
    KeyRejectedError,
    ModelError,
    ModelTimeoutError,
    NoApiKeyError,
    OfflineError,
    RateLimitedError,
    UnknownModelError,
)
from chat_engine.openai_llm import OpenAIResponsesLLM
from chat_engine.ports import ChatMessage, Completed, LLMRequest, TextDelta, ToolCall, ToolSpec

URL = "https://api.openai.com/v1/responses"
SPEC = ToolSpec("search_messages", "Find messages.", {"type": "object", "properties": {}})


class FakeStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = items
        self.closed = False

    def __iter__(self):
        for item in self._items:
            if isinstance(item, BaseException):
                raise item
            if self.closed:
                raise httpx.ReadError("stream closed")
            yield item

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.streams: list[FakeStream] = []
        self.responses = self

    def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        stream = FakeStream(outcome)
        self.streams.append(stream)
        return stream


def _request(*messages: ChatMessage, allow_tools: bool = True, cancel: CancelToken | None = None) -> LLMRequest:
    given = messages or (ChatMessage("system", "Be brief."), ChatMessage("user", "hi"))
    return LLMRequest(
        model="model-under-test", messages=given, tools=(SPEC,), max_output_tokens=512,
        allow_tools=allow_tools, cancel=cancel,
    )


def _usage(inp: int = 100, out: int = 20, cached: int = 40, reasoning: int = 5) -> NS:
    return NS(
        input_tokens=inp,
        output_tokens=out,
        input_tokens_details=NS(cached_tokens=cached),
        output_tokens_details=NS(reasoning_tokens=reasoning),
    )


def _delta(text: str, index: int = 0) -> NS:
    return NS(type="response.output_text.delta", delta=text, output_index=index)


def _completed(**extra: Any) -> NS:
    return NS(type="response.completed", response=NS(usage=_usage(), status="completed", **extra))


def _text_events(text: str = "Hello there") -> list[NS]:
    message = NS(type="message", id="msg_1", role="assistant", phase=None,
                 content=[NS(type="output_text", text=text)])
    return [
        NS(type="response.created"),
        NS(type="response.output_item.added", item=NS(type="message", phase=None), output_index=0),
        _delta(text[:5]),
        _delta(text[5:]),
        NS(type="response.output_item.done", item=message, output_index=0),
        _completed(),
    ]


def _status_error(
    cls: type,
    status: int,
    code: str | None = None,
    headers: dict | None = None,
    message: str = "provider says no (key sk-proj-abc123)",
) -> Exception:
    response = httpx.Response(status, request=httpx.Request("POST", URL), headers=headers or {})
    return cls(message, response=response, body={"code": code, "message": "x"})


def _llm(client: FakeClient, **settings: Any) -> OpenAIResponsesLLM:
    waits: list[float] = []
    llm = OpenAIResponsesLLM(client, sleep=waits.append, **settings)
    llm.waits = waits  # type: ignore[attr-defined]
    return llm


def test_request_uses_function_calling_streaming_no_storage_a_timeout_and_a_cap() -> None:
    client = FakeClient(_text_events())
    list(_llm(client, timeout_seconds=42.0).stream(_request()))
    sent = client.calls[0]
    assert sent["model"] == "model-under-test"
    assert sent["instructions"] == "Be brief."
    assert sent["input"] == [{"role": "user", "content": "hi"}]
    assert sent["store"] is False and sent["stream"] is True
    assert sent["max_output_tokens"] == 512
    assert sent["timeout"] == 42.0
    assert sent["tool_choice"] == "auto"
    assert sent["tools"] == [
        {
            "type": "function",
            "name": "search_messages",
            "description": "Find messages.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }
    ]
    assert "reasoning" not in sent and "text" not in sent and "include" not in sent


def test_optional_settings_are_sent_only_when_given() -> None:
    client = FakeClient(_text_events())
    llm = _llm(client, reasoning_effort="low", verbosity="low", carry_reasoning=False)
    list(llm.stream(_request(allow_tools=False)))
    sent = client.calls[0]
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["text"] == {"verbosity": "low"}
    assert sent["tool_choice"] == "none"
    assert "include" not in sent


def test_text_deltas_and_usage_come_through() -> None:
    events = list(_llm(FakeClient(_text_events())).stream(_request()))
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hello", " there"]
    end = events[-1]
    assert isinstance(end, Completed) and end.stop_reason == "stop"
    assert (end.usage.input_tokens, end.usage.output_tokens) == (100, 20)
    assert (end.usage.cached_input_tokens, end.usage.reasoning_tokens) == (40, 5)


def test_function_call_becomes_a_tool_call_and_items_are_carried_back() -> None:
    reasoning = NS(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque")
    function = NS(type="function_call", id="fc_1", call_id="call_9", name="search_messages",
                  arguments='{"query": "lease"}')
    first = [
        NS(type="response.output_item.done", item=reasoning, output_index=0),
        NS(type="response.output_item.done", item=function, output_index=1),
        _completed(),
    ]
    client = FakeClient(first, _text_events("Done"))
    llm = _llm(client)
    events = list(llm.stream(_request()))
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert calls == [ToolCall("call_9", "search_messages", '{"query": "lease"}')]
    carried = events[-1].provider_items
    assert carried == (
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
        {"type": "function_call", "call_id": "call_9", "name": "search_messages",
         "arguments": '{"query": "lease"}', "id": "fc_1"},
    )
    follow_up = _request(
        ChatMessage("system", "Be brief."),
        ChatMessage("user", "hi"),
        ChatMessage("assistant", "", tool_calls=tuple(calls), provider_items=carried),
        ChatMessage("tool", "1 message matches", tool_call_id="call_9"),
    )
    list(llm.stream(follow_up))
    assert client.calls[1]["input"] == [
        {"role": "user", "content": "hi"},
        *carried,
        {"type": "function_call_output", "call_id": "call_9", "output": "1 message matches"},
    ]


def test_ids_are_dropped_when_reasoning_cannot_be_carried() -> None:
    reasoning = NS(type="reasoning", id="rs_1", summary=[], encrypted_content=None)
    function = NS(type="function_call", id="fc_1", call_id="call_9", name="search_messages", arguments="{}")
    stream = [
        NS(type="response.output_item.done", item=reasoning, output_index=0),
        NS(type="response.output_item.done", item=function, output_index=1),
        _completed(),
    ]
    events = list(_llm(FakeClient(stream)).stream(_request()))
    assert events[-1].provider_items == (
        {"type": "function_call", "call_id": "call_9", "name": "search_messages", "arguments": "{}"},
    )


def test_plain_history_is_rebuilt_without_provider_items() -> None:
    client = FakeClient(_text_events())
    request = _request(
        ChatMessage("system", "One."),
        ChatMessage("system", "Two."),
        ChatMessage("user", "first"),
        ChatMessage("assistant", "answer", tool_calls=(ToolCall("c1", "search_messages", "{}"),)),
        ChatMessage("tool", "result", tool_call_id="c1"),
        ChatMessage("user", "second"),
    )
    list(_llm(client).stream(request))
    sent = client.calls[0]
    assert sent["instructions"] == "One.\n\nTwo."
    assert sent["input"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"type": "function_call", "call_id": "c1", "name": "search_messages", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "result"},
        {"role": "user", "content": "second"},
    ]


def test_commentary_phase_marks_the_deltas_and_is_carried_back() -> None:
    note = NS(type="message", id="msg_1", role="assistant", phase="commentary",
              content=[NS(type="output_text", text="Looking.")])
    stream = [
        NS(type="response.output_item.added", item=NS(type="message", phase="commentary"), output_index=0),
        _delta("Looking.", 0),
        NS(type="response.output_item.done", item=note, output_index=0),
        NS(type="response.output_item.added", item=NS(type="message", phase="final_answer"), output_index=1),
        _delta("Answer.", 1),
        _completed(),
    ]
    events = list(_llm(FakeClient(stream)).stream(_request()))
    deltas = [(e.text, e.commentary) for e in events if isinstance(e, TextDelta)]
    assert deltas == [("Looking.", True), ("Answer.", False)]
    assert events[-1].provider_items == ({"role": "assistant", "content": "Looking.", "phase": "commentary"},)


@pytest.mark.parametrize(("reason", "expected"), [("max_output_tokens", "length"), ("content_filter", "filtered")])
def test_incomplete_response_reports_why(reason: str, expected: str) -> None:
    stream = [
        _delta("Partial"),
        NS(type="response.incomplete", response=NS(usage=_usage(), incomplete_details=NS(reason=reason))),
    ]
    events = list(_llm(FakeClient(stream)).stream(_request()))
    assert events[-1].stop_reason == expected


def test_refusal_text_is_passed_on() -> None:
    stream = [NS(type="response.refusal.delta", delta="I cannot help with that."), _completed()]
    events = list(_llm(FakeClient(stream)).stream(_request()))
    assert events[0] == TextDelta("I cannot help with that.")


def test_retry_happens_before_the_first_token_only() -> None:
    limited = _status_error(openai.RateLimitError, 429, "rate_limit_exceeded")
    client = FakeClient(limited, _text_events())
    llm = _llm(client)
    events = list(llm.stream(_request()))
    assert len(client.calls) == 2
    assert llm.waits == [1.0]
    assert isinstance(events[-1], Completed)

    broken = FakeClient([_delta("Hel"), limited], _text_events())
    stream = _llm(broken).stream(_request())
    assert next(stream) == TextDelta("Hel")
    with pytest.raises(RateLimitedError):
        next(stream)
    assert len(broken.calls) == 1


def test_retries_are_bounded() -> None:
    limited = _status_error(openai.RateLimitError, 429, "rate_limit_exceeded")
    client = FakeClient(limited, limited, limited, _text_events())
    llm = _llm(client, max_retries=2)
    with pytest.raises(RateLimitedError):
        list(llm.stream(_request()))
    assert len(client.calls) == 3
    assert llm.waits == [1.0, 2.0]


def test_retry_after_header_is_honoured_or_reported() -> None:
    short = _status_error(openai.RateLimitError, 429, None, {"retry-after": "3"})
    client = FakeClient(short, _text_events())
    llm = _llm(client)
    list(llm.stream(_request()))
    assert llm.waits == [3.0]

    long = _status_error(openai.RateLimitError, 429, None, {"retry-after": "120"})
    client = FakeClient(long, _text_events())
    with pytest.raises(RateLimitedError) as caught:
        list(_llm(client).stream(_request()))
    assert caught.value.retry_after == 120.0
    assert len(client.calls) == 1


REQUEST = httpx.Request("POST", URL)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (_status_error(openai.AuthenticationError, 401, "invalid_api_key"), KeyRejectedError),
        (_status_error(openai.PermissionDeniedError, 403, None), KeyRejectedError),
        (_status_error(openai.PermissionDeniedError, 403, "model_not_found"), UnknownModelError),
        (_status_error(openai.NotFoundError, 404, "model_not_found"), UnknownModelError),
        (_status_error(openai.BadRequestError, 400, "model_not_found"), UnknownModelError),
        (_status_error(openai.BadRequestError, 400, "invalid_request_error"), ModelError),
        (_status_error(openai.RateLimitError, 429, "insufficient_quota"), RateLimitedError),
        (_status_error(openai.InternalServerError, 500, None), ModelError),
        (openai.APIConnectionError(request=REQUEST), OfflineError),
        (openai.APITimeoutError(request=REQUEST), ModelTimeoutError),
    ],
)
def test_each_provider_failure_maps_to_one_typed_error(raised: Exception, expected: type) -> None:
    client = FakeClient(*[raised] * 4)
    with pytest.raises(expected) as caught:
        list(_llm(client, max_retries=0).stream(_request()))
    assert type(caught.value) is expected
    assert "sk-proj" not in caught.value.detail
    assert "sk-proj" not in caught.value.message


def test_quota_error_is_not_retried() -> None:
    client = FakeClient(_status_error(openai.RateLimitError, 429, "insufficient_quota"), _text_events())
    with pytest.raises(RateLimitedError) as caught:
        list(_llm(client).stream(_request()))
    assert len(client.calls) == 1
    assert "credit" in caught.value.message


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (NS(type="response.failed", response=NS(error=NS(code="rate_limit_exceeded", message="slow down"))),
         RateLimitedError),
        (NS(type="response.failed", response=NS(error=NS(code="server_error", message="oops"))), ModelError),
        (NS(type="error", code="invalid_api_key", message="bad key"), KeyRejectedError),
        (NS(type="error", code="model_not_found", message="no such model"), UnknownModelError),
        (NS(type="error", code=None, message="boom"), ModelError),
    ],
)
def test_failure_events_in_the_stream_are_typed(event: NS, expected: type) -> None:
    with pytest.raises(expected):
        list(_llm(FakeClient([event]), max_retries=0).stream(_request()))


def test_stream_that_ends_without_a_completion_is_an_error() -> None:
    with pytest.raises(ModelError):
        list(_llm(FakeClient([_delta("Hel")])).stream(_request()))


def test_missing_key_is_typed_and_no_client_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("no client without a key")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    with pytest.raises(NoApiKeyError):
        list(OpenAIResponsesLLM().stream(_request()))


def test_client_is_built_once_with_the_key_no_sdk_retries_and_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict] = []
    client = FakeClient(_text_events(), _text_events())

    def factory(**kwargs: Any) -> FakeClient:
        built.append(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", factory)
    llm = OpenAIResponsesLLM(api_key="test-key-not-real", timeout_seconds=30.0)
    list(llm.stream(_request()))
    list(llm.stream(_request()))
    assert len(built) == 1
    transport = built[0].pop("http_client")
    assert transport.trust_env is False
    assert built == [
        {"api_key": "test-key-not-real", "base_url": "https://api.openai.com/v1", "max_retries": 0, "timeout": 30.0}
    ]


def test_cancel_closes_the_stream_and_ends_quietly() -> None:
    cancel = CancelToken()
    client = FakeClient([_delta("one"), _delta("two"), _delta("three"), _completed()])
    stream = _llm(client).stream(_request(cancel=cancel))
    assert next(stream) == TextDelta("one")
    cancel.cancel()
    assert client.streams[0].closed is True
    assert list(stream) == []


def test_cancel_during_the_wait_for_a_retry_ends_quietly() -> None:
    cancel = CancelToken()
    limited = _status_error(openai.RateLimitError, 429, "rate_limit_exceeded")
    client = FakeClient(limited, _text_events())
    llm = OpenAIResponsesLLM(client, sleep=lambda _seconds: cancel.cancel())
    assert list(llm.stream(_request(cancel=cancel))) == []
    assert len(client.calls) == 1


def test_closing_the_iterator_closes_the_provider_stream() -> None:
    client = FakeClient([_delta("one"), _delta("two"), _completed()])
    stream = _llm(client).stream(_request())
    next(stream)
    stream.close()
    assert client.streams[0].closed is True


# --- findings of the review ---------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "asked"),
    [
        ({}, False),
        ({"reasoning_effort": "low"}, True),
        ({"carry_reasoning": True}, True),
        ({"reasoning_effort": "low", "carry_reasoning": False}, False),
    ],
)
def test_encrypted_reasoning_is_asked_for_only_when_the_model_reasons(settings: dict, asked: bool) -> None:
    client = FakeClient(_text_events())
    list(_llm(client, **settings).stream(_request()))
    assert ("include" in client.calls[0]) is asked


def test_empty_tools_and_instructions_are_left_out() -> None:
    client = FakeClient(_text_events())
    request = LLMRequest(
        model="model-under-test", messages=(ChatMessage("user", "hi"),), tools=(), max_output_tokens=64,
        allow_tools=False,
    )
    list(_llm(client).stream(request))
    sent = client.calls[0]
    assert "tools" not in sent and "tool_choice" not in sent and "instructions" not in sent


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ReadTimeout("slow"), ModelTimeoutError),
        (httpx.ConnectTimeout("slow"), ModelTimeoutError),
        (httpx.RemoteProtocolError("peer closed the connection"), OfflineError),
        (httpx.ReadError("connection reset"), OfflineError),
    ],
)
def test_transport_failures_while_reading_the_stream_are_typed_and_retried(raised: Exception, expected: type) -> None:
    with pytest.raises(expected):
        list(_llm(FakeClient([raised]), max_retries=0).stream(_request()))
    client = FakeClient([raised], _text_events())
    events = list(_llm(client).stream(_request()))
    assert len(client.calls) == 2 and isinstance(events[-1], Completed)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (_status_error(openai.APIStatusError, 408), ModelTimeoutError),
        (_status_error(openai.ConflictError, 409), ModelError),
        (_status_error(openai.APIStatusError, 500, "server_error"), ModelError),
        (NS(type="response.failed", response=NS(error=NS(code="server_error", message="oops"))), ModelError),
    ],
)
def test_a_failure_that_may_pass_is_retried(raised: Any, expected: type) -> None:
    first = raised if isinstance(raised, BaseException) else [raised]
    client = FakeClient(first, _text_events())
    events = list(_llm(client).stream(_request()))
    assert len(client.calls) == 2 and isinstance(events[-1], Completed)
    again = raised if isinstance(raised, BaseException) else [raised]
    with pytest.raises(expected) as caught:
        list(_llm(FakeClient(again), max_retries=0).stream(_request()))
    assert type(caught.value) is expected


@pytest.mark.parametrize(
    ("headers", "waited"),
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after-ms": "250", "retry-after": "9"}, 0.25),
        ({"retry-after": "-5"}, 0.0),
        ({"retry-after": "nan"}, 1.0),
        ({"retry-after": "soon"}, 1.0),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, 0.0),
    ],
)
def test_the_wait_the_provider_asks_for_is_read_in_every_form(headers: dict, waited: float) -> None:
    limited = _status_error(openai.RateLimitError, 429, None, headers)
    client = FakeClient(limited, _text_events())
    llm = _llm(client)
    list(llm.stream(_request()))
    assert llm.waits == [waited]


def test_a_cut_function_call_is_not_run_and_not_carried() -> None:
    reasoning = NS(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque", status="completed")
    whole = NS(type="function_call", id="fc_1", call_id="call_1", name="search_messages",
               arguments='{"query": "lease"}', status="completed")
    cut = NS(type="function_call", id="fc_2", call_id="call_2", name="search_messages",
             arguments='{"query": "lea', status="incomplete")
    stream = [
        NS(type="response.output_item.done", item=reasoning, output_index=0),
        NS(type="response.output_item.done", item=whole, output_index=1),
        NS(type="response.output_item.done", item=cut, output_index=2),
        NS(type="response.incomplete",
           response=NS(usage=_usage(), incomplete_details=NS(reason="max_output_tokens"))),
    ]
    events = list(_llm(FakeClient(stream)).stream(_request()))
    assert [e for e in events if isinstance(e, ToolCall)] == [
        ToolCall("call_1", "search_messages", '{"query": "lease"}')
    ]
    assert events[-1].stop_reason == "length"
    assert events[-1].provider_items == (
        {"type": "function_call", "call_id": "call_1", "name": "search_messages",
         "arguments": '{"query": "lease"}'},
    )


def test_the_configured_key_is_never_in_the_detail() -> None:
    secret = "custom-gateway-key-12345"
    raised = _status_error(openai.BadRequestError, 400, None, message=f"bad request for key {secret}")
    llm = OpenAIResponsesLLM(FakeClient(raised), api_key=secret, max_retries=0)
    with pytest.raises(ModelError) as caught:
        list(llm.stream(_request()))
    assert secret not in caught.value.detail and "***" in caught.value.detail


@pytest.mark.parametrize(
    ("raised", "expected", "word"),
    [
        (_status_error(openai.NotFoundError, 404, None), ModelError, "failed"),
        (_status_error(openai.NotFoundError, 404, None, message="The model `x-1` does not exist"),
         UnknownModelError, "model"),
        (_status_error(openai.PermissionDeniedError, 403, "unsupported_country_region_territory"),
         ModelError, "region"),
        (_status_error(openai.PermissionDeniedError, 403, None), KeyRejectedError, "key"),
    ],
)
def test_not_found_and_forbidden_say_what_they_are(raised: Exception, expected: type, word: str) -> None:
    with pytest.raises(expected) as caught:
        list(_llm(FakeClient(raised), max_retries=0).stream(_request()))
    assert type(caught.value) is expected and word in caught.value.message
