"""The OpenAI adapter against a fake client. No request leaves the process.

Replies are built with the SDK's own Response type, and errors with the SDK's
own error classes, so the adapter is checked against the installed SDK.
"""

from typing import Any

import httpx
import openai
import pytest
from openai.types.responses import Response

from report_engine.errors import LLMError
from report_engine.openai_llm import OpenAIResponsesLLM, build_client
from report_engine.prompts import MAP_SCHEMA

MESSAGES = [
    {"role": "system", "content": "Be exact."},
    {"role": "user", "content": "Read this."},
]
REQUEST = httpx.Request("POST", "https://api.invalid/v1/responses")


def reply(
    text: str = '{"summary": "ok"}',
    *,
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    refusal: str | None = None,
    reason: str | None = None,
) -> Response:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "output_text", "text": text, "annotations": []})
    if refusal:
        content.append({"type": "refusal", "refusal": refusal})
    payload: dict[str, Any] = {
        "id": "resp_test",
        "created_at": 1_700_000_000,
        "model": "test-model-1",
        "object": "response",
        "status": status,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content,
            }
        ],
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 200},
            "output_tokens": 90,
            "output_tokens_details": {"reasoning_tokens": 40},
            "total_tokens": 1290,
        },
    }
    if reason:
        payload["incomplete_details"] = {"reason": reason}
    return Response.model_validate(payload)


def status_error(
    kind: type, code: int, *, headers: dict[str, str] | None = None, body: Any = None
) -> Exception:
    response = httpx.Response(code, headers=headers or {}, request=REQUEST)
    return kind("the service said no", response=response, body=body)


class Responses:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Client:
    def __init__(self, *script: Any) -> None:
        self.responses = Responses(list(script))


def adapter(client: Client, **overrides: Any) -> tuple[OpenAIResponsesLLM, list[float]]:
    slept: list[float] = []
    values: dict[str, Any] = {"timeout_s": 123.0, "sleep": slept.append, "jitter": lambda: 1.0}
    values.update(overrides)
    return OpenAIResponsesLLM(client, "test-model-1", **values), slept


def ask(llm: OpenAIResponsesLLM, **overrides: Any) -> Any:
    values: dict[str, Any] = {"schema": MAP_SCHEMA, "max_output_tokens": 700}
    values.update(overrides)
    return llm.generate(MESSAGES, **values)


def test_the_request_is_built_as_the_engine_needs_it() -> None:
    client = Client(reply())
    llm, _ = adapter(client)
    ask(llm, reasoning="low")
    (sent,) = client.responses.calls
    assert sent["model"] == "test-model-1"
    assert sent["store"] is False
    assert sent["max_output_tokens"] == 700
    assert sent["timeout"] == 123.0
    assert sent["instructions"] == "Be exact."
    assert sent["input"] == [{"role": "user", "content": "Read this."}]
    assert sent["reasoning"] == {"effort": "low"}
    form = sent["text"]["format"]
    assert (form["type"], form["name"], form["strict"]) == ("json_schema", "period_digest", True)
    assert "title" not in form["schema"]
    assert form["schema"]["required"] == ["summary", "carry_over", "moments"]
    assert "title" in MAP_SCHEMA
    assert "stream" not in sent and "truncation" not in sent


def test_without_a_schema_or_reasoning_those_fields_are_not_sent() -> None:
    client = Client(reply("plain words"))
    llm, _ = adapter(client)
    result = ask(llm, schema=None)
    (sent,) = client.responses.calls
    assert "text" not in sent and "reasoning" not in sent
    assert (result.text, result.parsed, result.status) == ("plain words", None, "completed")


def test_usage_is_read_from_the_reply() -> None:
    llm, _ = adapter(Client(reply()))
    result = ask(llm)
    assert (
        result.usage.input_tokens,
        result.usage.cached_input_tokens,
        result.usage.output_tokens,
    ) == (1200, 200, 90)
    assert result.parsed == {"summary": "ok"}
    assert result.attempts == 1


def test_a_reply_without_usage_counts_zero_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    bare = reply().model_copy(update={"usage": None})
    llm, _ = adapter(Client(bare))
    with caplog.at_level("WARNING"):
        result = ask(llm)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (0, 0)
    assert "usage" in caplog.text


def test_an_incomplete_reply_is_returned_once_and_never_sent_again() -> None:
    client = Client(
        reply('{"summary": "cut', status="incomplete", reason="max_output_tokens"), reply()
    )
    llm, slept = adapter(client)
    result = ask(llm)
    assert result.status == "incomplete"
    assert result.parsed is None
    assert result.usage.input_tokens == 1200
    assert len(client.responses.calls) == 1
    assert slept == []


def test_text_that_is_not_json_gives_no_parsed_value() -> None:
    llm, _ = adapter(Client(reply("not json at all")))
    result = ask(llm)
    assert (result.status, result.parsed, result.text) == ("completed", None, "not json at all")
    llm, _ = adapter(Client(reply("[1, 2]")))
    assert ask(llm).parsed is None


def test_an_empty_reply_is_returned_as_it_is() -> None:
    client = Client(reply(""), reply())
    llm, _ = adapter(client)
    result = ask(llm)
    assert (result.status, result.text, result.parsed) == ("completed", "", None)
    assert len(client.responses.calls) == 1


def test_a_refusal_is_reported_as_one() -> None:
    llm, _ = adapter(Client(reply("", refusal="I can not help with that.")))
    result = ask(llm)
    assert (result.status, result.parsed) == ("refused", None)
    assert result.text == "I can not help with that."


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued", "in_progress"])
def test_a_reply_that_did_not_finish_is_failed(status: str) -> None:
    llm, _ = adapter(Client(reply(status=status)))
    assert ask(llm).status == "failed"


def test_a_rate_limit_is_retried_after_the_time_the_service_names() -> None:
    limited = status_error(openai.RateLimitError, 429, headers={"retry-after": "7"})
    client = Client(limited, reply())
    llm, slept = adapter(client)
    result = ask(llm)
    assert slept == [7.0]
    assert result.attempts == 2
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0] == client.responses.calls[1]


def test_the_millisecond_header_wins_over_the_second_header() -> None:
    limited = status_error(
        openai.RateLimitError, 429, headers={"retry-after-ms": "1500", "retry-after": "9"}
    )
    llm, slept = adapter(Client(limited, reply()))
    ask(llm)
    assert slept == [1.5]


def test_a_wait_longer_than_the_cap_is_not_cut_short_but_gives_up() -> None:
    limited = status_error(openai.RateLimitError, 429, headers={"retry-after": "300"})
    client = Client(limited, reply())
    llm, slept = adapter(client, max_retry_wait_s=60.0)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert slept == []
    assert caught.value.retry_after_s == 300.0
    assert caught.value.status_code == 429
    assert len(client.responses.calls) == 1


def test_retries_are_bounded() -> None:
    errors = [status_error(openai.InternalServerError, 503) for _ in range(5)]
    client = Client(*errors)
    llm, slept = adapter(client, max_retries=2)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert len(client.responses.calls) == 3
    assert caught.value.attempts == 3
    assert caught.value.status_code == 503
    assert slept == [2.0, 4.0]


def test_backoff_has_jitter_and_a_cap() -> None:
    errors = [openai.APITimeoutError(REQUEST) for _ in range(4)]
    llm, slept = adapter(
        Client(*errors, reply()), max_retries=4, max_retry_wait_s=5.0, jitter=lambda: 0.0
    )
    result = ask(llm)
    assert slept == [1.0, 2.0, 2.5, 2.5]
    assert result.attempts == 5


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(REQUEST),
        openai.APIConnectionError(message="connection reset", request=REQUEST),
        status_error(openai.InternalServerError, 500),
        status_error(openai.APIStatusError, 408),
        status_error(openai.ConflictError, 409),
    ],
)
def test_passing_failures_are_retried(error: Exception) -> None:
    client = Client(error, reply())
    llm, slept = adapter(client)
    assert ask(llm).attempts == 2
    assert len(slept) == 1


@pytest.mark.parametrize(
    "error, code",
    [
        (status_error(openai.BadRequestError, 400, body={"code": "context_length_exceeded"}), 400),
        (status_error(openai.AuthenticationError, 401), 401),
        (status_error(openai.PermissionDeniedError, 403), 403),
        (status_error(openai.NotFoundError, 404), 404),
        (status_error(openai.UnprocessableEntityError, 422), 422),
    ],
)
def test_lasting_failures_are_not_retried(error: Exception, code: int) -> None:
    client = Client(error, reply())
    llm, slept = adapter(client)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert caught.value.status_code == code
    assert caught.value.attempts == 1
    assert slept == []
    assert len(client.responses.calls) == 1


def test_an_empty_account_is_not_retried() -> None:
    broke = status_error(
        openai.RateLimitError,
        429,
        headers={"retry-after": "1"},
        body={"code": "insufficient_quota"},
    )
    client = Client(broke, reply())
    llm, slept = adapter(client)
    with pytest.raises(LLMError) as caught:
        ask(llm)
    assert slept == []
    assert "insufficient_quota" in str(caught.value)


def test_an_error_message_holds_no_key_and_no_prompt() -> None:
    response = httpx.Response(401, request=REQUEST)
    leaky = openai.AuthenticationError(
        "Incorrect API key provided: sk-proj-abcDEF123456. Organization org-XyZ987 is over the limit.",
        response=response,
        body=None,
    )
    llm, _ = adapter(Client(leaky))
    with pytest.raises(LLMError) as caught:
        ask(llm)
    text = str(caught.value)
    assert "sk-proj-abcDEF123456" not in text
    assert "org-XyZ987" not in text
    assert "Read this." not in text
    assert "401" in text


def test_an_unknown_sdk_error_is_wrapped() -> None:
    llm, _ = adapter(Client(openai.OpenAIError("something else")))
    with pytest.raises(LLMError):
        ask(llm)


def test_only_user_and_system_roles_are_accepted() -> None:
    llm, _ = adapter(Client(reply()))
    with pytest.raises(ValueError):
        llm.generate([{"role": "assistant", "content": "x"}], max_output_tokens=10)  # type: ignore[list-item]


@pytest.mark.parametrize("bad", [0, -5])
def test_the_output_limit_must_be_positive(bad: int) -> None:
    llm, _ = adapter(Client(reply()))
    with pytest.raises(ValueError):
        ask(llm, max_output_tokens=bad)


def test_a_schema_name_is_made_safe() -> None:
    client = Client(reply())
    llm, _ = adapter(client)
    ask(llm, schema={"title": "my schema / v2!", "type": "object"})
    assert client.responses.calls[0]["text"]["format"]["name"] == "my_schema_v2"
    client = Client(reply())
    llm, _ = adapter(client)
    ask(llm, schema={"type": "object"})
    assert client.responses.calls[0]["text"]["format"]["name"] == "reply"


def test_build_client_turns_off_the_sdk_retries_and_sets_the_timeout(
    blocked_attempts: list[str],
) -> None:
    client = build_client("sk-test-dummy-not-a-real-key", timeout_s=45.0)
    assert client.max_retries == 0
    assert client.timeout == 45.0
    assert blocked_attempts == []


def test_a_request_to_a_host_that_does_not_exist_is_blocked_and_wrapped(
    blocked_attempts: list[str],
) -> None:
    """The SDK's own HTTP stack runs here. The host is under .invalid, which never
    resolves, and the socket guard stops the lookup before it starts."""
    before = len(blocked_attempts)
    client = build_client(
        "sk-test-dummy-not-a-real-key", timeout_s=5.0, base_url="https://api.invalid/v1"
    )
    llm = OpenAIResponsesLLM(client, "test-model-1", timeout_s=5.0, max_retries=0)
    with pytest.raises(LLMError) as caught:
        llm.generate(MESSAGES, max_output_tokens=16)
    assert caught.value.attempts == 1
    assert blocked_attempts[before:] == ["api.invalid"]
