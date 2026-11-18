"""The OpenAI adapter against a stand-in client. No request leaves the process.

Replies are built with the SDK's own reply type, and errors with the SDK's own
error classes, so the adapter is checked against the installed SDK.
"""

from __future__ import annotations

import email.utils
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from openai.types import CreateEmbeddingResponse

from meaning_index.build import build_index
from meaning_index.errors import (
    EmbeddingError,
    EmbeddingKeyError,
    EmbeddingNetworkError,
    EmbeddingRateLimited,
    EmbeddingReplyError,
    EmbeddingServiceError,
    EmbeddingTooLarge,
)
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.openai_embedder import OpenAIEmbedder
from meaning_index.search import MeaningIndex
from meaning_index.store import IndexStore
from tests import guards
from tests.meaning_index.support import ListSource, notes, words

REQUEST = httpx.Request("POST", "https://api.invalid/v1/embeddings")
MARK = "MARKED-TEXT-3390"
MODEL = "text-embedding-3-large"


def reply(vectors: Sequence[Sequence[float]], order: Sequence[int] | None = None) -> CreateEmbeddingResponse:
    places = list(order) if order is not None else list(range(len(vectors)))
    data = [{"object": "embedding", "index": place, "embedding": list(vectors[place])} for place in places]
    return CreateEmbeddingResponse.model_validate(
        {"object": "list", "model": MODEL, "data": data, "usage": {"prompt_tokens": 9, "total_tokens": 9}}
    )


def status_error(kind: type, code: int, *, headers: dict[str, str] | None = None, body: Any = None) -> Exception:
    response = httpx.Response(code, headers=headers or {}, request=REQUEST)
    return kind(f"the service said no to {MARK}", response=response, body=body)


class Embeddings:
    """Answers from a script: a reply, an error, or a function of the request."""

    def __init__(self, script: Sequence[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if callable(item):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        return item


class Client:
    def __init__(self, *script: Any) -> None:
        self.embeddings = Embeddings(script)


def test_the_request_has_the_shape_of_the_contract() -> None:
    client = Client(reply([[1.0, 0.0], [0.0, 1.0]]))
    embedder = OpenAIEmbedder(client, MODEL, 2)
    vectors = embedder.embed(("first text", "second text"))
    assert client.embeddings.calls == [{"model": MODEL, "input": ["first text", "second text"], "dimensions": 2}]
    assert [list(vector) for vector in vectors] == [[1.0, 0.0], [0.0, 1.0]]


def test_it_names_its_model_and_its_dimensions() -> None:
    embedder = OpenAIEmbedder(Client(reply([])), MODEL, 1024)
    assert (embedder.model, embedder.dimensions) == (MODEL, 1024)
    with pytest.raises(ValueError):
        OpenAIEmbedder(Client(reply([])), MODEL, 0)


def test_the_vectors_follow_the_index_of_the_reply_not_its_order() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
    client = Client(reply(vectors, order=[2, 0, 1]))
    assert [list(v) for v in OpenAIEmbedder(client, MODEL, 2).embed(["a", "b", "c"])] == vectors


def test_no_text_makes_no_request() -> None:
    client = Client(reply([]))
    assert list(OpenAIEmbedder(client, MODEL, 2).embed([])) == []
    assert client.embeddings.calls == []


FAILURES = [
    (status_error(openai.AuthenticationError, 401, body={"code": "invalid_api_key"}), EmbeddingKeyError),
    (status_error(openai.PermissionDeniedError, 403), EmbeddingKeyError),
    (status_error(openai.RateLimitError, 429, body={"code": "rate_limit_exceeded"}), EmbeddingRateLimited),
    (status_error(openai.RateLimitError, 429, body={"code": "insufficient_quota"}), EmbeddingServiceError),
    (openai.APIConnectionError(message=f"no route for {MARK}", request=REQUEST), EmbeddingNetworkError),
    (openai.APITimeoutError(request=REQUEST), EmbeddingNetworkError),
    (status_error(openai.BadRequestError, 400, body={"code": "context_length_exceeded"}), EmbeddingTooLarge),
    (status_error(openai.BadRequestError, 400, body={"code": "max_tokens_per_request"}), EmbeddingTooLarge),
    (status_error(openai.APIStatusError, 413), EmbeddingTooLarge),
    (status_error(openai.BadRequestError, 400, body={"code": "invalid_value"}), EmbeddingServiceError),
    (status_error(openai.NotFoundError, 404, body={"code": "model_not_found"}), EmbeddingServiceError),
    (status_error(openai.InternalServerError, 500), EmbeddingServiceError),
    (openai.OpenAIError(f"something else about {MARK}"), EmbeddingServiceError),
    (ValueError(f"No embedding data received for {MARK}"), EmbeddingServiceError),
]


@pytest.mark.parametrize(("raised", "expected"), FAILURES, ids=[f"{type(r).__name__}-{e.__name__}" for r, e in FAILURES])
def test_each_failure_of_the_service_is_one_typed_failure(raised: Exception, expected: type) -> None:
    embedder = OpenAIEmbedder(Client(raised), MODEL, 2)
    with pytest.raises(EmbeddingError) as caught:
        embedder.embed([f"a text with {MARK}"])
    assert type(caught.value) is expected
    assert MARK not in str(caught.value)
    assert MARK not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_a_quota_failure_is_not_a_rate_limit() -> None:
    embedder = OpenAIEmbedder(Client(FAILURES[3][0]), MODEL, 2)
    with pytest.raises(EmbeddingServiceError) as caught:
        embedder.embed(["a text"])
    assert not isinstance(caught.value, EmbeddingRateLimited)
    assert (caught.value.status_code, caught.value.code) == (429, "insufficient_quota")


def test_a_request_that_the_service_calls_too_large_in_words_only_is_too_large() -> None:
    body = {"code": None, "message": "This model's maximum context length is 8192 tokens, however you requested 9000"}
    embedder = OpenAIEmbedder(Client(status_error(openai.BadRequestError, 400, body=body)), MODEL, 2)
    with pytest.raises(EmbeddingTooLarge):
        embedder.embed(["a text"])


def test_the_error_names_the_status_the_code_and_the_request_and_nothing_else() -> None:
    raised = status_error(
        openai.RateLimitError, 429, headers={"x-request-id": "req_invented_1"}, body={"code": "rate_limit_exceeded"}
    )
    with pytest.raises(EmbeddingRateLimited) as caught:
        OpenAIEmbedder(Client(raised), MODEL, 2).embed(["a text"])
    assert str(caught.value) == "the embedding service failed (HTTP 429, rate_limit_exceeded, request req_invented_1)"
    assert (caught.value.status_code, caught.value.code) == (429, "rate_limit_exceeded")


def test_a_code_of_the_service_is_cut_to_plain_characters() -> None:
    raised = status_error(openai.BadRequestError, 400, body={"code": f"odd code {MARK} " + "x" * 300})
    with pytest.raises(EmbeddingServiceError) as caught:
        OpenAIEmbedder(Client(raised), MODEL, 2).embed(["a text"])
    assert len(str(caught.value)) < 160
    assert " code " not in str(caught.value)


def limited(headers: dict[str, str]) -> float | None:
    raised = status_error(openai.RateLimitError, 429, headers=headers, body={"code": "rate_limit_exceeded"})
    with pytest.raises(EmbeddingRateLimited) as caught:
        OpenAIEmbedder(Client(raised), MODEL, 2).embed(["a text"])
    return caught.value.retry_after_s


def test_a_rate_limit_carries_the_wait_that_the_service_names() -> None:
    assert limited({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    assert limited({"retry-after": "7"}) == 7.0
    assert limited({"retry-after": "0"}) == 0.0
    assert limited({}) is None
    assert limited({"retry-after": "soon"}) is None
    assert limited({"retry-after": "nan"}) is None
    assert limited({"retry-after": "-4"}) == 0.0
    in_a_minute = email.utils.formatdate(time.time() + 60, usegmt=True)
    assert limited({"retry-after": in_a_minute}) == pytest.approx(60.0, abs=3.0)


@pytest.mark.parametrize(
    "answer",
    [
        reply([[1.0, 0.0]]),  # one vector for two texts
        reply([[1.0, 0.0], [0.0, 1.0]], order=[0, 0]),  # one index two times
        reply([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),  # one vector too many
        object(),  # no data at all
    ],
)
def test_a_reply_that_does_not_answer_each_text_is_a_typed_failure(answer: object) -> None:
    with pytest.raises(EmbeddingReplyError):
        OpenAIEmbedder(Client(answer), MODEL, 2).embed(["a", "b"])


# --- through a build ---------------------------------------------------------


def answering(dimensions: int) -> Any:
    fake = FakeEmbedder(dimensions=dimensions)
    return lambda request: reply(fake.embed(request["input"]))


def test_a_build_runs_through_the_adapter(folder: Path) -> None:
    client = Client(answering(32))
    seen = len(guards.BLOCKED_ATTEMPTS)
    result = build_index(
        ListSource({"chat-a": notes(120)}), OpenAIEmbedder(client, MODEL, 32), IndexStore(folder), count_tokens=words
    )
    assert result.embedded == 120
    assert [len(call["input"]) for call in client.embeddings.calls] == [96, 24]
    assert all(call["model"] == MODEL and call["dimensions"] == 32 for call in client.embeddings.calls)
    index = MeaningIndex.load(IndexStore(folder))
    assert index is not None and (index.model, index.dimensions, len(index)) == (MODEL, 32, 120)
    assert guards.BLOCKED_ATTEMPTS[seen:] == []


def test_a_build_halves_a_request_that_the_service_calls_too_large(folder: Path) -> None:
    fake = FakeEmbedder(dimensions=8)

    def answer(request: dict[str, Any]) -> Any:
        if len(request["input"]) > 40:
            return status_error(openai.BadRequestError, 400, body={"code": "max_tokens_per_request"})
        return reply(fake.embed(request["input"]))

    client = Client(answer)
    result = build_index(
        ListSource({"chat-a": notes(96)}), OpenAIEmbedder(client, MODEL, 8), IndexStore(folder), count_tokens=words
    )
    assert result.embedded == 96
    assert [len(call["input"]) for call in client.embeddings.calls] == [96, 48, 24, 24, 48, 24, 24]


def test_a_build_waits_the_time_that_the_service_names(folder: Path) -> None:
    slept: list[float] = []
    busy = status_error(openai.RateLimitError, 429, headers={"retry-after": "11"}, body={"code": "rate_limit_exceeded"})
    client = Client(busy, busy, answering(8))
    result = build_index(
        ListSource({"chat-a": notes(5)}),
        OpenAIEmbedder(client, MODEL, 8),
        IndexStore(folder),
        count_tokens=words,
        sleep=slept.append,
    )
    assert slept == [11.0, 11.0]
    assert result.embedded == 5
    assert len(client.embeddings.calls) == 3
