"""No text of a message, no name and no handle reaches an error or a log line.

Each case sends marked text through a build that fails, or that warns, and
reads everything that the build said: the error, its chain, and every log line.
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from meaning_index import store_files
from meaning_index.batching import BatchLimits
from meaning_index.build import build_index, estimate_build
from meaning_index.errors import (
    BuildCancelled,
    EmbeddingError,
    EmbeddingKeyError,
    EmbeddingRateLimited,
    EmbeddingTooLarge,
    MeaningIndexError,
    SourceError,
    StoreError,
)
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import embedder_facts
from meaning_index.openai_embedder import OpenAIEmbedder
from meaning_index.search import MeaningIndex
from meaning_index.store import IndexStore
from tests.meaning_index.support import ListSource, ScriptedEmbedder, msg, stamp, words

TEXT_MARK = "MARKED-SECRET-TEXT-9917"
NAME_MARK = "Marked Person Name"
HANDLE = "+15550100999"
MAIL = "marked.person@example.com"
ID_MARK = "MARKED-MESSAGE-ID"
MARKS = (TEXT_MARK, NAME_MARK, HANDLE, MAIL, "marked.person", ID_MARK)


def marked() -> dict:
    """Every text, every conversation id and every message id carries a mark."""
    def talk(prefix: str, count: int) -> list:
        return [
            msg(f"{ID_MARK}-{prefix}-{n}", stamp(n * 180), f"{TEXT_MARK} from {NAME_MARK} line {n} of {prefix}", n % 2 == 0)
            for n in range(count)
        ]

    return {HANDLE: talk("phone", 120), MAIL: talk("mail", 5)}


def said(error: BaseException | None, caplog: pytest.LogCaptureFixture) -> str:
    """Everything a caller or a log file can get from a build."""
    parts = [record.getMessage() for record in caplog.records]
    parts += [str(record.args) for record in caplog.records]
    parts.append(caplog.text)
    if error is not None:
        parts += [str(error), repr(error), repr(error.args), repr(vars(error))]
        parts += traceback.format_exception(error)
        parts.append(repr(error.__cause__) + repr(error.__context__ if not error.__suppress_context__ else None))
    return "\n".join(parts)


def assert_clean(error: BaseException | None, caplog: pytest.LogCaptureFixture) -> None:
    everything = said(error, caplog)
    for mark in MARKS:
        assert mark not in everything, f"{mark!r} leaked"


def echoing(number: int, texts: Sequence[str]) -> Exception:
    return RuntimeError(f"could not embed: {' | '.join(texts)}")


def sdk_client(error: Exception) -> Any:
    class Embeddings:
        def create(self, **request: Any) -> Any:
            raise error

    class Client:
        embeddings = Embeddings()

    return Client()


def sdk_error(code: int) -> Exception:
    """An SDK error as the service could send it: its message repeats the input."""
    body = {"message": f"invalid input: {TEXT_MARK} from {NAME_MARK} ({HANDLE})", "code": "invalid_value"}
    request = httpx.Request("POST", "https://api.invalid/v1/embeddings", content=TEXT_MARK.encode())
    response = httpx.Response(code, request=request, json={"error": body})
    kind = openai.RateLimitError if code == 429 else openai.BadRequestError
    return kind(f"Error code: {code} - {{'error': {body}}}", response=response, body=body)


FAILING: list[tuple[str, Callable[[], Any], type]] = [
    ("an embedder that repeats its input", lambda: ScriptedEmbedder(echoing), EmbeddingError),
    ("a typed failure after two requests", lambda: ScriptedEmbedder({2: EmbeddingKeyError("no key")}), EmbeddingKeyError),
    ("always too large", lambda: ScriptedEmbedder(lambda n, texts: EmbeddingTooLarge("too large")), EmbeddingTooLarge),
    ("always rate limited", lambda: ScriptedEmbedder(lambda n, texts: EmbeddingRateLimited("wait")), EmbeddingRateLimited),
    ("a reply of the wrong size", lambda: FakeReply(), EmbeddingError),
    ("the SDK refuses the request", lambda: OpenAIEmbedder(sdk_client(sdk_error(400)), "text-embedding-3-large", 16), EmbeddingError),
    ("the SDK says wait", lambda: OpenAIEmbedder(sdk_client(sdk_error(429)), "text-embedding-3-large", 16), EmbeddingRateLimited),
]


class FakeReply:
    model, dimensions = "wrong-size", 16

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 2.0] for _ in texts]


@pytest.mark.parametrize(("case", "make", "expected"), FAILING, ids=[case for case, _, _ in FAILING])
def test_a_build_that_fails_in_the_embedder_says_nothing_of_the_messages(
    folder: Path, caplog: pytest.LogCaptureFixture, case: str, make: Callable[[], Any], expected: type
) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(expected) as caught:
        build_index(ListSource(marked()), make(), IndexStore(folder), count_tokens=words, sleep=lambda seconds: None)
    assert isinstance(caught.value, MeaningIndexError)
    assert_clean(caught.value, caplog)


def test_the_cases_above_do_send_the_marked_text_to_the_embedder(folder: Path) -> None:
    embedder = ScriptedEmbedder()
    build_index(ListSource(marked()), embedder, IndexStore(folder), count_tokens=words)
    assert all(TEXT_MARK in text and NAME_MARK in text for text in embedder.texts)
    assert len(embedder.texts) == 125


def test_a_source_that_fails_says_nothing_of_the_messages(folder: Path, caplog: pytest.LogCaptureFixture) -> None:
    class Broken(ListSource):
        def messages(self, conversation_id: str):  # type: ignore[no-untyped-def]
            if conversation_id == MAIL:
                raise RuntimeError(f"could not read {conversation_id}: {TEXT_MARK}")
            return super().messages(conversation_id)

    caplog.set_level(logging.DEBUG)
    with pytest.raises(SourceError) as caught:
        build_index(Broken(marked()), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    assert_clean(caught.value, caplog)
    with pytest.raises(SourceError) as caught:
        estimate_build(Broken(marked()), IndexStore(folder), embedder_facts(), words)
    assert_clean(caught.value, caplog)


def test_a_time_that_can_not_be_read_says_nothing_of_the_messages(folder: Path, caplog: pytest.LogCaptureFixture) -> None:
    data = marked()
    data[MAIL] = [*data[MAIL], msg(f"{ID_MARK}-bad", f"not a time {TEXT_MARK}", TEXT_MARK)]
    caplog.set_level(logging.DEBUG)
    with pytest.raises(SourceError) as caught:
        build_index(ListSource(data), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    assert_clean(caught.value, caplog)


def test_a_cancelled_build_says_nothing_of_the_messages(folder: Path, caplog: pytest.LogCaptureFixture) -> None:
    embedder = ScriptedEmbedder()
    caplog.set_level(logging.DEBUG)
    with pytest.raises(BuildCancelled) as caught:
        build_index(
            ListSource(marked()), embedder, IndexStore(folder), count_tokens=words, cancel=lambda: len(embedder.sizes) >= 1
        )
    assert caught.value.texts_embedded == 96
    assert_clean(caught.value, caplog)


def test_a_swap_that_fails_says_nothing_of_the_messages(
    folder: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(source: object, target: object) -> None:
        raise OSError("the disk said no")

    monkeypatch.setattr(store_files.os, "replace", refuse)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(StoreError) as caught:
        build_index(ListSource(marked()), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    assert_clean(caught.value, caplog)


def test_a_build_that_warns_and_ends_well_logs_nothing_of_the_messages(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def script(number: int, texts: Sequence[str]) -> Exception | None:
        if number == 1:
            return EmbeddingRateLimited("wait", retry_after_s=1.0)
        return EmbeddingTooLarge("too large") if len(texts) > 50 else None

    caplog.set_level(logging.DEBUG)
    store = IndexStore(folder)
    folder.mkdir(parents=True)
    (folder / "work.sqlite3").write_bytes(b"invented bytes that are no database" * 50)
    (folder / "meta.json").write_text("{not json", encoding="utf-8")
    result = build_index(
        ListSource(marked()),
        ScriptedEmbedder(script),
        store,
        count_tokens=words,
        sleep=lambda seconds: None,
        limits=BatchLimits(max_tries=3),
    )
    assert result.embedded == 125
    messages = [record.getMessage() for record in caplog.records if record.name.startswith("meaning_index")]
    assert any("too large" in message for message in messages)
    assert any("asks to wait" in message for message in messages)
    assert any("work file" in message for message in messages)
    assert any("index built" in message for message in messages)
    assert_clean(None, caplog)


def test_a_search_that_is_refused_says_nothing_of_the_messages(folder: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = IndexStore(folder)
    build_index(ListSource(marked()), FakeEmbedder(), store, count_tokens=words)
    index = MeaningIndex.load(store)
    assert index is not None
    (probe,) = FakeEmbedder().embed([TEXT_MARK])
    caplog.set_level(logging.DEBUG)
    for arguments in ({"since": f"{TEXT_MARK}"}, {"conversations": [HANDLE, 7]}, {"limit": 0}):
        with pytest.raises((ValueError, TypeError)) as caught:
            index.search(probe, **arguments)  # type: ignore[arg-type]
        assert_clean(caught.value, caplog)


def test_the_files_of_the_index_hold_no_text_of_a_message(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(marked()), FakeEmbedder(), store, count_tokens=words)
    for path in folder.iterdir():
        content = path.read_bytes()
        assert TEXT_MARK.encode() not in content, path.name
        assert NAME_MARK.encode() not in content, path.name
    # the segments file names the conversation and the messages: it is private, as the archive is
    assert HANDLE.encode() in (folder / "segments.sqlite3").read_bytes()
    assert os.stat(folder / "segments.sqlite3").st_mode & 0o077 == 0
