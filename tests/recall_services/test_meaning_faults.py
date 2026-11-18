"""meaning.py when the key, the provider, the archive or the folder of the index fails, and what it may say."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from chat_engine.cancel import CancelToken
from meaning_index import (
    BatchLimits,
    EmbeddingKeyError,
    EmbeddingNetworkError,
    EmbeddingRateLimited,
    EmbeddingReplyError,
    EmbeddingServiceError,
    EmbeddingTooLarge,
    FakeEmbedder,
    IndexStore,
    ModelFactsError,
)
from recall_services import meaning
from tests.recall_services import memory_kit as kit
from tests.recall_services.memory_kit import (
    FAKE_FACTS,
    MARK,
    Failing,
    Lake,
    Unreadable,
    build,
    estimate,
    failure,
    newer_format,
)
from tests.report_engine.synth import MINA, THEO

HANDLE_PARTS = (MINA, THEO, "2025550143", "theo@", "example.com")


@pytest.mark.parametrize(
    ("error", "code", "sentence"),
    [
        (
            EmbeddingKeyError("x", status_code=401),
            "key_rejected",
            "OpenAI rejected the API key. Check OPENAI_API_KEY in the file .env in Recall's data folder, "
            "then build the index again.",
        ),
        (
            EmbeddingRateLimited("x", retry_after_s=900.0, status_code=429),
            "rate_limit",
            "OpenAI is limiting requests, or the account has no credit left. Wait a moment, then build the index again.",
        ),
        (
            EmbeddingServiceError("x", status_code=429, code="insufficient_quota"),
            "rate_limit",
            "OpenAI is limiting requests, or the account has no credit left. Wait a moment, then build the index again.",
        ),
        (
            EmbeddingNetworkError("x"),
            "offline",
            "OpenAI could not be reached. Check the internet connection, then build the index again.",
        ),
        (
            EmbeddingServiceError("x", status_code=500),
            "model_error",
            "The embedding model did not answer. What was read is kept: build the index again.",
        ),
        (
            EmbeddingReplyError("x"),
            "model_error",
            "The embedding model did not answer. What was read is kept: build the index again.",
        ),
        (
            EmbeddingTooLarge("x", status_code=413),
            "model_error",
            "The embedding model did not answer. What was read is kept: build the index again.",
        ),
    ],
)
def test_a_failure_of_the_embedder_is_one_plain_sentence_and_the_index_is_as_it_was(
    lake: Lake, error: Exception, code: str, sentence: str
) -> None:
    told = failure(lake, Failing(error), limits=BatchLimits(max_tries=1))
    assert (told.code, str(told)) == (code, sentence)
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "none"


def test_a_second_build_at_the_same_time_is_told_to_wait(lake: Lake) -> None:
    with lake.store.building():
        told = failure(lake)
    assert (told.code, str(told)) == ("busy", "The index is being built now. Wait until that ends.")


@pytest.mark.parametrize("cancel", [lambda: True, "token"])
def test_a_build_that_is_stopped_says_so_and_keeps_the_index_as_it_was(lake: Lake, cancel: Any) -> None:
    token = CancelToken()
    token.cancel()
    told = failure(lake, cancel=token if cancel == "token" else cancel)
    assert (told.code, str(told)) == ("cancelled", "The build was stopped. What was read is kept for the next build.")
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "none"


def test_an_archive_that_can_not_be_read_is_one_plain_sentence(lake: Lake) -> None:
    broken = Unreadable(lake.archive)
    sentence = "The message archive could not be read. Update the archive, then build the index again."
    with pytest.raises(meaning.MeaningFailure) as built:
        meaning.build(broken, lake.names, store=lake.store, facts=FAKE_FACTS, embedder_factory=lambda *_: FakeEmbedder())
    with pytest.raises(meaning.MeaningFailure) as estimated:
        meaning.estimate(broken, lake.names, store=lake.store, facts=FAKE_FACTS)
    for told in (built.value, estimated.value):
        assert (told.code, str(told)) == ("archive_error", sentence)


def test_an_archive_that_fails_while_the_state_is_read_is_the_same_sentence(lake: Lake) -> None:
    build(lake)
    lake.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.status(Unreadable(lake.archive), lake.names, store=lake.store)
    assert raised.value.code == "archive_error"


def test_an_index_of_a_newer_build_of_the_app_is_not_read_and_not_replaced(lake: Lake) -> None:
    newer_format(lake.store)
    sentence = "A newer version of Recall made this index. Update Recall, then try again."
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "none"
    assert meaning.built(lake.archive.path, store=lake.store) is False
    with pytest.raises(meaning.MeaningFailure) as estimated:
        estimate(lake)
    told = failure(lake)
    assert (told.code, str(told)) == (estimated.value.code, str(estimated.value)) == ("index_newer", sentence)
    assert json.loads((lake.store.folder / "meta.json").read_text(encoding="utf-8")) == {"format_version": 99}


def test_a_folder_that_can_not_be_written_is_one_plain_sentence(lake: Lake, tmp_path: Path) -> None:
    (tmp_path / "a-file").write_text("not a folder", encoding="utf-8")
    told = failure(lake, store=IndexStore(tmp_path / "a-file" / "meaning"))
    assert (told.code, str(told)) == (
        "index_error",
        "The index could not be written. Check the free space on the disk, then build the index again.",
    )


def test_facts_that_can_not_be_read_are_a_failure_with_a_sentence(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise ModelFactsError("the embedding facts file can not be read")

    monkeypatch.setattr(meaning, "default_facts", broken)
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.build(lake.archive, lake.names, store=lake.store)
    assert (raised.value.code, str(raised.value)) == ("build_failed", "The index could not be built. Build it again.")


# --- privacy -----------------------------------------------------------------------


def test_no_log_line_no_job_line_and_no_sentence_holds_text_or_a_handle(
    lake: Lake, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    updates: list[dict[str, Any]] = []
    build(lake, on_progress=updates.append)
    estimate(lake)
    meaning.status(lake.archive, lake.names, store=lake.store)
    told = [failure(lake, Failing(EmbeddingServiceError(f"said {MARK}")), store=IndexStore(lake.store.folder / "x"))]
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.estimate(Unreadable(lake.archive), lake.names, store=lake.store, facts=FAKE_FACTS)
    told.append(raised.value)
    lines = [str(update.get("log") or "") for update in updates] + [
        str(update["progress"]["label"]) for update in updates if update.get("progress")
    ]
    everything = "\n".join([caplog.text, *lines, *(str(item) for item in told)])
    assert MARK not in everything and kit.FERRY_LINE not in everything
    assert not [part for part in HANDLE_PARTS if part in everything]
