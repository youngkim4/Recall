"""meaning.py: the status of the index of meanings, the cost of a build, and the build.

The archive is a real one with invented lines. The embedder is `FakeEmbedder`:
no request leaves this machine.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import recall_paths
from archive import ArchiveError
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
    MeaningIndex,
    embedder_facts,
)
from meaning_index.openai_embedder import OpenAIEmbedder
from recall_services import meaning
from tests.integration.report_data import Pulled, moment
from tests.recall_services import memory_kit as kit
from tests.recall_services.memory_kit import FAKE_FACTS, MARK, SEGMENTS, Lake
from tests.report_engine.synth import MINA, THEO

STATUS_KEYS = {"state", "segments", "model", "dimensions", "builtAt", "coveredUntil"}
ESTIMATE_KEYS = {"segments", "newSegments", "tokens", "estimatedCost"}
BUILD_KEYS = {"segments", "embedded", "reused", "removed", "tokens"}
NO_KEY = "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then build the index again."
HANDLE_PARTS = (MINA, THEO, "2025550143", "theo@", "example.com")


@pytest.fixture
def lake(tmp_path: Path) -> Iterator[Lake]:
    made = kit.build_lake(tmp_path)
    try:
        yield made
    finally:
        made.close()


class Failing:
    """An embedder that raises at each request."""

    model, dimensions = FAKE_FACTS.model, FAKE_FACTS.dimensions

    def __init__(self, error: Exception) -> None:
        self._error = error

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise self._error


def build(lake: Lake, embedder: Any = None, **more: Any) -> dict[str, Any]:
    made = embedder or FakeEmbedder()
    settings = {"store": lake.store, "facts": FAKE_FACTS, "count_tokens": kit.words, **more}
    return meaning.build(lake.archive, embedder_factory=lambda _model, _dimensions: made, **settings)


def estimate(lake: Lake) -> dict[str, Any]:
    return meaning.estimate(lake.archive, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)


def failure(lake: Lake, embedder: Any = None, **more: Any) -> meaning.MeaningFailure:
    with pytest.raises(meaning.MeaningFailure) as raised:
        build(lake, embedder, **more)
    return raised.value


def newer_format(store: IndexStore) -> None:
    """Make the folder hold an index that a newer build of the app wrote."""
    store.folder.mkdir(parents=True, exist_ok=True)
    (store.folder / "meta.json").write_text(json.dumps({"format_version": 99}), encoding="utf-8")


# --- the place ---------------------------------------------------------------------


def test_the_index_is_in_the_saves_folder_as_it_is_named_at_the_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "first")
    assert meaning.index_store().folder == tmp_path / "first" / "meaning"
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "second")
    assert meaning.index_store().folder == tmp_path / "second" / "meaning"


def test_the_app_builds_with_the_default_model_of_the_facts_file() -> None:
    assert meaning.default_facts() == embedder_facts()


# --- status ------------------------------------------------------------------------


def test_with_no_index_the_state_is_none_and_no_file_is_made(lake: Lake) -> None:
    assert meaning.status(lake.archive, store=lake.store) == {
        "state": "none", "segments": 0, "model": None, "dimensions": None, "builtAt": None, "coveredUntil": None,
    }  # fmt: skip
    assert not lake.store.folder.exists() and meaning.built(lake.store) is False


def test_after_a_build_the_state_is_ready_and_says_what_the_index_holds(lake: Lake) -> None:
    build(lake)
    found = meaning.status(lake.archive, store=lake.store)
    assert set(found) == STATUS_KEYS
    assert (found["state"], found["segments"]) == ("ready", SEGMENTS)
    assert (found["model"], found["dimensions"]) == (FAKE_FACTS.model, FAKE_FACTS.dimensions)
    assert found["coveredUntil"] == kit.LAST_TEXT and found["builtAt"].endswith("+00:00")
    assert meaning.built(lake.store) is True


def test_a_later_message_with_text_makes_the_state_behind_until_the_next_build(lake: Lake) -> None:
    build(lake)
    lake.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
    assert meaning.status(lake.archive, store=lake.store)["state"] == "behind"
    done = build(lake)
    assert (done["embedded"], done["reused"]) == (1, SEGMENTS)
    assert meaning.status(lake.archive, store=lake.store)["state"] == "ready"


def test_a_later_message_with_no_text_does_not_make_the_state_behind(lake: Lake) -> None:
    """The index holds messages with text only: a photo with no caption is in no segment."""
    build(lake)
    chat = lake.pulled.chat(MINA)
    photo = lake.pulled.builder.message(chat, when=moment("2024-07-01 08:00:00"), handle=lake.pulled.handle(MINA))
    lake.pulled.builder.attachment(photo)
    lake.add([])
    assert lake.archive.summary()["messages"] == len(kit.LINES) + 1
    assert meaning.status(lake.archive, store=lake.store)["state"] == "ready"


def test_an_index_with_no_segment_is_behind_as_soon_as_the_archive_holds_text(tmp_path: Path) -> None:
    pulled = Pulled(tmp_path / "empty")
    pulled.chat(MINA)
    pulled.pull()
    store = IndexStore(tmp_path / "meaning")
    with pulled.open() as archive:
        made = meaning.build(archive, store=store, facts=FAKE_FACTS, embedder_factory=lambda *_: FakeEmbedder())
        assert made["segments"] == 0
        assert meaning.status(archive, store=store)["state"] == "ready"
    kit.write(pulled, kit.LINES[:1])
    pulled.pull()
    with pulled.open() as archive:
        assert meaning.status(archive, store=store)["state"] == "behind"
    pulled.close()


# --- estimate ----------------------------------------------------------------------


def test_the_estimate_counts_what_a_build_would_send_and_makes_no_file(lake: Lake) -> None:
    found = estimate(lake)
    assert set(found) == ESTIMATE_KEYS
    assert (found["segments"], found["newSegments"]) == (SEGMENTS, SEGMENTS)
    assert found["tokens"] > 0
    assert found["estimatedCost"] == pytest.approx(found["tokens"] * FAKE_FACTS.usd_per_mtok / 1_000_000)
    assert not lake.store.folder.exists()


def test_the_build_sends_what_the_estimate_said_and_the_next_estimate_is_zero(lake: Lake) -> None:
    said = estimate(lake)
    embedder = FakeEmbedder()
    done = build(lake, embedder)
    assert set(done) == BUILD_KEYS
    assert (done["segments"], done["embedded"], done["reused"], done["removed"]) == (SEGMENTS, SEGMENTS, 0, 0)
    # two segments hold the same words: they are one text of the request
    assert done["tokens"] == said["tokens"] and len(embedder.texts) == SEGMENTS - 1
    after = estimate(lake)
    assert (after["segments"], after["newSegments"], after["tokens"], after["estimatedCost"]) == (SEGMENTS, 0, 0, 0.0)


def test_the_index_holds_the_ids_of_the_archive(lake: Lake) -> None:
    build(lake)
    index = MeaningIndex.load(lake.store)
    assert index is not None
    (vector,) = FakeEmbedder().embed([kit.BOAT_QUERY])
    best = index.search(vector, limit=1)[0].segment
    first, last = lake.archive.message(best.first_message_id), lake.archive.message(best.last_message_id)
    assert first is not None and last is not None
    assert (first["text"], last["text"], first["conversation"]) == (kit.FERRY_LINE, kit.FERRY_REPLY, MINA)


# --- no key ------------------------------------------------------------------------


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")


def test_with_no_key_the_status_and_the_estimate_work_and_a_build_says_that_no_key_is_set(
    lake: Lake, no_key: None
) -> None:
    assert meaning.status(lake.archive, store=lake.store)["state"] == "none"
    assert estimate(lake)["newSegments"] == SEGMENTS
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.build(lake.archive, store=lake.store, count_tokens=kit.words)
    assert (raised.value.code, str(raised.value)) == ("no_key", NO_KEY)
    assert meaning.status(lake.archive, store=lake.store)["state"] == "none"


def test_a_build_with_nothing_new_asks_for_no_key(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    build(lake)
    monkeypatch.delenv("OPENAI_API_KEY")
    done = meaning.build(lake.archive, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)
    assert (done["embedded"], done["reused"]) == (0, SEGMENTS)


def test_with_a_key_the_embedder_of_the_app_is_the_adapter_of_the_provider() -> None:
    made = meaning.embedder_for("text-embedding-3-large", 1024)
    assert isinstance(made, OpenAIEmbedder) and (made.model, made.dimensions) == ("text-embedding-3-large", 1024)


# --- failures ----------------------------------------------------------------------


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
    assert meaning.status(lake.archive, store=lake.store)["state"] == "none"


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
    assert meaning.status(lake.archive, store=lake.store)["state"] == "none"


class Unreadable:
    """An archive that fails when its messages are read."""

    def __init__(self, archive: Any) -> None:
        self._archive = archive

    def conversations(self, **more: Any) -> Any:
        return self._archive.conversations(**more)

    def iter_messages(self, *_args: Any, **_more: Any) -> Any:
        raise ArchiveError(f"the text {MARK} of {MINA}")


def test_an_archive_that_can_not_be_read_is_one_plain_sentence(lake: Lake) -> None:
    broken = Unreadable(lake.archive)
    sentence = "The message archive could not be read. Update the archive, then build the index again."
    with pytest.raises(meaning.MeaningFailure) as built:
        meaning.build(broken, store=lake.store, facts=FAKE_FACTS, embedder_factory=lambda *_: FakeEmbedder())
    with pytest.raises(meaning.MeaningFailure) as estimated:
        meaning.estimate(broken, store=lake.store, facts=FAKE_FACTS)
    for told in (built.value, estimated.value):
        assert (told.code, str(told)) == ("archive_error", sentence)


def test_an_archive_that_fails_while_the_state_is_read_is_the_same_sentence(lake: Lake) -> None:
    build(lake)
    lake.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.status(Unreadable(lake.archive), store=lake.store)
    assert raised.value.code == "archive_error"


def test_an_index_of_a_newer_build_of_the_app_is_not_read_and_not_replaced(lake: Lake) -> None:
    newer_format(lake.store)
    sentence = "A newer version of Recall made this index. Update Recall, then try again."
    assert meaning.status(lake.archive, store=lake.store)["state"] == "none"
    assert meaning.built(lake.store) is False
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
    from meaning_index import ModelFactsError

    def broken() -> Any:
        raise ModelFactsError("the embedding facts file can not be read")

    monkeypatch.setattr(meaning, "default_facts", broken)
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.build(lake.archive, store=lake.store)
    assert (raised.value.code, str(raised.value)) == ("build_failed", "The index could not be built. Build it again.")


# --- the answer to the route -------------------------------------------------------


def test_the_payload_is_the_status_and_the_estimate(lake: Lake) -> None:
    more = {"store": lake.store, "facts": FAKE_FACTS, "count_tokens": kit.words}
    before = meaning.payload(lake.archive, **more)
    assert set(before) == {"status", "estimate", "error", "code"}
    assert before["status"]["state"] == "none" and before["estimate"]["newSegments"] == SEGMENTS
    assert (before["error"], before["code"]) == (None, None)
    build(lake)
    after = meaning.payload(lake.archive, **more)
    assert after["status"] == meaning.status(lake.archive, store=lake.store)
    assert after["estimate"] == estimate(lake)


def test_the_payload_of_an_index_that_can_not_be_read_says_why_and_has_no_estimate(lake: Lake) -> None:
    newer_format(lake.store)
    found = meaning.payload(lake.archive, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)
    assert found["status"]["state"] == "none" and found["estimate"] is None
    assert (found["code"], found["error"]) == (
        "index_newer",
        "A newer version of Recall made this index. Update Recall, then try again.",
    )


# --- progress ----------------------------------------------------------------------


def test_the_listener_gets_the_three_steps_with_counts_and_lines_that_hold_counts_only(lake: Lake) -> None:
    updates: list[dict[str, Any]] = []
    done = build(lake, on_progress=updates.append)
    steps = [update["progress"] for update in updates if update.get("progress")]
    assert all(set(step) == {"step", "of", "label", "done", "total"} and step["of"] == 3 for step in steps)
    assert [step["step"] for step in steps] == sorted(step["step"] for step in steps)
    assert {step["step"]: step["label"] for step in steps} == {
        1: "Reading your conversations",
        2: "Reading the meaning of your messages",
        3: "Saving the index",
    }
    assert {(step["step"], step["done"], step["total"]) for step in steps} >= {(1, 2, 2), (2, 4, 4), (3, 5, 5)}
    lines = [update["log"] for update in updates if update.get("log")]
    assert lines == [
        "Reading 2 conversations",
        "Reading the meaning of 4 exchanges",
        "Saving 5 exchanges",
        "The index holds 5 exchanges: 5 new, 0 kept, 0 removed.",
    ]
    assert done["segments"] == 5


def test_a_build_with_nothing_new_says_so(lake: Lake) -> None:
    build(lake)
    updates: list[dict[str, Any]] = []
    build(lake, on_progress=updates.append)
    lines = [update["log"] for update in updates if update.get("log")]
    assert lines[1] == "No new exchange to read" and lines[-1] == "The index holds 5 exchanges: 0 new, 5 kept, 0 removed."


def test_a_listener_that_fails_does_not_stop_the_build(lake: Lake, caplog: pytest.LogCaptureFixture) -> None:
    def broken(_update: dict[str, Any]) -> None:
        raise RuntimeError(f"a listener that says {MARK}")

    assert build(lake, on_progress=broken)["segments"] == SEGMENTS
    assert len([record for record in caplog.records if "listener" in record.getMessage()]) == 1


# --- privacy -----------------------------------------------------------------------


def test_no_log_line_no_job_line_and_no_sentence_holds_text_or_a_handle(
    lake: Lake, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    updates: list[dict[str, Any]] = []
    build(lake, on_progress=updates.append)
    estimate(lake)
    meaning.status(lake.archive, store=lake.store)
    told = [failure(lake, Failing(EmbeddingServiceError(f"said {MARK}")), store=IndexStore(lake.store.folder / "x"))]
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.estimate(Unreadable(lake.archive), store=lake.store, facts=FAKE_FACTS)
    told.append(raised.value)
    lines = [str(update.get("log") or "") for update in updates] + [
        str(update["progress"]["label"]) for update in updates if update.get("progress")
    ]
    everything = "\n".join([caplog.text, *lines, *(str(item) for item in told)])
    assert MARK not in everything and kit.FERRY_LINE not in everything
    assert not [part for part in HANDLE_PARTS if part in everything]
