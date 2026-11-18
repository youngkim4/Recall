"""meaning.py: the status of the index of meanings, the cost of a build, and the build.

The archive is a real one with invented lines. The embedder is `FakeEmbedder`:
no request leaves this machine.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import recall_paths
from meaning_index import FakeEmbedder, IndexStore, MeaningIndex, StoreError, embedder_facts
from meaning_index.openai_embedder import OpenAIEmbedder
from recall_services import meaning
from tests.integration.report_data import Pulled, moment
from tests.recall_services import memory_kit as kit
from tests.recall_services.memory_kit import FAKE_FACTS, MARK, SEGMENTS, Lake, build, estimate, newer_format
from tests.report_engine.synth import MINA

STATUS_KEYS = {"state", "segments", "model", "dimensions", "builtAt", "coveredUntil"}
ESTIMATE_KEYS = {"segments", "newSegments", "tokens", "estimatedCost", "upperBound"}
BUILD_KEYS = {"segments", "embedded", "reused", "removed", "tokens"}
NO_KEY = "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then build the index again."



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


def test_an_estimate_that_counts_too_much_says_that_it_is_an_upper_bound(
    lake: Lake, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later meaning_index says so when the vectors of a stopped build could not be read."""
    assert estimate(lake)["upperBound"] is False
    real = meaning.estimate_build

    def too_much(*args: Any, **more: Any) -> Any:
        found = real(*args, **more)
        return SimpleNamespace(
            segments=found.segments, new_segments=found.new_segments, tokens=found.tokens, usd=found.usd, upper_bound=True
        )

    monkeypatch.setattr(meaning, "estimate_build", too_much)
    assert estimate(lake)["upperBound"] is True


def test_a_folder_whose_status_can_not_be_read_counts_as_no_index(lake: Lake, caplog: pytest.LogCaptureFixture) -> None:
    build(lake)

    class Broken(IndexStore):
        def status(self) -> Any:
            raise StoreError(f"a file that says {MARK}")

    broken = Broken(lake.store.folder)
    assert meaning.status(lake.archive, store=broken)["state"] == "none" and meaning.built(broken) is False
    assert "StoreError" in caplog.text and MARK not in caplog.text


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
