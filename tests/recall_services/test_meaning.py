"""meaning.py: the status of the index of meanings, the cost of a build, and the build.

The archive is a real one with invented lines. The embedder is `FakeEmbedder`:
no request leaves this machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import recall_paths
from meaning_index import FakeEmbedder, IndexStore, MeaningIndex, StoreError, embedder_facts
from meaning_index.openai_embedder import OpenAIEmbedder
from recall_services import meaning, reports
from recall_services.names import Names
from tests.integration.report_data import Pulled, moment
from tests.recall_services import memory_kit as kit
from tests.recall_services.memory_kit import FAKE_FACTS, MARK, SEGMENTS, Lake, build, estimate, newer_format
from tests import guards
from tests.report_engine.synth import MINA, THEO

NO_NAMES = Names.from_records([])
STATUS_KEYS = {"state", "segments", "model", "dimensions", "builtAt", "coveredUntil"}
ESTIMATE_KEYS = {"segments", "newSegments", "staleSegments", "tokens", "estimatedCost", "upperBound"}
BUILD_KEYS = {"segments", "embedded", "reused", "removed", "tokens"}
NO_KEY = "No API key is set. Add one in Settings > Archive > API key, then build the index again."



# --- the place ---------------------------------------------------------------------


def test_each_archive_has_an_index_folder_of_its_own_in_the_saves_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one, other = tmp_path / "a" / "archive.sqlite3", tmp_path / "b" / "archive.sqlite3"
    name = hashlib.sha256(str(one.resolve()).encode("utf-8")).hexdigest()[:16]
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "first")
    assert meaning.index_folder(one) == tmp_path / "first" / "meaning" / name
    assert meaning.index_store(one).folder == meaning.index_folder(one)
    assert meaning.index_folder(other).parent == meaning.index_folder(one).parent
    assert meaning.index_folder(other) != meaning.index_folder(one)
    # another spelling of the same file is the same folder
    assert meaning.index_folder(tmp_path / "a" / ".." / "a" / "archive.sqlite3") == meaning.index_folder(str(one))
    # the saves folder is read at each call
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "second")
    assert meaning.index_folder(one) == tmp_path / "second" / "meaning" / name


def test_an_index_that_is_built_for_one_archive_is_not_the_index_of_another(
    lake: Lake, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no store given, each function takes the folder from the archive that it was given."""
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "app-saves")
    other = kit.build_lake(tmp_path / "other")
    try:
        other.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
        same = {"facts": FAKE_FACTS, "embedder_factory": lambda *_: FakeEmbedder(), "count_tokens": kit.words}
        first = meaning.build(lake.archive, lake.names, **same)
        assert meaning.built(lake.archive.path) is True and meaning.built(other.archive.path) is False
        assert meaning.status(other.archive, other.names)["state"] == "none"
        second = meaning.build(other.archive, other.names, **same)
        assert (first["segments"], second["segments"], second["removed"]) == (SEGMENTS, SEGMENTS + 1, 0)
        assert meaning.status(lake.archive, lake.names)["segments"] == SEGMENTS
        assert meaning.status(other.archive, other.names)["segments"] == SEGMENTS + 1
        assert meaning.payload(lake.archive, lake.names, facts=FAKE_FACTS, count_tokens=kit.words)["estimate"]["newSegments"] == 0
        folders = sorted(path.name for path in (tmp_path / "app-saves" / "meaning").iterdir())
        assert folders == sorted([meaning.index_folder(lake.archive.path).name, meaning.index_folder(other.archive.path).name])
    finally:
        other.close()


def test_the_app_builds_with_the_default_model_of_the_facts_file() -> None:
    assert meaning.default_facts() == embedder_facts()


# --- status ------------------------------------------------------------------------


def test_with_no_index_the_state_is_none_and_no_file_is_made(lake: Lake) -> None:
    assert meaning.status(lake.archive, lake.names, store=lake.store) == {
        "state": "none", "segments": 0, "model": None, "dimensions": None, "builtAt": None, "coveredUntil": None,
    }  # fmt: skip
    assert not lake.store.folder.exists() and meaning.built(lake.archive.path, store=lake.store) is False


def test_after_a_build_the_state_is_ready_and_says_what_the_index_holds(lake: Lake) -> None:
    build(lake)
    found = meaning.status(lake.archive, lake.names, store=lake.store)
    assert set(found) == STATUS_KEYS
    assert (found["state"], found["segments"]) == ("ready", SEGMENTS)
    assert (found["model"], found["dimensions"]) == (FAKE_FACTS.model, FAKE_FACTS.dimensions)
    assert found["coveredUntil"] == kit.LAST_TEXT and found["builtAt"].endswith("+00:00")
    assert meaning.built(lake.archive.path, store=lake.store) is True


def test_a_later_message_with_text_makes_the_state_behind_until_the_next_build(lake: Lake) -> None:
    build(lake)
    lake.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "behind"
    done = build(lake)
    assert (done["embedded"], done["reused"]) == (1, SEGMENTS)
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "ready"


def test_a_later_message_with_no_text_does_not_make_the_state_behind(lake: Lake) -> None:
    """The index holds messages with text only: a photo with no caption is in no segment."""
    build(lake)
    chat = lake.pulled.chat(MINA)
    photo = lake.pulled.builder.message(chat, when=moment("2024-07-01 08:00:00"), handle=lake.pulled.handle(MINA))
    lake.pulled.builder.attachment(photo)
    lake.add([])
    assert lake.archive.summary()["messages"] == len(kit.LINES) + 1
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "ready"


def test_an_index_with_no_segment_is_behind_as_soon_as_the_archive_holds_text(tmp_path: Path) -> None:
    pulled = Pulled(tmp_path / "empty")
    pulled.chat(MINA)
    pulled.pull()
    store = IndexStore(tmp_path / "meaning")
    with pulled.open() as archive:
        made = meaning.build(archive, NO_NAMES, store=store, facts=FAKE_FACTS, embedder_factory=lambda *_: FakeEmbedder())
        assert made["segments"] == 0
        assert meaning.status(archive, NO_NAMES, store=store)["state"] == "ready"
    kit.write(pulled, kit.LINES[:1])
    pulled.pull()
    with pulled.open() as archive:
        assert meaning.status(archive, NO_NAMES, store=store)["state"] == "behind"
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
    assert meaning.status(lake.archive, lake.names, store=broken)["state"] == "none" and meaning.built(lake.archive.path, store=broken) is False
    assert "StoreError" in caplog.text and MARK not in caplog.text


# --- what a build sends ------------------------------------------------------------


class Recording(FakeEmbedder):
    """Keeps each text that it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    def embed(self, texts: Any) -> Any:
        self.sent.extend(texts)
        return super().embed(texts)


def test_a_build_sends_a_handle_that_a_person_typed_as_a_report_sends_it(lake: Lake) -> None:
    """Theo is not saved. Mina types his address into a message: the embedding provider gets the mask."""
    lake.add([(MINA, "2023-05-01 10:00:00", f"Write to {THEO} about the boat", False)])
    redact = lake.names.chat_labeler(lake.archive.conversations()).redact
    as_a_report = [row["text"] for row in reports._Masked(lake.archive, redact).messages(MINA, None, None)]
    (masked,) = [text for text in as_a_report if "about the boat" in text]
    embedder = Recording()
    build(lake, embedder)
    assert THEO not in masked and not [text for text in embedder.sent if THEO in text]
    assert [text for text in embedder.sent if f"Them: {masked}" in text]
    # the estimate reads the same text, so it agrees with the build
    assert estimate(lake)["newSegments"] == 0


def test_a_contact_that_is_saved_later_makes_the_segments_with_its_handle_new(lake: Lake) -> None:
    lake.add([(MINA, "2023-05-01 10:00:00", f"Write to {THEO} about the boat", False)])
    build(lake)
    saved = Names.from_records([kit.MINA_CONTACT, {"name": "Theo Lund", "phones": [], "emails": [THEO]}])
    found = meaning.estimate(lake.archive, saved, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)
    assert (found["newSegments"], found["staleSegments"]) == (1, 0)


# --- segments whose ids are not the ids of the store -------------------------------


def test_the_estimate_counts_the_segments_whose_id_the_store_does_not_hold(lake: Lake) -> None:
    assert estimate(lake)["staleSegments"] == SEGMENTS  # no index: no id is in the store
    build(lake)
    assert estimate(lake)["staleSegments"] == 0
    # one more message joins the last segment of Theo: that segment has a new id
    lake.add([(THEO, "2024-06-05 20:05:00", "Or we take the bus", False)])
    found = estimate(lake)
    assert (found["segments"], found["newSegments"], found["staleSegments"]) == (SEGMENTS, 1, 1)


def test_ids_that_changed_under_the_same_words_are_stale_and_cost_nothing(lake: Lake, tmp_path: Path) -> None:
    """After the first pull over a loaded export each message has a new id: the words are the same."""
    build(lake)
    again = Pulled(tmp_path / "again")
    again.builder.handle("+15550100000")  # one more row first: each message gets another guid
    again.builder.message(None, when=moment("2020-01-01 00:00:00"), text="no conversation")
    kit.write(again, kit.LINES)
    again.pull([kit.MINA_CONTACT])
    with again.open() as archive:
        assert not set(kit.ids_of(archive, kit.FERRY_LINE)) & set(kit.ids_of(lake.archive, kit.FERRY_LINE))
        more = {"store": lake.store, "facts": FAKE_FACTS, "count_tokens": kit.words}
        found = meaning.payload(archive, lake.names, **more)
        assert (found["estimate"]["newSegments"], found["estimate"]["staleSegments"]) == (0, SEGMENTS)
        assert found["estimate"]["estimatedCost"] == 0.0 and found["status"]["state"] == "behind"
        done = meaning.build(archive, lake.names, embedder_factory=lambda *_: FakeEmbedder(), **more)
        assert (done["embedded"], done["reused"]) == (0, SEGMENTS)
        assert meaning.payload(archive, lake.names, **more)["status"]["state"] == "ready"
    again.close()


def test_the_payload_says_behind_when_a_build_would_embed_segments(lake: Lake) -> None:
    """The cheap status looks for a later message only. The payload also knows what the estimate found."""
    build(lake)
    other = dataclasses.replace(FAKE_FACTS, model="fake-embedding-2")
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "ready"
    found = meaning.payload(lake.archive, lake.names, store=lake.store, facts=other, count_tokens=kit.words)
    assert (found["status"]["state"], found["estimate"]["newSegments"], found["estimate"]["staleSegments"]) == (
        "behind", SEGMENTS, 0,
    )  # fmt: skip


def test_after_a_change_of_the_time_zone_the_days_move_and_the_payload_says_behind(lake: Lake) -> None:
    """A known limit: the day of a segment is a local day. The estimate says what the change costs."""
    build(lake)
    with guards.time_zone("Asia/Seoul"):
        found = meaning.payload(lake.archive, lake.names, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)
    assert found["status"]["state"] == "behind" and found["estimate"]["newSegments"] > 0


@pytest.mark.parametrize(
    ("segments", "new", "small"),
    [(5, 5, True), (100, 2_000, True), (100, 2_001, False), (100_000, 10_000, True), (100_000, 10_001, False)],
)
def test_the_server_brings_an_index_up_to_date_by_itself_only_when_little_is_new(
    segments: int, new: int, small: bool
) -> None:
    assert (meaning.REFRESH_SEGMENTS, meaning.REFRESH_SHARE) == (2_000, 0.10)
    assert meaning.small_enough({"segments": segments, "newSegments": new}) is small


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
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "none"
    assert estimate(lake)["newSegments"] == SEGMENTS
    with pytest.raises(meaning.MeaningFailure) as raised:
        meaning.build(lake.archive, lake.names, store=lake.store, count_tokens=kit.words)
    assert (raised.value.code, str(raised.value)) == ("no_key", NO_KEY)
    assert meaning.status(lake.archive, lake.names, store=lake.store)["state"] == "none"


def test_a_build_with_nothing_new_asks_for_no_key(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    build(lake)
    monkeypatch.delenv("OPENAI_API_KEY")
    done = meaning.build(lake.archive, lake.names, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)
    assert (done["embedded"], done["reused"]) == (0, SEGMENTS)


def test_with_a_key_the_embedder_of_the_app_is_the_adapter_of_the_provider() -> None:
    made = meaning.embedder_for("text-embedding-3-large", 1024)
    assert isinstance(made, OpenAIEmbedder) and (made.model, made.dimensions) == ("text-embedding-3-large", 1024)


# --- the answer to the route -------------------------------------------------------


def test_the_payload_is_the_status_and_the_estimate(lake: Lake) -> None:
    more = {"store": lake.store, "facts": FAKE_FACTS, "count_tokens": kit.words}
    before = meaning.payload(lake.archive, lake.names, **more)
    assert set(before) == {"status", "estimate", "error", "code"}
    assert before["status"]["state"] == "none" and before["estimate"]["newSegments"] == SEGMENTS
    assert (before["error"], before["code"]) == (None, None)
    build(lake)
    after = meaning.payload(lake.archive, lake.names, **more)
    assert after["status"] == meaning.status(lake.archive, lake.names, store=lake.store)
    assert after["estimate"] == estimate(lake)


def test_the_payload_of_an_index_that_can_not_be_read_says_why_and_has_no_estimate(lake: Lake) -> None:
    newer_format(lake.store)
    found = meaning.payload(lake.archive, lake.names, store=lake.store, facts=FAKE_FACTS, count_tokens=kit.words)
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
