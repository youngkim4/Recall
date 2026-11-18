"""memory.py: the search by meaning of chat over the index. The embedder is `FakeEmbedder`."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import recall_paths
from meaning_index import EmbeddingNetworkError, FakeEmbedder, IndexStore, MeaningIndex
from recall_services import meaning, memory
from tests.recall_services import memory_kit as kit
from tests.recall_services.memory_kit import FAKE_FACTS, MARK, Lake, build
from tests.report_engine.synth import MINA, THEO

HIT_KEYS = {"conversation", "first_message_id", "last_message_id", "score"}
EVERYWHERE = {"conversations": None, "since": None, "until": None, "limit": 10}


@pytest.fixture(autouse=True)
def _nothing_kept() -> Iterator[None]:
    """No index and no vector of a query stays in the process for the next test."""
    memory.forget()
    yield
    memory.forget()


class Asked:
    """Stands in for the embedder of a query. It keeps what it was asked for."""

    def __init__(self, error: Exception | None = None) -> None:
        self.made: list[tuple[str, int]] = []
        self.embedder = FakeEmbedder()
        self._error = error

    def __call__(self, model: str, dimensions: int) -> Any:
        self.made.append((model, dimensions))
        return self

    @property
    def model(self) -> str:
        return self.embedder.model

    @property
    def dimensions(self) -> int:
        return self.embedder.dimensions

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._error is not None:
            raise self._error
        return self.embedder.embed(texts)

    @property
    def texts(self) -> list[str]:
        return self.embedder.texts


def port(lake: Lake, asked: Asked | None = None) -> memory.MeaningPort:
    found = memory.ports(lake.archive, lake.names, store=lake.store, embedder_factory=asked or Asked())["meaning"]
    assert isinstance(found, memory.MeaningPort)
    return found


def texts_of(lake: Lake, hit: dict[str, Any]) -> tuple[str, str]:
    first, last = lake.archive.message(hit["first_message_id"]), lake.archive.message(hit["last_message_id"])
    assert first is not None and last is not None
    assert first["conversation"] == last["conversation"] == hit["conversation"]
    return first["text"], last["text"]


# --- which ports are given ---------------------------------------------------------


def test_with_no_key_neither_port_is_given(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    build(lake)
    monkeypatch.delenv("OPENAI_API_KEY")
    assert memory.ports(lake.archive, lake.names, store=lake.store) == {"meaning": None, "digests": None}


def test_with_no_index_only_the_digest_port_is_given_and_no_file_is_made(lake: Lake) -> None:
    found = memory.ports(lake.archive, lake.names, store=lake.store)
    assert found["meaning"] is None and isinstance(found["digests"], memory.DigestPort)
    assert set(found) == {"meaning", "digests"} and not lake.store.folder.exists()


def test_the_index_of_the_app_is_the_one_in_the_saves_folder(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_paths, "SAVES_DIR", lake.store.folder.parent)
    assert memory.ports(lake.archive, lake.names)["meaning"] is None
    build(lake)
    assert isinstance(memory.ports(lake.archive, lake.names)["meaning"], memory.MeaningPort)


@pytest.mark.parametrize("damage", ["newer", "segments", "vectors", "times"])
def test_a_broken_index_folder_gives_no_port_and_log_lines_with_no_private_text(
    lake: Lake, damage: str, caplog: pytest.LogCaptureFixture
) -> None:
    build(lake)
    if damage == "newer":
        (lake.store.folder / "meta.json").write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    elif damage == "segments":
        (lake.store.folder / "segments.sqlite3").write_bytes(b"not a database " * 64)
    elif damage == "vectors":
        (lake.store.folder / "vectors.npy").write_bytes(b"not an array")
    else:  # the three files are one build, and a row of it can not be read
        with closing(sqlite3.connect(lake.store.folder / "segments.sqlite3")) as conn, conn:
            conn.execute("UPDATE segments SET first_timestamp = 'no time'")
    caplog.set_level(logging.DEBUG)
    assert memory.ports(lake.archive, lake.names, store=lake.store)["meaning"] is None
    assert caplog.records, "a folder that can not be used leaves a line in the log"
    assert not [part for part in (MARK, MINA, THEO, kit.FERRY_LINE, "2025550143") if part in caplog.text]


def test_a_digest_model_that_can_not_be_run_gives_no_digest_port(
    lake: Lake, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(memory, "DIGEST_MODEL", "a-model-nobody-offers")
    assert memory.ports(lake.archive, lake.names, store=lake.store)["digests"] is None
    assert "digest" in caplog.text


# --- a search ----------------------------------------------------------------------


def test_a_search_gives_the_exchanges_nearest_to_the_description_with_the_ids_of_the_archive(lake: Lake) -> None:
    build(lake)
    hits = port(lake).search(kit.BOAT_QUERY, **EVERYWHERE)
    assert all(set(hit) == HIT_KEYS for hit in hits)
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)
    assert texts_of(lake, hits[0]) == (kit.FERRY_LINE, kit.FERRY_REPLY) and hits[0]["conversation"] == MINA
    assert texts_of(lake, hits[1])[0] == kit.LATE_FERRY_LINE and hits[1]["conversation"] == THEO
    assert all(isinstance(hit["score"], float) for hit in hits)


def test_the_conversations_the_days_and_the_limit_bind_the_search(lake: Lake) -> None:
    build(lake)
    found = port(lake)
    only_theo = found.search(kit.BOAT_QUERY, conversations=[THEO], since=None, until=None, limit=10)
    assert {hit["conversation"] for hit in only_theo} == {THEO}
    in_march = found.search(kit.BOAT_QUERY, conversations=None, since="2023-03-01", until="2023-03-31", limit=10)
    assert {texts_of(lake, hit)[0][:10] for hit in in_march} == {kit.FERRY_LINE[:10], "The kettle"}
    assert len(found.search(kit.BOAT_QUERY, conversations=None, since=None, until=None, limit=1)) == 1
    assert found.search(kit.BOAT_QUERY, conversations=[], since=None, until=None, limit=10) == []


def test_the_same_words_said_again_are_one_hit(lake: Lake) -> None:
    """Two segments with the same content hash: the better one stays."""
    build(lake)
    hits = port(lake).search("happy birthday", **EVERYWHERE)
    birthdays = [hit for hit in hits if texts_of(lake, hit)[0] == kit.SAME_WORDS]
    assert len(birthdays) == 1 and hits[0] == birthdays[0]
    assert len(hits) == kit.SEGMENTS - 1


def test_only_the_description_goes_to_the_embedder_with_the_model_of_the_index(lake: Lake) -> None:
    build(lake)
    asked = Asked()
    port(lake, asked).search(kit.BOAT_QUERY, conversations=[MINA], since="2023-01-01", until=None, limit=5)
    assert asked.texts == [kit.BOAT_QUERY]
    assert asked.made == [(FAKE_FACTS.model, FAKE_FACTS.dimensions)]


def test_a_second_search_with_the_same_description_does_not_embed_again(lake: Lake) -> None:
    build(lake)
    asked = Asked()
    first = port(lake, asked).search(kit.BOAT_QUERY, **EVERYWHERE)
    # a new port, as the next question gets: the vectors are kept by the process
    again = port(lake, asked).search(kit.BOAT_QUERY, conversations=[MINA], since=None, until=None, limit=3)
    assert asked.texts == [kit.BOAT_QUERY] and again[0] == first[0]


def test_the_process_keeps_the_vectors_of_the_last_64_descriptions(lake: Lake) -> None:
    build(lake)
    asked = Asked()
    found = port(lake, asked)
    for number in range(memory.QUERY_VECTORS + 1):
        found.search(f"description {number}", **EVERYWHERE)
    found.search(f"description {memory.QUERY_VECTORS}", **EVERYWHERE)
    assert len(asked.texts) == memory.QUERY_VECTORS + 1
    found.search("description 0", **EVERYWHERE)
    assert len(asked.texts) == memory.QUERY_VECTORS + 2


def test_a_failure_of_the_embedder_is_raised_so_that_chat_falls_back_to_words(lake: Lake) -> None:
    build(lake)
    with pytest.raises(EmbeddingNetworkError):
        port(lake, Asked(EmbeddingNetworkError("no connection"))).search(kit.BOAT_QUERY, **EVERYWHERE)


def test_a_reply_that_is_not_one_vector_is_raised(lake: Lake) -> None:
    build(lake)

    class Two(Asked):
        def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return [*self.embedder.embed(texts), *self.embedder.embed(texts)]

    with pytest.raises(ValueError, match="one vector"):
        port(lake, Two()).search(kit.BOAT_QUERY, **EVERYWHERE)


# --- the index changes on disk -----------------------------------------------------


def test_the_index_is_loaded_one_time_while_the_build_on_disk_is_the_same(
    lake: Lake, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(lake)
    loads: list[int] = []
    load = MeaningIndex.load
    monkeypatch.setattr(MeaningIndex, "load", classmethod(lambda cls, store: loads.append(1) or load(store)))
    for _ in range(3):
        port(lake).search(kit.BOAT_QUERY, **EVERYWHERE)
    assert loads == [1]


def test_after_a_new_build_the_same_port_answers_from_the_new_index(lake: Lake) -> None:
    build(lake)
    found = port(lake)
    assert len(found.search("the lake is warm", **EVERYWHERE)) == kit.SEGMENTS - 1
    lake.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
    build(lake)
    hits = found.search("the lake is warm", **EVERYWHERE)
    assert texts_of(lake, hits[0])[0] == "The lake is warm now"


def test_while_the_files_are_swapped_the_loaded_index_still_answers(lake: Lake) -> None:
    """Until the meta file is in place the folder holds no whole build: a question must not fail then."""
    build(lake)
    found = port(lake)
    before = found.search(kit.BOAT_QUERY, **EVERYWHERE)
    meta = lake.store.folder / "meta.json"
    held = meta.read_bytes()
    meta.unlink()
    assert found.search(kit.BOAT_QUERY, **EVERYWHERE) == before
    assert isinstance(memory.ports(lake.archive, lake.names, store=lake.store)["meaning"], memory.MeaningPort)
    meta.write_bytes(held)
    assert found.search(kit.BOAT_QUERY, **EVERYWHERE) == before


def test_a_search_with_no_index_at_all_is_raised(lake: Lake) -> None:
    with pytest.raises(memory.MeaningNotReady):
        memory.MeaningPort(lake.store, Asked()).search(kit.BOAT_QUERY, **EVERYWHERE)


def test_forget_lets_go_of_the_index_and_of_the_vectors(lake: Lake) -> None:
    build(lake)
    asked = Asked()
    port(lake, asked).search(kit.BOAT_QUERY, **EVERYWHERE)
    memory.forget()
    port(lake, asked).search(kit.BOAT_QUERY, **EVERYWHERE)
    assert asked.texts == [kit.BOAT_QUERY, kit.BOAT_QUERY]


# --- a build that swaps the files while a question loads the index -------------------


def test_files_of_two_builds_are_never_read_as_one_index(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    """A load that ends on another build than it started on is thrown away and tried again."""
    build(lake)
    real = IndexStore.meta
    seen: list[str] = []

    def other_build_once(store: IndexStore) -> Any:
        found = real(store)
        seen.append("look")
        # the second look is the check after the first load: it finds another build there
        return replace(found, build_id="another-build") if len(seen) == 2 and found is not None else found

    monkeypatch.setattr(IndexStore, "meta", other_build_once)
    hits = port(lake).search(kit.BOAT_QUERY, **EVERYWHERE)
    assert texts_of(lake, hits[0])[0] == kit.FERRY_LINE and len(seen) >= 3


def test_a_folder_that_never_holds_one_whole_build_gives_no_port(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    build(lake)
    real = IndexStore.meta
    looks: list[int] = []

    def always_another(store: IndexStore) -> Any:
        looks.append(1)
        found = real(store)
        return replace(found, build_id=f"build-{len(looks)}") if found is not None else found

    monkeypatch.setattr(IndexStore, "meta", always_another)
    assert memory.ports(lake.archive, lake.names, store=lake.store)["meaning"] is None


def test_a_load_that_finds_the_folder_in_a_swap_gives_no_port_for_this_question(
    lake: Lake, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(lake)
    monkeypatch.setattr(MeaningIndex, "load", classmethod(lambda cls, store: None))
    assert memory.ports(lake.archive, lake.names, store=lake.store)["meaning"] is None


def test_a_new_index_that_can_not_be_read_leaves_the_loaded_one_to_answer(
    lake: Lake, caplog: pytest.LogCaptureFixture
) -> None:
    build(lake)
    found = port(lake)
    before = found.search(kit.BOAT_QUERY, **EVERYWHERE)
    lake.add([(MINA, "2024-07-01 08:00:00", "The lake is warm now", False)])
    build(lake)
    with closing(sqlite3.connect(lake.store.folder / "segments.sqlite3")) as conn, conn:
        conn.execute("UPDATE segments SET first_timestamp = 'no time'")
    assert found.search(kit.BOAT_QUERY, **EVERYWHERE) == before
    assert "the one before it answers" in caplog.text


# --- the edges ---------------------------------------------------------------------


def test_a_limit_below_one_gives_nothing_and_embeds_nothing(lake: Lake) -> None:
    build(lake)
    asked = Asked()
    assert port(lake, asked).search(kit.BOAT_QUERY, conversations=None, since=None, until=None, limit=0) == []
    assert asked.texts == []


def test_the_embedder_of_a_description_is_the_one_of_the_app_with_a_short_wait(
    lake: Lake, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(lake)
    made: list[tuple[str, int, float]] = []
    fake = FakeEmbedder()

    def stand_in(model: str, dimensions: int, *, timeout_s: float) -> Any:
        made.append((model, dimensions, timeout_s))
        return fake

    monkeypatch.setattr(meaning, "embedder_for", stand_in)
    found = memory.ports(lake.archive, lake.names, store=lake.store)["meaning"]
    assert found.search(kit.BOAT_QUERY, **EVERYWHERE)
    assert made == [(FAKE_FACTS.model, FAKE_FACTS.dimensions, memory.QUERY_TIMEOUT_SECONDS)]
    assert memory.QUERY_TIMEOUT_SECONDS < meaning.BUILD_TIMEOUT_SECONDS


@pytest.mark.parametrize("bad", ["not a number", "short"])
def test_a_vector_that_a_search_can_not_use_is_raised_and_not_kept(lake: Lake, bad: str) -> None:
    build(lake)

    class Broken(Asked):
        broken = True

        def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            (vector,) = self.embedder.embed(texts)
            if not self.broken:
                return [vector]
            return [[float("nan"), *vector[1:]]] if bad == "not a number" else [vector[:-1]]

    asked = Broken()
    found = port(lake, asked)
    with pytest.raises(ValueError, match="finite numbers with the dimensions"):
        found.search(kit.BOAT_QUERY, **EVERYWHERE)
    asked.broken = False
    assert texts_of(lake, found.search(kit.BOAT_QUERY, **EVERYWHERE)[0])[0] == kit.FERRY_LINE
