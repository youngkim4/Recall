"""Rare states: files that are not what their name says, and replies with a part missing."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import openai
import pytest

from meaning_index import store_files
from meaning_index.build import build_index
from meaning_index.errors import (
    EmbeddingRateLimited,
    IndexFormatError,
    ModelFactsError,
    SourceError,
    StoreError,
)
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import FACTS_PATH, embedder_facts, load_facts
from meaning_index.openai_embedder import OpenAIEmbedder
from meaning_index.search import MeaningIndex
from meaning_index.shapes import IndexMeta
from meaning_index.store import IndexStore
from tests.meaning_index.store_kit import FILES, content, fill, make, torn
from tests.meaning_index.support import ListSource, kept, notes, unit, words


@pytest.mark.parametrize("name", ["segments.sqlite3", "vectors.npy"])
def test_a_file_of_the_index_that_holds_other_bytes_is_no_index(folder: Path, name: str) -> None:
    store = make(folder)
    fill(store)
    (folder / name).write_bytes(b"invented bytes, not what the name says" * 30)
    assert store.status().state == "none"
    assert store.read() is None


def test_a_vectors_file_that_is_cut_short_is_no_index(folder: Path) -> None:
    store = make(folder)
    fill(store)
    whole = (folder / "vectors.npy").read_bytes()
    (folder / "vectors.npy").write_bytes(whole[:-16])
    assert store.status().state == "none"


def test_a_vectors_file_that_goes_away_after_the_check_is_a_typed_error(folder: Path) -> None:
    fill(make(folder))
    with pytest.raises(StoreError):
        store_files.open_vectors(folder / "missing.npy", 5, 8)


def test_a_swap_that_can_not_put_the_old_files_back_says_so_and_the_next_build_repairs(
    folder: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = make(folder)
    fill(store)
    before = content(folder)
    real, calls = os.replace, []

    def replace(source: object, target: object) -> None:
        calls.append(str(target))
        if len(calls) >= 2:
            raise OSError("the disk said no")
        real(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(store_files.os, "replace", replace)
    with pytest.raises(StoreError), store.building():
        store.replace("model-b", 8, [kept(n) for n in range(4)], [unit(4, 8)])
    monkeypatch.undo()
    assert any("could not be put back" in record.getMessage() for record in caplog.records)
    assert store.status().state == "none"
    with store.building():
        assert content(folder) == before
    assert store.status().model == "model-a"


def test_a_repair_leaves_the_files_of_a_newer_format_alone(folder: Path) -> None:
    store = make(folder)
    torn(folder, store, 1)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    (folder / "meta.json").write_text(json.dumps({**meta, "format_version": 2}), encoding="utf-8")
    kept_files = {name: (folder / name).read_bytes() for name in FILES}
    with store.building():
        pass
    assert {name: (folder / name).read_bytes() for name in FILES} == kept_files
    with pytest.raises(IndexFormatError):
        store.status()


def test_a_stored_time_that_can_not_be_read_is_a_typed_error() -> None:
    meta = IndexMeta(1, "invented", "invented-model", 8, "2026-11-17T00:00:00+00:00", 1, None)
    with pytest.raises(StoreError):
        MeaningIndex(meta, [kept(0, day="not-a-day!")], unit(1, 8))


def test_a_rate_limit_with_no_reply_names_no_wait() -> None:
    class Embeddings:
        def create(self, **request: object) -> object:
            error = openai.RateLimitError.__new__(openai.RateLimitError)
            Exception.__init__(error, "limited")
            error.status_code, error.code, error.request_id, error.message = 429, None, None, "limited"
            raise error

    class Client:
        embeddings = Embeddings()

    with pytest.raises(EmbeddingRateLimited) as caught:
        OpenAIEmbedder(Client(), "text-embedding-3-large", 8).embed(["a text"])
    assert caught.value.retry_after_s is None


def test_a_typed_failure_of_the_source_is_given_as_it_is(folder: Path) -> None:
    class Typed(ListSource):
        def conversation_ids(self) -> Sequence[str]:
            raise SourceError("the archive is closed")

    with pytest.raises(SourceError, match="the archive is closed"):
        build_index(Typed({}), FakeEmbedder(), IndexStore(folder), count_tokens=words)


def test_a_work_file_that_loses_a_vector_during_the_build_is_a_typed_error(folder: Path) -> None:
    store = IndexStore(folder)

    class Forgetful(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            store.work_file(self.dimensions).clear()
            return super().embed(texts)

    with pytest.raises(StoreError):
        build_index(ListSource({"chat-a": notes(200)}), Forgetful(), store, count_tokens=words)
    assert store.status().state == "none"


def test_facts_with_a_part_missing_are_refused(tmp_path: Path) -> None:
    raw = json.loads(FACTS_PATH.read_text(encoding="utf-8"))

    def write(name: str, **changes: object) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({**raw, **changes}), encoding="utf-8")
        return path

    no_size = [{**entry, "native_dimensions": None} for entry in raw["models"]]
    with pytest.raises(ModelFactsError):
        load_facts(write("no-size.json", models=no_size))
    with pytest.raises(ModelFactsError):
        load_facts(write("no-default.json", default=None))
    with pytest.raises(ModelFactsError):
        embedder_facts("text-embedding-3-large", path=write("list.json", models=["text-embedding-3-large"]))
