"""A build on a folder that is not whole: a damaged index, an index of a newer format, a damaged work file."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from meaning_index.build import build_index, estimate_build
from meaning_index.errors import IndexFormatError, StoreError
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import EmbedderFacts
from meaning_index.store import IndexStore
from meaning_index.workfile import WorkFile
from tests.meaning_index.support import ListSource, notes, words

FILES = ("segments.sqlite3", "vectors.npy", "meta.json")
FACTS = EmbedderFacts("fake-embedding-1", 64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="")


def built(folder: Path, count: int = 10) -> ListSource:
    source = ListSource({"chat-a": notes(count)})
    build_index(source, FakeEmbedder(), IndexStore(folder), count_tokens=words)
    return source


def lose_a_row(folder: Path) -> None:
    conn = sqlite3.connect(folder / "segments.sqlite3")
    conn.execute('DELETE FROM segments WHERE "row" = 3')
    conn.commit()
    conn.close()


def test_a_build_over_an_index_that_can_not_be_read_embeds_everything_and_says_so(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = built(folder)
    lose_a_row(folder)
    embedder = FakeEmbedder()
    with caplog.at_level(logging.WARNING, logger="meaning_index"):
        result = build_index(source, embedder, IndexStore(folder), count_tokens=words)
    assert (result.segments, result.embedded, result.reused, result.removed) == (10, 10, 0, 0)
    assert len(embedder.texts) == 10
    assert any("can not be read" in record.getMessage() for record in caplog.records)
    assert IndexStore(folder).read() is not None


def test_an_estimate_over_an_index_that_can_not_be_read_counts_everything(folder: Path) -> None:
    source = built(folder)
    lose_a_row(folder)
    estimate = estimate_build(source, IndexStore(folder), FACTS, words)
    assert (estimate.segments, estimate.new_segments) == (10, 10)


def test_an_index_of_a_newer_format_is_not_built_over(folder: Path) -> None:
    source = built(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    (folder / "meta.json").write_text(json.dumps({**meta, "format_version": 2}), encoding="utf-8")
    before = {name: (folder / name).read_bytes() for name in FILES}
    embedder = FakeEmbedder()
    with pytest.raises(IndexFormatError) as caught:
        build_index(source, embedder, IndexStore(folder), count_tokens=words)
    assert isinstance(caught.value, StoreError)
    assert embedder.calls == []
    assert {name: (folder / name).read_bytes() for name in FILES} == before
    with pytest.raises(IndexFormatError):
        estimate_build(source, IndexStore(folder), FACTS, words)
    with pytest.raises(IndexFormatError):
        IndexStore(folder).status()


def test_a_work_file_that_is_no_database_is_removed_and_the_build_goes_on(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "work.sqlite3").write_bytes(b"invented bytes that are no database" * 50)
    result = build_index(ListSource({"chat-a": notes(5)}), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    assert result.embedded == 5
    assert not (folder / "work.sqlite3").exists()


def test_reading_an_empty_work_file_does_not_write_to_it(tmp_path: Path) -> None:
    path = tmp_path / "work.sqlite3"
    path.write_bytes(b"")
    work = WorkFile(path, 8)
    assert work.hashes() == frozenset()
    assert work.get_many(["a" * 64]) == {}
    assert path.read_bytes() == b""


def test_an_estimate_leaves_the_work_file_of_a_stopped_build_as_it_is(folder: Path) -> None:
    from meaning_index.errors import EmbeddingKeyError
    from tests.meaning_index.support import ScriptedEmbedder

    source = ListSource({"chat-a": notes(150)})
    embedder = ScriptedEmbedder({2: EmbeddingKeyError("no key")}, dimensions=64, model="fake-embedding-1")
    with pytest.raises(EmbeddingKeyError):
        build_index(source, embedder, IndexStore(folder), count_tokens=words)
    before = (folder / "work.sqlite3").read_bytes()
    listed = sorted(path.name for path in folder.iterdir())
    assert estimate_build(source, IndexStore(folder), FACTS, words).new_segments == 54
    assert (folder / "work.sqlite3").read_bytes() == before
    assert sorted(path.name for path in folder.iterdir()) == listed
