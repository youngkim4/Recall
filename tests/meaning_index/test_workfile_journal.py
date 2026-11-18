"""The work file after a process died inside a write: nothing that was paid for is lost."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from meaning_index.build import build_index, estimate_build
from meaning_index.errors import EmbeddingKeyError, StoreError
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import EmbedderFacts
from meaning_index.store import IndexStore
from meaning_index.workfile import WorkFile
from tests.meaning_index.support import ListSource, ScriptedEmbedder, crashed_copy, cut_texts, notes, unit, words

FACTS = EmbedderFacts("fake-embedding-1", 64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="")
SAVED = [f"{n:064x}" for n in range(50)]


def saved_then_crashed(tmp_path: Path) -> Path:
    """A work file with 50 vectors that were paid for, and then a write that died."""
    live = tmp_path / "live"
    live.mkdir()
    WorkFile(live / "work.sqlite3", 8).put(SAVED, unit(50, 8))
    return crashed_copy(live, tmp_path / "crashed", 8) / "work.sqlite3"


def stopped_then_crashed(tmp_path: Path, source: ListSource) -> Path:
    """The folder of a build that paid for two requests, stopped, and then died inside a write."""
    live = tmp_path / "live"
    embedder = ScriptedEmbedder({3: EmbeddingKeyError("no key")}, dimensions=64, model="fake-embedding-1")
    with pytest.raises(EmbeddingKeyError):
        build_index(source, embedder, IndexStore(live), count_tokens=words)
    return crashed_copy(live, tmp_path / "crashed", 64)


def test_a_repair_takes_the_dead_write_out_and_keeps_every_saved_vector(tmp_path: Path) -> None:
    path = saved_then_crashed(tmp_path)
    work = WorkFile(path, 8)
    work.repair()
    assert path.exists()
    assert not path.with_name("work.sqlite3-journal").exists()
    assert work.hashes() == set(SAVED)
    assert np.array_equal(work.get_many(SAVED[:1])[SAVED[0]], unit(50, 8)[0])


def test_a_read_that_can_not_be_done_is_an_error_and_not_an_empty_file(tmp_path: Path) -> None:
    path = saved_then_crashed(tmp_path)
    before = path.read_bytes()
    with pytest.raises(StoreError):
        WorkFile(path, 8).hashes()
    with pytest.raises(StoreError):
        WorkFile(path, 8).get_many(SAVED)
    assert path.read_bytes() == before
    assert path.with_name("work.sqlite3-journal").exists()


def test_the_next_build_after_a_dead_write_does_not_pay_two_times(tmp_path: Path) -> None:
    source = ListSource({"chat-a": notes(250)})
    folder = stopped_then_crashed(tmp_path, source)
    embedder = FakeEmbedder()
    result = build_index(source, embedder, IndexStore(folder), count_tokens=words)
    assert embedder.texts == cut_texts(source)[192:]
    assert (result.embedded, result.reused) == (58, 192)


def test_an_estimate_that_can_not_read_the_work_file_says_that_it_is_an_upper_bound(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = ListSource({"chat-a": notes(250)})
    folder = stopped_then_crashed(tmp_path, source)
    listed = {path.name: path.read_bytes() for path in folder.iterdir()}
    with caplog.at_level(logging.WARNING, logger="meaning_index"):
        estimate = estimate_build(source, IndexStore(folder), FACTS, words)
    assert estimate.upper_bound is True
    assert (estimate.segments, estimate.new_segments) == (250, 250)
    assert any("work file" in record.getMessage() for record in caplog.records)
    # the estimate changed nothing: the build still finds what was paid
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == listed
    again = FakeEmbedder()
    build_index(source, again, IndexStore(folder), count_tokens=words)
    assert len(again.texts) == 58


def test_an_estimate_that_read_everything_is_no_upper_bound(tmp_path: Path) -> None:
    source = ListSource({"chat-a": notes(10)})
    assert estimate_build(source, IndexStore(tmp_path / "meaning"), FACTS, words).upper_bound is False


def test_a_work_file_that_is_no_database_is_removed_by_the_repair_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "work.sqlite3"
    path.write_bytes(b"this is no database at all, only invented bytes" * 40)
    work = WorkFile(path, 8)
    with pytest.raises(StoreError):
        work.hashes()
    assert path.exists()
    work.repair()
    assert not path.exists()
    assert any("removed" in record.getMessage() for record in caplog.records)
    assert work.hashes() == frozenset()


def test_a_repair_that_can_not_open_the_file_keeps_it_and_stops_the_build(tmp_path: Path) -> None:
    folder = tmp_path / "meaning"
    (folder / "work.sqlite3").mkdir(parents=True)  # a folder where the file must be: SQLite can not open it
    embedder = FakeEmbedder()
    with pytest.raises(StoreError):
        build_index(ListSource({"chat-a": notes(5)}), embedder, IndexStore(folder), count_tokens=words)
    assert embedder.calls == []
    assert (folder / "work.sqlite3").is_dir()


def test_the_work_file_must_take_a_row_before_the_first_request(tmp_path: Path) -> None:
    folder = tmp_path / "meaning"
    store = IndexStore(folder)
    with store.building():
        pass
    folder.chmod(0o500)  # the lock file is there; no new file can be made
    embedder = FakeEmbedder()
    try:
        with pytest.raises(StoreError):
            build_index(ListSource({"chat-a": notes(5)}), embedder, store, count_tokens=words)
    finally:
        folder.chmod(0o700)
    assert embedder.calls == []


def test_the_check_of_the_work_file_leaves_no_row(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    work.check_writable()
    assert work.hashes() == frozenset()
    work.put(SAVED[:2], unit(2, 8))
    work.check_writable()
    assert work.hashes() == set(SAVED[:2])


def test_a_row_with_a_value_that_is_no_number_is_not_kept(tmp_path: Path) -> None:
    rows = unit(2, 8).copy()
    rows[1, 3] = np.nan
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    with pytest.raises(StoreError):
        work.put(SAVED[:2], rows)
    assert work.hashes() == frozenset()
