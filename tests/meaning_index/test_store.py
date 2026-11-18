"""The files of the index: what they hold, who can read them, and what is read as no index."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from meaning_index.errors import IndexBusyError, StoreError
from meaning_index.shapes import IndexStatus
from meaning_index.store import FORMAT_VERSION
from tests.meaning_index.store_kit import FILES, fill, make, mode
from tests.meaning_index.support import unit


def test_a_folder_with_no_index_has_the_state_none_and_reading_makes_no_file(folder: Path) -> None:
    store = make(folder)
    assert store.status() == IndexStatus(state="none")
    assert store.read() is None
    assert not folder.exists()


def test_what_is_written_is_read_back_in_row_order(folder: Path) -> None:
    store = make(folder)
    segments, rows = fill(store)
    found = store.read()
    assert found is not None
    assert list(found.segments) == segments
    assert found.vectors.dtype == np.float16
    assert found.vectors.shape == (5, 8)
    assert np.array_equal(np.asarray(found.vectors), rows)
    assert found.meta.model == "model-a"
    assert found.meta.dimensions == 8


def test_the_status_says_what_the_index_holds(folder: Path) -> None:
    store = make(folder)
    fill(store)
    assert store.status() == IndexStatus(
        state="ready",
        segments=5,
        model="model-a",
        dimensions=8,
        built_at="2026-11-17T12:00:00+00:00",
        covered_until="2024-06-05 09:30:00",
    )


def test_the_meta_file_holds_the_facts_of_the_contract(folder: Path) -> None:
    fill(make(folder))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["format_version"] == FORMAT_VERSION == 1
    assert (meta["model"], meta["dimensions"], meta["segments"]) == ("model-a", 8, 5)
    assert meta["built_at"] == "2026-11-17T12:00:00+00:00"
    assert meta["covered_until"] == "2024-06-05 09:30:00"
    assert set(meta) == {"format_version", "build_id", "model", "dimensions", "built_at", "segments", "covered_until"}


def test_the_segments_file_has_one_row_for_each_segment_with_the_columns_of_the_contract(folder: Path) -> None:
    segments, _ = fill(make(folder))
    conn = sqlite3.connect(folder / "segments.sqlite3")
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(segments)")]
        query = 'SELECT segment_id, conversation_id, content_hash, "row" FROM segments ORDER BY "row"'
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()
    assert columns == [
        "segment_id",
        "conversation_id",
        "first_message_id",
        "last_message_id",
        "first_timestamp",
        "last_timestamp",
        "message_count",
        "content_hash",
        "row",
    ]
    assert rows == [(s.segment_id, s.conversation_id, s.content_hash, n) for n, s in enumerate(segments)]


def test_the_vectors_file_is_a_plain_array_of_float16(folder: Path) -> None:
    _, rows = fill(make(folder))
    loaded = np.load(folder / "vectors.npy", allow_pickle=False)
    assert loaded.dtype == np.float16
    assert np.array_equal(loaded, rows)


def test_every_file_is_for_its_owner_only(folder: Path) -> None:
    store = make(folder)
    fill(store)
    with store.building():
        store.work_file(8).put(["a" * 64], unit(1, 8))
        assert mode(folder / "build.lock") == 0o600
    assert mode(folder) == 0o700
    for name in (*FILES, "work.sqlite3"):
        assert mode(folder / name) == 0o600, name


def test_a_file_that_was_open_to_others_is_made_private_again(folder: Path) -> None:
    store = make(folder)
    fill(store)
    for name in FILES:
        os.chmod(folder / name, 0o644)
    fill(store, seed=8)
    assert [mode(folder / name) for name in FILES] == [0o600] * 3


def test_a_build_leaves_no_work_of_its_own_beside_the_files(folder: Path) -> None:
    fill(make(folder))
    assert sorted(path.name for path in folder.iterdir()) == ["build.lock", *sorted(FILES)]


def test_a_second_write_replaces_the_first(folder: Path) -> None:
    store = make(folder)
    fill(store, count=5)
    segments, rows = fill(store, count=3, seed=9)
    found = store.read()
    assert found is not None
    assert list(found.segments) == segments
    assert np.array_equal(np.asarray(found.vectors), rows)
    assert store.status().segments == 3


def test_an_index_with_no_segment_is_ready_and_empty(folder: Path) -> None:
    store = make(folder)
    with store.building():
        store.replace("model-a", 8, [], [])
    assert store.status() == IndexStatus(
        state="ready", segments=0, model="model-a", dimensions=8, built_at="2026-11-17T12:00:00+00:00"
    )
    found = store.read()
    assert found is not None
    assert found.vectors.shape == (0, 8)
    assert found.segments == ()


def test_a_meta_file_that_can_not_be_read_is_no_index(folder: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = make(folder)
    fill(store)
    (folder / "meta.json").write_text("{not json", encoding="utf-8")
    assert store.status().state == "none"
    assert store.read() is None
    assert any("meta" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("missing", ["segments.sqlite3", "vectors.npy"])
def test_an_index_with_a_file_missing_is_no_index(folder: Path, missing: str) -> None:
    store = make(folder)
    fill(store)
    (folder / missing).unlink()
    assert store.status().state == "none"
    assert store.read() is None


def test_a_vectors_file_of_another_shape_is_no_index(folder: Path) -> None:
    store = make(folder)
    fill(store)
    with open(folder / "vectors.npy", "wb") as handle:
        np.save(handle, unit(4, 8))
    assert store.status().state == "none"
    assert store.read() is None


def test_an_index_of_a_newer_format_is_a_typed_error(folder: Path) -> None:
    store = make(folder)
    fill(store)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    (folder / "meta.json").write_text(json.dumps({**meta, "format_version": FORMAT_VERSION + 1}), encoding="utf-8")
    with pytest.raises(StoreError):
        store.status()
    with pytest.raises(StoreError):
        store.read()


def test_one_build_holds_the_folder_at_a_time(folder: Path) -> None:
    first, second = make(folder), make(folder)
    with first.building():
        with pytest.raises(IndexBusyError), second.building():
            pass
    with second.building():
        pass


def test_the_files_are_replaced_only_inside_a_build(folder: Path) -> None:
    with pytest.raises(StoreError):
        make(folder).replace("model-a", 8, [], [])


def test_a_loaded_index_keeps_its_rows_when_a_build_replaces_the_files(folder: Path) -> None:
    store = make(folder)
    _, rows = fill(store)
    found = store.read()
    assert found is not None
    fill(store, count=3, seed=9)
    assert np.array_equal(np.asarray(found.vectors), rows)


def test_a_folder_that_can_not_be_made_is_a_typed_error(tmp_path: Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(StoreError), make(blocker / "meaning").building():
        pass


def test_a_failure_of_the_caller_inside_a_build_is_not_taken_for_a_failure_of_the_store(folder: Path) -> None:
    store = make(folder)
    with pytest.raises(OSError) as caught, store.building():
        raise OSError("the caller failed")
    assert not isinstance(caught.value, StoreError)
    with store.building():
        pass


def test_a_segments_file_with_fewer_rows_than_its_build_names_is_no_index(folder: Path) -> None:
    store = make(folder)
    fill(store)
    conn = sqlite3.connect(folder / "segments.sqlite3")
    conn.execute('DELETE FROM segments WHERE "row" = 4')
    conn.commit()
    conn.close()
    assert store.read() is None


def test_a_segments_file_with_a_row_missing_in_the_middle_is_no_index(folder: Path) -> None:
    store = make(folder)
    fill(store)
    conn = sqlite3.connect(folder / "segments.sqlite3")
    conn.execute('DELETE FROM segments WHERE "row" = 2')
    conn.commit()
    conn.close()
    assert store.read() is None
    assert store.status().state == "none"


def test_a_folder_with_odd_characters_in_its_name_is_read(tmp_path: Path) -> None:
    odd = tmp_path / "Application Support" / "a?b#c%20d" / "meaning"
    store = make(odd)
    segments, rows = fill(store)
    found = store.read()
    assert found is not None
    assert list(found.segments) == segments
    assert store.folder == odd
    assert store.status().state == "ready"
