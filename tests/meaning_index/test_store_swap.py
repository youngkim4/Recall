"""How a new set of index files takes the place of the old one, and what a swap that stops leaves."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from meaning_index import store_files
from meaning_index.errors import StoreError
from meaning_index.shapes import IndexStatus
from tests.meaning_index.store_kit import FILES, content, fill, make, torn
from tests.meaning_index.support import kept, unit


def test_rows_that_stop_in_the_middle_leave_the_old_index_as_it_was(folder: Path) -> None:
    store = make(folder)
    fill(store)
    before = content(folder)

    def stream():
        yield unit(2, 8)
        raise RuntimeError("the rows stopped")

    with pytest.raises(RuntimeError), store.building():
        store.replace("model-b", 8, [kept(n) for n in range(4)], stream())
    assert content(folder) == before
    assert not list(folder.glob("*.new"))


@pytest.mark.parametrize("rows", [[unit(3, 8)], [unit(5, 8)], [unit(4, 6)], [unit(4, 8).astype(np.float32)]])
def test_rows_that_do_not_fit_the_segments_are_refused_and_nothing_changes(folder: Path, rows: list) -> None:
    store = make(folder)
    fill(store)
    before = content(folder)
    with pytest.raises(StoreError), store.building():
        store.replace("model-b", 8, [kept(n) for n in range(4)], rows)
    assert content(folder) == before


@pytest.mark.parametrize("fails_at", [1, 2, 3])
def test_a_swap_that_fails_puts_the_old_files_back(
    folder: Path, monkeypatch: pytest.MonkeyPatch, fails_at: int
) -> None:
    store = make(folder)
    fill(store)
    before = content(folder)
    real, calls = os.replace, []

    def replace(source: object, target: object) -> None:
        calls.append(str(target))
        if len(calls) == fails_at:
            raise OSError("the disk said no")
        real(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(store_files.os, "replace", replace)
    with pytest.raises(StoreError), store.building():
        store.replace("model-b", 8, [kept(n) for n in range(4)], [unit(4, 8)])
    monkeypatch.undo()
    assert content(folder) == before
    assert store.status().model == "model-a"
    assert not list(folder.glob("*.new")) and not list(folder.glob("*.old"))


def test_a_first_swap_that_fails_leaves_no_index(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make(folder)
    real, calls = os.replace, []

    def replace(source: object, target: object) -> None:
        calls.append(str(target))
        if len(calls) == 2:
            raise OSError("the disk said no")
        real(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(store_files.os, "replace", replace)
    with pytest.raises(StoreError), store.building():
        store.replace("model-a", 8, [kept(n) for n in range(4)], [unit(4, 8)])
    monkeypatch.undo()
    assert store.status().state == "none"
    assert store.read() is None
    assert not [path.name for path in folder.iterdir() if path.name != "build.lock"]


def test_the_segments_file_is_replaced_first_and_the_meta_file_last(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make(folder)
    real, order = os.replace, []

    def replace(source: object, target: object) -> None:
        order.append(Path(str(target)).name)
        real(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(store_files.os, "replace", replace)
    fill(store)
    assert order == list(FILES) == ["segments.sqlite3", "vectors.npy", "meta.json"]


@pytest.mark.parametrize("replaced", [1, 2])
def test_a_swap_that_did_not_end_still_gives_the_build_before_it(folder: Path, replaced: int) -> None:
    store = make(folder)
    torn(folder, store, replaced)
    assert store.status().model == "model-a"
    found = store.read()
    assert found is not None and found.meta.segments == 5
    for name in FILES:
        (folder / f"{name}.old").unlink(missing_ok=True)
    assert store.status() == IndexStatus(state="none")
    assert store.read() is None


@pytest.mark.parametrize("replaced", [1, 2])
def test_the_next_build_puts_back_the_files_of_a_swap_that_did_not_end(folder: Path, replaced: int) -> None:
    store = make(folder)
    before = torn(folder, store, replaced)
    with store.building():
        assert content(folder) == before
        assert store.status().model == "model-a"
    assert not list(folder.glob("*.old"))


def test_a_swap_that_ended_keeps_the_new_files_when_its_backups_are_still_there(folder: Path) -> None:
    store = make(folder)
    torn(folder, store, 3)
    after = content(folder)
    with store.building():
        assert content(folder) == after
        assert store.status().model == "model-b"
    assert not list(folder.glob("*.old"))


def test_a_disk_that_refuses_the_new_files_is_a_typed_error(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make(folder)
    fill(store)
    before = content(folder)

    def refuse(path: object, meta: object) -> None:
        raise OSError("no space left")

    monkeypatch.setattr(store_files, "write_meta", refuse)
    with pytest.raises(StoreError), store.building():
        store.replace("model-b", 8, [kept(n) for n in range(4)], [unit(4, 8)])
    assert content(folder) == before
    assert not list(folder.glob("*.new"))


def test_a_repair_that_fails_is_a_typed_error(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make(folder)
    torn(folder, store, 1)

    def refuse(source: object, target: object) -> None:
        raise OSError("the disk said no")

    monkeypatch.setattr(store_files.os, "replace", refuse)
    with pytest.raises(StoreError), store.building():
        pass
