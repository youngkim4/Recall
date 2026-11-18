"""A reader and a build at the same time: the reader gets one whole build, never parts of two.

Each test swaps a second build into the folder at one exact point of a read,
through a hook on one of the three file readers. The two builds have the same
number of segments and the same dimensions, so only the id of a build tells
them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meaning_index import store_files
from meaning_index.search import MeaningIndex
from tests.meaning_index.store_kit import FILES, make, put, side, two_builds

READERS = ("read_meta", "read_segments", "open_vectors")


def swap_during(monkeypatch: pytest.MonkeyPatch, folder: Path, reader: str, build: Any, *, before: bool) -> list[int]:
    """Swap `build` into the folder one time, just before or just after the first call of one reader."""
    real, done = getattr(store_files, reader), []

    def hooked(*args: Any, **named: Any) -> Any:
        if before and not done:
            done.append(1)
            put(make(folder), build)
        found = real(*args, **named)
        if not before and not done:
            done.append(1)
            put(make(folder), build)
        return found

    monkeypatch.setattr(store_files, reader, hooked)
    return done


@pytest.mark.parametrize("before", [False, True], ids=["after", "before"])
@pytest.mark.parametrize("reader", READERS)
def test_a_swap_in_the_middle_of_a_read_gives_one_whole_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, reader: str, before: bool
) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    done = swap_during(monkeypatch, folder, reader, b, before=before)
    found = store.read()
    assert done == [1]
    assert found is not None
    for segments, rows in (a, b):
        if list(found.segments) == segments:
            assert np.array_equal(np.asarray(found.vectors), rows)
            break
    else:
        raise AssertionError("the segments are of no build")
    assert side(store, a, b) == "b"


@pytest.mark.parametrize("reader", READERS)
def test_a_swap_in_the_middle_of_a_status_gives_one_whole_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    done = swap_during(monkeypatch, folder, reader, b, before=False)
    assert store.status().state == "ready"
    assert done == [1]


@pytest.mark.parametrize("reader", READERS)
def test_a_search_never_answers_with_the_rows_of_another_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    swap_during(monkeypatch, folder, reader, b, before=False)
    index = MeaningIndex.load(store)
    assert index is not None
    segments, rows = a if index.meta.build_id != store.read().meta.build_id else b  # type: ignore[union-attr]
    for row, segment in enumerate(segments):
        assert index.search(rows[row].tolist(), limit=1)[0].segment == segment


def test_the_vectors_file_names_its_build(folder: Path) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    other = folder.parent / "other-build"
    put(make(other), b)
    # the vectors of another build with the same shape, and nothing else changed
    (folder / "vectors.npy").write_bytes((other / "vectors.npy").read_bytes())
    assert store.status().state == "none"
    assert store.read() is None


@pytest.mark.parametrize("name", FILES)
def test_one_file_of_another_build_of_the_same_shape_is_never_a_whole_index(folder: Path, name: str) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    other = folder.parent / "other-build"
    put(make(other), b)
    (folder / name).write_bytes((other / name).read_bytes())
    assert side(store, a, b) == "none"


def test_the_vectors_file_is_still_an_array_that_numpy_loads(folder: Path) -> None:
    a, _ = two_builds()
    put(make(folder), a)
    assert np.array_equal(np.load(folder / "vectors.npy", allow_pickle=False), a[1])
    assert np.array_equal(np.load(folder / "vectors.npy", mmap_mode="r"), a[1])
