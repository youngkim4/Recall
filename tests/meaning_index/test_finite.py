"""A value that is no number never reaches a search: not from a reply, not from a file, not from a query."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from meaning_index.batching import unit_rows
from meaning_index.build import build_index
from meaning_index.errors import EmbeddingReplyError, StoreError
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.search import MeaningIndex
from meaning_index.shapes import IndexMeta
from meaning_index.store import IndexStore
from tests.meaning_index.store_kit import content, fill, make
from tests.meaning_index.support import ListSource, kept, notes, unit, words

META = IndexMeta(1, "invented", "invented-model", 16, "2026-11-17T00:00:00+00:00", 5, None)


def parts() -> tuple[list, np.ndarray]:
    return [kept(n) for n in range(5)], unit(5, 16, seed=3)


def spoil(folder: Path, value: float) -> None:
    """Put one value that is no number into the stored vectors, in place."""
    whole = bytearray((folder / "vectors.npy").read_bytes())
    stored = np.load(folder / "vectors.npy", allow_pickle=False)
    start = whole.index(stored.tobytes()[:32])
    whole[start + 6 : start + 8] = np.array([value], dtype="<f2").tobytes()
    (folder / "vectors.npy").write_bytes(bytes(whole))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_an_index_from_parts_with_a_value_that_is_no_number_is_refused(value: float) -> None:
    segments, rows = parts()
    rows = rows.copy()
    rows[3, 0] = value
    with pytest.raises(ValueError):
        MeaningIndex(META, segments, rows)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_stored_value_that_is_no_number_is_a_typed_error_at_the_read(folder: Path, value: float) -> None:
    store = make(folder)
    fill(store)
    spoil(folder, value)
    assert store.status().state == "ready"  # the status does not read the rows
    with pytest.raises(StoreError):
        store.read()
    with pytest.raises(StoreError):
        MeaningIndex.load(store)


def test_a_build_over_vectors_that_hold_no_number_embeds_everything_again(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = ListSource({"chat-a": notes(10)})
    build_index(source, FakeEmbedder(), IndexStore(folder), count_tokens=words)
    spoil(folder, np.nan)
    embedder = FakeEmbedder()
    with caplog.at_level(logging.WARNING, logger="meaning_index"):
        result = build_index(source, embedder, IndexStore(folder), count_tokens=words)
    assert (result.embedded, result.reused) == (10, 0)
    assert any("can not be read" in record.getMessage() for record in caplog.records)
    assert MeaningIndex.load(IndexStore(folder)) is not None


def test_rows_with_a_value_that_is_no_number_are_not_written(folder: Path) -> None:
    store = make(folder)
    fill(store)
    before = content(folder)
    rows = unit(4, 8).copy()
    rows[2, 5] = np.nan
    with pytest.raises(StoreError), store.building():
        store.replace("model-b", 8, [kept(n) for n in range(4)], [rows])
    assert content(folder) == before


@pytest.mark.parametrize("reply", [[[1e200, 1e200, 0.0, 0.0]], [[1e-200, 0.0, 0.0, 0.0]]])
def test_a_reply_whose_length_can_not_be_worked_out_is_refused(reply: list) -> None:
    with pytest.raises(EmbeddingReplyError):
        unit_rows(reply, 1, 4)


def test_every_row_of_a_reply_that_is_kept_is_finite_and_has_unit_length() -> None:
    rows = unit_rows([[1e30, 1e30, 0.0, 0.0], [1e-30, 0.0, 0.0, 0.0]], 2, 4)
    assert np.isfinite(rows).all()
    assert np.allclose(np.linalg.norm(rows.astype(np.float64), axis=1), 1.0, atol=1e-3)


@pytest.mark.parametrize("scale", [1e30, 1e300, 1e-30, 1e-300])
def test_a_query_of_any_length_gives_the_hits_of_the_same_query_at_unit_length(scale: float) -> None:
    segments, rows = parts()
    index = MeaningIndex(META, segments, rows)
    plain = rows[1].astype(np.float64)
    expected = index.search(plain.tolist())
    found = index.search((plain * scale).tolist())
    assert [hit.segment for hit in found] == [hit.segment for hit in expected]
    assert [hit.score for hit in found] == pytest.approx([hit.score for hit in expected], abs=1e-5)
    assert found[0].score == pytest.approx(1.0, abs=2e-3)
    assert len(found) == 5


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_query_with_a_value_that_is_no_number_is_refused(value: float) -> None:
    segments, rows = parts()
    index = MeaningIndex(META, segments, rows)
    query = rows[1].astype(np.float64)
    query[4] = value
    with pytest.raises(ValueError):
        index.search(query.tolist())


def test_every_hit_is_given_whatever_the_limit(folder: Path) -> None:
    segments, rows = parts()
    index = MeaningIndex(META, segments, rows)
    for limit in (1, 2, 4, 5, 9):
        assert len(index.search(rows[0].tolist(), limit=limit)) == min(limit, 5)
