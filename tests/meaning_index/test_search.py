"""Search: the nearest segments for a query vector, with filters by conversation and by day."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from meaning_index.build import build_index
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.search import MeaningIndex
from meaning_index.shapes import IndexMeta, SearchHit, Segment
from meaning_index.store import IndexStore
from tests.meaning_index.support import ListSource, kept, msg, unit, words

LAKE, HOME, WORK = "chat-lake", "chat-home", "chat-work"


def talks() -> dict:
    return {
        LAKE: [
            msg("l1", "2024-06-03 09:00:00", "did you buy the ferry tickets"),
            msg("l2", "2024-06-03 09:02:00", "yes two tickets for saturday", True),
            msg("l3", "2024-06-10 23:58:00", "the canoe is on the roof"),
            msg("l4", "2024-06-11 00:03:00", "great, the lake will be calm", True),
        ],
        HOME: [
            msg("h1", "2024-06-05 18:00:00", "what should we cook for dinner"),
            msg("h2", "2024-06-05 18:01:00", "pasta with mushrooms", True),
            msg("h3", "2024-07-01 08:00:00", "the dog needs a walk"),
        ],
        WORK: [
            msg("w1", "2024-06-05 10:00:00", "the report is due on friday"),
            msg("w2", "2024-08-20 10:00:00", "can you book a ferry for the offsite", True),
        ],
    }


@pytest.fixture
def index(folder: Path) -> MeaningIndex:
    build_index(ListSource(talks()), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    loaded = MeaningIndex.load(IndexStore(folder))
    assert loaded is not None
    return loaded


def query(text: str) -> list[float]:
    return list(FakeEmbedder().embed([text])[0])


def firsts(hits: list[SearchHit]) -> list[str]:
    return [hit.segment.first_message_id for hit in hits]


def test_a_folder_with_no_index_loads_as_none(folder: Path) -> None:
    assert MeaningIndex.load(IndexStore(folder)) is None


def test_the_nearest_segment_comes_first(index: MeaningIndex) -> None:
    hits = index.search(query("boat ticket"))
    assert firsts(hits)[:2] == ["l1", "w2"]
    assert firsts(index.search(query("supper")))[0] == "h1"
    assert firsts(index.search(query("puppy")))[0] == "h3"


def test_a_hit_is_a_stored_segment_and_its_score(index: MeaningIndex) -> None:
    hit = index.search(query("ferry tickets"))[0]
    assert isinstance(hit, SearchHit)
    assert isinstance(hit.segment, Segment)
    assert hit.segment.conversation_id == LAKE
    assert (hit.segment.first_message_id, hit.segment.last_message_id, hit.segment.message_count) == ("l1", "l2", 2)
    assert (hit.segment.first_timestamp, hit.segment.last_timestamp) == ("2024-06-03 09:00:00", "2024-06-03 09:02:00")
    assert isinstance(hit.score, float)
    assert 0.3 < hit.score <= 1.001


def test_hits_are_in_order_of_score_and_every_segment_can_be_given(index: MeaningIndex) -> None:
    hits = index.search(query("ferry"))
    assert len(hits) == len(index) == 6
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_the_score_is_the_dot_product_with_a_query_of_unit_length(index: MeaningIndex) -> None:
    plain = index.search(query("ferry tickets"))
    scaled = index.search([value * 25.0 for value in query("ferry tickets")])
    assert firsts(plain) == firsts(scaled)
    assert [hit.score for hit in plain] == pytest.approx([hit.score for hit in scaled], abs=1e-6)
    text = "2024-06-03\nThem: did you buy the ferry tickets\nMe: yes two tickets for saturday"
    assert plain[0].score == pytest.approx(float(np.dot(query(text), query("ferry tickets"))), abs=2e-3)


def test_limit_cuts_the_hits(index: MeaningIndex) -> None:
    assert len(index.search(query("ferry"), limit=2)) == 2
    assert firsts(index.search(query("ferry"), limit=2)) == firsts(index.search(query("ferry")))[:2]
    assert len(index.search(query("ferry"), limit=500)) == 6


def test_the_default_limit_is_20(folder: Path) -> None:
    segments = [kept(n) for n in range(50)]
    index = MeaningIndex(meta_for(50, 8), segments, unit(50, 8))
    assert len(index.search(unit(1, 8, seed=3)[0].tolist())) == 20


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "3"])
def test_a_limit_that_is_no_count_is_refused(index: MeaningIndex, limit: object) -> None:
    with pytest.raises(ValueError):
        index.search(query("ferry"), limit=limit)  # type: ignore[arg-type]


def test_a_filter_by_conversation_gives_only_its_segments(index: MeaningIndex) -> None:
    hits = index.search(query("ferry"), conversations=[WORK, HOME])
    assert {hit.segment.conversation_id for hit in hits} == {WORK, HOME}
    assert len(hits) == 4
    assert firsts(hits)[0] == "w2"
    assert {hit.segment.conversation_id for hit in index.search(query("ferry"), conversations=LAKE)} == {LAKE}


def test_no_conversation_and_an_unknown_conversation_match_nothing(index: MeaningIndex) -> None:
    assert index.search(query("ferry"), conversations=[]) == []
    assert index.search(query("ferry"), conversations=["chat-nobody"]) == []
    assert len(index.search(query("ferry"), conversations=["chat-nobody", WORK, WORK])) == 2


@pytest.mark.parametrize("bad", [7, b"chat", {"chat-lake": 1}, [LAKE, 7]])
def test_conversations_that_are_no_ids_are_refused(index: MeaningIndex, bad: object) -> None:
    with pytest.raises(TypeError):
        index.search(query("ferry"), conversations=bad)  # type: ignore[arg-type]


def test_a_filter_by_day_keeps_the_segments_that_overlap_the_days(index: MeaningIndex) -> None:
    assert sorted(firsts(index.search(query("ferry"), since="2024-06-05", until="2024-06-05"))) == ["h1", "w1"]
    assert sorted(firsts(index.search(query("ferry"), since="2024-07-01"))) == ["h3", "w2"]
    assert sorted(firsts(index.search(query("ferry"), until="2024-06-03"))) == ["l1"]
    assert index.search(query("ferry"), since="2024-06-06", until="2024-06-09") == []
    assert index.search(query("ferry"), since="2024-09-01") == []
    assert index.search(query("ferry"), until="2024-01-01") == []


def test_a_segment_that_crosses_midnight_matches_both_of_its_days(index: MeaningIndex) -> None:
    assert firsts(index.search(query("canoe"), since="2024-06-11", until="2024-06-11")) == ["l3"]
    assert firsts(index.search(query("canoe"), since="2024-06-10", until="2024-06-10")) == ["l3"]
    assert firsts(index.search(query("canoe"), since="2024-06-11", until="2024-06-30")) == ["l3"]
    assert firsts(index.search(query("canoe"), since="2024-06-12", until="2024-06-30")) == []


def test_the_two_filters_work_together(index: MeaningIndex) -> None:
    hits = index.search(query("ferry"), conversations=[WORK], since="2024-06-01", until="2024-06-30")
    assert firsts(hits) == ["w1"]


def test_a_far_day_is_a_bound_with_nothing_beyond_it(index: MeaningIndex) -> None:
    assert len(index.search(query("ferry"), since="1900-01-01", until="9999-12-31")) == 6


@pytest.mark.parametrize("bad", ["2024-6-3", "yesterday", "2024-02-30", "2024-06-03 10:00:00", "", 20240603])
def test_a_day_that_is_no_day_is_refused(index: MeaningIndex, bad: object) -> None:
    with pytest.raises(ValueError):
        index.search(query("ferry"), since=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        index.search(query("ferry"), until=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [[1.0, 0.0], [0.0] * 64, [float("nan")] * 64, "ferry", None])
def test_a_query_that_is_no_vector_of_the_index_is_refused(index: MeaningIndex, bad: object) -> None:
    with pytest.raises(ValueError):
        index.search(bad)  # type: ignore[arg-type]


def test_the_index_says_what_it_holds(index: MeaningIndex) -> None:
    assert (index.model, index.dimensions, len(index)) == ("fake-embedding-1", 64, 6)
    assert index.meta.segments == 6


def test_the_stored_vectors_have_unit_length(folder: Path, index: MeaningIndex) -> None:
    stored = IndexStore(folder).read()
    assert stored is not None
    lengths = np.linalg.norm(np.asarray(stored.vectors, dtype=np.float64), axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3)


def test_an_index_with_no_segment_gives_no_hit(folder: Path) -> None:
    build_index(ListSource({}), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    index = MeaningIndex.load(IndexStore(folder))
    assert index is not None
    assert len(index) == 0
    assert index.search(query("ferry")) == []
    assert index.search(query("ferry"), conversations=[LAKE], since="2024-06-01") == []


def test_search_and_store_agree_on_rows_after_a_build_that_removes_segments(folder: Path) -> None:
    data = talks()
    store = IndexStore(folder)
    build_index(ListSource(data), FakeEmbedder(), store, count_tokens=words)
    del data[LAKE]
    data[HOME] = [*data[HOME], msg("h4", "2024-09-09 09:00:00", "the automobile is at the garage")]
    result = build_index(ListSource(data), FakeEmbedder(), store, count_tokens=words)
    assert (result.removed, result.embedded, result.reused) == (2, 1, 4)
    index = MeaningIndex.load(store)
    assert index is not None
    assert firsts(index.search(query("ferry")))[0] == "w2"
    assert firsts(index.search(query("supper")))[0] == "h1"
    assert firsts(index.search(query("car")))[0] == "h4"
    assert firsts(index.search(query("dog")))[0] == "h3"
    assert firsts(index.search(query("report friday")))[0] == "w1"
    assert LAKE not in {hit.segment.conversation_id for hit in index.search(query("canoe"))}


def test_a_loaded_index_answers_from_its_own_build_after_a_new_build(folder: Path, index: MeaningIndex) -> None:
    before = [(hit.segment.segment_id, hit.score) for hit in index.search(query("ferry"))]
    build_index(ListSource({WORK: talks()[WORK]}), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    assert [(hit.segment.segment_id, hit.score) for hit in index.search(query("ferry"))] == before
    again = MeaningIndex.load(IndexStore(folder))
    assert again is not None and len(again) == 2


# --- an index made from parts ------------------------------------------------


def meta_for(count: int, dimensions: int) -> IndexMeta:
    stamp = datetime(2026, 11, 17, tzinfo=UTC).isoformat(timespec="seconds")
    return IndexMeta(1, "invented", "invented-model", dimensions, stamp, count, None)


def test_equal_scores_come_in_the_order_of_the_rows() -> None:
    rows = np.tile(unit(1, 8), (30, 1))
    index = MeaningIndex(meta_for(30, 8), [kept(n) for n in range(30)], rows)
    hits = index.search(rows[0].tolist(), limit=5)
    assert [hit.segment.first_message_id for hit in hits] == [f"first-{n}" for n in range(5)]


@pytest.mark.parametrize("limit", [1, 7, 20, 6000])
def test_the_hits_are_the_ones_of_a_plain_dot_product_over_every_row(limit: int) -> None:
    count, dimensions = 9000, 24  # more rows than one chunk of the search
    rows = unit(count, dimensions, seed=11)
    segments = [kept(n, conversation=f"chat-{n % 3}", day=f"2024-{1 + n % 12:02d}-{1 + n % 28:02d}") for n in range(count)]
    index = MeaningIndex(meta_for(count, dimensions), segments, rows)
    probe = unit(1, dimensions, seed=5)[0].astype(np.float64)
    probe /= np.linalg.norm(probe)
    scores = rows.astype(np.float64) @ probe

    def check(found: list[SearchHit], mask: np.ndarray) -> None:
        """The hits are the best rows of the mask. Two scores that differ by less than the rounding can change places."""
        places = np.flatnonzero(mask)
        best = np.sort(scores[places])[::-1][:limit]
        rows_found = [int(hit.segment.segment_id, 16) for hit in found]
        assert len(set(rows_found)) == len(rows_found) == len(best)
        assert all(mask[row] for row in rows_found)
        assert scores[rows_found] == pytest.approx(best, abs=1e-5)
        assert [hit.score for hit in found] == pytest.approx(best, abs=1e-4)

    check(index.search(probe.tolist(), limit=limit), np.ones(count, dtype=bool))
    in_chat = np.array([n % 3 == 1 for n in range(count)])
    check(index.search(probe.tolist(), conversations=["chat-1"], limit=limit), in_chat)
    in_march = np.array([n % 12 == 2 for n in range(count)])
    check(index.search(probe.tolist(), since="2024-03-01", until="2024-03-31", limit=limit), in_march)
    both = in_chat & in_march
    check(index.search(probe.tolist(), conversations=["chat-1"], since="2024-03-01", until="2024-03-31", limit=limit), both)


def test_parts_that_do_not_fit_are_refused() -> None:
    with pytest.raises(ValueError):
        MeaningIndex(meta_for(3, 8), [kept(n) for n in range(3)], unit(4, 8))
    with pytest.raises(ValueError):
        MeaningIndex(meta_for(3, 8), [kept(n) for n in range(3)], unit(3, 6))
