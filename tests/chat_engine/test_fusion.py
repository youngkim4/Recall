"""Reciprocal rank fusion: two rankings of exchanges become one."""

import pytest

from chat_engine.fusion import RRF_K, fuse


def test_an_item_that_both_rankings_found_comes_first() -> None:
    fused = fuse([["a", "b", "c"], ["c", "d"]])
    assert [item.key for item in fused] == ["c", "a", "b", "d"]  # b and d tie: the first ranking wins
    assert fused[0].ranks == (3, 1)
    assert fused[0].score == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))


def test_an_item_that_only_one_ranking_found_still_counts() -> None:
    fused = fuse([["a"], ["b"]])
    assert [(item.key, item.ranks) for item in fused] == [("a", (1, None)), ("b", (None, 1))]
    assert fused[0].score == fused[1].score == pytest.approx(1 / (RRF_K + 1))


def test_a_tie_goes_to_the_better_single_rank_then_to_the_first_ranking() -> None:
    fused = fuse([["a", "b"], ["c", "a", "b"]])
    assert [item.key for item in fused] == ["a", "b", "c"]
    assert [item.key for item in fuse([["x", "y"], ["y", "x"]])] == ["x", "y"]


def test_an_item_counts_once_in_one_ranking_at_its_best_place() -> None:
    fused = fuse([["a", "b", "a", "a"], []])
    assert [(item.key, item.ranks) for item in fused] == [("a", (1, None)), ("b", (2, None))]


def test_no_ranking_and_empty_rankings_give_nothing() -> None:
    assert fuse([]) == []
    assert fuse([[], []]) == []


def test_the_inputs_are_not_changed_and_keys_can_be_tuples() -> None:
    first, second = [("c1", "m1", "m4")], [("w", "m9"), ("c1", "m1", "m4")]
    fused = fuse([first, second], k=1)
    assert first == [("c1", "m1", "m4")] and len(second) == 2
    assert fused[0].key == ("c1", "m1", "m4") and fused[0].score == pytest.approx(1 / 2 + 1 / 3)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        fuse([["a"]], k=0)
