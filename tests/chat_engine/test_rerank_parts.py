"""The small parts of a ranking: the pool, the order that is kept, the text of a candidate."""

from __future__ import annotations

import pytest

from chat_engine.fake_archive import make_message
from chat_engine.rerank import RERANK_CHARS, RERANK_MOST, candidate_text, kept_order, pool_size
from chat_engine.tool_limits import SEARCH_LIMIT_MAX
from tests.chat_engine.memory_bench import MemoryBench
from tests.chat_engine.tiny import MAYA_PHONE, UNSAVED_PHONE


@pytest.mark.parametrize(("limit", "pool"), [(1, 3), (2, 6), (8, 24), (10, 30), (SEARCH_LIMIT_MAX, RERANK_MOST)])
def test_the_pool_is_three_times_the_limit_up_to_a_cap(limit: int, pool: int) -> None:
    assert pool_size(limit) == pool and pool_size(limit) >= limit


@pytest.mark.parametrize(
    ("given", "keep", "expected"),
    [
        (["3", "1", "2"], 3, (2, 0, 1)),
        (["3"], 3, (2, 0, 1)),
        (["3", "3", "x", "0", "-1", "", "4"], 2, (2, 0)),
        ([], 2, (0, 1)),
        (["2", "1", "3"], 1, (1,)),
        (["1"], 0, ()),
    ],
)
def test_the_kept_order(given: list[str], keep: int, expected: tuple[int, ...]) -> None:
    assert kept_order(3, given, keep) == expected


def test_kept_order_reads_only_strings() -> None:
    assert kept_order(3, [2, None, "3"], 3) == (2, 0, 1)  # type: ignore[list-item]


def test_a_candidate_text_holds_the_hit_its_neighbours_and_the_conversation() -> None:
    bench = MemoryBench()
    ctx = bench.box.context
    rows = [
        make_message("x1", MAYA_PHONE, "2021-03-04 18:20:00", "before " * 10, is_from_me=True),
        make_message("x2", MAYA_PHONE, "2021-03-04 18:22:00", "the hit " + "word " * 200, sender=MAYA_PHONE),
        make_message("x3", MAYA_PHONE, "2021-03-04 18:23:00", "after " * 100, is_from_me=True),
    ]
    text = candidate_text(ctx, rows, frozenset({"x2"}), {})
    assert len(text) <= RERANK_CHARS
    assert "Maya Okafor" in text and "2021-03-04" in text and "the hit" in text
    assert "before" in text and "after" in text
    assert not ctx.registry.checkpoint()  # nothing is registered: no reference is given


def test_a_candidate_text_names_an_unsaved_person_by_a_mask() -> None:
    bench = MemoryBench()
    ctx = bench.box.context
    ctx.listed()
    rows = [make_message("y1", UNSAVED_PHONE, "2025-01-16 08:00:00", "call +17205550199", sender=UNSAVED_PHONE)]
    text = ctx.labeler.redact(candidate_text(ctx, rows, frozenset({"y1"}), {}))
    assert "7205550199" not in text and "call" in text
