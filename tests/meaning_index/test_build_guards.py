"""What a build refuses or leaves out before it pays: a repeated segment, a cancel during a wait."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from meaning_index.batching import BatchLimits, embed_in_batches
from meaning_index.build import build_index, estimate_build
from meaning_index.errors import BuildCancelled, EmbeddingRateLimited
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import EmbedderFacts
from meaning_index.store import IndexStore
from tests.meaning_index.support import ListSource, ScriptedEmbedder, msg, notes, stamp, words

FACTS = EmbedderFacts("fake-embedding-1", 64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="")
MARK = "MARKED-TEXT-4410"


def repeated() -> ListSource:
    """One conversation whose source gives one message id two times, five hours apart."""
    return ListSource({"chat-a": [msg("m1", stamp(0), f"hello {MARK}"), msg("m1", stamp(300), f"again {MARK}")]})


def test_a_repeated_segment_is_left_out_before_anything_is_embedded(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    embedder = FakeEmbedder()
    with caplog.at_level(logging.WARNING, logger="meaning_index"):
        result = build_index(repeated(), embedder, IndexStore(folder), count_tokens=words)
    assert (result.segments, result.embedded) == (1, 1)
    assert embedder.texts == [f"2024-06-03\nThem: hello {MARK}"]
    said = [record.getMessage() for record in caplog.records]
    assert any("1 segment" in line and "left out" in line for line in said)
    assert MARK not in "\n".join(said) and "m1" not in "\n".join(said) and "chat-a" not in "\n".join(said)


def test_a_second_build_with_a_repeated_segment_pays_nothing(folder: Path) -> None:
    build_index(repeated(), FakeEmbedder(), IndexStore(folder), count_tokens=words)
    again = FakeEmbedder()
    result = build_index(repeated(), again, IndexStore(folder), count_tokens=words)
    assert again.calls == []
    assert (result.segments, result.reused, result.removed) == (1, 1, 0)


def test_an_estimate_leaves_the_repeated_segment_out_too(folder: Path) -> None:
    estimate = estimate_build(repeated(), IndexStore(folder), FACTS, words)
    assert (estimate.segments, estimate.new_segments, estimate.tokens) == (1, 1, 4)


def test_a_wait_for_a_rate_limit_is_slept_in_parts_of_one_second_at_most() -> None:
    slept: list[float] = []
    embedder = ScriptedEmbedder({1: EmbeddingRateLimited("wait", retry_after_s=7.5)})
    parts = list(embed_in_batches(embedder, ["one", "two"], [1, 1], sleep=slept.append))
    assert slept == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    assert len(parts) == 1


def test_the_caller_is_asked_between_the_parts_of_a_wait() -> None:
    slept: list[float] = []
    asked: list[int] = []

    def check() -> None:
        asked.append(len(slept))
        if len(slept) >= 3:
            raise BuildCancelled("the build was cancelled")

    embedder = ScriptedEmbedder({1: EmbeddingRateLimited("wait", retry_after_s=30.0)})
    with pytest.raises(BuildCancelled):
        list(embed_in_batches(embedder, ["one"], [1], sleep=slept.append, check=check))
    assert slept == [1.0, 1.0, 1.0]
    assert asked == [0, 1, 2, 3]
    assert embedder.sizes == [1]


def test_a_cancel_during_a_wait_stops_the_build_within_one_second(folder: Path) -> None:
    slept: list[float] = []

    def limited(number: int, texts: Sequence[str]) -> Exception | None:
        return EmbeddingRateLimited("wait", retry_after_s=120.0) if number == 2 else None

    embedder = ScriptedEmbedder(limited)
    with pytest.raises(BuildCancelled) as caught:
        build_index(
            ListSource({"chat-a": notes(150)}),
            embedder,
            IndexStore(folder),
            count_tokens=words,
            sleep=slept.append,
            cancel=lambda: len(slept) >= 2,
        )
    assert slept == [1.0, 1.0]
    assert embedder.sizes == [96, 54]
    assert (caught.value.texts_embedded, caught.value.tokens) == (96, 96 * 4)
    assert IndexStore(folder).work_file(16).hashes()


def test_the_waits_of_a_request_still_add_up_to_what_the_service_named(folder: Path) -> None:
    slept: list[float] = []
    busy = EmbeddingRateLimited("wait", retry_after_s=2.25)
    embedder = ScriptedEmbedder({1: busy, 2: busy})
    list(embed_in_batches(embedder, ["one"], [1], sleep=slept.append, limits=BatchLimits(max_tries=3)))
    assert slept == [1.0, 1.0, 0.25, 1.0, 1.0, 0.25]
