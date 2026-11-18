"""The estimate of a build: how many segments, how many of them are new, and what they cost."""

from __future__ import annotations

from pathlib import Path

import pytest

from meaning_index.build import build_index, estimate_build
from meaning_index.errors import EmbeddingKeyError
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import EmbedderFacts, embedder_facts
from meaning_index.shapes import BuildEstimate
from meaning_index.store import IndexStore
from tests.meaning_index.support import ListSource, ScriptedEmbedder, cut_texts, msg, notes, run, stamp, words

FACTS = EmbedderFacts(model="fake-embedding-1", dimensions=64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="")


def small() -> dict:
    return {"chat-a": run(3, prefix="a") + run(3, prefix="a-late", start=600), "chat-b": run(40, prefix="b")}


def test_before_a_build_every_segment_is_new(folder: Path) -> None:
    source = ListSource(small())
    tokens = sum(words(text) for text in cut_texts(source))
    estimate = estimate_build(source, IndexStore(folder), FACTS, words)
    assert estimate == BuildEstimate(segments=5, new_segments=5, tokens=tokens, usd=pytest.approx(tokens * 0.13 / 1_000_000))
    assert estimate.usd > 0


def test_an_estimate_makes_no_file(folder: Path) -> None:
    estimate_build(ListSource(small()), IndexStore(folder), FACTS, words)
    assert not folder.exists()


def test_after_a_build_nothing_is_new_and_nothing_is_paid(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(small()), FakeEmbedder(), store, count_tokens=words)
    assert estimate_build(ListSource(small()), store, FACTS, words) == BuildEstimate(5, 0, 0, 0.0)


def test_only_what_the_next_build_embeds_is_counted(folder: Path) -> None:
    store, data = IndexStore(folder), small()
    build_index(ListSource(data), FakeEmbedder(), store, count_tokens=words)
    data["chat-c"] = [msg("c0", stamp(0), "one two three")]
    estimate = estimate_build(ListSource(data), store, FACTS, words)
    assert (estimate.segments, estimate.new_segments, estimate.tokens) == (6, 1, 5)
    result = build_index(ListSource(data), FakeEmbedder(), store, count_tokens=words)
    assert (result.embedded, result.tokens) == (estimate.new_segments, estimate.tokens)


def test_another_model_makes_every_segment_new(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(small()), FakeEmbedder(), store, count_tokens=words)
    other = EmbedderFacts(model="another", dimensions=64, usd_per_mtok=0.02, max_input_tokens=8192, tokenizer_family="")
    assert estimate_build(ListSource(small()), store, other, words).new_segments == 5
    wider = EmbedderFacts(model="fake-embedding-1", dimensions=128, usd_per_mtok=0.02, max_input_tokens=8192, tokenizer_family="")
    assert estimate_build(ListSource(small()), store, wider, words).new_segments == 5


def test_what_a_stopped_build_paid_for_is_not_counted_again(folder: Path) -> None:
    store, source = IndexStore(folder), ListSource({"chat-a": notes(250)})
    embedder = ScriptedEmbedder({3: EmbeddingKeyError("no key")}, dimensions=64, model="fake-embedding-1")
    with pytest.raises(EmbeddingKeyError):
        build_index(source, embedder, store, count_tokens=words)
    estimate = estimate_build(source, store, FACTS, words)
    assert (estimate.segments, estimate.new_segments, estimate.tokens) == (250, 58, 58 * 4)


def test_one_text_of_two_segments_is_paid_one_time(folder: Path) -> None:
    same = [msg("x", stamp(0), "ok")]
    estimate = estimate_build(ListSource({"chat-a": same, "chat-b": same}), IndexStore(folder), FACTS, words)
    assert (estimate.segments, estimate.new_segments, estimate.tokens) == (2, 2, 3)


def test_the_price_is_the_one_of_the_facts_file(folder: Path) -> None:
    source = ListSource({"chat-a": notes(10)})
    large = estimate_build(source, IndexStore(folder), embedder_facts(), lambda text: 100_000)
    small_model = estimate_build(source, IndexStore(folder), embedder_facts("text-embedding-3-small"), lambda text: 100_000)
    assert large.usd == pytest.approx(0.13)
    assert small_model.usd == pytest.approx(0.02)
