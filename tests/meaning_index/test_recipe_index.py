"""The recipe in the index: the meta file names it, and a change of recipe is a change to build."""

from __future__ import annotations

import json
from pathlib import Path

from meaning_index.build import build_index, estimate_build
from meaning_index.fake_embedder import FakeEmbedder
from meaning_index.model_facts import EmbedderFacts
from meaning_index.recipe import DEFAULT_NAME, Recipe
from meaning_index.store import IndexStore
from tests.meaning_index.support import ListSource, run, words

FACTS = EmbedderFacts("fake-embedding-1", 64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="")
OVERLAP = Recipe(overlap=True)
ADAPTIVE = Recipe(adaptive=True)


def data() -> dict:
    return {"chat-a": run(40, prefix="a"), "chat-b": run(3, prefix="b")}


def meta_of(folder: Path) -> dict:
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))


def test_a_build_with_the_default_recipe_writes_the_meta_file_as_before(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(data()), FakeEmbedder(), store, count_tokens=words)
    assert "recipe" not in meta_of(folder)
    assert store.meta().recipe == DEFAULT_NAME


def test_a_build_with_another_recipe_names_it_in_the_meta_file_and_the_status(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(data()), FakeEmbedder(), store, count_tokens=words, recipe=OVERLAP)
    assert meta_of(folder)["recipe"] == "v1+overlap"
    assert store.meta().recipe == "v1+overlap"


def test_a_meta_file_with_a_recipe_that_is_not_text_is_no_index(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(data()), FakeEmbedder(), store, count_tokens=words, recipe=OVERLAP)
    (folder / "meta.json").write_text(json.dumps({**meta_of(folder), "recipe": 3}), encoding="utf-8")
    assert store.status().state == "none"


def test_the_estimate_says_when_the_recipe_changed_and_what_the_new_recipe_embeds(folder: Path) -> None:
    store = IndexStore(folder)
    build_index(ListSource(data()), FakeEmbedder(), store, count_tokens=words)
    same = estimate_build(ListSource(data()), store, FACTS, words)
    assert (same.new_segments, same.recipe_changed) == (0, False)
    changed = estimate_build(ListSource(data()), store, FACTS, words, recipe=OVERLAP)
    # the segments of chat-a after the first carry two more messages: their text changed
    assert (changed.segments, changed.new_segments, changed.recipe_changed) == (4, 2, True)
    result = build_index(ListSource(data()), FakeEmbedder(), store, count_tokens=words, recipe=OVERLAP)
    assert (result.embedded, result.reused) == (2, 2)
    again = estimate_build(ListSource(data()), store, FACTS, words, recipe=OVERLAP)
    assert (again.new_segments, again.recipe_changed) == (0, False)


def test_a_new_recipe_with_the_same_texts_still_says_that_it_changed(folder: Path) -> None:
    store = IndexStore(folder)
    # one short talk: the rhythm is not known, so the adaptive cut is the default cut
    build_index(ListSource({"chat-b": run(3, prefix="b")}), FakeEmbedder(), store, count_tokens=words)
    found = estimate_build(ListSource({"chat-b": run(3, prefix="b")}), store, FACTS, words, recipe=ADAPTIVE)
    assert (found.new_segments, found.tokens, found.recipe_changed) == (0, 0, True)


def test_with_no_index_the_recipe_has_not_changed(folder: Path) -> None:
    found = estimate_build(ListSource(data()), IndexStore(folder), FACTS, words, recipe=OVERLAP)
    assert found.recipe_changed is False and found.new_segments == found.segments


def test_the_context_of_the_caller_is_embedded(folder: Path) -> None:
    store = IndexStore(folder)
    seen: list[str] = []

    class Keeping(FakeEmbedder):
        def embed(self, texts):  # type: ignore[no-untyped-def]
            seen.extend(texts)
            return super().embed(texts)

    recipe = Recipe(context=True)
    build_index(
        ListSource(data()), Keeping(), store, count_tokens=words, recipe=recipe, context=lambda chat, _first: f"about {chat}"
    )
    assert len(seen) == 4
    assert all(text.startswith("Context: June 2024; about chat-") for text in seen)
    found = estimate_build(
        ListSource(data()), store, FACTS, words, recipe=recipe, context=lambda chat, _first: f"about {chat}"
    )
    assert (found.new_segments, found.recipe_changed) == (0, False)
