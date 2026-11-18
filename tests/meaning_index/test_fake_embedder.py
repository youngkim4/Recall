"""The fixed embedder of the tests: no network, and vectors that rank by shared words."""

from __future__ import annotations

import numpy as np
import pytest

from meaning_index.fake_embedder import FakeEmbedder


def dot(left: object, right: object) -> float:
    return float(np.dot(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)))


def test_it_gives_one_vector_of_unit_length_for_each_text_in_order() -> None:
    embedder = FakeEmbedder(dimensions=32)
    texts = ["the ferry leaves at nine", "dinner on friday", "the ferry leaves at nine"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert all(len(vector) == 32 for vector in vectors)
    assert all(abs(dot(vector, vector) - 1.0) < 1e-9 for vector in vectors)
    assert list(vectors[0]) == list(vectors[2])
    assert list(vectors[0]) != list(vectors[1])


def test_it_names_its_model_and_its_dimensions() -> None:
    embedder = FakeEmbedder()
    assert embedder.model == "fake-embedding-1"
    assert embedder.dimensions == 64
    assert FakeEmbedder(dimensions=8, model="other").model == "other"


def test_the_same_text_gives_the_same_vector_in_every_instance() -> None:
    first = FakeEmbedder().embed(["a walk by the lake"])[0]
    second = FakeEmbedder().embed(["a walk by the lake"])[0]
    assert list(first) == list(second)


def test_words_that_mean_the_same_give_the_same_vector() -> None:
    embedder = FakeEmbedder()
    boat, ferry, dinner = embedder.embed(["boat tickets", "ferry ticket", "dinner plans"])
    assert dot(boat, ferry) == pytest.approx(1.0)
    assert dot(boat, dinner) < 0.5


def test_a_query_is_nearest_to_the_text_that_shares_its_meaning() -> None:
    embedder = FakeEmbedder()
    texts = [
        "2024-06-03\nThem: did you buy the boat tickets\nMe: yes for saturday",
        "2024-06-04\nThem: what should we cook for supper\nMe: pasta",
        "2024-06-05\nMe: the dog needs a walk\nThem: on it",
    ]
    vectors = embedder.embed(texts)
    (ferry,) = embedder.embed(["ferry ticket"])
    (meal,) = embedder.embed(["dinner"])
    assert max(range(3), key=lambda n: dot(ferry, vectors[n])) == 0
    assert max(range(3), key=lambda n: dot(meal, vectors[n])) == 1


def test_the_labels_of_a_segment_and_its_day_do_not_count_as_words() -> None:
    embedder = FakeEmbedder()
    plain, labelled = embedder.embed(["canoe", "2024-06-03\nMe: canoe\nThem: canoe"])
    assert dot(plain, labelled) == pytest.approx(1.0)


def test_more_synonyms_can_be_given() -> None:
    embedder = FakeEmbedder(synonyms={"kayak": "canoe"})
    kayak, canoe = embedder.embed(["kayak", "canoe"])
    assert dot(kayak, canoe) == pytest.approx(1.0)


def test_a_text_with_no_word_still_has_a_vector_of_unit_length() -> None:
    (vector,) = FakeEmbedder(dimensions=8).embed(["?!"])
    assert dot(vector, vector) == pytest.approx(1.0)


def test_each_call_is_recorded() -> None:
    embedder = FakeEmbedder()
    embedder.embed(["one", "two"])
    embedder.embed([])
    assert embedder.calls == [("one", "two"), ()]
    assert embedder.texts == ["one", "two"]


def test_dimensions_below_one_are_refused() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dimensions=0)
