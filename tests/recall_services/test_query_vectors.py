"""query_vectors.py: one cache of the vectors of descriptions for the process, for the people and the meaning search."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from meaning_index import FakeEmbedder
from recall_services import query_vectors
from recall_services.meaning_port import MeaningPort




@pytest.fixture(autouse=True)
def _empty() -> Iterator[None]:
    query_vectors.forget()
    yield
    query_vectors.forget()


def factory_of(embedder: FakeEmbedder) -> Any:
    asked: list[tuple[str, int]] = []

    def factory(model: str, dimensions: int) -> FakeEmbedder:
        asked.append((model, dimensions))
        return embedder

    factory.asked = asked  # type: ignore[attr-defined]
    return factory


def test_a_text_is_embedded_once_for_one_model_and_size() -> None:
    embedder = FakeEmbedder(dimensions=8, model="m")
    factory = factory_of(embedder)
    first = query_vectors.query_vector("a ferry ride", "m", 8, factory)
    again = query_vectors.query_vector("a ferry ride", "m", 8, factory)
    assert first == again and len(first) == 8 and embedder.texts == ["a ferry ride"]
    assert factory.asked == [("m", 8)]


def test_another_model_another_size_or_another_text_is_embedded_again() -> None:
    embedder = FakeEmbedder(dimensions=8, model="m")
    query_vectors.query_vector("a", "m", 8, factory_of(embedder))
    query_vectors.query_vector("a", "n", 8, factory_of(embedder))
    query_vectors.query_vector("b", "m", 8, factory_of(embedder))
    query_vectors.query_vector("a", "m", 4, factory_of(FakeEmbedder(dimensions=4, model="m")))
    assert embedder.texts == ["a", "a", "b"]


def test_the_cache_keeps_the_last_vectors_only() -> None:
    embedder = FakeEmbedder(dimensions=4, model="m")
    for number in range(query_vectors.QUERY_VECTORS + 1):
        query_vectors.query_vector(f"text {number}", "m", 4, factory_of(embedder))
    query_vectors.query_vector("text 0", "m", 4, factory_of(embedder))
    assert embedder.texts.count("text 0") == 2
    assert query_vectors.QUERY_VECTORS == 64


class _Wrong:
    def __init__(self, reply: Sequence[Sequence[float]]) -> None:
        self.reply = reply
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        return self.reply


@pytest.mark.parametrize("reply", [[], [[1.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]] * 2, [[float("nan"), 0.0, 0.0, 0.0]]])
def test_a_reply_that_a_search_can_not_use_raises_and_is_not_kept(reply: Any) -> None:
    wrong = _Wrong(reply)
    for _ in range(2):
        with pytest.raises(ValueError):
            query_vectors.query_vector("x", "m", 4, lambda _model, _dimensions: wrong)
    assert wrong.calls == 2


def test_a_failure_of_the_embedder_is_raised() -> None:
    def broken(_model: str, _dimensions: int) -> Any:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        query_vectors.query_vector("x", "m", 4, broken)


def test_the_people_search_and_the_meaning_port_embed_a_question_once(tmp_path: Any) -> None:
    """The seeded people search and the meaning search of one question share the vector of its text."""
    from tests.recall_services import memory_kit
    from tests.recall_services.people_kit import FAKE_FACTS, build_town, settings

    from recall_services import people

    lake = memory_kit.build_lake(tmp_path / "lake")
    town = build_town(tmp_path / "town")
    try:
        embedder = FakeEmbedder(dimensions=FAKE_FACTS.dimensions, model=FAKE_FACTS.model)
        memory_kit.build(lake)
        town.read_memory()
        people.build(town.archive, town.names, **settings(town))
        factory = factory_of(embedder)
        people.search(town.archive, town.names, "a boat ride", store=settings(town)["store"], embedder_factory=factory)
        MeaningPort(lake.store, factory).search("a boat ride", conversations=None, since=None, until=None, limit=3)
        assert embedder.texts == ["a boat ride"]
    finally:
        lake.close()
        town.close()
