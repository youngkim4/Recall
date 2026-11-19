"""people.search: the profiles ranked by meaning, filtered by kind and by days. It never fails a question."""

from __future__ import annotations

from typing import Any

import pytest

from meaning_index import EmbeddingNetworkError, FakeEmbedder
from recall_services import meaning, people
from recall_services.meaning_failures import NoKey
from recall_services.people_store import PeopleStore
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.recall_services.people_kit import (
    FAKE_FACTS,
    GROUP,
    MINA,
    MINA_MAIL,
    THEO,
    profile_reply,
    settings,
    town,  # noqa: F401
)

ROW_KEYS = {"key", "conversations", "label", "kind", "status", "summary", "turning_points", "figures", "score"}
FIGURE_KEYS = {
    "first_day", "last_day", "messages", "sent", "received", "active_days",
    "longest_gap_days", "busiest_month", "wrote_first_share",
}  # fmt: skip


def responder(call: FakeCall) -> Any:
    user = call.messages[-1]["content"]
    if "This is a group conversation:" in user:
        return profile_reply(summary="Evening walks by the harbor with friends.")
    if "conversation: Mina." in user:
        return profile_reply(
            kind="talking_stage",
            status="faded",
            summary="Coffee dates and a valentine, then it faded.",
            turning_points=[{"date": "2023-02-14", "what": "You sent a valentine."}],
        )
    return profile_reply(kind="service", summary="Theo tried to sell you a ferry.")


@pytest.fixture
def built(town: Any) -> Any:
    town.read_memory()
    people.build(town.archive, town.names, **settings(town, FakeLLM(responder)))
    return town


def find(town: Any, description: str, embedder: Any = None, **more: Any) -> people.Found:
    chosen = settings(town, embedder=embedder)
    return people.search(
        town.archive, town.names, description,
        store=chosen["store"], embedder_factory=chosen["embedder_factory"], **more,
    )  # fmt: skip


def test_with_no_index_the_search_gives_nothing_and_says_so(town) -> None:
    embedder = FakeEmbedder(dimensions=FAKE_FACTS.dimensions, model=FAKE_FACTS.model)
    found = find(town, "a valentine", embedder)
    assert found == [] and isinstance(found, list)
    assert (found.reason, found.index_missing) == ("no_index", True)
    assert embedder.calls == []


def test_the_nearest_profile_comes_first_with_its_whole_row(built) -> None:
    found = find(built, "a boat for sale")
    assert found.reason is None and found.index_missing is False
    first = found[0]
    assert set(first) == ROW_KEYS and set(first["figures"]) == FIGURE_KEYS
    assert (first["conversations"], first["kind"], first["label"]) == ([THEO], "service", "an unsaved email contact")
    assert first["score"] > found[1]["score"]
    assert all(isinstance(row["score"], float) for row in found)


def test_a_row_gives_the_turning_points_and_both_threads(built) -> None:
    (mina,) = [row for row in find(built, "valentine") if MINA in row["conversations"]]
    assert mina["conversations"] == [MINA, MINA_MAIL]
    assert mina["turning_points"] == [{"date": "2023-02-14", "what": "You sent a valentine."}]
    assert (mina["figures"]["first_day"], mina["figures"]["messages"]) == ("2023-01-02", 8)


def test_kinds_filter_the_profiles(built) -> None:
    found = find(built, "anything", kinds=["talking_stage", "romantic"])
    assert [row["kind"] for row in found] == ["talking_stage"]


def test_days_keep_the_profiles_whose_span_overlaps_them(built) -> None:
    assert {row["conversations"][0] for row in find(built, "x", since="2023-06-10", until="2023-06-10")} == {GROUP}
    assert {row["conversations"][0] for row in find(built, "x", since="2024-01-01")} == {THEO}
    assert {row["conversations"][0] for row in find(built, "x", until="2023-01-02")} == {MINA}


def test_the_limit_counts_rows(built) -> None:
    assert len(find(built, "x", limit=2)) == 2


@pytest.mark.parametrize(
    "more",
    [
        {"kinds": ["situationship"]},
        {"kinds": "friend"},
        {"since": "June 2023"},
        {"since": "2023-02-30"},
        {"since": "2024-01-01", "until": "2023-01-01"},
        {"limit": 0},
        {"limit": 51},
        {"limit": True},
    ],
)
def test_arguments_that_can_not_be_used_raise(built, more: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        find(built, "x", **more)


def test_a_description_that_can_not_be_embedded_gives_nothing_and_says_why(built) -> None:
    class Offline:
        model, dimensions = FAKE_FACTS.model, FAKE_FACTS.dimensions

        def embed(self, texts: Any) -> Any:
            raise EmbeddingNetworkError("no network")

    found = find(built, "x", Offline())
    assert (found, found.reason, found.index_missing) == ([], "embed_failed", False)


def test_with_no_key_the_search_gives_nothing_and_says_why(built) -> None:
    def no_key(_model: str, _dimensions: int) -> Any:
        raise NoKey("no key is set")

    chosen = settings(built)
    found = people.search(built.archive, built.names, "x", store=chosen["store"], embedder_factory=no_key)
    assert (found, found.reason) == ([], "no_key")


def test_the_description_is_embedded_with_the_model_and_the_dimensions_of_the_index(built) -> None:
    asked: list[tuple[str, int]] = []
    embedder = FakeEmbedder(dimensions=FAKE_FACTS.dimensions, model=FAKE_FACTS.model)

    def factory(model: str, dimensions: int) -> Any:
        asked.append((model, dimensions))
        return embedder

    chosen = settings(built)
    people.search(built.archive, built.names, "a ferry", store=chosen["store"], embedder_factory=factory)
    assert asked == [(FAKE_FACTS.model, FAKE_FACTS.dimensions)] and embedder.texts == ["a ferry"]


def test_a_store_that_can_not_be_read_is_no_index(built) -> None:
    store: PeopleStore = settings(built)["store"]
    store.path.write_text("not a database", encoding="utf-8")
    assert find(built, "x").reason == "no_index"


def test_by_default_the_store_of_the_archive_and_the_embedder_of_the_app(built, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    target = people.people_folder(built.archive.path)
    shutil.copytree(settings(built)["store"].folder, target)
    monkeypatch.setattr(meaning, "embedder_for", lambda model, dimensions, **_more: FakeEmbedder(dimensions, model))
    found = people.search(built.archive, built.names, "a boat for sale")
    assert found.reason is None and found[0]["conversations"] == [THEO]
    shutil.rmtree(target)
