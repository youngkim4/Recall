"""people.status, estimate and payload: what is built and what a build would cost, with no call."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from recall_services import memory_prep, models, people, people_prompt
from recall_services.people_inputs import Planner
from recall_services.people_store import PeopleStore
from report_engine import LLMUsage
from report_engine.pricing import call_cost_usd
from report_engine.tokens import meter_for
from tests.recall_services.people_kit import FAKE_FACTS, THEO, settings, town  # noqa: F401
from tests.report_engine.synth import WordCounter

PAYLOAD_KEYS = {
    "state", "model", "people", "profiles", "current", "toWrite", "notRead",
    "estimatedCost", "builtAt", "error", "code",
}  # fmt: skip
STATUS_KEYS = {"state", "profiles", "builtAt", "model", "embedding"}


def look_settings(town: Any) -> dict[str, Any]:
    chosen = settings(town)
    return {name: chosen[name] for name in ("places", "options", "counter", "store", "facts")}


def payload(town: Any) -> dict[str, Any]:
    return people.payload(town.archive, town.names, **look_settings(town))


def build(town: Any) -> dict[str, Any]:
    return people.build(town.archive, town.names, **settings(town))


def test_before_any_build_the_status_is_none(town) -> None:
    found = people.status(town.archive, town.names, store=settings(town)["store"])
    assert found == {"state": "none", "profiles": 0, "builtAt": None, "model": None, "embedding": None}


def test_after_a_build_the_status_is_ready_and_names_the_models(town) -> None:
    town.read_memory()
    build(town)
    found = people.status(town.archive, town.names, store=settings(town)["store"])
    assert set(found) == STATUS_KEYS and (found["state"], found["profiles"]) == ("ready", 3)
    assert found["model"] == memory_prep.model()
    assert found["embedding"] == {"model": FAKE_FACTS.model, "dimensions": FAKE_FACTS.dimensions}


def test_a_status_of_a_store_that_can_not_be_read_is_none(town) -> None:
    store = settings(town)["store"]
    store.folder.mkdir(parents=True)
    store.path.write_text("not a database", encoding="utf-8")
    assert people.status(town.archive, town.names, store=store)["state"] == "none"


def test_with_nothing_read_the_payload_says_none_and_counts_who_waits(town) -> None:
    found = payload(town)
    assert set(found) == PAYLOAD_KEYS
    assert (found["state"], found["people"], found["profiles"], found["toWrite"], found["notRead"]) == ("none", 3, 0, 0, 3)
    assert (found["estimatedCost"], found["error"], found["code"]) == (0.0, None, None)
    assert not settings(town)["store"].folder.exists()


def test_the_cost_is_the_request_as_counted_and_the_fullest_reply_the_limits_allow(town) -> None:
    town.read_memory()
    found = payload(town)
    assert (found["state"], found["toWrite"], found["notRead"]) == ("none", 3, 0)
    planner = Planner(town.archive, town.names, model=memory_prep.model(), **town.notes())
    info = models.model_info(memory_prep.model())
    meter = meter_for(info, WordCounter())
    expected = 0.0
    for subject in planner.subjects():
        planned = planner.plan(subject)
        if planned.input_hash is None:
            continue
        size = meter.count("\n".join(part["content"] for part in planned.messages))
        expected += call_cost_usd(info, LLMUsage(input_tokens=size, output_tokens=people_prompt.OUTPUT_TOKENS))
        expected += people_prompt.OUTPUT_TOKENS * FAKE_FACTS.usd_per_mtok / 1_000_000
    assert found["estimatedCost"] == pytest.approx(expected) and expected > 0


def test_the_estimate_is_not_below_what_the_build_spends(town) -> None:
    """The fake replies are much shorter than the fullest reply, so only the lower side is pinned here."""
    town.read_memory()
    cost = payload(town)["estimatedCost"]
    assert build(town)["usd"] <= cost


def test_after_a_build_the_payload_is_ready_and_costs_nothing(town) -> None:
    town.read_memory()
    build(town)
    found = payload(town)
    assert (found["state"], found["profiles"], found["current"], found["toWrite"]) == ("ready", 3, 3, 0)
    assert found["estimatedCost"] == 0.0 and found["builtAt"]


def test_a_changed_person_makes_the_state_partial(town) -> None:
    town.read_memory()
    build(town)
    town.add([(THEO, "2024-03-04 09:00:00", "Ok then", False, "")])
    assert (payload(town)["state"], payload(town)["notRead"]) == ("partial", 1)
    town.read_memory([THEO])
    found = payload(town)
    assert (found["state"], found["toWrite"], found["current"]) == ("partial", 1, 2) and found["estimatedCost"] > 0


def test_the_payload_makes_no_call_and_changes_no_file(town) -> None:
    town.read_memory()
    with sqlite3.connect(town.places.cache_path) as conn:
        before = sorted(conn.execute("SELECT key, last_used_at FROM period_results"))
    payload(town)
    with sqlite3.connect(town.places.cache_path) as conn:
        assert sorted(conn.execute("SELECT key, last_used_at FROM period_results")) == before


def test_a_store_of_a_newer_format_gives_zeros_and_a_sentence(town) -> None:
    town.read_memory()
    build(town)
    store: PeopleStore = settings(town)["store"]
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE meta SET format_version = 99")
    found = payload(town)
    assert (found["state"], found["people"], found["code"]) == ("none", 0, "store_newer")
    assert found["error"] == "A newer version of Recall made the people index. Update Recall, then try again."


def test_a_model_that_can_not_be_run_gives_zeros_and_a_sentence(town, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_prep, "model", lambda: "no-such-model")
    found = payload(town)
    assert found["state"] == "none" and found["error"] == "The model of the people index can not be run. Nothing was written."
    assert found["code"]


def test_a_build_that_completed_is_known(town) -> None:
    store = settings(town)["store"]
    assert people.completed_before(town.archive.path, store=store) is False
    build(town)
    assert people.completed_before(town.archive.path, store=store) is True


def test_the_server_builds_by_itself_only_what_is_cheap() -> None:
    assert people.PEOPLE_AUTO_USD == 0.10
    assert people.worth_starting({"toWrite": 2, "estimatedCost": 0.05, "error": None}) is True
    assert people.worth_starting({"toWrite": 2, "estimatedCost": 0.11, "error": None}) is False
    assert people.worth_starting({"toWrite": 0, "estimatedCost": 0.0, "error": None}) is False
    assert people.worth_starting({"toWrite": 2, "estimatedCost": 0.01, "error": "no"}) is False
    assert people.worth_starting({"toWrite": 2, "estimatedCost": True, "error": None}) is False
