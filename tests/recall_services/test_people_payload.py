"""people.status, estimate and payload: what is built and what a build would cost, with no call."""

from __future__ import annotations

from typing import Any

import pytest

from recall_services import memory_prep, models, people, people_prompt
from recall_services.people_inputs import Planner
from recall_services.people_store import PeopleStore
from report_engine import LLMUsage
from report_engine.pricing import call_cost_usd
from report_engine.tokens import meter_for
from tests.recall_services.people_kit import opened, FAKE_FACTS, THEO, settings, town  # noqa: F401
from tests.report_engine.synth import WordCounter

PAYLOAD_KEYS = {
    "state", "model", "people", "profiles", "current", "toWrite", "toEmbed", "notRead",
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
    """With no notes file nothing is planned: a profile with no text is counted too, as not read."""
    found = payload(town)
    assert set(found) == PAYLOAD_KEYS
    assert (found["state"], found["people"], found["profiles"], found["toWrite"], found["notRead"]) == ("none", 4, 0, 0, 4)
    assert (found["estimatedCost"], found["error"], found["code"]) == (0.0, None, None)
    assert not settings(town)["store"].folder.exists()


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
    with opened(town.places.cache_path) as conn:
        before = sorted(conn.execute("SELECT key, last_used_at FROM period_results"))
    payload(town)
    with opened(town.places.cache_path) as conn:
        assert sorted(conn.execute("SELECT key, last_used_at FROM period_results")) == before


def test_a_store_of_a_newer_format_gives_zeros_and_a_sentence(town) -> None:
    town.read_memory()
    build(town)
    store: PeopleStore = settings(town)["store"]
    with opened(store.path) as conn:
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


def test_the_server_builds_by_itself_only_what_is_cheap_with_every_try() -> None:
    """A profile can take TRIES calls: the server starts a build only when TRIES times the estimate is cheap."""
    assert (people.PEOPLE_AUTO_USD, people.TRIES) == (0.10, 2)
    assert people.worth_starting({"toWrite": 2, "toEmbed": 0, "estimatedCost": 0.05, "error": None}) is True
    assert people.worth_starting({"toWrite": 2, "toEmbed": 0, "estimatedCost": 0.051, "error": None}) is False
    assert people.worth_starting({"toWrite": 0, "toEmbed": 3, "estimatedCost": 0.0001, "error": None}) is True
    assert people.worth_starting({"toWrite": 0, "toEmbed": 0, "estimatedCost": 0.0, "error": None}) is False
    assert people.worth_starting({"toWrite": 2, "toEmbed": 0, "estimatedCost": 0.01, "error": "no"}) is False
    assert people.worth_starting({"toWrite": 2, "toEmbed": 0, "estimatedCost": True, "error": None}) is False


# --- the review of the join of Part G ------------------------------------------------


class _Down:
    model, dimensions = FAKE_FACTS.model, FAKE_FACTS.dimensions

    def embed(self, texts: Any) -> Any:
        raise RuntimeError("down")


def test_rows_with_no_vector_are_to_embed_and_a_build_repairs_them(town) -> None:
    town.read_memory()
    with pytest.raises(people.PeopleFailure):
        people.build(town.archive, town.names, **settings(town, embedder=_Down()))
    found = payload(town)
    assert (found["state"], found["toWrite"], found["toEmbed"], found["current"]) == ("partial", 0, 3, 3)
    assert 0 < found["estimatedCost"] < 0.001 and people.worth_starting(found)
    build(town)
    assert (payload(town)["state"], payload(town)["toEmbed"]) == ("ready", 0)


def _long_memory(town: Any) -> None:
    """Notes of the size of real ones: a summary of about 250 words and a few moments for each part."""
    from report_engine.fake_llm import FakeCall, FakeLLM, default_map_reply

    def map_reply(call: FakeCall) -> Any:
        reply = default_map_reply(call)
        words = " ".join(f"the talk about the lantern festival and plans number {n}" for n in range(28))
        moments = [{**reply["moments"][0], "detail": "You and the other side made a plan. " * 6}] * 3
        return {**reply, "summary": words, "moments": moments}

    town.read_memory(llm=FakeLLM(map_reply))


def _realistic(retry: bool) -> Any:
    """A model that reports usage as a real one would: about 4 characters for one token, in and out."""
    import json
    import math

    from report_engine.fake_llm import FakeCall, FakeLLM, completed
    from report_engine.ports import LLMUsage as Spent
    from tests.recall_services.people_kit import profile_reply

    seen: dict[str, int] = {}

    def reply(call: FakeCall) -> Any:
        key = call.messages[-1]["content"]
        seen[key] = seen.get(key, 0) + 1
        # a reply of a real size: a summary near its limit and three turning points
        points = [{"date": "2023-01-05", "what": "You planned the festival together and set a day. " * 2}] * 3
        body = profile_reply(summary="You and this person " + "talked about plans and the festival " * 15, turning_points=points)
        body = {"wrong": "shape " * 200} if retry and seen[key] == 1 else body
        text = json.dumps(body)
        usage = Spent(input_tokens=math.ceil(len(call.text) / 4), output_tokens=math.ceil(len(text) / 4))
        return completed(call, body, text=text, usage=usage)

    return FakeLLM(reply)


def test_with_realistic_usage_one_try_spends_no_more_than_the_estimate(town) -> None:
    _long_memory(town)
    cost = payload(town)["estimatedCost"]
    found = people.build(town.archive, town.names, **settings(town, _realistic(retry=False)))
    assert found["usage"]["input_tokens"] > 3 * 1_000 and found["usage"]["calls"] == 3
    assert found["usd"] <= cost <= 1.6 * found["usd"]


def test_with_a_retry_for_each_profile_the_spend_stays_under_tries_times_the_estimate(town) -> None:
    _long_memory(town)
    cost = payload(town)["estimatedCost"]
    found = people.build(town.archive, town.names, **settings(town, _realistic(retry=True)))
    assert found["usage"]["calls"] == 6 and found["built"] == 3
    assert cost < found["usd"] <= people.TRIES * cost


def _counted_plans(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from recall_services import people_look

    planned: list[str] = []
    real = people_look.span_plan

    def counted(source: Any, conversation: str, *args: Any, **more: Any) -> Any:
        planned.append(conversation)
        return real(source, conversation, *args, **more)

    monkeypatch.setattr(people_look, "span_plan", counted)
    return planned


def test_the_payload_plans_a_conversation_once_and_again_only_when_it_changed(town, monkeypatch: pytest.MonkeyPatch) -> None:
    town.read_memory()
    planned = _counted_plans(monkeypatch)
    first = payload(town)
    assert len(planned) == 5
    assert payload(town) == first and len(planned) == 5
    town.add([(THEO, "2024-03-04 09:00:00", "Ok then", False, "")])
    payload(town)
    assert planned[5:] == [THEO]


def test_the_payload_reads_no_message_for_the_figures(town, monkeypatch: pytest.MonkeyPatch) -> None:
    from recall_services import people_inputs

    town.read_memory()
    monkeypatch.setattr(people_inputs, "read_figures", lambda *_args: pytest.fail("figures in a payload"))
    assert payload(town)["toWrite"] == 3


def test_with_no_notes_nothing_is_planned(town, monkeypatch: pytest.MonkeyPatch) -> None:
    planned = _counted_plans(monkeypatch)
    found = payload(town)
    assert planned == [] and (found["people"], found["notRead"]) == (4, 4)


def test_the_plans_are_kept_for_their_owner_only(town) -> None:
    import stat

    town.read_memory()
    payload(town)
    plans = settings(town)["store"].folder / "plans.sqlite3"
    assert stat.S_IMODE(plans.stat().st_mode) == 0o600
    assert stat.S_IMODE(plans.parent.stat().st_mode) == 0o700


def test_a_build_after_a_payload_plans_only_what_it_writes(town, monkeypatch: pytest.MonkeyPatch) -> None:
    town.read_memory()
    build(town)
    planned = _counted_plans(monkeypatch)
    town.add([(THEO, "2024-03-04 09:00:00", "Ok then", False, "")])
    town.read_memory([THEO])
    found = build(town)
    assert found["built"] == 1 and planned.count(THEO) >= 1 and set(planned) == {THEO}
