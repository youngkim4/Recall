"""people.build: the profiles, made with the fake model and the fake embedder, and kept when nothing changed."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import numpy as np
import pytest

from meaning_index import EmbeddingKeyError, FakeEmbedder
from recall_services import people
from recall_services.people_store import Held, PeopleStore
from recall_services.report_failures import ReportFailure
from report_engine import CancelToken, LLMError
from report_engine.fake_llm import FakeCall, FakeLLM
from tests.recall_services.people_kit import (
    opened,
    FAKE_FACTS,
    GROUP,
    MARK,
    MINA,
    MINA_FIGURES,
    MINA_MAIL,
    THEO,
    by_key,
    profile_reply,
    settings,
    town,  # noqa: F401
)

RESULT_KEYS = {"people", "built", "kept", "failed", "notRead", "removed", "embedded", "usd", "usage", "embedTokens"}


def who(call: FakeCall) -> str:
    """The label of the profile that a call is about."""
    user = call.messages[-1]["content"]
    if "This is a group conversation:" in user:
        return "group"
    return "mina" if "conversation: Mina." in user else "theo"


def replies(**by_who: Any) -> FakeLLM:
    """A model that answers each profile with its own reply, or the default reply."""

    def responder(call: FakeCall) -> Any:
        found = by_who.get(who(call), profile_reply())
        if isinstance(found, Exception):
            raise found
        return found() if callable(found) else found

    return FakeLLM(responder)


def build(town: Any, llm: Any = None, embedder: Any = None, **more: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []
    found = people.build(town.archive, town.names, None, updates.append, **settings(town, llm, embedder, **more))
    return found, updates


def stored(town: Any) -> dict[str, Any]:
    held = settings(town)["store"].read()
    return {profile.key: profile for profile in held.profiles} if held else {}


def key_of(town: Any, conversation: str) -> str:
    return next(key for key, profile in stored(town).items() if conversation in profile.conversations)


def said(updates: list[dict[str, Any]]) -> str:
    return json.dumps(updates)


# --- a first build ---------------------------------------------------------------------


def test_a_build_writes_one_profile_for_each_person_whose_memory_is_read(town) -> None:
    town.read_memory()
    llm = replies(mina=profile_reply(kind="talking_stage", status="faded", summary="You and Mina flirted."))
    found, updates = build(town, llm)
    assert set(found) == RESULT_KEYS
    assert (found["people"], found["built"], found["kept"], found["failed"], found["notRead"]) == (3, 3, 0, 0, 0)
    assert (found["removed"], found["embedded"]) == (0, 3)
    assert len(llm.calls) == 3 and found["usage"]["calls"] == 3 and found["usd"] > 0
    mina = stored(town)[key_of(town, MINA)]
    assert (mina.kind, mina.status, mina.summary, mina.label) == ("talking_stage", "faded", "You and Mina flirted.", "Mina")
    assert mina.conversations == (MINA, MINA_MAIL) and mina.figures.as_dict() == MINA_FIGURES
    assert mina.vector is not None and np.isclose(np.linalg.norm(np.frombuffer(mina.vector, "<f4")), 1.0)
    assert stored(town)[f"conversation:{GROUP}"].kind == "group"


def test_each_call_asks_for_the_shape_with_no_reasoning_and_a_bounded_reply(town) -> None:
    town.read_memory()
    llm = replies()
    build(town, llm)
    for call in llm.calls:
        assert call.schema["title"] == "person_profile"
        assert (call.reasoning, call.max_output_tokens) == ("none", people.MAX_OUTPUT_TOKENS)
        assert THEO not in call.text and MINA not in call.text and MINA_MAIL not in call.text


def test_with_no_memory_read_nobody_is_written_and_the_build_says_so(town) -> None:
    llm = replies()
    found, updates = build(town, llm)
    # with no notes file nothing is laid out: the one with no text is counted too, as not read
    assert (found["people"], found["built"], found["notRead"], found["usd"]) == (4, 0, 4, 0.0)
    assert llm.calls == [] and stored(town) == {}
    assert "0 profiles to write, 0 kept as they are, 4 left out because their memory is not read" in said(updates)


def test_the_lines_and_the_progress_hold_counts_only(town) -> None:
    town.read_memory()
    _found, updates = build(town)
    text = said(updates)
    for private in ("Mina", MINA, MINA_MAIL, THEO, MARK, "Harbor", "unsaved"):
        assert private not in text
    progress = [update["progress"] for update in updates if "progress" in update]
    assert progress[-1] == {"step": 3, "of": 3, "label": "Saving the people index", "done": 1, "total": 1}
    lines = [update["log"] for update in updates if "log" in update]
    assert lines[0] == "Looking at 3 people and groups"
    assert lines[-1] == "The people index is saved: 3 profiles written now, 0 kept, 0 failed, 0 left out, 0 removed."


# --- a rebuild -------------------------------------------------------------------------


def test_a_rebuild_with_nothing_changed_makes_no_call(town) -> None:
    town.read_memory()
    build(town)
    llm, embedder = replies(), FakeEmbedder(dimensions=FAKE_FACTS.dimensions, model=FAKE_FACTS.model)
    found, _updates = build(town, llm, embedder)
    assert (found["built"], found["kept"], found["embedded"], found["usd"]) == (0, 3, 0, 0.0)
    assert llm.calls == [] and embedder.calls == []


def test_a_rebuild_calls_the_model_only_for_the_profile_whose_inputs_changed(town) -> None:
    town.read_memory()
    build(town)
    town.add([(THEO, "2024-03-04 09:00:00", "Ok then", False, "")])
    town.read_memory([THEO])
    llm = replies()
    found, _updates = build(town, llm)
    assert [who(call) for call in llm.calls] == ["theo"]
    assert (found["built"], found["kept"], found["embedded"]) == (1, 2, 1)
    assert stored(town)[key_of(town, THEO)].figures.last_day == "2024-03-04"


def test_a_person_whose_new_part_is_not_read_keeps_the_old_profile_and_is_left_out(town) -> None:
    town.read_memory()
    build(town)
    before = stored(town)[key_of(town, THEO)]
    town.add([(THEO, "2024-03-04 09:00:00", "Ok then", False, "")])
    llm = replies()
    found, _updates = build(town, llm)
    assert llm.calls == [] and (found["notRead"], found["kept"]) == (1, 2)
    assert stored(town)[key_of(town, THEO)] == before


def test_a_profile_of_no_one_in_the_archive_is_removed(town) -> None:
    town.read_memory()
    build(town)
    store = settings(town)["store"]
    held = store.read()
    gone = held.profiles[0].__class__(**{**held.profiles[0].__dict__, "key": "person:999"})
    store.write(Held(held.meta, (*held.profiles, gone)))
    found, _updates = build(town)
    assert found["removed"] == 1 and "person:999" not in stored(town)


def test_another_embedding_model_embeds_again_with_no_model_call(town) -> None:
    town.read_memory()
    build(town)
    other = FAKE_FACTS.__class__(**{**FAKE_FACTS.__dict__, "dimensions": 32})
    llm, embedder = replies(), FakeEmbedder(dimensions=32, model=FAKE_FACTS.model)
    found, _updates = build(town, llm, embedder, facts=other)
    assert llm.calls == [] and (found["embedded"], found["kept"]) == (3, 3)
    assert settings(town)["store"].read().meta.dimensions == 32


# --- the replies -----------------------------------------------------------------------


def test_a_wrong_shape_then_a_good_one_writes_the_profile(town) -> None:
    town.read_memory()
    answers = iter([{"kind": "friend"}, profile_reply(summary="Second try.")])
    llm = replies(theo=lambda: next(answers))
    found, _updates = build(town, llm)
    assert [who(call) for call in llm.calls].count("theo") == 2
    assert (found["built"], found["failed"]) == (3, 0)
    assert stored(town)[key_of(town, THEO)].summary == "Second try."


def test_two_replies_in_the_wrong_shape_fail_the_person_not_the_build(town) -> None:
    town.read_memory()
    llm = replies(theo={"summary": "no shape"})
    found, _updates = build(town, llm)
    assert [who(call) for call in llm.calls].count("theo") == 2
    assert (found["built"], found["failed"], found["usage"]["calls"]) == (2, 1, 4)
    assert all(THEO not in profile.conversations for profile in stored(town).values())


def test_an_unknown_kind_a_day_outside_the_span_a_reference_and_a_handle(town) -> None:
    town.read_memory()
    reply = profile_reply(
        kind="situationship",
        summary="You sold Theo a boat (m2) and wrote to theo@example.com.",
        turning_points=[
            {"date": "2024-03-01", "what": "Theo asked m1 about the boat."},
            {"date": "2025-01-01", "what": "After the last message."},
        ],
    )
    build(town, replies(theo=reply))
    theo = stored(town)[key_of(town, THEO)]
    assert (theo.kind, theo.summary) == ("unclear", "You sold Theo a boat and wrote to an unsaved email contact.")
    assert [(point.date, point.what) for point in theo.turning_points] == [("2024-03-01", "Theo asked about the boat.")]


def test_a_call_that_fails_on_its_own_fails_its_person_only(town) -> None:
    town.read_memory()
    found, _updates = build(town, replies(theo=LLMError("HTTP 500", status_code=500)))
    assert (found["built"], found["failed"]) == (2, 1)


# --- failures that stop the build ------------------------------------------------------


def test_a_refused_key_stops_the_build_and_keeps_what_was_paid(town) -> None:
    town.read_memory()
    llm = replies(theo=LLMError("HTTP 401", status_code=401))
    with pytest.raises(people.PeopleFailure) as stopped:
        build(town, llm, parallel=1)
    assert stopped.value.code == "key_rejected" and "rejected the API key" in str(stopped.value)
    assert stopped.value.usage["calls"] == 1
    assert [who(call) for call in llm.calls] == ["mina", "theo"]
    assert set(stored(town)) == {key_of(town, MINA)}


def test_with_no_key_nothing_is_called(town) -> None:
    town.read_memory()

    def no_key(_info: Any) -> Any:
        raise ReportFailure("no_key", "No API key is set.")

    with pytest.raises(people.PeopleFailure) as stopped:
        people.build(town.archive, town.names, **{**settings(town), "llm_factory": no_key})
    assert stopped.value.code == "no_key" and stored(town) == {}


def test_with_nothing_to_write_no_key_is_needed(town) -> None:
    town.read_memory()
    build(town)

    def no_key(_info: Any) -> Any:
        raise ReportFailure("no_key", "No API key is set.")

    found = people.build(town.archive, town.names, **{**settings(town), "llm_factory": no_key})
    assert found["kept"] == 3


class _NoKeyEmbedder:
    model, dimensions = FAKE_FACTS.model, FAKE_FACTS.dimensions

    def embed(self, texts: Any) -> Any:
        raise EmbeddingKeyError("HTTP 401", status_code=401)


def test_an_embedding_that_fails_keeps_the_profiles_and_the_next_build_embeds_them(town) -> None:
    town.read_memory()
    with pytest.raises(people.PeopleFailure) as stopped:
        build(town, embedder=_NoKeyEmbedder())
    assert stopped.value.code == "key_rejected" and stopped.value.usage["calls"] == 3
    assert len(stored(town)) == 3 and all(profile.vector is None for profile in stored(town).values())
    llm = replies()
    found, _updates = build(town, llm)
    assert llm.calls == [] and (found["embedded"], found["kept"]) == (3, 3)


def test_a_cancelled_build_sends_nothing_more(town) -> None:
    town.read_memory()
    cancel = CancelToken()
    cancel.cancel()
    llm = replies()
    with pytest.raises(people.PeopleFailure) as stopped:
        people.build(town.archive, town.names, cancel, None, **settings(town, llm))
    assert stopped.value.code == "cancelled" and llm.calls == []


def test_a_store_of_a_newer_format_is_not_replaced(town) -> None:
    town.read_memory()
    build(town)
    store: PeopleStore = settings(town)["store"]
    with opened(store.path) as conn:
        conn.execute("UPDATE meta SET format_version = 99")
    with pytest.raises(people.PeopleFailure) as stopped:
        build(town)
    assert stopped.value.code == "store_newer"


def test_each_end_of_a_build_writes_one_line_of_the_spend_log(town, caplog: pytest.LogCaptureFixture) -> None:
    town.read_memory()
    with caplog.at_level(logging.INFO, logger="recall.spend"):
        build(town)
    (line,) = [record.getMessage() for record in caplog.records if record.name == "recall.spend"]
    assert line.startswith("the people job: 3 profiles written, 0 kept, 0 failed, 3 model calls,")
    assert "Mina" not in line and MINA not in line


def test_a_listener_that_fails_does_not_stop_the_build(town) -> None:
    town.read_memory()

    def broken(_update: dict[str, Any]) -> None:
        raise RuntimeError("the page went away")

    found = people.build(town.archive, town.names, None, broken, **settings(town))
    assert found["built"] == 3


def test_two_builds_at_once_are_not_needed_but_a_build_is_thread_safe(town) -> None:
    town.read_memory()
    lock = threading.Lock()
    seen: list[str] = []

    def responder(call: FakeCall) -> Any:
        with lock:
            seen.append(who(call))
        return profile_reply()

    found, _updates = build(town, FakeLLM(responder), parallel=3)
    assert sorted(seen) == ["group", "mina", "theo"] and found["built"] == 3
    assert by_key([{"key": key} for key in stored(town)]).keys() == stored(town).keys()
