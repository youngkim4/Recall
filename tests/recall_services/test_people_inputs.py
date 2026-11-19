"""people_inputs.py: who has a profile, what the model reads of each, and the hash of those inputs."""

from __future__ import annotations

from typing import Any

import pytest

from recall_services import memory_prep, people_inputs, people_prompt
from recall_services.names import Names
from recall_services.people_inputs import EMPTY, NOT_READ, READ, Planner, Subject
from tests.recall_services.people_kit import (
    opened,
    GROUP,
    GROUP_TITLE,
    MINA,
    MINA_FIGURES,
    MINA_MAIL,
    MINA_RECORD,
    RAVI,
    THEO,
    town,  # noqa: F401
)


def planner(town: Any, names: Names | None = None) -> Planner:
    return Planner(town.archive, names or town.names, model=memory_prep.model(), **town.notes())


def person_id(town: Any, name: str) -> int:
    (found,) = [person["id"] for person in town.archive.people() if person["name"] == name]
    return int(found)


def test_one_profile_for_each_person_and_each_group(town) -> None:
    found = {subject.key: subject for subject in planner(town).subjects()}
    mina = f"person:{person_id(town, 'Mina')}"
    theo = next(key for key, subject in found.items() if subject.conversations == (THEO,))
    ravi = next(key for key, subject in found.items() if subject.conversations == (RAVI,))
    assert found[mina] == Subject(mina, (MINA, MINA_MAIL), False, "Mina")
    assert found[f"conversation:{GROUP}"] == Subject(f"conversation:{GROUP}", (GROUP,), True, GROUP_TITLE)
    assert (theo.startswith("person:"), found[theo].label) == (True, "an unsaved email contact")
    assert (ravi.startswith("person:"), found[ravi].label) == (True, "a number ending in 0171")
    assert len(found) == 4
    assert planner(town).subjects()[0].key == mina  # the busiest first


class _Stub:
    """An archive with one direct conversation that belongs to no person."""

    def conversations(self, **_more: Any) -> list[dict[str, Any]]:
        return [{"id": "+12025550199", "is_group": False, "name": "", "members": ["+12025550199"], "person_id": None}]


def test_a_direct_conversation_of_no_person_is_a_profile_of_its_own() -> None:
    found = people_inputs.subjects_of(_Stub(), Names())
    assert found == (Subject("conversation:+12025550199", ("+12025550199",), False, "a number ending in 0199"),)


def test_with_nothing_read_no_one_has_notes(town) -> None:
    reader = planner(town)
    assert {subject.key: reader.plan(subject).notes.state for subject in reader.subjects()} == {
        subject.key: (EMPTY if subject.conversations == (RAVI,) else NOT_READ) for subject in reader.subjects()
    }
    assert not town.places.cache_path.exists()


def test_a_person_with_one_thread_not_read_has_no_notes(town) -> None:
    town.read_memory([MINA])
    reader = planner(town)
    (mina,) = [subject for subject in reader.subjects() if subject.conversations == (MINA, MINA_MAIL)]
    assert reader.plan(mina).notes.state == NOT_READ


def test_the_notes_of_both_threads_are_read_in_the_order_of_their_days(town) -> None:
    town.read_memory()
    reader = planner(town)
    (mina,) = [subject for subject in reader.subjects() if subject.conversations == (MINA, MINA_MAIL)]
    planned = reader.plan(mina)
    assert planned.notes.state == READ
    assert [period.first_day for period in planned.notes.periods] == ["2023-01-02", "2023-01-05"]
    assert planned.figures.as_dict() == MINA_FIGURES
    assert planned.messages and planned.input_hash is None


def test_reading_the_notes_calls_no_model_and_marks_no_entry(town) -> None:
    town.read_memory()
    with opened(town.places.cache_path) as conn:
        before = sorted(conn.execute("SELECT key, last_used_at FROM period_results"))
    reader = planner(town)
    for subject in reader.subjects():
        reader.plan(subject)
    with opened(town.places.cache_path) as conn:
        assert sorted(conn.execute("SELECT key, last_used_at FROM period_results")) == before


def _fingerprint(town: Any, conversation: str, names: Names | None = None) -> str | None:
    from recall_services.people_look import look_cheaply
    from recall_services.people_plans import PlanStore

    reader = planner(town, names)
    seen = look_cheaply(reader, PlanStore(town.folder / "plans"))
    (found,) = [one for one in seen if conversation in one.subject.conversations]
    return found.fingerprint


def test_the_fingerprint_of_the_inputs_follows_each_input(town, monkeypatch: pytest.MonkeyPatch) -> None:
    town.read_memory()
    first = _fingerprint(town, MINA)
    assert first is not None and _fingerprint(town, MINA) == first
    assert _fingerprint(town, MINA, Names.from_records([{**MINA_RECORD, "name": "Mina Lund"}])) != first
    monkeypatch.setattr(people_prompt, "PROMPT_VERSION", "another")
    assert _fingerprint(town, MINA) != first


def test_a_new_message_changes_the_fingerprint_once_its_part_is_read(town) -> None:
    town.read_memory()
    first = _fingerprint(town, THEO)
    town.add([(THEO, "2024-03-04 09:00:00", "Ok", False, "")])
    assert _fingerprint(town, THEO) is None  # its new part is not read
    town.read_memory([THEO])
    assert _fingerprint(town, THEO) not in (None, first)


def test_a_group_member_who_is_saved_is_named_in_the_notes(town) -> None:
    town.read_memory()
    reader = planner(town)
    (group,) = [subject for subject in reader.subjects() if subject.is_group]
    user = reader.plan(group).messages[-1]["content"]
    assert "This is a group conversation: Harbor Walkers." in user
    assert THEO not in user and MINA not in user


def test_every_handle_in_the_notes_is_masked_before_it_leaves(town) -> None:
    """A summary of a digest can hold any handle: the notes are masked as the chat digest masks them."""
    from report_engine.fake_llm import FakeCall, FakeLLM, default_map_reply

    def map_reply(call: FakeCall) -> Any:
        reply = default_map_reply(call)
        return {**reply, "summary": "Mina gave 415-555-0199, someone@else.org and theo@example.com on 2023-01-05."}

    town.read_memory(llm=FakeLLM(map_reply))
    reader = planner(town)
    (mina,) = [subject for subject in reader.subjects() if MINA in subject.conversations]
    user = reader.plan(mina).messages[-1]["content"]
    assert "415-555-0199" not in user and "someone@else.org" not in user and THEO not in user
    assert "a phone number" in user and "an email address" in user and "an unsaved email contact" in user
    assert "2023-01-05" in user
