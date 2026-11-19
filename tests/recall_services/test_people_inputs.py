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
    assert planned.input_hash and planned.messages


def test_reading_the_notes_calls_no_model_and_marks_no_entry(town) -> None:
    town.read_memory()
    with opened(town.places.cache_path) as conn:
        before = sorted(conn.execute("SELECT key, last_used_at FROM period_results"))
    reader = planner(town)
    for subject in reader.subjects():
        reader.plan(subject)
    with opened(town.places.cache_path) as conn:
        assert sorted(conn.execute("SELECT key, last_used_at FROM period_results")) == before


def test_the_hash_of_the_inputs_follows_each_input(town, monkeypatch: pytest.MonkeyPatch) -> None:
    town.read_memory()

    def mina_hash(names: Names | None = None) -> str:
        reader = planner(town, names)
        (mina,) = [subject for subject in reader.subjects() if MINA in subject.conversations]
        return str(reader.plan(mina).input_hash)

    first = mina_hash()
    assert mina_hash() == first
    assert mina_hash(Names.from_records([{**MINA_RECORD, "name": "Mina Lund"}])) != first
    monkeypatch.setattr(people_prompt, "PROMPT_VERSION", "another")
    assert mina_hash() != first


def test_a_new_message_changes_the_hash(town) -> None:
    town.read_memory()

    def theo_hash() -> str | None:
        reader = planner(town)
        (theo,) = [subject for subject in reader.subjects() if subject.conversations == (THEO,)]
        return reader.plan(theo).input_hash

    first = theo_hash()
    town.add([(THEO, "2024-03-04 09:00:00", "Ok", False, "")])
    town.read_memory([THEO])
    assert theo_hash() not in (None, first)


def test_a_group_member_who_is_saved_is_named_in_the_notes(town) -> None:
    town.read_memory()
    reader = planner(town)
    (group,) = [subject for subject in reader.subjects() if subject.is_group]
    user = reader.plan(group).messages[-1]["content"]
    assert "This is a group conversation: Harbor Walkers." in user
    assert THEO not in user and MINA not in user
