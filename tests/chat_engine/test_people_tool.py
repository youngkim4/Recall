"""find_people: people found by the kind of relationship and its story, inside the scope, masked."""

from __future__ import annotations

from typing import Any

import pytest

from chat_engine.fake_people import FakePeople, person, turning_point
from chat_engine.ports import PERSON_KINDS
from chat_engine.render import Limits, estimate_tokens
from chat_engine.scope import Scope
from chat_engine.tool_people import PEOPLE_LIMIT_DEFAULT, PEOPLE_LIMIT_MAX
from tests.chat_engine.memory_bench import RAW_HANDLES, MemoryBench
from tests.chat_engine.test_tools import Bench
from tests.chat_engine.tiny import GROUP, INES_PHONE, JONAS_PHONE, MAYA_EMAIL, MAYA_PHONE, UNSAVED_PHONE

FIGURES = {
    "first_day": "2021-03-04", "last_day": "2025-12-24", "messages": 11, "sent": 5, "received": 6,
    "active_days": 4, "longest_gap_days": 530.5, "busiest_month": "2021-03", "wrote_first_share": 0.5,
}
MAYA = person(
    "person:1", [MAYA_PHONE, MAYA_EMAIL], label="Maya Okafor", kind="talking_stage", status="faded",
    summary="They flirted over a lease and it faded. Her work line is +1 (303) 555-0142, mail maya.w@example.net.",
    turning_points=[
        turning_point("2021-03-04", "They sign a lease together"),
        turning_point("2023-07-11", "She asks you to call 303-555-0142"),
    ],
    figures=FIGURES, score=0.9,
)
JONAS = person(
    "person:2", [JONAS_PHONE], label="Jonas Weber", kind="friend", status="ongoing", summary="Bread friends.",
    figures={"first_day": "2024-02-10", "last_day": "2026-08-01", "messages": 4, "sent": 2, "received": 2},
    score=0.5,
)
HEADER = (
    "People like “a talking stage that faded”: 2 found, best first. A story is a summary, not a quote. "
    "Check a person in their own conversation before you name them."
)


def bench(*rows: Any, error: Exception | None = None, **options: Any) -> MemoryBench:
    return MemoryBench(people=FakePeople(list(rows) if rows else [MAYA, JONAS], error=error), **options)


def asked(made: MemoryBench) -> list[dict]:
    return made.box.context.memory.people.calls


def test_the_fake_returns_the_scripted_people_and_records_each_call() -> None:
    fake = FakePeople([MAYA, JONAS])
    found = fake.search("a fling", kinds=["talking_stage"], since="2021-01-01", until=None, limit=1)
    assert [row["key"] for row in found] == ["person:1"]
    assert fake.calls == [
        {"description": "a fling", "kinds": ["talking_stage"], "since": "2021-01-01", "until": None, "limit": 1}
    ]
    found[0]["conversations"].append("tampered")
    assert fake.search("again", kinds=None, since=None, until=None, limit=5)[0]["conversations"] == [
        MAYA_PHONE, MAYA_EMAIL,
    ]
    by_query = FakePeople(by_query={"bread": [JONAS]})
    assert by_query.search("lease", kinds=None, since=None, until=None, limit=5) == []
    with pytest.raises(RuntimeError):
        FakePeople(error=RuntimeError("no index")).search("x", kinds=None, since=None, until=None, limit=5)


def test_find_people_is_a_tool_only_with_the_port() -> None:
    refused = Bench().run("find_people", description="a talking stage")
    assert not refused.ok and 'There is no tool named "find_people".' in refused.text
    names = [spec.name for spec in bench().box.specs]
    assert names[-1] == "find_people" and "find_people" in bench().box.offered
    assert PERSON_KINDS[1] == "talking_stage" and len(PERSON_KINDS) == 11


def test_each_person_shows_the_aliases_the_kind_the_status_the_span_the_counts_and_the_story() -> None:
    made = bench()
    outcome = made.run("find_people", description="a talking stage that faded")
    lines = outcome.text.splitlines()
    assert lines[0] == HEADER
    assert lines[2] == "1. Maya Okafor · talking stage · faded · conversations c1, c2"
    assert lines[3] == (
        "   2021-03-04 to 2025-12-24 · 11 messages (you 5, them 6) · active on 4 days · longest gap 530 days · "
        "you wrote first on 50% of the active days"
    )
    assert lines[4].startswith("   Story: They flirted over a lease and it faded.")
    assert lines[5].startswith("   Turning points: 2021-03-04 They sign a lease together; 2023-07-11 She asks")
    assert "2. Jonas Weber · friend · ongoing · conversations c3" in outcome.text
    assert made.aliases.resolve("c1") == MAYA_PHONE and made.aliases.resolve("c3") == JONAS_PHONE
    assert outcome.summary == "Found 2 people like “a talking stage that faded”"
    assert asked(made) == [
        {"description": "a talking stage that faded", "kinds": None, "since": None, "until": None,
         "limit": PEOPLE_LIMIT_DEFAULT},
    ]
    assert made.prepare("find_people", description="a talking stage that faded").status == (
        "Finding people like “a talking stage that faded”"
    )


def test_a_phone_number_or_an_address_of_a_profile_is_masked_and_so_is_the_description() -> None:
    made = bench()
    outcome = made.run("find_people", description="who wrote from 720-555-0199 or rk.unsaved@example.org")
    seen = "\n".join([outcome.text, outcome.summary, asked(made)[0]["description"]])
    for raw in [*RAW_HANDLES, "303", "555-0142", "maya.w", "example.net"]:
        assert raw not in seen, raw
    assert "Her work line is a phone number, mail an email address." in outcome.text
    assert "She asks you to call a phone number" in outcome.text
    assert asked(made)[0]["description"] == "who wrote from a number ending in 0199 or an unsaved email contact"


def test_a_person_with_no_conversation_in_the_scope_never_appears() -> None:
    made = bench(scope=Scope(conversations=(JONAS_PHONE,)))
    outcome = made.run("find_people", description="a talking stage that faded")
    assert "Maya" not in outcome.text and "c2" not in outcome.text
    assert "1. Jonas Weber · friend · ongoing · conversations c1" in outcome.text
    assert "in the scope the user chose: 1 found" in outcome.text
    # the figures cover conversations outside the scope: they are not shown
    assert "4 messages" not in outcome.text
    assert "   Figures: call conversation_stats inside the scope." in outcome.text
    assert asked(made)[0]["limit"] == 3 * PEOPLE_LIMIT_DEFAULT  # some people of the port fall outside the scope


def test_only_the_conversations_inside_the_scope_are_named() -> None:
    made = bench(scope=Scope(conversations=(MAYA_PHONE, INES_PHONE)))
    outcome = made.run("find_people", description="a talking stage that faded")
    assert "1. Maya Okafor · talking stage · faded · conversations c1" in outcome.text
    assert made.aliases.resolve("c2") is None  # the email thread got no alias
    assert "Jonas" not in outcome.text


def test_days_reach_the_port_and_a_person_or_a_turning_point_outside_them_is_left_out() -> None:
    made = bench()
    outcome = made.run("find_people", description="a talking stage that faded", since="2023-01-01", until="2023-12-31")
    assert asked(made)[0]["since"] == "2023-01-01" and asked(made)[0]["until"] == "2023-12-31"
    assert "Jonas" not in outcome.text  # his messages start in 2024
    assert "They sign a lease" not in outcome.text and "2023-07-11 She asks" in outcome.text
    scoped = bench(scope=Scope(since="2024-01-01"))
    text = scoped.run("find_people", description="a talking stage that faded").text
    assert asked(scoped)[0]["since"] == "2024-01-01" and "Turning points" not in text


def test_kinds_are_checked_and_passed_on() -> None:
    made = bench()
    wrong = made.run("find_people", description="x", kinds=["situationship"])
    assert not wrong.ok and "kinds must be a list of: romantic, talking_stage" in wrong.text
    assert asked(made) == []
    made.run("find_people", description="x", kinds=["talking_stage", "romantic"])
    assert asked(made)[0]["kinds"] == ["talking_stage", "romantic"]
    made.run("find_people", description="y", kinds="friend")
    assert asked(made)[1]["kinds"] == ["friend"]
    assert not made.run("find_people", description="").ok
    assert not made.run("find_people", description="x", kinds=[3]).ok


def test_the_limit_is_bounded() -> None:
    made = bench()
    made.run("find_people", description="x", limit=99)
    made.run("find_people", description="y", limit=1)
    assert [call["limit"] for call in asked(made)] == [PEOPLE_LIMIT_MAX, 1]


def test_a_port_that_fails_gives_one_line_and_the_question_goes_on() -> None:
    outcome = bench(error=RuntimeError("the index is gone")).run("find_people", description="x")
    assert not outcome.ok
    assert outcome.text == "The people index was not available: nobody was looked up. Use the other tools."
    assert outcome.summary == "Could not look for people"


def test_nobody_found_says_so() -> None:
    made = MemoryBench(people=FakePeople([]))
    outcome = made.run("find_people", description="a pen pal")
    assert outcome.text == "People like “a pen pal”: nobody found. Use the other tools."
    assert outcome.summary == "Looked for people like “a pen pal”: nobody found"


def test_rows_of_another_form_and_people_the_archive_does_not_hold_are_left_out() -> None:
    odd = [
        {"key": "x"},
        "not a row",
        person("person:7", ["+19995550000"], label="Ghost", summary="Nobody."),
        person("person:8", [GROUP], label="Lisbon crew", kind="party animals", status="?", summary="Trips."),
    ]
    outcome = bench(*odd).run("find_people", description="x")
    assert "Ghost" not in outcome.text
    assert "1. Lisbon crew · unclear · unclear · conversations c1" in outcome.text
    assert "found" in outcome.text.splitlines()[0] and ": 1 found" in outcome.text.splitlines()[0]


def test_a_person_with_an_unsaved_number_is_named_by_its_mask() -> None:
    row = person("conversation:9", [UNSAVED_PHONE], label="+17205550199", kind="acquaintance", summary="Climbing.")
    outcome = bench(row).run("find_people", description="x")
    assert "1. a number ending in 0199 · acquaintance · unclear · conversations c1" in outcome.text
    for raw in RAW_HANDLES:
        assert raw not in outcome.text


def test_a_result_that_does_not_fit_is_cut_with_a_note() -> None:
    rows = [MAYA, JONAS, person("person:3", [INES_PHONE], label="Inês", summary="Long story. " * 80)]
    made = bench(*rows, limits=Limits(count_tokens=estimate_tokens))
    outcome = made.run("find_people", budget=260, description="x")
    assert estimate_tokens(outcome.text) <= 260
    assert outcome.text.rstrip().endswith("not shown. Ask for fewer people, or narrow the days.]")


def test_a_cancel_while_the_port_searches_reads_nothing_more() -> None:
    made = bench()
    prepared = made.prepare("find_people", description="x")
    made.cancel.cancel()
    outcome = prepared.run(3000)
    assert outcome.text == "" and not outcome.ok
