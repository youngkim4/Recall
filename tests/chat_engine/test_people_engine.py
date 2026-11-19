"""Whole questions with the people port: the tool list, the prompt, the seeded lookup."""

from __future__ import annotations

from typing import Any

from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.fake_digests import FakeDigests
from chat_engine.fake_llm import FakeLLM, call, say
from chat_engine.fake_meaning import FakeMeaning
from chat_engine.fake_people import FakePeople, person
from chat_engine.prompt import PEOPLE_RULES
from chat_engine.scope import Scope
from chat_engine.tool_specs import FIND_PEOPLE_DESCRIPTION, tool_specs
from chat_engine.tools import TOOL_SPECS
from tests.chat_engine.conftest import FIXED_NOW
from tests.chat_engine.helpers import events_of, last_user, system_text, tool_results
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_EMAIL, MAYA_PHONE, tiny_archive, tiny_labeler

MAYA = person(
    "person:1", [MAYA_PHONE, MAYA_EMAIL], label="Maya Okafor", kind="talking_stage", status="faded",
    summary="A few weeks of late-night texts that faded.",
)
QUESTION = "Who is someone I had a talking stage with but didn't work out?"
SEED_LABEL = "[lookup 2] find_people(description=the question, limit=5)"


def engine_with(script: list, people: Any = None, **ports: Any) -> tuple[ChatEngine, FakeLLM]:
    fake = FakeLLM(script)
    config = EngineConfig(model="test-model", max_output_tokens=800, timezone="Test/Zone")
    engine = ChatEngine(
        tiny_archive(), fake, tiny_labeler(), config, people=people, clock=lambda: FIXED_NOW, **ports
    )
    return engine, fake


def test_the_tool_is_described_as_the_contract_says() -> None:
    assert FIND_PEOPLE_DESCRIPTION == (
        "Find people by the kind of relationship and its story: a talking stage that did not work out, a friend "
        "I lost touch with, a coworker who became a friend. Gives each person's conversations (aliases), the "
        "kind, the status, the span and the counts, and a short story. Confirm a person in their own "
        "conversation before you name them."
    )
    spec = tool_specs(people=True)[-1]
    assert spec.name == "find_people" and spec.description == FIND_PEOPLE_DESCRIPTION
    assert spec.parameters["required"] == ["description", "kinds", "since", "until", "limit"]
    assert spec.parameters["properties"]["kinds"]["items"]["enum"][1] == "talking_stage"
    assert tool_specs(people=True)[:-1] == TOOL_SPECS


def test_the_tool_and_the_rules_are_there_only_with_the_port() -> None:
    engine, fake = engine_with([say("Nobody.")], people=FakePeople([MAYA]))
    engine.ask(QUESTION)
    request = fake.requests[0]
    assert [spec.name for spec in request.tools][-1] == "find_people"
    for rule in PEOPLE_RULES:
        assert rule in system_text(request)
    plain, plain_fake = engine_with([say("Nobody.")])
    plain.ask(QUESTION)
    assert "find_people" not in system_text(plain_fake.requests[0])
    assert plain_fake.requests[0].tools == TOOL_SPECS


def test_the_rules_say_where_to_start_how_to_check_and_what_to_keep_to() -> None:
    joined = " ".join(PEOPLE_RULES)
    assert "start with find_people" in joined
    assert "Never name a person only from what was said about them in another conversation." in joined
    assert "do not name a relationship that did" in joined
    assert "take the correction" in joined


def test_the_rules_sit_with_the_other_tools_of_the_ports() -> None:
    engine, fake = engine_with(
        [say("Nobody.")], people=FakePeople([MAYA]), meaning=FakeMeaning(), digests=FakeDigests({})
    )
    engine.ask(QUESTION)
    request = fake.requests[0]
    assert [spec.name for spec in request.tools][-2:] == ["conversation_digest", "find_people"]
    assert system_text(request).count("start with find_people") == 1


def test_every_question_gets_the_seeded_lookup_with_the_question_as_the_description() -> None:
    port = FakePeople([MAYA])
    engine, fake = engine_with([say("Maya, maybe.")], people=port)
    events: list[dict] = []
    result = engine.ask(QUESTION, on_event=events.append)
    assert port.calls == [{"description": QUESTION, "kinds": None, "since": None, "until": None, "limit": 5}]
    block = last_user(fake.requests[0])
    assert SEED_LABEL in block
    assert "1. Maya Okafor · talking stage · faded · conversations c" in block
    assert "Finding people who fit your question" in [event["text"] for event in events_of("status", events)]
    step = next(step for step in result["steps"] if step["tool"] == "find_people")
    assert step == {"tool": "find_people", "text": "Found 1 person who fits the question", "seeded": True}


def test_a_seeded_lookup_that_fails_does_not_fail_the_question() -> None:
    engine, fake = engine_with([say("I could not tell.")], people=FakePeople(error=RuntimeError("gone")))
    result = engine.ask(QUESTION)
    assert result["answer"] == "I could not tell."
    assert "The people index was not available" in last_user(fake.requests[0])


def test_the_seeded_lookup_keeps_the_scope() -> None:
    port = FakePeople([MAYA])
    engine, fake = engine_with([say("Nobody here.")], people=port)
    engine.ask(QUESTION, scope=Scope(conversations=(JONAS_PHONE,), since="2024-01-01"))
    assert port.calls[0]["since"] == "2024-01-01" and port.calls[0]["limit"] == 15
    block = last_user(fake.requests[0])
    assert "People who fit the question in the scope the user chose, from 2024-01-01: nobody found." in block
    assert "Maya" not in block.split("find_people(")[1]


def test_a_call_of_the_model_reaches_the_port_and_its_result_the_model() -> None:
    port = FakePeople([MAYA])
    script = [[call("find_people", "p1", description="a fling that faded", kinds=["talking_stage"])], say("Maya.")]
    engine, fake = engine_with(script, people=port)
    engine.ask("Anyone else?")
    assert port.calls[-1] == {
        "description": "a fling that faded", "kinds": ["talking_stage"], "since": None, "until": None, "limit": 5,
    }
    assert tool_results(fake.requests[1])[0].startswith(
        "People like “a fling that faded” (kinds: talking stage): 1 found"
    )
