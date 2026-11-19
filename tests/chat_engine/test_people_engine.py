"""Whole questions with the people port: the tool list, the prompt, the seeded lookup."""

from __future__ import annotations

import time
from typing import Any

from chat_engine.cancel import CancelToken
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
    # who by a kind of relationship or a story: find_people; who by an amount: the figures
    assert "asks who by a kind of relationship or by a story" in joined and "starts with find_people" in joined
    assert "asks who by an amount (closest, most, who did I talk to) starts with the figures" in joined
    assert "Check at most 3 candidates, the best first." in joined
    assert "Never name a person only from what was said about them in another conversation." in joined
    assert "do not name a relationship that did" in joined and "take the correction" in joined
    assert "A story from find_people is not a quote and not a citation" in joined


def test_the_people_rules_sit_in_their_places() -> None:
    engine, fake = engine_with([say("Nobody.")], people=FakePeople([MAYA]))
    engine.ask(QUESTION)
    system = system_text(fake.requests[0])
    work, answer = system.split("How to answer")
    assert "starts with find_people" in work and "Check at most 3 candidates" in work
    assert "A story from find_people is not a quote" in answer and "take the correction" in answer
    assert work.index("People seldom write the words") < work.index("starts with find_people")
    assert work.index("1. Figures") < work.index("starts with find_people")


def test_the_rules_sit_with_the_other_tools_of_the_ports() -> None:
    engine, fake = engine_with(
        [say("Nobody.")], people=FakePeople([MAYA]), meaning=FakeMeaning(), digests=FakeDigests({})
    )
    engine.ask(QUESTION)
    request = fake.requests[0]
    assert [spec.name for spec in request.tools][-2:] == ["conversation_digest", "find_people"]
    assert system_text(request).count("starts with find_people") == 1
    answer = system_text(request).split("How to answer")[1]
    assert answer.index("A summary from conversation_digest") < answer.index("A story from find_people")


def test_every_question_gets_the_seeded_lookup_with_the_question_as_the_description() -> None:
    port = FakePeople([MAYA])
    engine, fake = engine_with([say("Maya, maybe.")], people=port)
    events: list[dict] = []
    result = engine.ask(QUESTION, on_event=events.append)
    assert port.calls == [
        {"description": QUESTION, "kinds": None, "since": None, "until": None, "limit": 5, "conversations": None}
    ]
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
    assert port.calls[0]["since"] == "2024-01-01" and port.calls[0]["limit"] == 5
    assert port.calls[0]["conversations"] == [JONAS_PHONE]
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
        "conversations": None,
    }
    assert tool_results(fake.requests[1])[0].startswith(
        "People like “a fling that faded” (kinds: talking stage): 1 found"
    )


class SlowPeople:
    """A port that waits as a provider that hangs would, then fails or answers. Each call is counted."""

    def __init__(self, seconds: float, *, fail: bool = True, during: Any = None) -> None:
        self.seconds = seconds
        self.fail = fail
        self.during = during
        self.calls = 0

    def search(self, description: str, **_: Any) -> list:
        self.calls += 1
        if self.during is not None:
            self.during()
        time.sleep(self.seconds)
        if self.fail:
            raise TimeoutError("the embedding provider did not answer")
        return [MAYA]


def test_a_port_that_fails_in_the_seed_is_not_asked_again_in_the_question() -> None:
    port = SlowPeople(0.3)
    script = [
        [call("find_people", "p1", description="a fling"), call("find_people", "p2", description="an old friend")],
        say("I could not look."),
    ]
    engine, fake = engine_with(script, people=port)
    started = time.monotonic()
    result = engine.ask(QUESTION)
    assert time.monotonic() - started < 0.55  # one wait of the port, not three
    assert port.calls == 1 and result["answer"] == "I could not look."
    results = tool_results(fake.requests[1])
    assert results == ["The people index was not available: nobody was looked up. Use the other tools."] * 2


def test_a_cancel_while_the_seeded_port_waits_ends_the_question_when_the_port_returns() -> None:
    token = CancelToken()
    port = SlowPeople(0.2, fail=False, during=token.cancel)
    engine, fake = engine_with([say("never")], people=port)
    result = engine.ask(QUESTION, cancel=token)
    assert result["cancelled"] is True and fake.requests == []
    assert port.calls == 1
