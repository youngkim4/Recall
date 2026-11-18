"""Part F4: no answer is promised for later. A story too long for one question is not read, and the payload says so."""

from __future__ import annotations

from typing import Any

from chat_engine.engine import EngineConfig
from chat_engine.fake_digests import FakeDigests, period
from chat_engine.fake_llm import call, say
from chat_engine.fake_meaning import FakeMeaning
from chat_engine.ports import DigestSource
from chat_engine.tool_context import Memory
from chat_engine.tool_specs import tool_specs
from tests.chat_engine.helpers import alias_of_line, last_user, system_text, tool_results
from tests.chat_engine.memory_bench import MemoryBench, bench_for
from tests.chat_engine.test_memory_engine import engine_with
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE

NOT_READY = (
    "The story of c1 · Maya Okafor · direct is not read yet, and it is too long to read inside one question: "
    "5 periods are not read yet, and one question reads at most 4 periods. Answer from the other tools."
)
LATER_WORDS = ("background", "later", "few minutes", "fuller answer", "come back")


def _many(count: int) -> list:
    return [period(f"P{n}", f"2010-01-{n + 1:02d}", f"2010-01-{n + 1:02d}", 5, f"Summary {n}.") for n in range(count)]


def _ask_digest(request: Any) -> list:
    alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
    return [call("conversation_digest", "d1", conversation=alias)]


class Prepares(FakeDigests):
    """An older source that still has `prepare`. The engine never calls it."""

    def prepare(self, conversation_id: str, **_days: object) -> bool:
        self.calls.append(("prepare", {"conversation_id": conversation_id}))
        return True


def test_a_question_reads_four_parts_and_the_cost_guard_stays() -> None:
    config = EngineConfig(model="m", max_output_tokens=100)
    assert (config.digest_parts_limit, config.digest_usd_limit) == (4, 0.5)
    assert Memory().digest_parts_limit == 4
    assert not hasattr(DigestSource, "prepare")


def test_four_parts_are_read_inside_the_question() -> None:
    bench = bench_for(_many(4), digest_parts_limit=4)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert outcome.ok and outcome.summary == "Read the story of your conversation with Maya Okafor: 4 periods"


def test_a_story_too_long_for_one_question_is_not_read_and_nothing_is_started() -> None:
    digests = Prepares({MAYA_PHONE: _many(5)}, usd=0.2)
    bench = MemoryBench(digests=digests, digest_parts_limit=4)
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    outcome = prepared.run(prepared.own_budget)
    assert prepared.status == "Your conversation with Maya Okafor is too long to read inside one question"
    assert prepared.fetch is None
    assert not outcome.ok and outcome.text == NOT_READY
    assert outcome.summary == "Did not read the conversation with Maya Okafor: 5 periods are not read yet"
    assert [name for name, _ in digests.calls] == ["estimate"]
    assert bench.box.memory_payload() == {"memory": {"notReady": [MAYA_PHONE]}}


def test_the_payload_names_each_conversation_that_is_not_read_once() -> None:
    digests = FakeDigests({MAYA_PHONE: _many(5), JONAS_PHONE: _many(2)}, usd=0.2)

    def both(request: Any) -> list:
        block = last_user(request)
        maya = alias_of_line(block, "Maya Okafor · direct, phone")
        jonas = alias_of_line(block, "Jonas Weber · direct")
        return [
            call("conversation_digest", "d1", conversation=maya),
            call("conversation_digest", "d2", conversation=maya, since="2010-01-02"),
            call("conversation_digest", "d3", conversation=jonas),
        ]

    engine, fake, _ = engine_with([both, say("From the figures.")], digests=digests)
    result = engine.ask("How close am I to Maya and Jonas?")
    assert result["memory"] == {"notReady": [MAYA_PHONE]}
    told = tool_results(fake.requests[1])
    assert told[0] == NOT_READY and told[1] == NOT_READY.replace("direct is", "direct, from 2010-01-02 is")
    assert "== P0" in told[2]
    assert not any(word in text for text in told for word in LATER_WORDS)


def test_the_payload_has_the_field_only_with_a_digest_source() -> None:
    engine, _, _ = engine_with([say("Nothing.")], digests=FakeDigests({}))
    assert engine.ask("anything about zanzibar?")["memory"] == {"notReady": []}
    engine, _, _ = engine_with([say("Nothing.")], meaning=FakeMeaning())
    assert "memory" not in engine.ask("anything about zanzibar?")
    engine, _, _ = engine_with([say("Nothing.")])
    assert "memory" not in engine.ask("anything about zanzibar?")


def test_a_story_too_long_tells_the_model_and_the_reader_nothing_about_later() -> None:
    digests = FakeDigests({MAYA_PHONE: _many(5)}, usd=0.2)
    engine, fake, _ = engine_with([_ask_digest, say("From the figures: you write often.")], digests=digests)
    events: list[dict] = []
    result = engine.ask("What is my friendship with Maya like?", on_event=events.append)
    shown = [event["text"] for event in events if event["type"] == "status"] + [step["text"] for step in result["steps"]]
    words = " ".join([*shown, *tool_results(fake.requests[1]), system_text(fake.requests[0])])
    assert not any(word in words.lower() for word in LATER_WORDS)
    assert "progress" not in [event["type"] for event in events]
    assert result["usage"]["digest_usd"] == 0.0 and result["memory"] == {"notReady": [MAYA_PHONE]}


def test_the_description_of_the_tool_promises_nothing_for_later() -> None:
    description = tool_specs(digest=True)[-1].description
    assert not any(word in description.lower() for word in LATER_WORDS)
    assert "A conversation that is too long to read inside one question is not read: the tool says so." in description
