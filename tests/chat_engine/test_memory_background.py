"""Part D1 as F4 left it: a digest in a question is bounded, a longer story is not read, the cost is counted."""

import logging
from typing import Any

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.engine import EngineConfig
from chat_engine.fake_digests import FakeDigests, period
from chat_engine.fake_llm import call, say
from chat_engine.fake_meaning import FakeMeaning, hit
from chat_engine.tool_specs import tool_specs
from tests.chat_engine.helpers import alias_of_line, last_user, system_text, tool_results
from tests.chat_engine.memory_bench import MemoryBench, bench_for, maya_story
from tests.chat_engine.test_memory_engine import engine_with
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE

NOT_READY = (
    "The story of c1 · Maya Okafor · direct is not read yet, and it is too long to read inside one question: "
    "3 periods are not read yet, and one question reads at most 2 periods. Answer from the other tools."
)
TOO_LONG = "Your conversation with Maya Okafor is too long to read inside one question"


def _five() -> list:
    return [period(f"Y{n}", f"202{n}-01-01", f"202{n}-12-31", 10, f"Summary {n}.") for n in range(5)]


# --- the fake --------------------------------------------------------------------------


def test_the_digest_fake_has_no_prepare_and_reports_each_part() -> None:
    fake = FakeDigests({MAYA_PHONE: _five()[:2]}, cached=1)
    assert not hasattr(fake, "prepare")
    told: list[dict] = []
    fake.digest(MAYA_PHONE, since=None, until=None, cancel=CancelToken(), progress=told.append)
    assert told == [
        {"part": 1, "of": 2, "label": "Y0", "state": "cached"},
        {"part": 2, "of": 2, "label": "Y1", "state": "start"},
        {"part": 2, "of": 2, "label": "Y1", "state": "done"},
    ]


def test_the_digest_fake_counts_cached_periods_and_names_what_a_digest_cost() -> None:
    fake = FakeDigests({MAYA_PHONE: _five()}, usd=0.3, cached=3, spent=0.012)
    assert fake.estimate(MAYA_PHONE, since=None, until=None) == {"periods": 5, "periods_cached": 3, "usd": 0.3}
    assert fake.digest(MAYA_PHONE, since=None, until=None, cancel=CancelToken())["usd"] == 0.012
    plain = FakeDigests({MAYA_PHONE: _five()})
    assert "usd" not in plain.digest(MAYA_PHONE, since=None, until=None, cancel=CancelToken())


# --- the limits ------------------------------------------------------------------------


def test_the_limits_of_a_digest_inside_a_question() -> None:
    config = EngineConfig(model="m", max_output_tokens=100)
    assert config.digest_parts_limit == 4 and config.digest_usd_limit == 0.5
    assert EngineConfig(model="m", max_output_tokens=100, digest_parts_limit=0).digest_parts_limit == 0
    with pytest.raises(ValueError, match="digest_parts_limit must not be negative"):
        EngineConfig(model="m", max_output_tokens=100, digest_parts_limit=-1)


# --- more periods to read than one question reads --------------------------------------


def test_a_long_story_of_some_days_is_not_read_in_the_question() -> None:
    bench = bench_for(digest_parts_limit=2, usd=0.01)
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE), since="2021-01-01")
    outcome = prepared.run(prepared.own_budget)
    assert prepared.status == TOO_LONG
    assert not outcome.ok and outcome.text == NOT_READY.replace("direct is", "direct, from 2021-01-01 is")
    assert outcome.summary == "Did not read the conversation with Maya Okafor: 3 periods are not read yet"
    assert bench.box.context.memory.digests.calls == [
        ("estimate", {"conversation_id": MAYA_PHONE, "since": "2021-01-01", "until": None}),
    ]
    assert bench.read("message") == [] and len(bench.registry) == 0


def test_a_long_story_says_what_a_question_reads(caplog: pytest.LogCaptureFixture) -> None:
    bench = bench_for(digest_parts_limit=2)
    prepared = bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    with caplog.at_level(logging.WARNING, logger="chat_engine"):
        outcome = prepared.run(prepared.own_budget)
    assert prepared.status == TOO_LONG
    assert not outcome.ok and outcome.text == NOT_READY and "\n" not in outcome.text
    assert outcome.summary == "Did not read the conversation with Maya Okafor: 3 periods are not read yet"
    assert "digest" not in [name for name, _ in bench.box.context.memory.digests.calls]
    assert caplog.records == []


@pytest.mark.parametrize(
    ("limit", "cached", "read"), [(2, 1, True), (2, 0, False), (3, 0, True), (0, 3, True), (0, 2, False), (1, 2, True)]
)
def test_the_parts_limit_counts_the_periods_that_are_not_read_yet(limit: int, cached: int, read: bool) -> None:
    bench = bench_for(digest_parts_limit=limit, cached=cached)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert outcome.ok is read
    assert ("digest" in [name for name, _ in bench.box.context.memory.digests.calls]) is read


def test_one_period_to_read_and_a_limit_of_none_read_well() -> None:
    bench = bench_for(digest_parts_limit=0, cached=2)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert "one question: 1 period is not read yet, and one question reads at most 0 periods." in outcome.text


def test_the_parts_limit_comes_before_the_cost_limit_and_after_the_estimate() -> None:
    costly = bench_for(digest_parts_limit=2, usd=9.0)
    assert "too long to read inside one question" in costly.run(
        "conversation_digest", conversation=costly.alias(MAYA_PHONE)
    ).text
    few = bench_for(digest_parts_limit=2, usd=9.0, cached=1)
    text = few.run("conversation_digest", conversation=few.alias(MAYA_PHONE)).text
    assert "would cost about $9.00: 2 periods are not built yet." in text
    assert few.box.memory_payload() == {"memory": {"notReady": []}}  # a cost refusal is no story too long
    broken = bench_for(digest_parts_limit=2, estimate_error=RuntimeError("down"))
    assert "could not be built now" in broken.run("conversation_digest", conversation=broken.alias(MAYA_PHONE)).text
    assert [name for name, _ in broken.box.context.memory.digests.calls] == ["estimate"]


# --- the cost of a digest that was read now --------------------------------------------


def _ask_digest(request: Any) -> list:
    alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
    return [call("conversation_digest", "d1", conversation=alias)]


def test_the_cost_of_a_digest_is_in_the_usage_and_in_the_step() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()[:2]}, usd=0.4, spent=0.0123)
    engine, _, _ = engine_with([_ask_digest, say("Close friends.")], digests=digests)
    result = engine.ask("What is my friendship with Maya like?")
    assert result["usage"]["digest_usd"] == pytest.approx(0.0123)
    assert result["steps"][-1]["text"] == "Read the story of your conversation with Maya Okafor: 2 periods, about $0.01"


@pytest.mark.parametrize(
    ("spent", "words", "counted"),
    [
        (None, "2 periods", 0.0),
        (0.0, "2 periods", 0.0),
        (0.004, "2 periods, less than $0.01", 0.004),
        (1.239, "2 periods, about $1.24", 1.239),
        ("a lot", "2 periods", 0.0),
        (float("nan"), "2 periods", 0.0),
        (-1.0, "2 periods", 0.0),
        (True, "2 periods", 0.0),
    ],
)
def test_the_step_names_the_cost_when_it_is_known(spent: Any, words: str, counted: float) -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()[:2]}, spent=spent)
    bench = MemoryBench(digests=digests)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert outcome.summary == f"Read the story of your conversation with Maya Okafor: {words}"
    assert bench.box.context.digest_usd == pytest.approx(counted)
    assert bench.box.usage({"model_calls": 1}) == {"model_calls": 1, "digest_usd": pytest.approx(counted)}


def test_the_usage_has_the_figure_only_with_a_digest_source() -> None:
    engine, _, _ = engine_with([say("Nothing.")])
    assert "digest_usd" not in engine.ask("anything about zanzibar?")["usage"]
    engine, _, _ = engine_with([say("Nothing.")], meaning=FakeMeaning())
    assert "digest_usd" not in engine.ask("anything about zanzibar?")["usage"]
    engine, _, _ = engine_with([say("Nothing.")], digests=FakeDigests({}))
    usage = engine.ask("anything about zanzibar?")["usage"]
    assert usage["digest_usd"] == 0.0 and isinstance(usage["digest_usd"], float)
    assert MemoryBench().box.usage({"model_calls": 1}) == {"model_calls": 1}


def test_the_cost_of_two_digests_is_added_and_a_cancelled_question_keeps_it() -> None:
    cancel = CancelToken()
    stories = {MAYA_PHONE: maya_story()[:2], JONAS_PHONE: maya_story()[2:]}
    digests = FakeDigests(stories, spent=0.02, honour_cancel=False)

    def both(request: Any) -> list:
        block = last_user(request)
        maya = alias_of_line(block, "Maya Okafor · direct, phone")
        jonas = alias_of_line(block, "Jonas Weber · direct")
        return [
            call("conversation_digest", "d1", conversation=maya),
            call("conversation_digest", "d2", conversation=jonas),
        ]

    engine, _, _ = engine_with([both, say("never")], digests=digests)
    seen = 0

    def on_event(event: dict) -> None:
        nonlocal seen
        seen += event["type"] == "status" and event["text"].startswith("Reading your whole conversation")
        if seen == 2:
            digests._during = cancel.cancel  # the cancel comes while the second digest is read

    result = engine.ask("How close am I to Maya Okafor and Jonas Weber?", cancel=cancel, on_event=on_event)
    assert result["cancelled"] is True
    assert result["usage"]["digest_usd"] == pytest.approx(0.04)  # the second digest was paid for, then cancelled


def test_a_long_story_in_a_question_tells_the_model_what_to_do() -> None:
    digests = FakeDigests({MAYA_PHONE: _five()}, usd=0.2)

    def go_on(request: Any) -> list:
        assert tool_results(request)[0] == NOT_READY.replace("3 periods", "5 periods")
        return say("From the figures: you write often.")

    engine, fake, _ = engine_with([_ask_digest, go_on], digests=digests, digest_parts_limit=2)
    events: list[dict] = []
    result = engine.ask("What is my friendship with Maya like?", on_event=events.append)
    statuses = [event["text"] for event in events if event["type"] == "status"]
    assert TOO_LONG in statuses
    assert result["steps"][-1] == {
        "tool": "conversation_digest",
        "text": "Did not read the conversation with Maya Okafor: 5 periods are not read yet",
        "seeded": False,
    }
    assert result["usage"]["digest_usd"] == 0.0 and result["usage"]["tool_calls"] == 1
    assert [name for name, _ in digests.calls] == ["estimate"]


def test_the_engine_gives_its_parts_limit_to_the_tool() -> None:
    digests = FakeDigests({MAYA_PHONE: _five()}, usd=0.2)
    engine, fake, _ = engine_with([_ask_digest, say("A long friendship.")], digests=digests, digest_parts_limit=5)
    result = engine.ask("What is my friendship with Maya like?")
    assert "== Y4 · 2024-01-01 to 2024-12-31 · 10 messages ==" in tool_results(fake.requests[1])[0]
    assert result["steps"][-1]["text"] == "Read the story of your conversation with Maya Okafor: 5 periods"


# --- an index that is older than the archive -------------------------------------------

STALE = (
    "The search by meaning found nothing that the archive still holds: its index must be brought up to date. "
    "These were found by their words only."
)


def test_hits_that_the_archive_no_longer_holds_are_named_in_the_header(caplog: pytest.LogCaptureFixture) -> None:
    gone = [hit(JONAS_PHONE, f"legacy:{n}", f"legacy:{n + 9}") for n in range(30)]
    bench = MemoryBench(meaning=FakeMeaning(gone))
    with caplog.at_level(logging.WARNING, logger="chat_engine"):
        outcome = bench.run("search_messages", query="sourdough starter", match="meaning")
    assert outcome.ok and outcome.text.splitlines()[0].endswith(f"best first. {STALE}")
    assert "sourdough starter is alive  <<< match" in outcome.text
    assert outcome.summary.endswith("2 exchanges shown, by words only")
    assert "24 hits" in " ".join(record.getMessage() for record in caplog.records)
    assert len(bench.read("message")) == 24  # one read for each hit: the first message is gone, the last is not asked
    nothing = MemoryBench(meaning=FakeMeaning(gone)).run("search_messages", query="zeppelin", match="meaning")
    assert nothing.text == f"Search by meaning for “zeppelin” in all conversations: nothing found. {STALE}"


def test_one_hit_that_the_archive_holds_is_enough_for_no_word_about_the_index() -> None:
    hits = [hit(JONAS_PHONE, "legacy:1", "legacy:9"), hit(JONAS_PHONE, "j1", "j3")]
    outcome = MemoryBench(meaning=FakeMeaning(hits)).run("search_messages", query="zeppelin", match="meaning")
    assert "brought up to date" not in outcome.text and "its name is Clint Yeastwood" in outcome.text
    outside = [hit(MAYA_PHONE, "a1", "a4")]  # held, and outside the conversation of the call: no word about the index
    bench = MemoryBench(meaning=FakeMeaning(outside))
    jonas = [bench.alias(JONAS_PHONE)]
    text = bench.run("search_messages", query="zeppelin", match="meaning", conversations=jonas).text
    assert "brought up to date" not in text and bench.read("message") == []
    assert "brought up to date" not in MemoryBench(meaning=FakeMeaning()).run(
        "search_messages", query="zeppelin", match="meaning"
    ).text


# --- the words the model reads ---------------------------------------------------------


def test_the_description_of_the_digest_tool_says_what_happens_to_a_long_conversation() -> None:
    description = tool_specs(digest=True)[-1].description
    assert "costs money" not in description and "above the limit" not in description
    assert "A conversation that is too long to read inside one question is not read: the tool says so." in description
    assert "One question builds at most 3 digests." in description


def test_the_prompt_says_what_to_do_when_the_tool_gives_no_digest() -> None:
    engine, fake, _ = engine_with([say("Nothing.")], digests=FakeDigests({}))
    engine.ask("anything about zanzibar?")
    system = system_text(fake.requests[0])
    assert "When the tool gives no digest, read that conversation at several points of the period" in system
    assert "could not be built" not in system
