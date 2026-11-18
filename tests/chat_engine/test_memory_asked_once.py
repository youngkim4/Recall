"""Part E: one question asks a digest source for little, a failed digest says what it cost, a status names its days."""

from typing import Any

import pytest

from chat_engine.fake_digests import DigestFailed, FakeDigests, period
from chat_engine.fake_llm import call, say
from chat_engine.tool_limits import DIGEST_ESTIMATES_PER_QUESTION, DIGESTS_PER_QUESTION
from tests.chat_engine.helpers import alias_of_line, last_user, tool_results
from tests.chat_engine.memory_bench import MemoryBench, bench_for, maya_story
from tests.chat_engine.test_memory_engine import engine_with
from tests.chat_engine.tiny import JONAS_PHONE, MAYA_PHONE


def _five() -> list:
    return [period(f"Y{n}", f"202{n}-01-01", f"202{n}-12-31", 10, f"Summary {n}.") for n in range(5)]


def _ask_digest(request: Any) -> list:
    alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
    return [call("conversation_digest", "d1", conversation=alias)]


@pytest.mark.parametrize(("prepare", "words"), [(True, "in the background"), (False, "one question reads at most")])
def test_a_repeat_after_a_long_story_gets_the_kept_line_and_asks_the_source_for_nothing(
    prepare: bool, words: str
) -> None:
    bench = bench_for(digest_parts_limit=2, prepare=prepare)
    alias = bench.alias(MAYA_PHONE)
    first = bench.run("conversation_digest", conversation=alias)
    bench.box.context.memory.digests._cached = 3  # the estimate would now be another: it is not asked again
    for arguments in ({}, {"since": None, "until": ""}, {}):
        again = bench.prepare("conversation_digest", conversation=alias, **arguments)
        outcome = again.run(again.own_budget)
        assert not again.repeated and again.status == ""
        assert words in outcome.text and (outcome.text, outcome.summary) == (first.text, first.summary)
    assert [name for name, _ in bench.box.context.memory.digests.calls] == ["estimate", "prepare"]
    assert bench.box.context.digests_built == 0


def test_other_days_of_a_long_story_are_a_new_call() -> None:
    bench = bench_for(digest_parts_limit=2)
    alias = bench.alias(MAYA_PHONE)
    bench.run("conversation_digest", conversation=alias)
    bench.box.context.memory.digests._cached = 2  # the narrower days hold one period that is not read
    narrower = bench.prepare("conversation_digest", conversation=alias, since="2023-01-01")
    assert narrower.status == "Reading your conversation with Maya Okafor (from 2023-01-01)"
    assert narrower.run(narrower.own_budget).ok
    names = [name for name, _ in bench.box.context.memory.digests.calls]
    assert names == ["estimate", "prepare", "estimate", "digest"]


def test_one_question_asks_for_a_handful_of_estimates_at_most() -> None:
    digests = FakeDigests({MAYA_PHONE: _five(), JONAS_PHONE: _five()}, usd=0.2)
    bench = MemoryBench(digests=digests, digest_parts_limit=2)
    alias = bench.alias(MAYA_PHONE)
    outcomes = [
        bench.run("conversation_digest", conversation=alias, since=f"2020-01-{day:02d}") for day in range(1, 11)
    ]
    names = [name for name, _ in digests.calls]
    assert DIGEST_ESTIMATES_PER_QUESTION == 6 == 2 * DIGESTS_PER_QUESTION
    assert names.count("estimate") == 6 and names.count("prepare") == 6 and names.count("digest") == 0
    assert all("in the background" in outcome.text for outcome in outcomes[:6])
    for outcome in outcomes[6:]:
        assert not outcome.ok and outcome.text == (
            "This question already asked for 6 digests. Answer from what you have, or use the other tools."
        )
        assert outcome.summary == "Did not ask for another digest"
    kept = bench.run("conversation_digest", conversation=alias, since="2020-01-03")
    assert "in the background" in kept.text and len(digests.calls) == 12


def test_a_digest_that_is_too_costly_is_estimated_a_handful_of_times_at_most() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()[:2]}, usd=9.0)
    bench = MemoryBench(digests=digests)
    alias = bench.alias(MAYA_PHONE)
    texts = [bench.run("conversation_digest", conversation=alias).text for _ in range(9)]
    assert [name for name, _ in digests.calls] == ["estimate"] * 6
    assert all("would cost about $9.00" in text for text in texts[:6])
    assert all(text.startswith("This question already asked for 6 digests.") for text in texts[6:])


def test_one_question_of_many_identical_calls_asks_the_source_one_time() -> None:
    digests = FakeDigests({MAYA_PHONE: _five(), JONAS_PHONE: _five()}, usd=0.2)

    def step(request: Any) -> list:
        alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
        return [call("conversation_digest", f"d{len(request.messages)}-{n}", conversation=alias) for n in range(8)]

    engine, fake, _ = engine_with([step] * 8 + [say("Later.")], digests=digests)
    result = engine.ask("What is it like with Maya?")
    assert [name for name, _ in digests.calls] == ["estimate", "prepare"]
    assert result["usage"]["tool_calls"] == 64 and result["usage"]["digest_usd"] == 0.0
    assert all("in the background" in text for text in tool_results(fake.requests[8]))


@pytest.mark.parametrize(
    ("usd", "counted", "step"),
    [
        (0.02, 0.02, "Could not read the story of your conversation with Maya Okafor, about $0.02 was spent"),
        (0.001, 0.001, "Could not read the story of your conversation with Maya Okafor, less than $0.01 was spent"),
        (0.0, 0.0, "Could not read the story of your conversation with Maya Okafor"),
        (None, 0.0, "Could not read the story of your conversation with Maya Okafor"),
        ("much", 0.0, "Could not read the story of your conversation with Maya Okafor"),
        (float("inf"), 0.0, "Could not read the story of your conversation with Maya Okafor"),
        (-0.5, 0.0, "Could not read the story of your conversation with Maya Okafor"),
        (True, 0.0, "Could not read the story of your conversation with Maya Okafor"),
    ],
)
def test_a_digest_that_fails_after_it_was_paid_for_counts_what_it_cost(usd: Any, counted: float, step: str) -> None:
    error = RuntimeError("no usd") if usd is None else DigestFailed("the parts were refused", usd=usd)
    bench = bench_for(error=error)
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert not outcome.ok and outcome.summary == step
    assert outcome.text.endswith("could not be built now. Go on with the other tools.")
    assert bench.box.context.digest_usd == pytest.approx(counted)


def test_the_cost_of_a_failed_digest_is_in_the_usage_of_the_question() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()[:2]}, error=DigestFailed("refused", usd=0.03))
    engine, _, _ = engine_with([_ask_digest, say("I could not read it.")], digests=digests)
    result = engine.ask("What is my friendship with Maya like?")
    assert result["usage"]["digest_usd"] == pytest.approx(0.03)
    assert result["steps"][-1]["text"].endswith("with Maya Okafor, about $0.03 was spent")


def test_a_digest_that_is_cancelled_as_it_fails_keeps_what_it_cost() -> None:
    bench = bench_for(error=DigestFailed("stopped late", usd=0.05), honour_cancel=False)
    bench.box.context.memory.digests._during = bench.cancel.cancel
    outcome = bench.run("conversation_digest", conversation=bench.alias(MAYA_PHONE))
    assert outcome.text == "" and bench.box.context.digest_usd == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("days", "words"),
    [
        ({}, ""),
        ({"since": "2021-01-01"}, " (from 2021-01-01)"),
        ({"until": "2023-12-31"}, " (up to 2023-12-31)"),
        ({"since": "2021-01-01", "until": "2023-12-31"}, " (2021-01-01 to 2023-12-31)"),
    ],
)
def test_a_status_line_names_the_days_of_the_call_and_says_whole_only_without_days(days: dict, words: str) -> None:
    def status(**options: Any) -> str:
        bench = bench_for(**options)
        return bench.prepare("conversation_digest", conversation=bench.alias(MAYA_PHONE), **days).status

    whole = "whole " if not days else ""
    assert status(usd=0.1) == f"Reading your {whole}conversation with Maya Okafor{words}"
    assert status(cached=True) == f"Recalling your conversation with Maya Okafor{words}"
    started = status(digest_parts_limit=2)
    assert started == f"Starting to read your conversation with Maya Okafor{words} in the background"
