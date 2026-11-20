"""A question reads until the model answers: no step or token budget stops it.

What can end it early: the model answers, a loop that finds nothing new
(`stall_rounds`), or the safety cap on rounds (`max_steps`). Older results
are condensed when the results held in the prompt pass `held_tool_tokens`.
"""

import re

from chat_engine.condense import CONDENSED_ABOVE, CONDENSED_MARK
from chat_engine.engine import EngineConfig
from chat_engine.fake_llm import call, say
from chat_engine.prompt import LIMIT_NOTE, STALL_NOTE
from tests.chat_engine.helpers import events_of, last_user, make_engine, ref_of_line, tool_results

QUERIES = ("sourdough", "lease", "starter", "bread", "flour", "oven", "rent", "keys", "move", "dinner")


def _distinct_rounds(count: int) -> list:
    return [[call("search_messages", f"s{n}", query=f"{QUERIES[n % len(QUERIES)]} {n}")] for n in range(count)]


def test_the_defaults_have_no_budget_that_a_normal_question_reaches() -> None:
    settings = EngineConfig(model="m", max_output_tokens=800)
    assert settings.max_steps >= 50
    assert settings.held_tool_tokens >= 150_000
    assert settings.stall_rounds == 3
    assert not hasattr(settings, "total_tool_tokens")


def test_a_question_reads_past_the_old_limit_until_the_model_answers() -> None:
    rounds = _distinct_rounds(20)
    engine, fake, _ = make_engine([*rounds, say("Done reading.")], stall_rounds=100)
    result = engine.ask("tell me everything")
    assert result["answer"] == "Done reading."
    assert result["limited"] is False
    assert all(request.allow_tools for request in fake.requests)
    assert all(LIMIT_NOTE not in last_user(request) for request in fake.requests)


def test_the_safety_cap_ends_with_an_answer_and_marks_it_limited() -> None:
    engine, fake, _ = make_engine([*_distinct_rounds(2), say("From what I have.")], max_steps=2, stall_rounds=100)
    result = engine.ask("bread?")
    assert [request.allow_tools for request in fake.requests] == [True, True, False]
    assert LIMIT_NOTE in last_user(fake.requests[2])
    assert result["limited"] is True


def test_lookups_that_find_nothing_new_end_the_question_without_marking_it_limited() -> None:
    same = [call("search_messages", query="sourdough")]
    script = [same, [call("search_messages", "b", query="sourdough")], [call("search_messages", "c", query="sourdough")]]
    engine, fake, _ = make_engine([*script, [call("search_messages", "d", query="sourdough")], say("That is all.")])
    result = engine.ask("bread?")
    assert result["answer"] == "That is all."
    assert result["limited"] is False
    assert fake.requests[-1].allow_tools is False
    assert STALL_NOTE in last_user(fake.requests[-1])


def test_a_near_same_lookup_counts_as_a_repeat() -> None:
    engine, fake, archive = make_engine(
        [[call("search_messages", "a", query="Sourdough ")], [call("search_messages", "b", query="sourdough")], say("ok")]
    )
    engine.ask("bread?")
    assert "Same lookup as before" in tool_results(fake.requests[2])[-1]
    searched = [a for name, a in archive.calls if name == "search" and str(a.get("query")).strip().lower() == "sourdough"]
    assert len(searched) == 1


def test_older_results_are_condensed_when_the_held_results_pass_the_mark() -> None:
    def condense(request):
        assert request.allow_tools is False and request.tools == ()
        ref = ref_of_line("\n".join(m.content for m in request.messages), "sourdough starter is alive")
        return say(f"{ref}: the starter is alive.")

    def answer(request):
        tools = tool_results(request)
        assert tools[0].startswith(CONDENSED_MARK)
        assert all(text == CONDENSED_ABOVE for text in tools[1:-1])
        ref = re.search(r"m\d+", tools[0]).group(0)
        return say(f"It is alive [{ref}].")

    rounds = [[call("search_messages", "a", query="sourdough")], [call("search_messages", "b", query="lease")]]
    engine, fake, _ = make_engine([*rounds, condense, answer], held_tool_tokens=50, stall_rounds=100)
    events: list[dict] = []
    result = engine.ask("bread?", on_event=events.append)
    assert result["answer"] == "It is alive [1]."
    assert len(result["citations"]) == 1
    assert "Condensing what I have read" in [event["text"] for event in events_of("status", events)]
    assert result["usage"]["model_calls"] == 4


def test_the_newest_round_is_never_condensed() -> None:
    def answer(request):
        assert all(not text.startswith(CONDENSED_MARK) for text in tool_results(request))
        return say("ok")

    engine, fake, _ = make_engine([[call("search_messages", "a", query="sourdough")], answer], held_tool_tokens=10)
    engine.ask("bread?")
    assert len(fake.requests) == 2
