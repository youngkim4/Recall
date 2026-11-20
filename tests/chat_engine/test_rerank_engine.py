"""Whole questions with a reranker: the switch, the seeded search, the status line, the usage."""

from __future__ import annotations

from typing import Any

import pytest

from chat_engine.engine import ChatEngine, EngineConfig
from chat_engine.fake_llm import FakeLLM, call, say
from chat_engine.fake_meaning import FakeMeaning, hit
from chat_engine.fake_rerank import FakeReranker, preferring
from chat_engine.rerank import RANKING
from tests.chat_engine.conftest import FIXED_NOW
from tests.chat_engine.helpers import all_text, events_of, last_user
from tests.chat_engine.tiny import JONAS_PHONE, tiny_archive, tiny_labeler

QUESTION = "What did Maya say about the lease?"


def engine_with(script: list, *, reranker: Any = None, meaning: Any = None, rerank: bool = True) -> tuple:
    fake = FakeLLM(script)
    config = EngineConfig(model="test-model", max_output_tokens=800, timezone="Test/Zone", rerank=rerank)
    engine = ChatEngine(
        tiny_archive(), fake, tiny_labeler(), config, meaning=meaning, reranker=reranker, clock=lambda: FIXED_NOW
    )
    return engine, fake


def asked(engine: ChatEngine, question: str = QUESTION) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    result = engine.ask(question, on_event=events.append)
    return result, events


def test_the_switch_is_off_by_default() -> None:
    assert EngineConfig(model="m", max_output_tokens=10).rerank is False


def test_the_seeded_search_is_ranked_against_the_whole_question() -> None:
    ranker = FakeReranker(preferring("Lisbon"), usd=0.0015)
    engine, fake = engine_with([say("She signed it.")], reranker=ranker)
    result, events = asked(engine)
    assert result["answer"] == "She signed it."
    assert [made["query"] for made in ranker.calls] == [QUESTION]
    prompt = last_user(fake.requests[0])
    assert prompt.index("lease for the Lisbon flat") < prompt.index("lease got signed")
    assert {"type": "status", "text": RANKING} in events
    assert result["usage"]["rerank_usd"] == pytest.approx(0.0015)


def test_a_tool_search_is_ranked_and_shows_the_status_line() -> None:
    ranker = FakeReranker(preferring("Lisbon"), usd=0.001)
    script = [[call("search_messages", "s1", query="lease", match="all", order=None, limit=2)], say("Done.")]
    engine, fake = engine_with(script, reranker=ranker)
    result, events = asked(engine, "anything?")
    statuses = [event["text"] for event in events_of("status", events)]
    at = statuses.index("Searching for “lease”")
    assert statuses[at + 1] == RANKING
    assert [made["query"] for made in ranker.calls] == ["lease"]
    tool_text = [m.content for m in fake.requests[1].messages if m.role == "tool"][0]
    assert "lease for the Lisbon flat" in tool_text and "lease scan" not in tool_text
    assert result["usage"]["rerank_usd"] == pytest.approx(0.001)


def test_a_failing_reranker_does_not_fail_the_question() -> None:
    ranker = FakeReranker(error=RuntimeError("the ranking model is down"), usd=0.5)
    engine, _ = engine_with([say("She signed it.")], reranker=ranker)
    result, events = asked(engine)
    assert result["answer"] == "She signed it." and events[-1]["type"] == "done"
    assert result["usage"]["rerank_usd"] == 0.0


def test_the_seeded_search_by_meaning_is_ranked_against_the_question() -> None:
    ranker = FakeReranker()
    meaning = FakeMeaning([hit(JONAS_PHONE, "j1", "j3"), hit(JONAS_PHONE, "j4", "j4")])
    engine, _ = engine_with([say("Nothing.")], reranker=ranker, meaning=meaning)
    asked(engine, "Did we talk about sourdough?")
    assert [made["query"] for made in ranker.calls] == ["Did we talk about sourdough?"]


def _run(script: list, **more: Any) -> tuple[dict, list[str], list[dict]]:
    engine, fake = engine_with(script, **more)
    result, events = asked(engine)
    return result, [all_text(request) for request in fake.requests], events


def test_with_the_switch_off_nothing_changes() -> None:
    script = [[call("search_messages", "s1", query="lease", limit=2)], say("She signed it [m1].")]
    ranker = FakeReranker(preferring("Lisbon"), usd=1.0)
    plain = _run(script, rerank=False)
    switched_off = _run(script, reranker=ranker, rerank=False)
    no_port = _run(script, reranker=None, rerank=True)
    assert plain == switched_off == no_port
    assert ranker.calls == [] and "rerank_usd" not in plain[0]["usage"]
    assert RANKING not in [event.get("text") for event in plain[2]]
