"""chat.py gives the engine a reranker over the cheap model when the switch is on. Every model is a script."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest

from archive import Archive
from chat_engine import errors as engine_errors
from chat_engine.fake_llm import FakeLLM, say
from chat_engine.ports import Completed, LLMEvent, LLMRequest, TextDelta, Usage
from recall_services import chat, memory, memory_prep, models, rerank_port
from recall_services.models import ChatModelConfig
from recall_services.names import Names
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

NOW = datetime(2026, 11, 17, 9, 30)
QUESTION = "Who said the wheelbarrow has a flat tire?"
RAW = ("15550179999", "555-017-9999", "stranger@example.net")


class ScriptRanker:
    """A model port that answers every ranking call with the same JSON."""

    def __init__(self, answer: str = '{"order": []}') -> None:
        self.answer = answer
        self.requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        self.requests.append(request)
        return iter([TextDelta(self.answer), Completed(usage=Usage(input_tokens=1000, output_tokens=50))])


@pytest.fixture(autouse=True)
def _no_memory_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "ports", lambda _archive, _names: {"meaning": None, "digests": None})


def ask(archive: Archive, saved: Names, **more: Any) -> dict[str, Any]:
    llm = FakeLLM([say("Someone did.")])
    settings = {"model": "gpt-6.1-sol", "llm_factory": lambda _config: llm, "clock": lambda: NOW, **more}
    events = list(chat.stream(QUESTION, archive=archive, names=saved, **settings))
    assert events[-1]["type"] == "done", events[-1]
    return {**events[-1]["payload"], "statuses": [e["text"] for e in events if e["type"] == "status"]}


def recording(ranker: Any, made: list[ChatModelConfig]) -> Any:
    def factory(config: ChatModelConfig) -> Any:
        made.append(config)
        return ranker

    return factory


@pytest.mark.parametrize(
    ("environ", "on"),
    [({"RECALL_RERANK": "1"}, True), ({}, True), ({"RECALL_RERANK": "0"}, False), ({"RECALL_RERANK": " 0 "}, False),
     ({"RECALL_RERANK": "yes"}, True)],
)
def test_the_switch_comes_from_the_environment(environ: dict[str, str], on: bool) -> None:
    assert chat.rerank_switch(environ) is on


def test_the_switch_is_on_in_the_app_and_off_in_the_tests() -> None:
    assert chat.rerank_switch({}) is True
    assert chat.RERANK is False  # tests/guards.py sets RECALL_RERANK=0


def test_the_ranking_model_is_the_cheap_model_of_the_memory_with_no_reasoning() -> None:
    config = rerank_port.config()
    assert config.model == memory_prep.model()
    assert config.reasoning_effort == "none"
    assert config.max_output_tokens == rerank_port.OUTPUT_TOKENS
    assert config.timeout_seconds == rerank_port.TIMEOUT_SECONDS


def test_with_the_switch_on_the_seeded_search_is_ranked_and_paid_for(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat, "RERANK", True)
    ranker, made = ScriptRanker(), []
    payload = ask(archive, saved, rerank_factory=recording(ranker, made))
    assert [config.model for config in made] == [memory_prep.model()]
    assert len(ranker.requests) == 1
    sent = "\n".join(message.content for message in ranker.requests[0].messages)
    assert QUESTION in sent and "flat tire" in sent
    assert not [raw for raw in RAW if raw in sent]
    one_call = call_cost_usd(models.model_info(memory_prep.model()), LLMUsage(1000, 0, 50))
    assert payload["usage"]["rerank_usd"] == pytest.approx(one_call) and one_call > 0
    assert "Ranking what I found" in payload["statuses"]


def test_with_the_switch_off_no_ranking_model_is_made(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat, "RERANK", False)
    made: list[ChatModelConfig] = []
    payload = ask(archive, saved, rerank_factory=recording(ScriptRanker(), made))
    assert made == [] and "rerank_usd" not in payload["usage"]
    assert "Ranking what I found" not in payload["statuses"]


def test_a_ranking_model_that_can_not_be_made_leaves_the_question_as_it_was(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat, "RERANK", True)

    def no_key(_config: ChatModelConfig) -> Any:
        raise engine_errors.NoApiKeyError()

    payload = ask(archive, saved, rerank_factory=no_key)
    assert payload["answer"] == "Someone did." and "rerank_usd" not in payload["usage"]


def test_a_ranking_answer_that_can_not_be_read_still_answers(
    archive: Archive, saved: Names, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat, "RERANK", True)
    payload = ask(archive, saved, rerank_factory=recording(ScriptRanker("not json"), []))
    assert payload["answer"] == "Someone did." and payload["usage"]["rerank_usd"] > 0
