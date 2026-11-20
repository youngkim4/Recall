"""ModelReranker: the prompt it sends, and how it reads the answer. The model is a script."""

from __future__ import annotations

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.fake_llm import FakeLLM, done, say
from chat_engine.model_reranker import ModelReranker, parse_order
from chat_engine.ports import RerankItem, Usage

TEXTS = ("we signed the lease", "pizza later?", "rent up 50")
ITEMS = tuple(RerankItem(str(number), text) for number, text in enumerate(TEXTS, 1))
IDS = ("1", "2", "3")


def ranker(*steps: list, price: object = None) -> tuple[ModelReranker, FakeLLM]:
    llm = FakeLLM(list(steps))
    priced = price if price is not None else (lambda usage: usage.input_tokens * 0.001)
    return ModelReranker(llm, "cheap-model", price=priced, max_output_tokens=300), llm  # type: ignore[arg-type]


def test_the_prompt_holds_the_question_and_the_numbered_candidates() -> None:
    model, llm = ranker([*say('{"order": ["3", "1"]}'), done(200, 12)])
    cancel = CancelToken()
    model.rerank("When did the rent go up?", ITEMS, 2, cancel)
    [request] = llm.requests
    assert request.model == "cheap-model" and request.max_output_tokens == 300
    assert request.tools == () and request.allow_tools is False and request.cancel is cancel
    system, user = request.messages
    assert system.role == "system" and "JSON" in system.content and '"order"' in system.content
    assert user.role == "user" and "When did the rent go up?" in user.content
    for item in ITEMS:
        assert f"[{item.id}] {item.text}" in user.content
    assert "at most 2" in user.content


def test_the_answer_gives_the_order_and_the_cost() -> None:
    model, _ = ranker([*say('{"order": ["3", "1"]}'), done(200, 12)])
    result = model.rerank("When did the rent go up?", ITEMS, 2)
    assert result.ids == ("3", "1") and not result.failed
    assert result.usage == Usage(input_tokens=200, output_tokens=12)
    assert result.usd == pytest.approx(0.2)


def test_missing_ids_follow_in_their_order_up_to_keep() -> None:
    model, _ = ranker([*say('{"order": ["3"]}'), done(10, 2)])
    assert model.rerank("q", ITEMS, 3).ids == ("3", "1", "2")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"order": ["3", "1", "2"]}', ("3", "1", "2")),
        ('{"order": [3, 1]}', ("3", "1", "2")),
        ('{"order": ["9", "3", "3", "x", null, "1"]}', ("3", "1", "2")),
        ('```json\n{"order": ["2"]}\n```', ("2", "1", "3")),
        ('Here it is: {"order": ["2", "3"]} done', ("2", "3", "1")),
        ('["3", "2"]', ("3", "2", "1")),
        ('{"order": []}', ("1", "2", "3")),
    ],
)
def test_the_answer_is_read_defensively(text: str, expected: tuple[str, ...]) -> None:
    assert parse_order(text, IDS) is not None
    model, _ = ranker([*say(text), done(5, 5)])
    assert model.rerank("q", ITEMS, 3).ids == expected


@pytest.mark.parametrize("text", ["", "no idea", '{"order": "3"}', '{"best": ["3"]}', "{broken", "[1, 2"])
def test_an_answer_that_can_not_be_read_keeps_the_order_and_says_so(text: str) -> None:
    assert parse_order(text, IDS) is None
    model, _ = ranker([*say(text), done(40, 3)])
    result = model.rerank("q", ITEMS, 2)
    assert result.ids == ("1", "2") and result.failed and result.reason
    assert result.usd == pytest.approx(0.04)  # the call was paid for


def test_a_failing_model_keeps_the_order_and_never_raises() -> None:
    model, _ = ranker([RuntimeError("the provider is down")])
    result = model.rerank("q", ITEMS, 2)
    assert result.ids == ("1", "2") and result.failed and result.reason == "RuntimeError"
    assert result.usd == 0.0


def test_a_failing_price_costs_nothing_and_keeps_the_order() -> None:
    def broken(_usage: Usage) -> float:
        raise KeyError("no price")

    model, _ = ranker([*say('{"order": ["2"]}'), done(5, 5)], price=broken)
    result = model.rerank("q", ITEMS, 3)
    assert result.ids == ("2", "1", "3") and result.usd == 0.0


def test_no_candidates_make_no_call() -> None:
    model, llm = ranker()
    assert model.rerank("q", (), 3).ids == () and llm.requests == []
    assert model.rerank("q", ITEMS, 0).ids == () and llm.requests == []


def test_a_cancelled_question_makes_no_call() -> None:
    model, llm = ranker()
    cancel = CancelToken()
    cancel.cancel()
    result = model.rerank("q", ITEMS, 2, cancel)
    assert result.ids == ("1", "2") and result.failed and llm.requests == []
