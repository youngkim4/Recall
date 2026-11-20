"""Scoring math on fixed inputs, the reading of a judge reply, and the references of a raw answer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from chat_engine.errors import ModelError
from chat_engine.fake_llm import FakeLLM, done, say
from chat_engine.ports import LLMEvent, LLMRequest, Usage
from evals.judge import RUBRIC, JudgeCase, ModelJudge, parse_verdict
from evals.recorder import references
from evals.score import (
    Metrics,
    by_kind,
    citation_precision,
    citation_recall,
    citation_scores,
    cited_guids,
    spread,
    spread_table,
    summarize,
    table,
)


class Lookup:
    def __init__(self, held: set[str]) -> None:
        self.held = held

    def message(self, message_id: str) -> dict[str, Any] | None:
        return {"id": message_id} if message_id in self.held else None


BASE = Metrics(
    correct=True, score=1.0, citation_recall=1.0, citation_precision=0.5, citations=2, unverified_quotes=0,
    invalid_references=0, leaked_handle=False, limited=False, truncated=False, model_calls=2, tool_calls=1,
    input_tokens=1000, cached_input_tokens=200, output_tokens=100, reasoning_tokens=0, chat_usd=0.01,
    judge_usd=0.002, wall_seconds=2.0,
)  # fmt: skip


def test_recall_is_capped_at_three_gold_messages() -> None:
    assert citation_recall(["a", "b"], ["a"]) == 0.5
    assert citation_recall(["a", "b", "c", "d", "e"], ["a", "b", "c"]) == 1.0
    assert citation_recall(["a", "b", "c", "d", "e"], ["a"]) == pytest.approx(1 / 3)
    assert citation_recall(["a"], []) == 0.0
    assert citation_recall([], ["a"]) is None


def test_precision_is_the_share_of_cited_that_is_gold() -> None:
    assert citation_precision(["a", "b"], ["a", "x", "y", "z"]) == 0.25
    assert citation_precision(["a"], []) is None
    assert citation_precision([], ["a"]) is None


def test_citations_map_to_guids_through_the_archive() -> None:
    citations = [{"messageId": "G1"}, {"messageId": "nope"}, {"messageId": "G2"}, {"messageId": "G1"}, {}]
    assert cited_guids(citations, Lookup({"G1", "G2"})) == ("G1", "G2")


def test_the_summary_adds_and_averages() -> None:
    other = replace(BASE, correct=False, score=0.0, citation_recall=None, chat_usd=0.03, error="timeout")
    found = summarize([BASE, other])
    assert (found.questions, found.correct, found.errors) == (2, 1, 1)
    assert found.accuracy == 0.5 and found.mean_score == 0.5
    assert found.mean_recall == 1.0  # a question with no recall is left out
    assert found.usd == pytest.approx(0.044)
    assert found.model_calls == 4 and found.input_tokens == 2000


def test_the_table_has_a_row_per_kind_and_all() -> None:
    summaries = by_kind(("fact", "count", "absent"), [("fact", BASE), ("count", BASE)])
    assert list(summaries) == ["fact", "count", "all"]
    text = table(summaries)
    assert text.splitlines()[0].startswith("kind")
    assert len(text.splitlines()) == 5


@pytest.mark.parametrize(
    ("reply", "correct", "score", "error"),
    [
        ('{"correct": true, "score": 1, "missing": [], "wrong": []}', True, 1.0, ""),
        ('Here: {"correct": false, "score": 0.5, "missing": ["the date"], "wrong": []} done', False, 0.5, ""),
        ('{"correct": true, "score": 7}', True, 1.0, ""),
        ("no json at all", False, 0.0, "the judge gave no JSON"),
        ('{"correct": "yes", "score": 1}', False, 0.0, "the judge gave no true or false for correct"),
        ('{"correct": true, "score": "high"}', False, 0.0, "the judge gave no number for score"),
        ('{"correct": true, "score": 1,}', False, 0.0, "the judge gave JSON that can not be read"),
    ],
)
def test_a_judge_reply_is_read_strictly(reply: str, correct: bool, score: float, error: str) -> None:
    verdict = parse_verdict(reply)
    assert (verdict.correct, verdict.score, verdict.error) == (correct, score, error)


CASE = JudgeCase("Where?", "fact", "Mile End.", ("Mile End",), "In Mile End [1].")


def test_the_model_judge_reads_the_reply_and_prices_it() -> None:
    llm = FakeLLM([[*say('{"correct": true, "score": 1, "missing": [], "wrong": []}', chunk=7), done(500, 40)]])
    verdict = ModelJudge(llm, "judge-model", lambda usage: usage.input_tokens / 1000).judge(CASE)
    assert verdict.correct and verdict.cost_usd == 0.5
    request = llm.requests[0]
    assert not request.allow_tools and request.tools == ()
    assert "Mile End" in request.messages[1].content


class Failing:
    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        raise ModelError()


def test_a_judge_call_that_fails_is_no_pass() -> None:
    verdict = ModelJudge(Failing(), "judge-model", lambda usage: 0.0).judge(CASE)
    assert not verdict.correct and verdict.score == 0.0 and "failed" in verdict.error
    assert verdict.usage == Usage()


def test_references_of_a_raw_answer() -> None:
    assert references("A [m3] and [m3, m7] and [m10-m12] but not [3] or m9") == ("m3", "m7", "m10", "m11", "m12")
    assert references("") == ()


def test_a_count_question_has_no_citation_scores() -> None:
    assert citation_scores("count", ["a", "b"], []) == (None, None)
    assert citation_scores("fact", ["a", "b"], ["a"]) == (0.5, 1.0)


def test_the_spread_gives_the_accuracy_of_each_round() -> None:
    wrong = replace(BASE, correct=False)
    rows = [("fact", 0, BASE), ("fact", 0, BASE), ("fact", 1, BASE), ("fact", 1, wrong), ("count", 0, wrong),
            ("count", 1, BASE)]  # fmt: skip
    found = spread(("fact", "count", "absent"), rows)
    assert list(found) == ["fact", "count", "all"]
    assert found["fact"].rounds == (1.0, 0.5) and found["fact"].mean == 0.75
    assert (found["count"].low, found["count"].high) == (0.0, 1.0)
    assert found["all"].rounds == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert spread(("fact",), []) == {}
    assert spread_table(found).splitlines()[0].split() == ["kind", "rounds", "mean", "min", "max"]


def test_the_rubric_is_fair_on_wording_and_strict_on_facts() -> None:
    for rule in ("Equivalent descriptions count", "light hedging", "equally likely", "arithmetic yourself",
                 "leave out the year", "could not find", "invented fact or a contradiction"):  # fmt: skip
        assert rule in RUBRIC, rule
