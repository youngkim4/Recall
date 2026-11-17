"""Every question of the evaluation set, run with the scripted model.

The scripted model reads only the request it gets, so a pass shows that the
engine and its tools carry the facts to the model.
"""

from typing import Any, Mapping

import pytest

from chat_engine.engine import EngineConfig
from chat_engine.eval.harness import Row, evaluate, unsaved_handles, with_roots
from chat_engine.eval.oracle import SCRIPTS
from chat_engine.eval.questions import Question, build_questions
from chat_engine.eval.synthetic import build_synthetic_archive
from chat_engine.fake_llm import FakeLLM

SYNTHETIC = build_synthetic_archive()
QUESTIONS = build_questions(SYNTHETIC)
CONFIG = EngineConfig(model="scripted", max_output_tokens=600, timezone="Test/Zone")


class Run:
    """One scripted pass over all questions, with what the model was sent."""

    def __init__(self) -> None:
        self.fakes: dict[str, FakeLLM] = {}
        self.results: dict[str, Mapping[str, Any]] = {}
        self.rows = {row.id: row for row in evaluate(SYNTHETIC, QUESTIONS, self._llm, CONFIG, self._keep)}

    def _llm(self, question: Question) -> FakeLLM:
        self.fakes[question.id] = FakeLLM(SCRIPTS[question.id]())
        return self.fakes[question.id]

    def _keep(self, question: Question, result: Mapping[str, Any]) -> None:
        self.results[question.id] = result


@pytest.fixture(scope="module")
def run() -> Run:
    return Run()


def test_the_set_is_large_and_covers_every_kind() -> None:
    assert len(QUESTIONS) >= 25
    assert len({question.id for question in QUESTIONS}) == len(QUESTIONS)
    kinds = {question.kind for question in QUESTIONS}
    assert kinds >= {
        "recall", "count", "first_last", "who_is_this", "compare", "relative_date", "follow_up",
        "not_in_archive", "scoped", "other_language", "long_message", "group", "two_threads",
    }
    assert set(SCRIPTS) == {question.id for question in QUESTIONS}
    for question in QUESTIONS:
        if question.follows:
            earlier = [q.id for q in QUESTIONS[: QUESTIONS.index(question)]]
            assert question.follows in earlier


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda question: question.id)
def test_scripted_run_of_each_question_passes(run: Run, question: Question) -> None:
    row: Row = run.rows[question.id]
    assert row.error == "", row
    assert row.passed, (row.checks, run.results[question.id]["answer"])
    for tool in question.expect.tools:
        assert tool in row.tools, (tool, row.tools)


def test_recall_questions_end_in_one_model_call(run: Run) -> None:
    for name in ["recall_starter", "recall_wifi", "long_message", "group_flights", "portuguese_question"]:
        assert run.rows[name].model_calls == 1, name
        assert run.rows[name].tool_calls == 0, name


def test_counting_answers_come_from_exact_counts(run: Run) -> None:
    facts = SYNTHETIC.facts
    top = run.results["count_top"]
    assert str(facts["top_conversation_total"]) in top["answer"].replace(",", "")
    busiest = run.results["count_busiest_month"]
    assert str(facts["maya_busiest_month_2023_total"]) in busiest["answer"]
    volume = run.results["compare_volume"]["answer"].replace(",", "")
    assert str(facts["maya_total"]) in volume and str(facts["jonas_total"]) in volume
    assert facts["sourdough_total"] > 8, "more hits than one search result shows"


def test_follow_ups_reuse_the_evidence_of_their_chat(run: Run) -> None:
    first = run.results["recall_starter"]["citations"][0]
    again = run.results["follow_when"]["citations"][0]
    assert again["id"] == first["id"] and again["reference"] == first["reference"]
    assert run.rows["follow_when"].model_calls == 1
    request = run.fakes["follow_when"].requests[0]
    assert [message.role for message in request.messages][:3] == ["system", "user", "assistant"]


def test_scoped_questions_cite_only_their_scope(run: Run) -> None:
    assert run.results["scoped_absent"]["citations"] == []
    lease = run.results["scoped_lease"]["citations"]
    assert {citation["conversation"] for citation in lease} == {SYNTHETIC.keys["maya_phone"]}
    seen = "\n".join(m.content for request in run.fakes["scoped_absent"].requests for m in request.messages)
    assert "lease got signed" not in seen and "landlord" not in seen


def test_no_request_of_the_whole_run_shows_a_raw_unsaved_handle(run: Run) -> None:
    raw = [*unsaved_handles(SYNTHETIC), "7205550199", "555-0199", "3035550142", "rk.unsaved"]
    assert len(unsaved_handles(SYNTHETIC)) == 3
    for name, fake in run.fakes.items():
        for request in fake.requests:
            for message in request.messages:
                text = message.content + " ".join(call.arguments for call in message.tool_calls)
                for handle in raw:
                    assert handle not in text, (name, handle)
    assert "a number ending in 0199" in run.results["who_number"]["citations"][0]["sender_label"]


def test_long_message_is_in_the_prompt_whole(run: Run) -> None:
    from chat_engine.eval.synthetic_data import HANDOVER

    request = run.fakes["long_message"].requests[0]
    assert HANDOVER in request.messages[-1].content


def test_with_roots_adds_the_question_a_follow_up_continues() -> None:
    chosen = with_roots(QUESTIONS, ["follow_count"])
    assert [question.id for question in chosen] == ["count_sourdough", "follow_count"]
    assert with_roots(QUESTIONS, []) == QUESTIONS
