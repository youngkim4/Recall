"""The runner end to end with a scripted model and a fake judge, and the dry run. No call leaves the machine."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import recall_paths
from archive import Archive
from chat_engine.fake_llm import FakeLLM, done, say
from chat_engine.ports import LLMRequest
from evals import places
from evals.judge import JudgeCase, Verdict
from evals.questions import build_questions
from evals.run import main
from evals.runner import Settings, run_all
from evals.world import Built, world
from recall_services import chat, keys, models
from recall_services.models import ChatModelConfig

QUESTIONS = {question.id: question for question in build_questions(world())}
# words of a gold message for each question the script answers
NEEDLES = {
    "What did the plumber come to fix at the new apartment?": "plumber coming thursday",
    "Who found it?": "FOUND IT",
    "Which city did Theo move to?": "denver job",
}
NOT_FOUND = "I could not find that in the messages."


def _asked(request: LLMRequest) -> str:
    return next(message.content for message in reversed(request.messages) if message.role == "user")


def _shown_ref(request: LLMRequest, words: str) -> str | None:
    for message in request.messages:
        for line in message.content.splitlines():
            found = re.match(r"\s*(m\d+) ", line)
            if found and words.casefold() in line.casefold():
                return found.group(1)
    return None


def answer(request: LLMRequest) -> list[Any]:
    """Cite the shown message with the words of the question's needle, or say it is not there."""
    asked = _asked(request)
    needle = next((words for question, words in NEEDLES.items() if question in asked), None)
    ref = _shown_ref(request, needle) if needle else None
    text = f"Here it is [{ref}]." if ref else NOT_FOUND
    return [*say(text, chunk=9), done(1200, 80, 300)]


class Factory:
    """The chat model port: a fresh scripted model for each question. It keeps every request."""

    def __init__(self, step: Callable[[LLMRequest], list[Any]] = answer) -> None:
        self.step = step
        self.models: list[FakeLLM] = []
        self.configs: list[ChatModelConfig] = []

    def __call__(self, config: ChatModelConfig) -> FakeLLM:
        self.configs.append(config)
        made = FakeLLM([self.step])
        self.models.append(made)
        return made


class FakeJudge:
    def __init__(self) -> None:
        self.cases: list[JudgeCase] = []

    def judge(self, case: JudgeCase) -> Verdict:
        self.cases.append(case)
        right = case.answer != NOT_FOUND
        return Verdict(right, 1.0 if right else 0.0, cost_usd=0.001)


class Exploding:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no model may be called")

    def judge(self, case: JudgeCase) -> Verdict:
        raise AssertionError("no judge may be called")


@pytest.fixture
def run_few(built: Built, archive: Archive, tmp_path: Path) -> Callable[..., Any]:
    def run(ids: tuple[str, ...], factory: Factory, judge: FakeJudge) -> dict[str, Any]:
        with places.isolated_saves(tmp_path):
            found = run_all(
                [QUESTIONS[one] for one in ids], archive=archive, names=built.names,
                settings=Settings(models.default_id()), factory=factory, judge=judge,
            )  # fmt: skip
        return {result.question.id: result for result in found}

    return run


def test_a_cited_gold_message_scores_full_recall(run_few: Callable[..., Any]) -> None:
    factory, judge = Factory(), FakeJudge()
    results = run_few(("fact_plumber",), factory, judge)
    found = results["fact_plumber"]
    assert found.cited == QUESTIONS["fact_plumber"].gold_guids
    assert found.metrics.citation_recall == 1.0 and found.metrics.citation_precision == 1.0
    assert found.metrics.correct and found.metrics.model_calls == 1
    assert found.metrics.input_tokens == 1200 and found.metrics.cached_input_tokens == 300
    assert found.metrics.chat_usd > 0 and found.metrics.judge_usd == 0.001
    assert not found.metrics.leaked_handle and found.metrics.invalid_references == 0
    assert judge.cases[0].key_facts == QUESTIONS["fact_plumber"].key_facts


def test_a_follow_up_sends_the_history_rows_of_the_app(run_few: Callable[..., Any]) -> None:
    factory = Factory()
    results = run_few(("follow_who_found",), factory, FakeJudge())
    question = QUESTIONS["follow_who_found"]
    assert len(factory.models) == 2
    second = "\n".join(message.content for message in factory.models[1].requests[0].messages)
    assert question.prior in second and results["follow_who_found"].prior_answer in second
    assert results["follow_who_found"].metrics.model_calls == 2


def test_an_absent_question_has_no_recall(run_few: Callable[..., Any]) -> None:
    results = run_few(("absent_dog",), Factory(), FakeJudge())
    assert results["absent_dog"].metrics.citation_recall is None
    assert results["absent_dog"].answer == NOT_FOUND


def test_a_reference_to_no_message_is_counted(run_few: Callable[..., Any]) -> None:
    factory = Factory(lambda request: [*say("It was Denver [m999]."), done(10, 5)])
    results = run_few(("fact_theo_city",), factory, FakeJudge())
    assert results["fact_theo_city"].metrics.invalid_references == 1


def test_an_error_is_recorded_and_not_judged(run_few: Callable[..., Any]) -> None:
    def broken(request: LLMRequest) -> list[Any]:
        raise RuntimeError("the script broke")

    judge = FakeJudge()
    results = run_few(("fact_salary",), Factory(broken), judge)
    assert results["fact_salary"].metrics.error and not results["fact_salary"].metrics.correct
    assert judge.cases == []


def test_main_writes_the_results_and_prints_the_table(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    said: list[str] = []
    factory, judge = Factory(), FakeJudge()
    assert main(["--kinds", "fact", "--limit", "3", "--out", str(out)], factory=factory, judge=judge, say=said.append) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["meta"]["model"] == models.default_id() and body["meta"]["questions"] == 3
    assert [row["kind"] for row in body["results"]] == ["fact"] * 3
    assert set(body["summary"]) == {"fact", "all"}
    assert body["summary"]["all"]["questions"] == 3
    assert any(line.startswith("kind") for line in said) and said[-1] == f"results: {out}"
    assert {config.model for config in factory.configs} == {models.default_id()}


def test_the_dry_run_calls_nothing_and_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "default_llm", Exploding())
    said: list[str] = []
    assert main(["--dry-run", "--limit", "2"], factory=Exploding(), judge=Exploding(), say=said.append) == 0
    text = "\n".join(said)
    assert "dry run: 2 questions" in text and "total: $" in text
    assert not places.results_folder().exists()


def test_a_run_with_no_key_stops_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keys, "api_key", lambda: "")
    monkeypatch.setattr(chat, "default_llm", Exploding())
    said: list[str] = []
    assert main(["--limit", "1"], say=said.append) == 2
    assert keys.NO_KEY in said[0]


@pytest.mark.parametrize("argv", [["--model", "no-such-model"], ["--kinds", "vibes"], ["--limit", "0"],
                                  ["--effort", "sideways"]])
def test_a_bad_option_is_one_line_and_no_call(argv: list[str]) -> None:
    said: list[str] = []
    assert main(argv, factory=Exploding(), judge=Exploding(), say=said.append) == 2
    assert len(said) == 1


def test_results_never_go_into_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    assert places.results_folder() == Path(recall_paths.DATA_DIR) / "evals"
    monkeypatch.setattr(recall_paths, "DATA_DIR", recall_paths.ROOT)
    assert places.results_folder() == places.APP_DATA / "evals"


def test_the_saves_point_into_the_world_folder_only_while_a_run_lasts(tmp_path: Path) -> None:
    before: Mapping[str, Path] = {"saves": recall_paths.SAVES_DIR, "out": recall_paths.OUT_DIR}
    with places.isolated_saves(tmp_path) as saves:
        assert recall_paths.SAVES_DIR == saves == tmp_path / "saves"
    assert {"saves": recall_paths.SAVES_DIR, "out": recall_paths.OUT_DIR} == before
