"""The evaluation runner. The real-model path is tested with a stand-in: nothing is sent."""

from pathlib import Path
from typing import Any, Iterator

import pytest

from chat_engine.errors import RateLimitedError
from chat_engine.eval import runner
from chat_engine.eval.oracle import SCRIPTS
from chat_engine.eval.questions import build_questions
from chat_engine.eval.synthetic import build_synthetic_archive
from chat_engine.fake_llm import FakeLLM
from chat_engine.ports import LLMEvent, LLMRequest

SYNTHETIC = build_synthetic_archive()
QUESTIONS = {question.id: question for question in build_questions(SYNTHETIC)}
ANSWER_TEXT = ["Clint Yeastwood", "pinecone", "Taberna", "Bainbridge", "Dana", "sourdough starter"]


class StandIn:
    """Takes the place of the OpenAI adapter and plays the scripts by question text."""

    created: list[dict[str, Any]] = []
    fail_with: BaseException | None = None

    def __init__(self, **settings: Any) -> None:
        StandIn.created.append(settings)
        self._fakes: dict[str, FakeLLM] = {}

    def stream(self, request: LLMRequest) -> Iterator[LLMEvent]:
        if StandIn.fail_with is not None:
            raise StandIn.fail_with
        asked = [m.content for m in request.messages if m.role == "user"][-1].rsplit("Question: ", 1)[1].strip()
        name = next(q.id for q in QUESTIONS.values() if q.text == asked)
        fake = self._fakes.setdefault(name, FakeLLM(SCRIPTS[name]()))
        return fake.stream(request)


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> type[StandIn]:
    StandIn.created = []
    StandIn.fail_with = None
    monkeypatch.setattr(runner, "OpenAIResponsesLLM", StandIn)
    return StandIn


def _run(*arguments: str) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = runner.main(list(arguments), out=lines.append)
    return code, lines


def test_scripted_run_prints_scores_only() -> None:
    code, lines = _run("--scripted")
    text = "\n".join(lines)
    assert code == 0
    assert f"{len(QUESTIONS)} of {len(QUESTIONS)} passed." in text
    assert "By kind:" in text and "recall 3/3" in text
    assert "recall_starter" in text and "FAIL" not in text
    for fragment in ANSWER_TEXT:
        assert fragment not in text
    for question in QUESTIONS.values():
        assert question.text not in text


def test_only_runs_the_chosen_questions_and_what_they_continue() -> None:
    code, lines = _run("--scripted", "--only", "follow_count, recall_wifi")
    rows = [line.split()[0] for line in lines[1:4]]
    assert code == 0 and rows == ["recall_wifi", "count_sourdough", "follow_count"]
    assert "3 of 3 passed." in "\n".join(lines)


def test_real_run_needs_a_model_and_an_output_cap() -> None:
    with pytest.raises(SystemExit) as stopped:
        _run("--model", "some-model")
    assert stopped.value.code == 2


def test_real_run_builds_the_adapter_from_the_arguments(stand_in: type[StandIn]) -> None:
    code, lines = _run(
        "--model", "some-model", "--max-output-tokens", "900", "--timeout", "45",
        "--reasoning-effort", "low", "--only", "recall_starter,count_sourdough",
    )
    assert code == 0
    assert stand_in.created == [
        {"api_key": "test-key-not-real", "timeout_seconds": 45.0, "reasoning_effort": "low", "verbosity": None}
    ]
    assert "2 of 2 passed." in "\n".join(lines)


def test_key_can_come_from_one_named_file(stand_in: type[StandIn], tmp_path: Path) -> None:
    env_file = tmp_path / "eval.env"
    env_file.write_text("# comment\nOTHER=1\nexport OPENAI_API_KEY='file-key-not-real'\n", encoding="utf-8")
    _run("--model", "m", "--max-output-tokens", "500", "--env-file", str(env_file), "--only", "recall_wifi")
    assert stand_in.created[0]["api_key"] == "file-key-not-real"
    empty = tmp_path / "empty.env"
    empty.write_text("OTHER=1\n", encoding="utf-8")
    assert runner.read_key(str(empty)) == ""


def test_failed_model_call_is_a_row_with_its_error_code(stand_in: type[StandIn]) -> None:
    stand_in.fail_with = RateLimitedError()
    code, lines = _run("--model", "m", "--max-output-tokens", "500", "--only", "recall_wifi")
    assert code == 1
    assert "error: rate_limit" in lines[1] and "FAIL" in lines[1]
    assert "0 of 1 passed." in "\n".join(lines)


def test_only_refuses_an_id_that_is_not_a_question() -> None:
    with pytest.raises(SystemExit) as stopped:
        _run("--scripted", "--only", "recall_wifi,no_such_question")
    assert stopped.value.code == 2


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("OPENAI_API_KEY=plain-key-not-real", "plain-key-not-real"),
        ("OPENAI_API_KEY=plain-key-not-real # the eval key", "plain-key-not-real"),
        ('OPENAI_API_KEY="quoted key # not a comment" # a comment', "quoted key # not a comment"),
        ("export OPENAI_API_KEY = 'spaced-key-not-real'  ", "spaced-key-not-real"),
        ("OPENAI_API_KEY=", ""),
        ("#OPENAI_API_KEY=commented-out", ""),
    ],
)
def test_read_key_reads_one_line_of_an_env_file(tmp_path: Path, line: str, key: str) -> None:
    env_file = tmp_path / "eval.env"
    env_file.write_text(f"OTHER=1\n{line}\n", encoding="utf-8")
    assert runner.read_key(str(env_file)) == key
