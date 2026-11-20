"""--rejudge, --repeat and the reading of a results file. A scripted model and fake judges only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.judge import JudgeCase, Verdict
from evals.rejudge import default_out
from evals.results_file import ResultsFileError, load, metrics_of, structured
from evals.run import main
from recall_services import chat, keys
from tests.evals.test_runner import Exploding, Factory, FakeJudge


class PassAll:
    def __init__(self) -> None:
        self.cases: list[JudgeCase] = []

    def judge(self, case: JudgeCase) -> Verdict:
        self.cases.append(case)
        return Verdict(True, 1.0, cost_usd=0.002)


def _run(tmp_path: Path, *argv: str) -> Path:
    out = tmp_path / "run.json"
    assert main([*argv, "--out", str(out)], factory=Factory(), judge=FakeJudge(), say=lambda _line: None) == 0
    return out


def test_the_results_file_holds_json_objects_and_lists(tmp_path: Path) -> None:
    body = json.loads(_run(tmp_path, "--kinds", "count", "--limit", "2").read_text(encoding="utf-8"))
    for row in body["results"]:
        assert isinstance(row["metrics"], dict) and isinstance(row["verdict"], dict)
        assert isinstance(row["cited_guids"], list) and isinstance(row["key_facts"], list)
        assert row["metrics"]["citation_recall"] is None  # a count answer cites no single message
    assert body["summary"]["count"]["mean_recall"] is None


def test_rejudge_grades_the_stored_answers_with_no_chat_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _run(tmp_path, "--kinds", "fact", "--limit", "3")
    monkeypatch.setattr(chat, "default_llm", Exploding())
    judge = PassAll()
    said: list[str] = []
    assert main(["--rejudge", str(source)], factory=Exploding(), judge=judge, say=said.append) == 0
    target = default_out(source)
    assert target.name == "run-rejudged.json"
    body = json.loads(target.read_text(encoding="utf-8"))
    before = json.loads(source.read_text(encoding="utf-8"))
    assert len(judge.cases) == 3
    assert [case.answer for case in judge.cases] == [row["answer"] for row in before["results"]]
    assert body["summary"]["all"]["correct"] == 3 and body["meta"]["rejudged_from"] == str(source)
    assert [row["metrics"]["chat_usd"] for row in body["results"]] == [row["metrics"]["chat_usd"] for row in before["results"]]
    assert all(row["metrics"]["judge_usd"] == 0.002 for row in body["results"])
    assert said[-1] == f"results: {target}" and any(line.startswith("kind") for line in said)


def test_rejudge_reads_an_older_file_with_repr_values(tmp_path: Path) -> None:
    source = _run(tmp_path, "--kinds", "count", "--limit", "1")
    body = json.loads(source.read_text(encoding="utf-8"))
    row = body["results"][0]
    old = {**row, "metrics": repr({**row["metrics"], "citation_recall": 0.0}), "cited_guids": repr(row["cited_guids"])}
    older = tmp_path / "older.json"
    older.write_text(json.dumps({**body, "results": [old]}), encoding="utf-8")
    out = tmp_path / "again.json"
    assert main(["--rejudge", str(older), "--out", str(out)], judge=PassAll(), say=lambda _line: None) == 0
    again = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    assert isinstance(again["metrics"], dict) and again["metrics"]["citation_recall"] is None


def test_a_row_that_ended_in_a_chat_error_is_not_judged_again(tmp_path: Path) -> None:
    source = _run(tmp_path, "--kinds", "fact", "--limit", "1")
    body = json.loads(source.read_text(encoding="utf-8"))
    row = body["results"][0]
    failed = {**row, "metrics": {**row["metrics"], "error": "timeout", "correct": False}}
    source.write_text(json.dumps({**body, "results": [failed]}), encoding="utf-8")
    judge = PassAll()
    assert main(["--rejudge", str(source)], judge=judge, say=lambda _line: None) == 0
    assert judge.cases == []


@pytest.mark.parametrize("text", ["not json", '{"results": 3}', '{"results": [{"id": "x"}]}'])
def test_a_results_file_that_can_not_be_used_is_one_line(tmp_path: Path, text: str) -> None:
    source = tmp_path / "bad.json"
    source.write_text(text, encoding="utf-8")
    said: list[str] = []
    assert main(["--rejudge", str(source)], judge=Exploding(), say=said.append) == 2
    assert len(said) == 1
    with pytest.raises(ResultsFileError):
        load(source)


def test_rejudge_with_no_key_stops_before_any_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _run(tmp_path, "--kinds", "fact", "--limit", "1")
    monkeypatch.setattr(keys, "api_key", lambda: "")
    monkeypatch.setattr(chat, "default_llm", Exploding())
    said: list[str] = []
    assert main(["--rejudge", str(source)], say=said.append) == 2
    assert keys.NO_KEY in said[0]


def test_structured_reads_json_and_python_reprs() -> None:
    assert structured('{"a": [1, 2]}') == {"a": [1, 2]}
    assert structured("{'a': True, 'b': None}") == {"a": True, "b": None}
    assert structured([1]) == [1]
    with pytest.raises(ResultsFileError):
        structured("__import__('os')")
    with pytest.raises(ResultsFileError):
        metrics_of({"id": "x", "metrics": {"correct": True}})


def test_repeat_asks_each_question_n_times_and_reports_the_spread(tmp_path: Path) -> None:
    out = tmp_path / "rounds.json"
    said: list[str] = []
    argv = ["--kinds", "fact", "--limit", "2", "--repeat", "3", "--out", str(out)]
    assert main(argv, factory=Factory(), judge=FakeJudge(), say=said.append) == 0
    body: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert [row["round"] for row in body["results"]] == [0, 0, 1, 1, 2, 2]
    assert body["meta"]["repeat"] == 3
    assert len(body["spread"]["fact"]["rounds"]) == 3
    assert body["spread"]["all"]["min"] <= body["spread"]["all"]["mean"] <= body["spread"]["all"]["max"]
    assert any(line.startswith("kind") and "rounds" in line for line in said)


def test_one_round_has_no_spread(tmp_path: Path) -> None:
    body = json.loads(_run(tmp_path, "--kinds", "fact", "--limit", "1").read_text(encoding="utf-8"))
    assert "spread" not in body


def test_the_dry_run_counts_every_round(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "default_llm", Exploding())
    once: list[str] = []
    thrice: list[str] = []
    assert main(["--dry-run", "--limit", "2"], factory=Exploding(), judge=Exploding(), say=once.append) == 0
    assert main(["--dry-run", "--limit", "2", "--repeat", "3"], factory=Exploding(), judge=Exploding(),
                say=thrice.append) == 0  # fmt: skip

    def total(lines: list[str]) -> float:
        return float(lines[0].split("total: $")[1].split()[0])

    assert total(["\n".join(thrice)]) == pytest.approx(3 * total(["\n".join(once)]), rel=1e-3)
    assert "x 3 rounds" in "\n".join(thrice)


@pytest.mark.parametrize("argv", [["--repeat", "0"]])
def test_a_bad_repeat_is_refused(argv: list[str]) -> None:
    said: list[str] = []
    assert main(argv, factory=Exploding(), judge=Exploding(), say=said.append) == 2
