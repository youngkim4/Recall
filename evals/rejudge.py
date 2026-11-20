"""--rejudge: grade the stored answers of a results file again, with the current rubric. No chat call.

Each row keeps its answer, citations, tokens and chat cost. The judge runs
again on each answer, and the citation scores are counted again with the
current rules (a count question has none). A row whose question ended in a
chat error has no answer: it is kept as it is and not judged. The judge cost
of a row is the cost of the new judge call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from evals.judge import Judge, JudgeCase
from evals.report import metrics_row, verdict_row
from evals.results_file import metrics_of
from evals.runner import JUDGE_ERROR
from evals.score import citation_scores

SUFFIX = "-rejudged"


def default_out(source: Path) -> Path:
    """`run.json` gives `run-rejudged.json`, in the same folder."""
    return source.with_name(f"{source.stem}{SUFFIX}{source.suffix or '.json'}")


def _chat_failed(error: str) -> bool:
    return bool(error) and not error.startswith(JUDGE_ERROR)


def rejudge_row(row: Mapping[str, Any], judge: Judge) -> dict[str, Any]:
    metrics = metrics_of(row)
    recall, precision = citation_scores(str(row["kind"]), row["gold_guids"], row["cited_guids"])
    metrics = replace(metrics, citation_recall=recall, citation_precision=precision)
    if _chat_failed(metrics.error):
        return {**row, "metrics": metrics_row(metrics)}
    case = JudgeCase(
        str(row["question"]), str(row["kind"]), str(row["gold"]), tuple(str(fact) for fact in row["key_facts"]),
        str(row["answer"]), str(row.get("prior") or ""),
    )  # fmt: skip
    verdict = judge.judge(case)
    judged = replace(
        metrics, correct=verdict.correct, score=verdict.score, judge_usd=verdict.cost_usd,
        error=f"{JUDGE_ERROR}{verdict.error}" if verdict.error else "",
    )  # fmt: skip
    return {
        **row,
        "verdict": verdict_row(verdict.correct, verdict.score, verdict.missing, verdict.wrong, verdict.error),
        "metrics": metrics_row(judged),
    }


def rejudge_rows(
    rows: tuple[Mapping[str, Any], ...],
    judge: Judge,
    on_row: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    done: list[dict[str, Any]] = []
    for row in rows:
        new = rejudge_row(row, judge)
        if on_row is not None:
            on_row(new)
        done.append(new)
    return tuple(done)
