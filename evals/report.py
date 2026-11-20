"""The results file of a run: what was asked, what came back, the scores and the money.

Everything in it is invented: the world holds no real message. The file is
JSON with three keys: `meta` (the settings of the run), `summary` (per kind
and "all") and `results` (one row for each question).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.question_types import KINDS
from evals.runner import Result
from evals.score import Summary, by_kind

FORMAT = 1


def result_row(result: Result) -> dict[str, Any]:
    question, verdict = result.question, result.verdict
    return {
        "id": question.id,
        "kind": question.kind,
        "question": question.text,
        "prior": question.prior,
        "prior_answer": result.prior_answer,
        "answer": result.answer,
        "gold": question.gold,
        "key_facts": list(question.key_facts),
        "gold_guids": list(question.gold_guids),
        "cited_guids": list(result.cited),
        "verdict": {
            "correct": verdict.correct,
            "score": verdict.score,
            "missing": list(verdict.missing),
            "wrong": list(verdict.wrong),
            "error": verdict.error,
        },
        "metrics": {**asdict(result.metrics), "usd": result.metrics.usd},
    }


def summaries(results: Sequence[Result]) -> dict[str, Summary]:
    return by_kind(KINDS, [(result.question.kind, result.metrics) for result in results])


def summary_rows(found: Mapping[str, Summary]) -> dict[str, Any]:
    return {kind: {**asdict(summary), "accuracy": summary.accuracy} for kind, summary in found.items()}


def document(meta: Mapping[str, Any], results: Sequence[Result]) -> dict[str, Any]:
    return {
        "meta": {"format": FORMAT, **meta},
        "summary": summary_rows(summaries(results)),
        "results": [result_row(result) for result in results],
    }


def write(path: Path, body: Mapping[str, Any]) -> Path:
    """Write the JSON through a temp file and a rename, so a half file is never left."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".chat-eval-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(body, file, ensure_ascii=False, indent=1)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return target
