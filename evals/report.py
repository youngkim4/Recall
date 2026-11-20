"""The results file of a run: what was asked, what came back, the scores and the money.

Everything in it is invented: the world holds no real message. The file is
JSON with these keys: `meta` (the settings of the run), `summary` (per kind
and "all"), `results` (one row for each question and round) and, for a run
of more than one round, `spread` (the accuracy of each round per kind).

Every value is a JSON object, list, number or text: never the repr of a
Python object. `evals.results_file` reads a file back, also an older one.
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
from evals.results_file import metrics_of
from evals.runner import Result
from evals.score import Metrics, Spread, Summary, by_kind, spread

FORMAT = 2


def verdict_row(correct: bool, score: float, missing: Sequence[str], wrong: Sequence[str], error: str) -> dict[str, Any]:
    return {"correct": correct, "score": score, "missing": list(missing), "wrong": list(wrong), "error": error}


def metrics_row(metrics: Metrics) -> dict[str, Any]:
    return {**asdict(metrics), "usd": metrics.usd}


def result_row(result: Result) -> dict[str, Any]:
    question, verdict = result.question, result.verdict
    return {
        "id": question.id,
        "kind": question.kind,
        "round": result.round,
        "question": question.text,
        "prior": question.prior,
        "prior_answer": result.prior_answer,
        "answer": result.answer,
        "gold": question.gold,
        "key_facts": list(question.key_facts),
        "gold_guids": list(question.gold_guids),
        "cited_guids": list(result.cited),
        "verdict": verdict_row(verdict.correct, verdict.score, verdict.missing, verdict.wrong, verdict.error),
        "metrics": metrics_row(result.metrics),
    }


def _scored(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, Metrics]]:
    return [(str(row["kind"]), int(row.get("round") or 0), metrics_of(row)) for row in rows]


def summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Summary]:
    return by_kind(KINDS, [(kind, metrics) for kind, _round, metrics in _scored(rows)])


def spreads(rows: Sequence[Mapping[str, Any]]) -> dict[str, Spread]:
    return spread(KINDS, _scored(rows))


def summary_rows(found: Mapping[str, Summary]) -> dict[str, Any]:
    return {kind: {**asdict(summary), "accuracy": summary.accuracy} for kind, summary in found.items()}


def spread_rows(found: Mapping[str, Spread]) -> dict[str, Any]:
    return {
        kind: {"rounds": list(one.rounds), "mean": one.mean, "min": one.low, "max": one.high}
        for kind, one in found.items()
    }


def document(meta: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The whole file. `rows` are the rows of `result_row`, or rows read back from a file."""
    body: dict[str, Any] = {
        "meta": {**meta, "format": FORMAT},
        "summary": summary_rows(summaries(rows)),
        "results": [dict(row) for row in rows],
    }
    found = spreads(rows)
    if found and len(found["all"].rounds) > 1:
        body = {**body, "spread": spread_rows(found)}
    return body


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
