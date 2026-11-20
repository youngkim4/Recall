"""Read a results file back: for --rejudge, and to compare runs.

A value that should be an object or a list may be text in a file a tool
rewrote: the JSON of it, or the repr of a Python dict. Both are read
(`json.loads`, then `ast.literal_eval`, which runs no code). A row that
still can not be read is a ResultsFileError with its place.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from evals.score import Metrics

_METRIC_FIELDS = {field.name for field in fields(Metrics)}


class ResultsFileError(ValueError):
    """A results file that can not be used. The message says where."""


def structured(value: object) -> object:
    """An object or a list as it was meant: text holding JSON or a Python repr is read."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ResultsFileError(f"a value is neither JSON nor a Python literal: {value[:60]!r}") from error


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    found = structured(row.get(key))
    if not isinstance(found, Mapping):
        raise ResultsFileError(f"row {row.get('id', '?')}: {key} is not an object")
    return found


def metrics_of(row: Mapping[str, Any]) -> Metrics:
    """The metrics of one row of a results file."""
    stored = _mapping(row, "metrics")
    try:
        return Metrics(**{name: stored[name] for name in _METRIC_FIELDS if name in stored})
    except TypeError as error:
        raise ResultsFileError(f"row {row.get('id', '?')}: the metrics miss a field") from error


def _row(raw: object, place: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ResultsFileError(f"result {place} is not an object")
    row = dict(raw)
    for key in ("id", "kind", "question", "answer", "gold"):
        if not isinstance(row.get(key), str):
            raise ResultsFileError(f"result {place} has no text {key}")
    lists = {key: structured(row.get(key) or []) for key in ("key_facts", "gold_guids", "cited_guids")}
    if not all(isinstance(value, list) for value in lists.values()):
        raise ResultsFileError(f"result {place}: key_facts, gold_guids and cited_guids must be lists")
    return {
        **row,
        **lists,
        "prior": str(row.get("prior") or ""),
        "round": int(row.get("round") or 0),
        "verdict": dict(_mapping(row, "verdict")),
        "metrics": dict(_mapping(row, "metrics")),
    }


def load(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """The meta and the result rows of a results file."""
    try:
        body = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except OSError as error:
        raise ResultsFileError(f"the results file can not be read: {type(error).__name__}") from error
    except ValueError as error:
        raise ResultsFileError("the results file is not JSON") from error
    if not isinstance(body, Mapping) or not isinstance(body.get("results"), list):
        raise ResultsFileError("the results file has no list of results")
    meta = structured(body.get("meta") or {})
    if not isinstance(meta, Mapping):
        raise ResultsFileError("the meta of the results file is not an object")
    return dict(meta), tuple(_row(raw, place) for place, raw in enumerate(body["results"], start=1))
