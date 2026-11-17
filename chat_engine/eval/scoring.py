"""Score one result of the engine against what the question expects."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..textnorm import fold
from .questions import Question

_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NOT_FOUND_MARKERS = (
    "no message", "not find", "n't find", "no mention", "not mention", "n't mention", "nothing",
    "does not hold", "doesn't hold", "not in the archive", "no record", "not found", "could not",
    "couldn't", "unable to", "no evidence", "do not have", "don't have", "does not contain",
    "doesn't contain", "does not say", "doesn't say", "never", "not appear", "n't appear",
)


def plain(text: str) -> str:
    """Lower case, no diacritics, no thousands separators, single spaces."""
    return " ".join(fold(_THOUSANDS_RE.sub("", str(text or ""))).split())


def score(result: Mapping[str, Any], question: Question, forbidden: Sequence[str] = ()) -> dict[str, bool]:
    """One named check per expectation. The question passes when all are True."""
    expect = question.expect
    answer = plain(str(result.get("answer") or ""))
    cited = {str(citation.get("id")) for citation in result.get("citations") or []}
    checks = {
        "answered": bool(answer) and not result.get("cancelled"),
        "quotes_verified": not result.get("unverified_quotes"),
        "references_valid": not result.get("invalid_references"),
        "no_raw_handle": not any(plain(raw) in answer for raw in forbidden),
    }
    if expect.contains:
        checks["contains"] = all(plain(item) in answer for item in expect.contains)
    if expect.any_of:
        checks["any_of"] = all(any(plain(item) in answer for item in group) for group in expect.any_of)
    if expect.cites:
        checks["cites"] = bool(cited & set(expect.cites))
    if expect.absent:
        checks["absent"] = not any(plain(item) in answer for item in expect.absent)
    if expect.not_found:
        checks["says_not_found"] = any(marker in answer for marker in _NOT_FOUND_MARKERS)
    return checks


def passed(checks: Mapping[str, bool]) -> bool:
    return all(checks.values())
