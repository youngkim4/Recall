"""Score one result of the engine against what the question expects.

The checks look for facts in the text of the answer. They prove that the
right names, numbers and days are there. They do not prove which number the
answer gives to which side ("you sent 630" against "you sent 686" with both
numbers in the answer): read the answers of a real run before you trust it.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Mapping, Sequence

from ..labeler import MaskingLabeler
from ..textnorm import fold
from .questions import Question

_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_WORD_START = r"(?<![^\W_])"
_NUMBER_START = r"(?<![^\W_])(?<!\d[.,])"
_NUMBER_END = r"(?!\d|[.,]\d)"
_ORDINAL = r"(?:st|nd|rd|th)?"
_LONGER_SHORT_FORMS = {9: "sept"}
_NOT_FOUND_MARKERS = (
    "no message", "not find", "n't find", "unable to find", "found nothing", "nothing about", "nothing on",
    "no mention", "not mention", "n't mention", "does not hold", "doesn't hold", "not in the archive",
    "no record", "not found", "no evidence", "does not contain", "doesn't contain", "does not say",
    "doesn't say", "not appear", "n't appear", "no match", "no result",
)


def plain(text: str) -> str:
    """Lower case, no diacritics, no thousands separators, single spaces."""
    return " ".join(fold(_THOUSANDS_RE.sub("", str(text or ""))).split())


def holds(answer: str, item: str) -> bool:
    """True when the plain `answer` has `item` at the start of a word.

    An item that ends in a digit must be the whole number: "12" is not in
    "312", "120" or "12.5". An item that ends in a letter may go on: "scan"
    is in "scanned".
    """
    wanted = plain(item)
    if not wanted:
        return False
    start = _NUMBER_START if wanted[0].isdigit() else (_WORD_START if wanted[0].isalnum() else "")
    end = _NUMBER_END if wanted[-1].isdigit() else ""
    return re.search(start + re.escape(wanted) + end, answer) is not None


def says_day(answer: str, day: str) -> bool:
    """True when the plain `answer` gives the day "YYYY-MM-DD" with its year.

    "2021-03-04", "March 4, 2021", "4th of March 2021" and "Mar 4 2021" count.
    A day without its year does not: the archive holds many years.
    """
    value = date.fromisoformat(day)
    month = value.strftime("%B").lower()
    names = {month, month[:3], _LONGER_SHORT_FORMS.get(value.month, month[:3])}
    name = rf"(?:{'|'.join(sorted(names, key=len, reverse=True))})\.?"
    number = rf"0?{value.day}{_ORDINAL}"
    year = rf"{value.year}(?!\d)"
    patterns = (
        rf"(?<!\d){re.escape(day)}(?!\d)",
        rf"{_WORD_START}{name}\s+{number}(?!\d)\s*,?\s*{year}",
        rf"(?<!\d){number}\s+(?:of\s+)?{name}\s*,?\s*{year}",
    )
    return any(re.search(pattern, answer) for pattern in patterns)


def _no_handle(answer: str, forbidden: Sequence[str]) -> bool:
    """True when the answer holds none of the handles, in any spelling."""
    if not forbidden:
        return True
    return MaskingLabeler(known_handles=forbidden).redact(answer) == answer


def score(result: Mapping[str, Any], question: Question, forbidden: Sequence[str] = ()) -> dict[str, bool]:
    """One named check per expectation. The question passes when all are True.

    A result that misses a key of the contract raises KeyError: a missing
    key must not read as a passed check.
    """
    expect = question.expect
    raw = str(result["answer"] or "")
    answer = plain(raw)
    cited = {str(citation["id"]) for citation in result["citations"]}
    checks = {
        "answered": bool(answer) and not result["cancelled"],
        "quotes_verified": not result["unverified_quotes"],
        "references_valid": not result["invalid_references"],
        "no_raw_handle": _no_handle(raw, forbidden),
    }
    if expect.contains:
        checks["contains"] = all(holds(answer, item) for item in expect.contains)
    if expect.any_of:
        checks["any_of"] = all(any(holds(answer, item) for item in group) for group in expect.any_of)
    if expect.days:
        checks["days"] = all(says_day(answer, day) for day in expect.days)
    if expect.cites:
        checks["cites"] = bool(cited & set(expect.cites))
    if expect.absent:
        checks["absent"] = not any(holds(answer, item) for item in expect.absent)
    if expect.not_found:
        checks["says_not_found"] = any(marker in answer for marker in _NOT_FOUND_MARKERS)
        checks["cites_nothing"] = not cited
    return checks


def passed(checks: Mapping[str, bool]) -> bool:
    return all(checks.values())
