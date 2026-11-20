"""The time a question is about: its first and its last day.

A question names its time with years ("in 2021", "2019-2021"), with a month,
a season or a part of a year and its year ("March 2024", "fall 2022", "early
2023"), or relative to today ("last summer", "two years ago", "recently").
`time_phrases` says what each phrase means.

A named time wins over a relative one: "last year, I mean 2021" is 2021.
Several named times give one period from the first day to the last.

Digits of a phone number or an email address never become a date: a handle
that `redact` masks gives nothing, and an unmasked handle keeps only a date
or a year written beside it. Only a time that overlaps the years the archive
holds counts, so an amount such as 2000 is seldom taken for a year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Optional, Sequence

from .handles import DIGITS_RE, MIN_PHONE_DIGITS, YEAR_RE, is_dates
from .labeler import handle_spans
from .ports import ArchiveSummary
from .textnorm import fold, surface_words
from .time_phrases import Found, named_times, relative_times

ALL_YEARS = (1990, 2099)
_MAX_PHRASE = 60  # a longer phrase is left out of the line the model reads
_EDGES = " \t\n,.;:!?\"'()[]"


@dataclass(frozen=True)
class Period:
    """The days a question is about.

    phrase: the words of the question that name the time.
    relative: worked out from today, so it can be a wrong guess.
    words: the folded words of those phrases, kept out of a word search.
    """

    since: str
    until: str
    phrase: str
    relative: bool
    words: tuple[str, ...] = ()


def archive_years(summary: ArchiveSummary) -> tuple[int, int]:
    """The first and the last year the archive holds. With no dated message: every year."""
    first, last = (str(summary.get(key) or "")[:4] for key in ("first_timestamp", "last_timestamp"))
    if YEAR_RE.fullmatch(first) and YEAR_RE.fullmatch(last):
        return int(first), int(last)
    return ALL_YEARS


def question_time(
    question: str,
    now: datetime,
    held: tuple[int, int] = ALL_YEARS,
    redact: Optional[Callable[[str], str]] = None,
) -> Optional[Period]:
    """The period the question is about, or None when it names no time the archive holds."""
    text = without_handles(question, redact)
    every = _named(text)
    named = [one for one in every if _overlaps(one.first, one.last, held)]
    if named:
        return _period(text, named, relative=False)
    found = relative_times(_blank(text, every), now.date())  # "March 2030" is no "in March"
    first = min((one.first for one in found), default=None)
    last = max((one.last for one in found), default=None)
    if first is None or last is None or not _overlaps(first, last, held):
        return None
    return _period(text, found, relative=True)


def question_period(
    question: str, held: tuple[int, int] = ALL_YEARS, redact: Optional[Callable[[str], str]] = None
) -> tuple[Optional[str], Optional[str]]:
    """The first and the last day of the years the question names. No year: (None, None).

    Only a year the archive holds counts.
    """
    text = without_handles(question, redact)
    years = sorted(year for year in map(int, YEAR_RE.findall(text)) if held[0] <= year <= held[1])
    if not years:
        return None, None
    return f"{years[0]}-01-01", f"{years[-1]}-12-31"


def time_line(period: Optional[Period]) -> str:
    """The line of the question block that names the days. Empty when there is no period."""
    if period is None:
        return ""
    named = f" ({period.phrase})" if period.phrase else ""
    return (
        f"The question is about {period.since} to {period.until}{named}. "
        "Set since and until to these days when a lookup is about that time."
    )


def without_handles(question: str, redact: Optional[Callable[[str], str]] = None) -> str:
    """The question with each phone number and email address taken out.

    A handle that `redact` masks leaves nothing. Another keeps a date or a
    range of years whole, and a year written beside a phone number.
    """
    kept = question
    for start, end in reversed(handle_spans(question)):
        span = question[start:end]
        masked = redact is not None and redact(span) != span
        kept = f"{kept[:start]} {'' if masked else _years_beside_a_handle(span)} {kept[end:]}"
    return kept


def _named(text: str) -> list[Found]:
    """Months, seasons and parts of a year with their year, then the years left alone."""
    named = named_times(text)
    years = [
        Found(match.start(), match.end(), date(int(match[0]), 1, 1), date(int(match[0]), 12, 31))
        for match in YEAR_RE.finditer(_blank(text, named))
    ]
    return sorted([*named, *years], key=lambda one: one.start)


def _blank(text: str, found: Sequence[Found]) -> str:
    """The text with spaces in place of the spans already read. Every other span keeps its place."""
    for one in found:
        text = text[: one.start] + " " * (one.end - one.start) + text[one.end :]
    return text


def _overlaps(first: date, last: date, held: tuple[int, int]) -> bool:
    return first.year <= held[1] and last.year >= held[0]


def _period(text: str, found: Sequence[Found], *, relative: bool) -> Period:
    phrase = re.sub(r"\s+", " ", text[min(one.start for one in found) : max(one.end for one in found)])
    phrase = phrase.strip(_EDGES)
    words = tuple(dict.fromkeys(fold(word) for one in found for word in surface_words(text[one.start : one.end])))
    return Period(
        since=min(one.first for one in found).isoformat(),
        until=max(one.last for one in found).isoformat(),
        phrase=phrase if len(phrase) <= _MAX_PHRASE else "",
        relative=relative,
        words=words,
    )


def _years_beside_a_handle(span: str) -> str:
    """What a span that looks like a phone number or an email address keeps for the search of years.

    A range of years or a date looks like a phone number: it is kept whole. A
    year written beside a phone number is kept; the number is not.
    """
    if "@" in span or "+" in span:
        return ""
    if is_dates(span):
        return span
    parts = span.split()
    years: list[str] = []
    for end in (-1, 0):
        while len(parts) > 1 and YEAR_RE.fullmatch(parts[end]) and _digit_count(parts) - 4 >= MIN_PHONE_DIGITS:
            years.append(parts.pop(end))
    return " ".join(years)


def _digit_count(parts: Sequence[str]) -> int:
    return sum(len(run) for part in parts for run in DIGITS_RE.findall(part))
