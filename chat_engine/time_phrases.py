"""Words that name a time, and the days they mean.

Each finder reads text that has no handle left in it, and gives the spans it
read with their first and last day. Named times ("March 2024", "fall 2022")
need no clock. Relative times ("last summer", "two years ago") are worked out
from today.

Seasons follow the northern meteorological calendar: spring is March to May,
summer June to August, fall (or autumn) September to November. Winter 2021
runs from 1 December 2021 to the end of February 2022. "Early 2023" is
January to April; "late 2023" is September to December.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable, Iterator, Optional

RECENT_DAYS = 90  # "recently" and "lately"
_MAX_MONTHS_BACK = 24
_MAX_YEARS_BACK = 30

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_SHORT_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_SEASONS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}  # the first month of each
_PARTS = {"early": (1, 4), "late": (9, 12)}
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _either(words: Iterable[str]) -> str:
    """One regex choice of the words. A longer word is tried first: "sept" before "sep"."""
    return "|".join(sorted(words, key=len, reverse=True))


_YEAR = r"(?P<year>199[0-9]|20[0-9]{2})(?![0-9])"
_COUNT = rf"(?P<count>[0-9]{{1,2}}|{_either(_NUMBERS)})"
_FLAGS = re.IGNORECASE | re.ASCII

_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<month>{_either([*_MONTHS, *_SHORT_MONTHS])})\b\.?,?\s+(?:of\s+)?{_YEAR}", _FLAGS
)
_SEASON_YEAR_RE = re.compile(rf"\b(?P<season>{_either(_SEASONS)})\s+(?:of\s+)?{_YEAR}", _FLAGS)
_PART_YEAR_RE = re.compile(rf"\b(?P<part>{_either(_PARTS)})\s+(?:in\s+)?{_YEAR}", _FLAGS)

_LAST_MONTHS_RE = re.compile(
    rf"\b(?:(?:in|over|during)\s+)?(?:the\s+)?(?:last|past)\s+{_COUNT}\s+months?\b", _FLAGS
)
_YEARS_AGO_RE = re.compile(rf"\b{_COUNT}\s+years?\s+ago\b", _FLAGS)
_LAST_SEASON_RE = re.compile(rf"\blast\s+(?P<season>{_either(_SEASONS)})\b", _FLAGS)
_MONTH_ALONE_RE = re.compile(rf"\b(?:in|during|around|last)\s+(?P<month>{_either(_MONTHS)})\b", _FLAGS)
_LAST_YEAR_RE = re.compile(r"\blast\s+year\b", _FLAGS)
_THIS_YEAR_RE = re.compile(r"\bthis\s+year\b", _FLAGS)
_LAST_MONTH_RE = re.compile(r"\blast\s+month\b", _FLAGS)
_LAST_WEEK_RE = re.compile(r"\blast\s+week\b", _FLAGS)
_RECENTLY_RE = re.compile(r"\b(?:recently|lately)\b", _FLAGS)


@dataclass(frozen=True)
class Found:
    """One span of the text that names a time, and its first and last day."""

    start: int
    end: int
    first: date
    last: date


def month_days(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _months_days(year: int, first_month: int, last_month: int) -> tuple[date, date]:
    """From the first day of one month to the last day of a later one. A month past 12 is in the next year."""
    end_year, end_month = divmod(year * 12 + last_month - 1, 12)
    return date(year, first_month, 1), month_days(end_year, end_month + 1)[1]


def season_days(year: int, season: str) -> tuple[date, date]:
    first = _SEASONS[season.lower()]
    return _months_days(year, first, first + 2)


def _month_of(word: str) -> int:
    key = word.lower()
    return _MONTHS.get(key) or _SHORT_MONTHS[key]


def _count(word: str) -> int:
    key = word.lower()
    return _NUMBERS[key] if key in _NUMBERS else int(key)


def _months_back(today: date, months: int) -> date:
    year, month = divmod(today.year * 12 + today.month - 1 - months, 12)
    return date(year, month + 1, min(today.day, calendar.monthrange(year, month + 1)[1]))


Reader = Callable[["re.Match[str]"], Optional[tuple[date, date]]]


def _scan(text: str, pattern: "re.Pattern[str]", read: Reader) -> Iterator[Found]:
    for match in pattern.finditer(text):
        days = read(match)
        if days is not None:
            yield Found(match.start(), match.end(), *days)


def named_times(text: str) -> list[Found]:
    """Months, seasons and parts of a year written with their year."""

    def month(match: "re.Match[str]") -> tuple[date, date]:
        return month_days(int(match["year"]), _month_of(match["month"]))

    def season(match: "re.Match[str]") -> tuple[date, date]:
        return season_days(int(match["year"]), match["season"])

    def part(match: "re.Match[str]") -> tuple[date, date]:
        return _months_days(int(match["year"]), *_PARTS[match["part"].lower()])

    found = [
        *_scan(text, _MONTH_YEAR_RE, month),
        *_scan(text, _SEASON_YEAR_RE, season),
        *_scan(text, _PART_YEAR_RE, part),
    ]
    return sorted(found, key=lambda one: one.start)


def relative_times(text: str, today: date) -> list[Found]:
    """Times worked out from today. A span read once is not read again by a shorter phrase."""
    readers: list[tuple["re.Pattern[str]", Reader]] = [
        (_LAST_MONTHS_RE, lambda m: _last_months(today, _count(m["count"]))),
        (_YEARS_AGO_RE, lambda m: _years_ago(today, _count(m["count"]))),
        (_LAST_SEASON_RE, lambda m: _last_season(today, m["season"])),
        (_MONTH_ALONE_RE, lambda m: _latest_month(today, _MONTHS[m["month"].lower()])),
        (_LAST_YEAR_RE, lambda m: _years_ago(today, 1)),
        (_THIS_YEAR_RE, lambda m: (date(today.year, 1, 1), today)),
        (_LAST_MONTH_RE, lambda m: month_days(*_year_month(_months_back(today, 1)))),
        (_LAST_WEEK_RE, lambda m: _last_week(today)),
        (_RECENTLY_RE, lambda m: (today - timedelta(days=RECENT_DAYS), today)),
    ]
    found: list[Found] = []
    for pattern, read in readers:
        for one in _scan(text, pattern, read):
            if not any(one.start < known.end and known.start < one.end for known in found):
                found.append(one)
    return sorted(found, key=lambda one: one.start)


def _year_month(day: date) -> tuple[int, int]:
    return day.year, day.month


def _last_months(today: date, months: int) -> Optional[tuple[date, date]]:
    if not 1 <= months <= _MAX_MONTHS_BACK:
        return None
    return _months_back(today, months), today


def _years_ago(today: date, years: int) -> Optional[tuple[date, date]]:
    if not 1 <= years <= _MAX_YEARS_BACK:
        return None
    return date(today.year - years, 1, 1), date(today.year - years, 12, 31)


def _last_season(today: date, season: str) -> tuple[date, date]:
    """The latest such season that ended before today."""
    year = today.year
    while season_days(year, season)[1] >= today:
        year -= 1
    return season_days(year, season)


def _latest_month(today: date, month: int) -> tuple[date, date]:
    """The latest such month that began on or before today: this month counts."""
    return month_days(today.year if month <= today.month else today.year - 1, month)


def _last_week(today: date) -> tuple[date, date]:
    """Monday to Sunday of the week before this one."""
    monday = today - timedelta(days=today.weekday())
    return monday - timedelta(days=7), monday - timedelta(days=1)
