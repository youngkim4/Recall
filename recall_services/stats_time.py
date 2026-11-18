"""The time rows of /api/stats: every bucket, empty ones as 0, in time order.

The keys are those of `Archive.count`: day `YYYY-MM-DD`, week the Monday that
starts it, month `YYYY-MM`, year `YYYY`, weekday `0` (Monday) to `6`, hour
`00` to `23`, weekday_hour `W-HH`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

RANGED = ("day", "week", "month", "year")
TIME_GROUPS = (*RANGED, "weekday", "hour", "weekday_hour")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
HOURS = 24
MONTHS = 12
COUNTS = ("total", "sent", "received")


@dataclass(frozen=True)
class Span:
    """Local days, both included."""

    first: date
    last: date


def _days(span: Span, step: int) -> Iterator[date]:
    day = span.first
    while day <= span.last:
        yield day
        day += timedelta(days=step)


def _months(span: Span) -> Iterator[str]:
    number = span.first.year * MONTHS + span.first.month - 1
    while number <= span.last.year * MONTHS + span.last.month - 1:
        yield f"{number // MONTHS:04d}-{number % MONTHS + 1:02d}"
        number += 1


def _ranged_keys(by: str, span: Span) -> list[str]:
    if by == "day":
        return [day.isoformat() for day in _days(span, 1)]
    if by == "week":
        monday = span.first - timedelta(days=span.first.weekday())
        return [day.isoformat() for day in _days(Span(monday, span.last), 7)]
    if by == "month":
        return list(_months(span))
    return [f"{year:04d}" for year in range(span.first.year, span.last.year + 1)]


def _fixed_keys(by: str) -> list[tuple[str, str]]:
    """(key, label) of every bucket of weekday, hour and weekday_hour."""
    if by == "weekday":
        return [(str(number), name) for number, name in enumerate(WEEKDAYS)]
    if by == "hour":
        return [(f"{hour:02d}", f"{hour:02d}:00") for hour in range(HOURS)]
    return [
        (f"{number}-{hour:02d}", f"{name} {hour:02d}:00") for number, name in enumerate(WEEKDAYS) for hour in range(HOURS)
    ]


def time_rows(by: str, counted: Sequence[Mapping[str, Any]], span: Span | None) -> list[dict[str, Any]]:
    """Every bucket of `by` with its count. A ranged grouping with no span (nothing to cover) has no row."""
    if by in RANGED:
        buckets = [] if span is None or span.first > span.last else [(key, key) for key in _ranged_keys(by, span)]
    else:
        buckets = _fixed_keys(by)
    found = {str(row["key"]): row for row in counted}
    return [{"key": key, "label": label, **_sums(found.get(key))} for key, label in buckets]


def _sums(row: Mapping[str, Any] | None) -> dict[str, int]:
    """The three counts of a row as the archive gives them; an empty bucket is 0."""
    return {name: int(row[name]) if row else 0 for name in COUNTS}
