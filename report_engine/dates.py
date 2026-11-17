"""Day and timestamp strings. The engine works in local time with no zone."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

DAY_FORMAT = "%Y-%m-%d"
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DAY_LENGTH = 10
STAMP_LENGTH = 19
DAY_START = " 00:00:00"
DAY_END = " 23:59:59"

_STAMP_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2} ([01]\d|2[0-3]):[0-5]\d:[0-5]\d", re.ASCII)


def parse_day(value: str) -> date:
    """Parse YYYY-MM-DD in plain digits. Raises ValueError for anything else."""
    if not isinstance(value, str) or len(value) != DAY_LENGTH or not value.isascii():
        raise ValueError(f"not a day in YYYY-MM-DD form: {value!r}")
    return datetime.strptime(value, DAY_FORMAT).date()


def parse_stamp(value: str) -> datetime:
    """Parse YYYY-MM-DD HH:MM:SS in plain digits. Raises ValueError for anything else."""
    if not isinstance(value, str) or len(value) != STAMP_LENGTH or not value.isascii():
        raise ValueError(f"not a timestamp in YYYY-MM-DD HH:MM:SS form: {value!r}")
    return datetime.strptime(value, STAMP_FORMAT)


def is_day(value: object) -> bool:
    try:
        parse_day(value)  # type: ignore[arg-type]
    except ValueError:
        return False
    return True


def has_stamp_shape(value: object) -> bool:
    """True for text shaped like YYYY-MM-DD HH:MM:SS. The day is not checked."""
    return isinstance(value, str) and bool(_STAMP_SHAPE.fullmatch(value))


def is_stamp(value: object) -> bool:
    """True for a real YYYY-MM-DD HH:MM:SS. Timestamps of this shape compare as text."""
    return has_stamp_shape(value) and is_day(value[:DAY_LENGTH])  # type: ignore[index]


def shift_day(day: str, days: int) -> str:
    """The day `days` later. Raises ValueError when that day is outside the calendar."""
    try:
        return (parse_day(day) + timedelta(days=days)).strftime(DAY_FORMAT)
    except OverflowError as exc:
        raise ValueError(f"no day {days} days from {day}") from exc


def days_between(first: str, second: str) -> int:
    """Whole days from `first` to `second`. Negative when `second` is earlier."""
    return (parse_day(second) - parse_day(first)).days


def lower_bound(since: str | None) -> str | None:
    """A day or a timestamp as the first timestamp it includes."""
    if since is None:
        return None
    return since + DAY_START if len(since) == DAY_LENGTH else since


def upper_bound(until: str | None) -> str | None:
    """A day or a timestamp as the last timestamp it includes."""
    if until is None:
        return None
    return until + DAY_END if len(until) == DAY_LENGTH else until
