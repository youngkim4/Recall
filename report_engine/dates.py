"""Day and timestamp strings. The engine works in local time with no zone."""

from __future__ import annotations

from datetime import date, datetime, timedelta

DAY_FORMAT = "%Y-%m-%d"
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DAY_LENGTH = 10
STAMP_LENGTH = 19


def parse_day(value: str) -> date:
    """Parse YYYY-MM-DD. Raises ValueError for anything else."""
    if not isinstance(value, str) or len(value) != DAY_LENGTH:
        raise ValueError(f"not a day in YYYY-MM-DD form: {value!r}")
    return datetime.strptime(value, DAY_FORMAT).date()


def parse_stamp(value: str) -> datetime:
    """Parse YYYY-MM-DD HH:MM:SS. Raises ValueError for anything else."""
    if not isinstance(value, str) or len(value) != STAMP_LENGTH:
        raise ValueError(f"not a timestamp in YYYY-MM-DD HH:MM:SS form: {value!r}")
    return datetime.strptime(value, STAMP_FORMAT)


def is_day(value: object) -> bool:
    try:
        parse_day(value)  # type: ignore[arg-type]
    except ValueError:
        return False
    return True


def shift_day(day: str, days: int) -> str:
    return (parse_day(day) + timedelta(days=days)).strftime(DAY_FORMAT)


def days_between(first: str, second: str) -> int:
    """Whole days from `first` to `second`. Negative when `second` is earlier."""
    return (parse_day(second) - parse_day(first)).days
