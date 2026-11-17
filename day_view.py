"""One day of a conversation: the messages of that day, the part of a long
day that is shown, and the nearest days with messages on either side."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# how many messages of a day are given at once, and the most that can be asked for
DAY_LIMIT = 200
DAY_LIMIT_MAX = 400
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class DayWindow:
    day: str
    rows: pd.DataFrame
    # how many messages the day holds, and where among them the rows start
    total: int
    offset: int
    previous_day: str | None
    next_day: str | None


def _start_of(day: str) -> pd.Timestamp:
    """The start of the day that "YYYY-MM-DD" names. Anything else is refused."""
    text = str(day or "")
    if not _DAY.fullmatch(text):
        raise ValueError(f"Not a day: {text!r}")
    try:
        return pd.Timestamp(text)
    except ValueError as error:
        raise ValueError(f"Not a day: {text!r}") from error


def _day_of(stamp) -> str | None:
    return None if pd.isna(stamp) else str(stamp)[:10]


def _size_of(limit: int) -> int:
    """How many messages one answer holds. Fewer than one is not a part of a
    day; more than the largest part is the largest part."""
    size = int(limit)
    if size < 1:
        raise ValueError(f"Not a number of messages: {limit!r}")
    return min(size, DAY_LIMIT_MAX)


def _before(stamps: pd.Series, around: str) -> int:
    """How many of the stamps come before the time in `around`. A zone in it
    is dropped: every time in an export is the time on the clock."""
    at = pd.to_datetime(around, errors="coerce")
    if pd.isna(at):
        return 0
    if at.tzinfo is not None:
        at = at.tz_localize(None)
    return int((stamps < at).sum())


def day_window(
    conv: pd.DataFrame,
    day: str,
    around: str | None = None,
    limit: int = DAY_LIMIT,
    offset: int | None = None,
) -> DayWindow:
    """The messages of one day, oldest first. Messages sent at the same time
    keep the order they have in `conv`; one with no readable time is in no day.

    A day longer than the limit is given in part. With `offset`, the part
    starts at that place and is not moved: a reader who asks for what follows
    the rows they hold gets those rows and no others, and a place past the end
    gives none. Without it, the part is a whole window around the time in
    `around`, or from the start of the day, and never runs past either end."""
    start = _start_of(day)
    end = start + pd.Timedelta(days=1)
    size = _size_of(limit)
    ordered = conv.sort_values("timestamp", kind="stable")
    stamps = ordered["timestamp"]
    of_day = ordered[(stamps >= start) & (stamps < end)]
    total = int(len(of_day))

    if offset is not None:
        begin = max(0, min(int(offset), total))
    else:
        wanted = _before(of_day["timestamp"], around) - size // 2 if around else 0
        begin = max(0, min(wanted, total - size))

    return DayWindow(
        day=str(start.date()),
        rows=of_day.iloc[begin : begin + size],
        total=total,
        offset=begin,
        previous_day=_day_of(stamps[stamps < start].max()),
        next_day=_day_of(stamps[stamps >= end].min()),
    )
