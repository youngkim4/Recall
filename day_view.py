"""One day of a conversation: the messages of that day, the part of a long
day that is shown, and the nearest days with messages on either side."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# how many messages of a day are given at once
DAY_LIMIT = 200
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


def day_window(
    conv: pd.DataFrame,
    day: str,
    around: str | None = None,
    limit: int = DAY_LIMIT,
    offset: int | None = None,
) -> DayWindow:
    """The messages of one day, oldest first. A day longer than the limit is
    given in part: from `offset` when one is named, else around the time in
    `around`, else from its start. The part never runs past either end."""
    start = _start_of(day)
    end = start + pd.Timedelta(days=1)
    ordered = conv.sort_values("timestamp", kind="stable")
    stamps = ordered["timestamp"]
    of_day = ordered[(stamps >= start) & (stamps < end)]
    total = int(len(of_day))
    size = max(1, int(limit))

    if offset is not None:
        begin = int(offset)
    elif around:
        at = pd.to_datetime(around, errors="coerce")
        before_it = 0 if pd.isna(at) else int((of_day["timestamp"] < at).sum())
        begin = before_it - size // 2
    else:
        begin = 0
    begin = max(0, min(begin, total - size))

    return DayWindow(
        day=str(start.date()),
        rows=of_day.iloc[begin : begin + size],
        total=total,
        offset=begin,
        previous_day=_day_of(stamps[stamps < start].max()),
        next_day=_day_of(stamps[stamps >= end].min()),
    )
