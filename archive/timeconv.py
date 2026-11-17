"""Time rules of the archive.

Stored time is UTC. Local time is the time zone of the process, and it is
only made at the edge: when a row is shown or when a day is turned into a range.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta

APPLE_EPOCH_UNIX = 978_307_200  # 2001-01-01T00:00:00Z
NS_PER_SECOND = 1_000_000_000
# Apple writes seconds (old rows) or nanoseconds since 2001. A value this large
# cannot be seconds: it is year 5170.
NANOSECOND_THRESHOLD = 10**11
MAX_UNIX_NS = 7_258_118_400 * NS_PER_SECOND  # 2200-01-01, far below the 64-bit limit

LOCAL_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def apple_to_unix_ns(value: object) -> int | None:
    """Apple time to UTC unix nanoseconds. None when the value is not a usable date."""
    if value is None or isinstance(value, (bytes, str)) and not str(value).strip():
        return None
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if number >= NANOSECOND_THRESHOLD:
        unix_ns = number + APPLE_EPOCH_UNIX * NS_PER_SECOND
    else:
        unix_ns = (number + APPLE_EPOCH_UNIX) * NS_PER_SECOND
    return unix_ns if unix_ns <= MAX_UNIX_NS else None


def apple_to_unix_seconds(value: object) -> int | None:
    unix_ns = apple_to_unix_ns(value)
    return None if unix_ns is None else unix_ns // NS_PER_SECOND


def parse_day(day: str) -> date:
    """A local day written as YYYY-MM-DD. Anything else is a ValueError."""
    if not isinstance(day, str) or _DAY.fullmatch(day) is None:
        raise ValueError(f"a day must be written as YYYY-MM-DD, got {day!r}")
    return date.fromisoformat(day)


def _local_midnight(day: date) -> int:
    return int(datetime(day.year, day.month, day.day).timestamp())


def day_start(day: str) -> int:
    """UTC unix seconds of the first second of a local day."""
    return _local_midnight(parse_day(day))


def day_end(day: str) -> int:
    """UTC unix seconds of the first second after a local day."""
    return _local_midnight(parse_day(day) + timedelta(days=1))


def local_timestamp(unix_seconds: int) -> str:
    return time.strftime(LOCAL_TIMESTAMP_FORMAT, time.localtime(unix_seconds))


def sql_local(pattern: str, seconds_sql: str) -> str:
    """SQL text that formats UTC unix seconds as local time. Same result as local_timestamp."""
    return f"strftime('{pattern}', {seconds_sql}, 'unixepoch', 'localtime')"


def sql_local_timestamp(seconds_sql: str) -> str:
    return sql_local(LOCAL_TIMESTAMP_FORMAT, seconds_sql)
