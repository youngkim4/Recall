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
    if value is None or (isinstance(value, (bytes, str)) and not str(value).strip()):
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


# -- local time by arithmetic ------------------------------------------------
#
# Grouping a million rows by local day with strftime(..., 'localtime') converts
# every row through the C library. The UTC offset of a time zone changes only
# a few times a year, so the code below finds those moments once and lets SQL
# add the right offset with integer arithmetic.

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
OFFSET_TABLE_START = APPLE_EPOCH_UNIX  # no iMessage is older than 2001
OFFSET_TABLE_YEARS_AHEAD = 2
OFFSET_PROBE_STEP = 7 * SECONDS_PER_DAY  # offset changes are weeks apart or more
# moments whose offsets tell one time zone from another: mid January and mid July of four years
ZONE_PROBES = (
    1_105_790_400,
    1_121_428_800,
    1_326_628_800,
    1_342_353_600,
    1_547_553_600,
    1_563_192_000,
    1_705_320_000,
    1_721_044_800,
)

OffsetChanges = list[tuple[int, int]]
_offset_table: tuple[tuple[int, ...], int, OffsetChanges] | None = None


def utc_offset(unix_seconds: int) -> int:
    """Seconds that local time is ahead of UTC at a moment, from the C library."""
    return int(time.localtime(unix_seconds).tm_gmtoff)


def _first_change(low: int, high: int, low_offset: int) -> int:
    """The first second in (low, high] whose offset is not `low_offset`."""
    while high - low > 1:
        middle = (low + high) // 2
        if utc_offset(middle) == low_offset:
            low = middle
        else:
            high = middle
    return high


def _build_offset_table(end: int) -> OffsetChanges:
    changes = [(OFFSET_TABLE_START, utc_offset(OFFSET_TABLE_START))]
    moment = OFFSET_TABLE_START
    while moment < end:
        following = min(moment + OFFSET_PROBE_STEP, end)
        if utc_offset(following) != changes[-1][1]:
            start = _first_change(moment, following, changes[-1][1])
            changes.append((start, utc_offset(start)))
        moment = following
    return changes


def offset_changes(lower: int | None = None, upper: int | None = None) -> OffsetChanges:
    """`[(unix second, UTC offset from then on), ...]` for the local time zone, oldest first.

    With `lower` and `upper` only the entries that matter inside [lower, upper)
    are given. The first entry also stands for every earlier moment.
    The table is rebuilt when the time zone of the process changes.
    """
    global _offset_table
    now = int(time.time())
    zone = (*(utc_offset(probe) for probe in ZONE_PROBES), utc_offset(now))
    cached = _offset_table
    if cached is None or cached[0] != zone or cached[1] < now + 366 * SECONDS_PER_DAY:
        end = now + OFFSET_TABLE_YEARS_AHEAD * 366 * SECONDS_PER_DAY
        cached = (zone, end, _build_offset_table(end))
        _offset_table = cached
    changes = cached[2]
    first = 0
    if lower is not None:
        first = max((index for index, (start, _) in enumerate(changes) if start <= lower), default=0)
    last = len(changes) if upper is None else sum(1 for start, _ in changes if start < upper)
    return changes[first : max(last, first + 1)]


def sql_local_seconds(column_ns: str, lower: int | None = None, upper: int | None = None) -> str:
    """SQL for local wall-clock seconds since 1970 of a UTC nanosecond column.

    `lower` and `upper` are the unix-second range the query can see. A narrow
    range gives a short expression.
    """
    return f"({column_ns} / {NS_PER_SECOND} + {_offset_sql(column_ns, offset_changes(lower, upper))})"


def _offset_sql(column_ns: str, changes: OffsetChanges) -> str:
    """SQL for the UTC offset in force: a CASE that halves the table at each step."""
    if len(changes) == 1:
        return str(changes[0][1])
    middle = len(changes) // 2
    later = _offset_sql(column_ns, changes[middle:])
    earlier = _offset_sql(column_ns, changes[:middle])
    return f"CASE WHEN {column_ns} >= {changes[middle][0] * NS_PER_SECOND} THEN {later} ELSE {earlier} END"
