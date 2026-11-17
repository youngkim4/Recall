"""Time rules of the archive.

Stored time is UTC. Local time is the time zone of the process, and it is
only made at the edge: when a row is shown or when a day is turned into a range.
"""

from __future__ import annotations

import os
import re
import time
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta
from typing import NamedTuple

APPLE_EPOCH_UNIX = 978_307_200  # 2001-01-01T00:00:00Z
NS_PER_SECOND = 1_000_000_000
SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
# Apple writes seconds (old rows) or nanoseconds since 2001. A value this large
# cannot be seconds: it is year 5170.
NANOSECOND_THRESHOLD = 10**11
# 2200-01-01: the end of stored time, far below the 64-bit limit. Every stored moment is before it.
MAX_UNIX_NS = 7_258_118_400 * NS_PER_SECOND
MAX_UNIX_SECONDS = MAX_UNIX_NS // NS_PER_SECOND

LOCAL_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
# The days that become a local time as they are. A local midnight of these days is after 1970-01-01T00:00Z
# and before MAX_UNIX_NS in every time zone. A day outside them is a bound: the start or the end of stored time.
FIRST_DAY = date(1970, 1, 2)
LAST_DAY = date(2199, 12, 30)
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
# local time as local_timestamp writes it, with or without a fraction of a second
_LOCAL_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} (?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?")
_LOCAL_TIME_CHARS = len("YYYY-MM-DD HH:MM:SS")
_FIRST_LOCAL_TIME = datetime(FIRST_DAY.year, FIRST_DAY.month, FIRST_DAY.day)
_END_LOCAL_TIME = datetime(LAST_DAY.year, LAST_DAY.month, LAST_DAY.day) + timedelta(days=1)


def apple_to_unix_ns(value: object) -> int | None:
    """Apple time to UTC unix nanoseconds. None when the value is not a usable date.

    It does not raise. Text that is not a whole number, infinity, nan and a value of another type give None.
    """
    if value is None:
        return None
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return None
    if number <= 0:
        return None
    if number >= NANOSECOND_THRESHOLD:
        unix_ns = number + APPLE_EPOCH_UNIX * NS_PER_SECOND
    else:
        unix_ns = (number + APPLE_EPOCH_UNIX) * NS_PER_SECOND
    # before the end, not at it: a filter up to a far day is `sent_ns < MAX_UNIX_NS` and must leave no date out
    return unix_ns if unix_ns < MAX_UNIX_NS else None


def apple_to_unix_seconds(value: object) -> int | None:
    """Apple time to UTC unix seconds. None when the value is not a usable date. It does not raise."""
    unix_ns = apple_to_unix_ns(value)
    return None if unix_ns is None else unix_ns // NS_PER_SECOND


def parse_day(day: str) -> date:
    """A local day written as YYYY-MM-DD. Anything else is a ValueError."""
    if not isinstance(day, str) or _DAY.fullmatch(day) is None:
        raise ValueError(f"a day must be written as YYYY-MM-DD, got {day!r}")
    return date.fromisoformat(day)


def _local_midnight(day: date) -> int:
    return int(datetime(day.year, day.month, day.day).timestamp())


def _standard_midnight(day: date) -> int:
    """Midnight of a day in the standard time of the zone, by arithmetic."""
    return (day.toordinal() - _EPOCH_ORDINAL) * SECONDS_PER_DAY + time.timezone


def _day_edge(day: date, days_later: int) -> int:
    """UTC unix seconds of the local midnight `days_later` days after the start of `day`.

    A day before FIRST_DAY gives 0 and a day after LAST_DAY gives MAX_UNIX_SECONDS.
    """
    if day < FIRST_DAY:
        return 0
    if day > LAST_DAY:
        return MAX_UNIX_SECONDS
    edge = day + timedelta(days=days_later)
    try:
        moment = _local_midnight(edge)
    except (OverflowError, OSError):  # the C library has no local time for this day
        moment = _standard_midnight(edge)
    return min(max(moment, 0), MAX_UNIX_SECONDS)


def day_start(day: str) -> int:
    """UTC unix seconds of the first second of a local day.

    Text that is not a day is a ValueError. A day before 1970-01-02 gives 0. A day after 2199-12-30
    gives the end of stored time, MAX_UNIX_SECONDS. No other error comes out.
    """
    return _day_edge(parse_day(day), 0)


def day_end(day: str) -> int:
    """UTC unix seconds of the first second after a local day. The same limits as day_start."""
    return _day_edge(parse_day(day), 1)


def local_timestamp(unix_seconds: int) -> str:
    return time.strftime(LOCAL_TIMESTAMP_FORMAT, time.localtime(unix_seconds))


def local_to_unix(text: object) -> int | None:
    """UTC unix seconds of a local time written as YYYY-MM-DD HH:MM:SS. The reverse of local_timestamp.

    A fraction of a second can follow the seconds and is dropped. White space around the text is ignored.
    None for anything else, and for a time outside the days from FIRST_DAY to LAST_DAY. It does not raise.
    An hour that happens twice, when the clocks go back, gives the first of the two moments.
    An hour that does not exist, when the clocks go forward, gives the moment that much later.
    """
    if not isinstance(text, str):
        return None
    trimmed = text.strip()
    if _LOCAL_TIME.fullmatch(trimmed) is None:
        return None
    try:
        moment = datetime.fromisoformat(trimmed[:_LOCAL_TIME_CHARS])
    except ValueError:  # the shape is right but there is no such day
        return None
    if not _FIRST_LOCAL_TIME <= moment < _END_LOCAL_TIME:
        return None
    try:
        return int(moment.timestamp())
    except (OverflowError, OSError):  # the C library has no local time for this moment
        return None


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

OFFSET_TABLE_START = APPLE_EPOCH_UNIX  # no iMessage is older than 2001
OFFSET_TABLE_END = MAX_UNIX_SECONDS  # the importer takes no later date
# The step of the table builder. The builder assumes that two offset changes of one zone are never closer
# than this. The closest two in the tz database (2026e) from 2001 to 2200 are 7 days less one hour apart
# (Asia/Gaza and Asia/Hebron in 2040).
OFFSET_PROBE_STEP = 3 * SECONDS_PER_DAY
LOCALTIME_LINK = "/etc/localtime"  # a link to the file of the system time zone
# One moment in each year from 2001 to 2199: `years` after 2001-01-01, and then some days into that year.
# The time of the year moves on by 139 days from each year to the next, because zones differ in the months
# in which their clocks change.
ZONE_PROBE_FIRST_YEAR = 2001
DAYS_PER_400_YEARS = 146_097
ZONE_PROBES: tuple[int, ...] = tuple(
    APPLE_EPOCH_UNIX + (years * DAYS_PER_400_YEARS // 400 + years * 139 % 365) * SECONDS_PER_DAY
    for years in range(LAST_DAY.year - ZONE_PROBE_FIRST_YEAR + 1)
)

OffsetChanges = list[tuple[int, int]]
ZoneKey = tuple[object, ...]


class _ZoneTable(NamedTuple):
    """The offset changes of one time zone, with the key of that zone."""

    key: ZoneKey
    starts: tuple[int, ...]
    changes: tuple[tuple[int, int], ...]


_offset_table: _ZoneTable | None = None


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
    """Every change of the UTC offset from OFFSET_TABLE_START to `end`, read from the C library.

    The offset is read once for each OFFSET_PROBE_STEP. A step whose two ends have different offsets is
    searched for the second of the change. This assumes that two changes are never closer than one step.
    Two changes inside one step are not seen when the offset comes back to its old value. When it does
    not, the second change is put one second after the end of that step.
    """
    changes = [(OFFSET_TABLE_START, utc_offset(OFFSET_TABLE_START))]
    moment = OFFSET_TABLE_START
    while moment < end:
        following = min(moment + OFFSET_PROBE_STEP, end)
        if utc_offset(following) != changes[-1][1]:
            start = _first_change(moment, following, changes[-1][1])
            changes.append((start, utc_offset(start)))
        moment = following
    return changes


def _localtime_link() -> str | None:
    """Where LOCALTIME_LINK points. It names the system time zone. None when it is not a link."""
    try:
        return os.readlink(LOCALTIME_LINK)
    except OSError:
        return None


def _zone_key() -> ZoneKey:
    """What tells the time zone of this process from another one.

    The TZ variable and the link name the zone. The names and the offsets from the C library are for a
    zone that changes while these two stay the same. The offsets are those of ZONE_PROBES from 2001 to
    next year.
    """
    years = time.gmtime().tm_year - ZONE_PROBE_FIRST_YEAR + 2
    probes = ZONE_PROBES[: max(years, 0)]
    return (
        os.environ.get("TZ"),
        _localtime_link(),
        time.tzname,
        time.timezone,
        time.altzone,
        time.daylight,
        *(utc_offset(probe) for probe in probes),
    )


def _zone_table() -> _ZoneTable:
    """The offset table of the local time zone. It is built again when the time zone of the process changes."""
    global _offset_table
    key = _zone_key()
    cached = _offset_table
    if cached is None or cached.key != key:
        changes = tuple(_build_offset_table(OFFSET_TABLE_END))
        cached = _ZoneTable(key, tuple(start for start, _ in changes), changes)
        _offset_table = cached
    return cached


def offset_changes(lower: int | None = None, upper: int | None = None) -> OffsetChanges:
    """`[(unix second, UTC offset from then on), ...]` for the local time zone, oldest first.

    The table goes from 2001 to MAX_UNIX_NS. With `lower` and `upper` only the entries that
    matter inside [lower, upper) are given. The first entry also stands for every earlier moment.
    The table is built again when the time zone of the process changes.
    """
    table = _zone_table()
    first = 0 if lower is None else max(bisect_right(table.starts, lower) - 1, 0)
    last = len(table.changes) if upper is None else bisect_left(table.starts, upper)
    return list(table.changes[first : max(last, first + 1)])


def sql_local_seconds(column_ns: str, lower: int | None = None, upper: int | None = None) -> str:
    """SQL for local wall-clock seconds since 1970 of a UTC nanosecond column.

    `column_ns` goes into the SQL as it is, in parentheses. It must be a trusted SQL fragment: a column
    or an expression that this package wrote, never text from a user.
    `lower` and `upper` are the unix-second range the query can see. A narrow range gives a short
    expression. With no range the expression has a branch for each offset change up to 2200.
    """
    return f"(({column_ns}) / {NS_PER_SECOND} + {_offset_sql(column_ns, offset_changes(lower, upper))})"


def _offset_sql(column_ns: str, changes: OffsetChanges) -> str:
    """SQL for the UTC offset in force: a CASE that halves the table at each step.

    `column_ns` must be a trusted SQL fragment, as in sql_local_seconds. `changes` has one entry or more.
    """
    if not changes:
        raise ValueError("an offset table has one entry or more")
    if len(changes) == 1:
        return str(changes[0][1])
    middle = len(changes) // 2
    later = _offset_sql(column_ns, changes[middle:])
    earlier = _offset_sql(column_ns, changes[:middle])
    return f"CASE WHEN ({column_ns}) >= {changes[middle][0] * NS_PER_SECOND} THEN {later} ELSE {earlier} END"
