"""Counts and per-conversation statistics.

Days, months, weekdays and hours are local time. A day starts at local
midnight. Group events and tapbacks are not messages and are not counted.

Local time is not converted row by row. SQL adds the UTC offset that was in
force (timeconv.sql_local_seconds) and groups by whole days or hours. Python
then names the few buckets that come back.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from archive.filters import Scope, check_choice, in_conversations, time_clauses
from archive.rows import CONVERSATION_INDEX, TIME_INDEX, VISIBLE
from archive.timeconv import (
    NS_PER_SECOND,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    local_timestamp,
    sql_local_seconds,
)

GROUPS = ("day", "month", "year", "weekday_hour", "conversation", "sender")
OWNER_KEY = ""  # count(group_by="sender"): the owner's messages
UNKNOWN_SENDER_KEY = "unknown"  # received messages that have no sender in the source
DAY_KEY_LENGTH = {"day": 10, "month": 7, "year": 4}  # prefix of "YYYY-MM-DD"
EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
EPOCH_WEEKDAY = 3  # 1970-01-01 was a Thursday. Monday is 0, like datetime.weekday().
HOURS_PER_WEEK = 168
GAP_CHUNK = 10_000
SUMS = "COUNT(*), COALESCE(SUM(m.is_from_me), 0)"
Bucket = tuple[str, int, int]


def _unix(nanoseconds: int | None) -> int | None:
    return None if nanoseconds is None else nanoseconds // NS_PER_SECOND


def _day_name(day_number: int) -> str:
    """Local day number (days since 1970-01-01) to YYYY-MM-DD."""
    return date.fromordinal(EPOCH_ORDINAL + day_number).isoformat()


def _table(where: Scope, *, reads_rows: bool = False, by_conversation: bool = False) -> str:
    """The messages table with the index that fits the scope.

    Both indexes hold everything a count needs, so a count never reads the table.
    """
    if where.searches_text:
        return "messages m"
    if where.in_conversations or by_conversation:
        return f"messages m INDEXED BY {CONVERSATION_INDEX}"
    if where.counts_system or reads_rows:
        return "messages m"
    return f"messages m INDEXED BY {TIME_INDEX}"


def _grouped(conn: sqlite3.Connection, where: Scope, key_sql: str, table: str) -> list[tuple[Any, int, int]]:
    found = conn.execute(
        f"SELECT {key_sql} AS bucket, {SUMS} FROM {table} {where.joins} WHERE {where.where} GROUP BY bucket",
        where.params,
    )
    return [(bucket, int(total), int(sent)) for bucket, total, sent in found.fetchall()]


def _summed(buckets: Iterable[Bucket]) -> list[Bucket]:
    """Add up buckets that have the same key."""
    totals: dict[str, list[int]] = {}
    for key, total, sent in buckets:
        entry = totals.setdefault(key, [0, 0])
        entry[0] += total
        entry[1] += sent
    return [(key, total, sent) for key, (total, sent) in totals.items()]


def _time_buckets(
    conn: sqlite3.Connection, where: Scope, group_by: str, lower: int | None, upper: int | None
) -> list[Bucket]:
    # With no day range the whole offset table is used: a group event or a message with no
    # conversation can lie outside the time range that the totals of the conversations give.
    local = sql_local_seconds("m.sent_ns", _unix(lower), _unix(upper))
    if group_by == "weekday_hour":
        hours = _grouped(conn, where, f"({local} / {SECONDS_PER_HOUR}) % {HOURS_PER_WEEK}", _table(where))
        named = [(f"{(EPOCH_WEEKDAY + hour // 24) % 7}-{hour % 24:02d}", total, sent) for hour, total, sent in hours]
    else:
        days = _grouped(conn, where, f"{local} / {SECONDS_PER_DAY}", _table(where))
        length = DAY_KEY_LENGTH[group_by]
        named = _summed((_day_name(day)[:length], total, sent) for day, total, sent in days)
    return sorted(named)


def _sender_key(key: int, handles: dict[int, str]) -> str:
    """Name of a sender bucket. Handle ids start at 1: -1 and 0 are the rows with no sender handle."""
    if key == -1:
        return OWNER_KEY
    return str(handles.get(key, UNKNOWN_SENDER_KEY)) if key else UNKNOWN_SENDER_KEY


def _named_buckets(conn: sqlite3.Connection, where: Scope, group_by: str) -> list[Bucket]:
    """Buckets by conversation or by sender, largest first."""
    if group_by == "conversation":
        names = dict(conn.execute("SELECT id, chat_identifier FROM conversations").fetchall())
        found = _grouped(conn, where, "m.conversation_id", _table(where, by_conversation=True))
        named = _summed((str(names.get(key, "")), total, sent) for key, total, sent in found)
    else:
        names = dict(conn.execute("SELECT id, handle FROM handles").fetchall())
        # the owner's messages have no sender handle: split the rows without one by direction
        found = _grouped(conn, where, "COALESCE(m.sender_handle_id, -m.is_from_me)", _table(where, reads_rows=True))
        named = _summed((_sender_key(key, names), total, sent) for key, total, sent in found)
    return sorted(named, key=lambda bucket: (-bucket[1], bucket[0]))


def count(
    conn: sqlite3.Connection, where: Scope, group_by: str | None, lower: int | None = None, upper: int | None = None
) -> int | list[dict[str, Any]]:
    """Number of messages in `where`, or one row per bucket of `group_by`.

    `lower` and `upper` are the UTC nanosecond bounds that `where` already applies.
    """
    if group_by is None:
        if where.matches_nothing:
            return 0
        found = conn.execute(f"SELECT COUNT(*) FROM {_table(where)} {where.joins} WHERE {where.where}", where.params)
        return int(found.fetchone()[0])
    check_choice("group_by", group_by, GROUPS)
    if where.matches_nothing:
        return []
    if group_by in ("conversation", "sender"):
        buckets = _named_buckets(conn, where, group_by)
    else:
        buckets = _time_buckets(conn, where, group_by, lower, upper)
    return [{"key": key, "total": total, "sent": sent, "received": total - sent} for key, total, sent in buckets]


def count_from_totals(
    conn: sqlite3.Connection, conversation_keys: Sequence[int] | None, from_me: bool | None, by_conversation: bool
) -> int | list[dict[str, Any]]:
    """count() for a question with no day, no text and no group events.

    Each conversation row holds its own totals, so no message is read.
    `conversation_keys` None means every conversation.
    """
    if conversation_keys is not None and not conversation_keys:
        return [] if by_conversation else 0
    limit = (
        "" if conversation_keys is None else f"WHERE id IN ({', '.join(str(int(key)) for key in conversation_keys)})"
    )
    found = conn.execute(f"SELECT chat_identifier, sent_count, received_count FROM conversations {limit}").fetchall()
    buckets = [
        (str(name), 0 if from_me is False else int(sent), 0 if from_me is True else int(received))
        for name, sent, received in found
    ]
    if not by_conversation:
        return sum(sent + received for _, sent, received in buckets)
    kept = sorted((bucket for bucket in buckets if bucket[1] + bucket[2]), key=lambda b: (-(b[1] + b[2]), b[0]))
    return [
        {"key": name, "total": sent + received, "sent": sent, "received": received} for name, sent, received in kept
    ]


def _longest_gap(moments: sqlite3.Cursor) -> tuple[int, int] | None:
    """(start, end) of the longest gap between two moments that follow each other. The earliest wins a tie.

    `moments` gives unix seconds in order. The rows are read in chunks: an ordered read with a
    loop here is twice as fast as a window function in SQL, and it holds no more than one chunk.
    """
    longest = -1
    found: tuple[int, int] | None = None
    previous: int | None = None
    while True:
        chunk = moments.fetchmany(GAP_CHUNK)
        if not chunk:
            return found
        for (moment,) in chunk:
            if previous is not None and moment - previous > longest:
                longest = moment - previous
                found = (previous, moment)
            previous = moment


def _empty_stats(label: object) -> dict[str, Any]:
    return {
        "conversation": label,
        "total_messages": 0,
        "sent_count": 0,
        "received_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "active_days": 0,
        "avg_messages_per_day": 0.0,
        "busiest_day": None,
        "busiest_day_count": 0,
        "longest_gap_days": 0.0,
        "longest_gap_start": None,
        "longest_gap_end": None,
    }


def stats(
    conn: sqlite3.Connection,
    label: object,
    conversation_keys: Sequence[int],
    lower: int | None,
    upper: int | None,
    include_system: bool,
) -> dict[str, Any]:
    """The figures of compute_stats in conversation.py, for one or more conversations.

    Ties go to the earliest day and the earliest gap. The gap is elapsed time.
    `conn` must be inside one read transaction so the figures agree.
    """
    if not conversation_keys:
        return _empty_stats(label)
    time_sql, params = time_clauses(lower, upper)
    where = " AND ".join(
        [in_conversations(tuple(conversation_keys)), *time_sql, *([] if include_system else [VISIBLE])]
    )
    scoped = f"FROM messages m INDEXED BY {CONVERSATION_INDEX} WHERE {where}"
    total, sent, first, last = conn.execute(
        f"SELECT {SUMS}, MIN(m.sent_ns), MAX(m.sent_ns) {scoped}", params
    ).fetchone()
    if not total:
        return _empty_stats(label)
    first_unix, last_unix = first // NS_PER_SECOND, last // NS_PER_SECOND
    local_day = f"{sql_local_seconds('m.sent_ns', first_unix, last_unix + 1)} / {SECONDS_PER_DAY}"
    busiest_day, busiest_count, active_days = conn.execute(
        f"SELECT day, n, COUNT(*) OVER () FROM (SELECT {local_day} AS day, COUNT(*) AS n {scoped} GROUP BY day)"
        " ORDER BY n DESC, day ASC LIMIT 1",
        params,
    ).fetchone()
    gap = _longest_gap(conn.execute(f"SELECT m.sent_ns / {NS_PER_SECOND} {scoped} ORDER BY m.sent_ns, m.id", params))
    return {
        "conversation": label,
        "total_messages": int(total),
        "sent_count": int(sent),
        "received_count": int(total - sent),
        "first_timestamp": local_timestamp(first_unix),
        "last_timestamp": local_timestamp(last_unix),
        "active_days": int(active_days),
        "avg_messages_per_day": total / active_days,
        "busiest_day": _day_name(busiest_day),
        "busiest_day_count": int(busiest_count),
        "longest_gap_days": 0.0 if gap is None else (gap[1] - gap[0]) / SECONDS_PER_DAY,
        "longest_gap_start": None if gap is None else local_timestamp(gap[0]),
        "longest_gap_end": None if gap is None else local_timestamp(gap[1]),
    }
