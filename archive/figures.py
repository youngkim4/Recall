"""Counts and per-conversation statistics.

Days, months, weekdays and hours are local time. A day starts at local
midnight. Group events and tapbacks are not messages and are not counted.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from archive.filters import Scope, check_choice, in_conversations, time_clauses
from archive.rows import VISIBLE
from archive.timeconv import local_timestamp, sql_local

SECONDS_PER_DAY = 86_400.0
LOCAL_DAY = sql_local("%Y-%m-%d", "m.sent_at")
# strftime %w has Sunday as 0. The key has Monday as 0, like datetime.weekday().
WEEKDAY_HOUR = f"((CAST({sql_local('%w', 'm.sent_at')} AS INTEGER) + 6) % 7) || '-' || {sql_local('%H', 'm.sent_at')}"
BY_TOTAL = "COUNT(*) DESC, bucket"

# group_by -> (key expression, join it needs, order)
GROUPS: dict[str, tuple[str, str, str]] = {
    "day": (LOCAL_DAY, "", "bucket"),
    "month": (sql_local("%Y-%m", "m.sent_at"), "", "bucket"),
    "year": (sql_local("%Y", "m.sent_at"), "", "bucket"),
    "weekday_hour": (WEEKDAY_HOUR, "", "bucket"),
    "conversation": ("c.chat_identifier", "JOIN conversations c ON c.id = m.conversation_id", BY_TOTAL),
    "sender": ("COALESCE(h.handle, '')", "LEFT JOIN handles h ON h.id = m.sender_handle_id", BY_TOTAL),
}


def count(conn: sqlite3.Connection, where: Scope, group_by: str | None) -> int | list[dict[str, Any]]:
    """Number of messages in `where`, or one row per bucket of `group_by`."""
    if group_by is None:
        if where.matches_nothing:
            return 0
        found = conn.execute(f"SELECT COUNT(*) FROM messages m {where.joins} WHERE {where.where}", where.params)
        return int(found.fetchone()[0])
    key, join, order = GROUPS[check_choice("group_by", group_by, GROUPS)]
    if where.matches_nothing:
        return []
    buckets = conn.execute(
        f"SELECT {key} AS bucket, COUNT(*), COALESCE(SUM(m.is_from_me), 0)"
        f" FROM messages m {join} {where.joins} WHERE {where.where} GROUP BY bucket ORDER BY {order}",
        where.params,
    ).fetchall()
    return [
        {"key": str(bucket), "total": int(total), "sent": int(sent), "received": int(total - sent)}
        for bucket, total, sent in buckets
    ]


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
    where = " AND ".join([in_conversations(tuple(conversation_keys)), *time_sql, *([] if include_system else [VISIBLE])])
    total, sent, first, last = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM(m.is_from_me), 0), MIN(m.sent_at), MAX(m.sent_at) FROM messages m WHERE {where}",
        params,
    ).fetchone()
    if not total:
        return _empty_stats(label)
    busiest_day, busiest_count, active_days = conn.execute(
        f"SELECT day, n, COUNT(*) OVER () FROM"
        f" (SELECT {LOCAL_DAY} AS day, COUNT(*) AS n FROM messages m WHERE {where} GROUP BY day)"
        " ORDER BY n DESC, day ASC LIMIT 1",
        params,
    ).fetchone()
    gap = conn.execute(
        f"SELECT previous_at, at FROM"
        f" (SELECT m.sent_at AS at, LAG(m.sent_at) OVER (ORDER BY m.sent_ns, m.id) AS previous_at"
        f"    FROM messages m WHERE {where})"
        " WHERE previous_at IS NOT NULL ORDER BY at - previous_at DESC, at ASC LIMIT 1",
        params,
    ).fetchone()
    return {
        "conversation": label,
        "total_messages": int(total),
        "sent_count": int(sent),
        "received_count": int(total - sent),
        "first_timestamp": local_timestamp(first),
        "last_timestamp": local_timestamp(last),
        "active_days": int(active_days),
        "avg_messages_per_day": total / active_days,
        "busiest_day": str(busiest_day),
        "busiest_day_count": int(busiest_count),
        "longest_gap_days": 0.0 if gap is None else (gap[1] - gap[0]) / SECONDS_PER_DAY,
        "longest_gap_start": None if gap is None else local_timestamp(gap[0]),
        "longest_gap_end": None if gap is None else local_timestamp(gap[1]),
    }
