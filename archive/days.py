"""One local day of a conversation: its messages in order, how many it holds,
and the nearest days with messages on either side."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from archive import rows
from archive.filters import check_int, in_conversations
from archive.timeconv import (
    NS_PER_SECOND,
    day_end,
    day_start,
    local_timestamp,
    local_to_unix,
    parse_day,
    sql_local_timestamp,
)

# how many messages of a day are given at once, and the most that can be asked for
DAY_LIMIT = 200
DAY_LIMIT_MAX = 400


def _first_row(total: int, limit: int, offset: int | None, before_anchor: int | None) -> int:
    """Where the given part of a long day starts.

    A place named with `offset` is used as it is: a reader who asks for what follows the rows
    they hold gets those rows and no others, and a place past the end gives none. Without it,
    the part is a whole window around the anchor, or from the start, and never runs past either end.
    """
    if offset is not None:
        return min(offset, total)
    wanted = 0 if before_anchor is None else before_anchor - limit // 2
    return max(0, min(wanted, total - limit))


def _anchor_text(around: object) -> str | None:
    """`around` as "YYYY-MM-DD HH:MM:SS". Text that is not a local time gives None: the part starts at the start."""
    if around is None:
        return None
    if not isinstance(around, str):
        raise TypeError(f"around must be text, got {type(around).__name__}")
    return around.strip()[:19] if local_to_unix(around) is not None else None


def _day_of(unix_ns: int | None) -> str | None:
    return None if unix_ns is None else local_timestamp(unix_ns // NS_PER_SECOND)[:10]


def day(
    conn: sqlite3.Connection,
    conversation_keys: Sequence[int],
    day_text: str,
    *,
    limit: int,
    offset: int | None,
    around: str | None,
    include_system: bool,
) -> dict[str, Any]:
    """The messages of one local day, oldest first. `conn` must be inside one read transaction.

    A day longer than `limit` is given in part: from `offset` when one is
    named, else around the local time `around`, else from its start.
    A `limit` above DAY_LIMIT_MAX is DAY_LIMIT_MAX.
    """
    limit = min(check_int("limit", limit, 1), DAY_LIMIT_MAX)
    if offset is not None:
        check_int("offset", offset, 0)
    lower, upper = day_start(day_text) * NS_PER_SECOND, day_end(day_text) * NS_PER_SECOND
    name = parse_day(day_text).isoformat()
    if not conversation_keys:
        return {"day": name, "total": 0, "offset": 0, "messages": [], "previous_day": None, "next_day": None}
    where = in_conversations(tuple(conversation_keys)) + ("" if include_system else f" AND {rows.VISIBLE}")
    scoped = f"FROM messages m INDEXED BY {rows.CONVERSATION_INDEX} WHERE {where}"
    in_day = "m.sent_ns >= ? AND m.sent_ns < ?"
    total = int(conn.execute(f"SELECT COUNT(*) {scoped} AND {in_day}", (lower, upper)).fetchone()[0])
    before_anchor = None
    anchor = _anchor_text(around)
    if anchor is not None and offset is None:
        # by wall clock, as the day view does: in the hour that the clocks repeat, two moments have one text
        earlier = conn.execute(
            f"SELECT COUNT(*) {scoped} AND {in_day} AND {sql_local_timestamp('m.sent_at')} < ?",
            (lower, upper, anchor),
        )
        before_anchor = int(earlier.fetchone()[0])
    begin = _first_row(total, limit, offset, before_anchor)
    found = conn.execute(
        f"{rows.PAGED_MESSAGE_SELECT} WHERE {where} AND {in_day} ORDER BY m.sent_ns, m.id LIMIT ? OFFSET ?",
        (lower, upper, limit, begin),
    ).fetchall()
    previous = conn.execute(f"SELECT MAX(m.sent_ns) {scoped} AND m.sent_ns < ?", (lower,)).fetchone()[0]
    following = conn.execute(f"SELECT MIN(m.sent_ns) {scoped} AND m.sent_ns >= ?", (upper,)).fetchone()[0]
    return {
        "day": name,
        "total": total,
        "offset": begin,
        "messages": [rows.message(row) for row in found],
        "previous_day": _day_of(previous),
        "next_day": _day_of(following),
    }
