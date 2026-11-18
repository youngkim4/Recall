"""Full-text search over message text.

`ranges` in a hit are [start, end) offsets in Unicode code points of
`message["text"]`, the same as Python string indexes. JavaScript strings
count UTF-16 units: convert with utf16_ranges() before the ranges leave Python.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any

from archive import rows
from archive.filters import Scope, check_bool, check_choice, check_int, scope
from archive.schema import INDEXED_TEXT_CHARS
from archive.search_query import ParsedQuery, ascii_lower, parse_query

__all__ = ["parse_query", "search", "utf16_ranges"]

ORDERS = ("newest", "oldest", "relevance")
NEWEST = rows.ORDER_NEWEST
# `rank` is the bm25 score of the index, best match first. The function bm25() cannot be used
# here: the query also has a window function, and FTS5 functions do not work across one.
ORDER_SQL = {"newest": NEWEST, "oldest": rows.ORDER_OLDEST, "relevance": f"messages_fts.rank, {NEWEST}"}
# highlight() puts these around each match. A text that has one of them is read with the next pair.
# U+FDD0 to U+FDEF are Unicode noncharacters: no valid text has them.
MARKERS = (*((chr(code), chr(code + 1)) for code in range(0xFDD0, 0xFDF0, 2)), ("\x01", "\x02"))


def utf16_ranges(text: str, ranges: Sequence[Sequence[int]]) -> list[list[int]]:
    """Convert code point ranges of `text` to UTF-16 unit ranges, for JavaScript."""
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + (2 if ord(char) > 0xFFFF else 1))
    return [[offsets[start], offsets[end]] for start, end in ranges]


def _ranges_from_marked(marked: str, opener: str, closer: str) -> list[list[int]]:
    """Read match ranges from highlight() output: the text with markers around each match."""
    ranges: list[list[int]] = []
    position = 0
    start = 0
    for char in marked:
        if char == opener:
            start = position
        elif char == closer:
            ranges.append([start, position])
        else:
            position += 1
    return ranges


def _index_ranges(conn: sqlite3.Connection, match: str, texts: Mapping[int, str]) -> dict[int, list[list[int]]]:
    """Ranges of the indexed terms, by message row id."""
    found: dict[int, list[list[int]]] = {}
    pending = dict(texts)
    for opener, closer in MARKERS:
        usable = [key for key, text in pending.items() if opener not in text and closer not in text]
        if usable:
            marks = ", ".join("?" for _ in usable)
            marked_rows = conn.execute(
                f"SELECT rowid, highlight(messages_fts, 0, ?, ?) FROM messages_fts"
                f" WHERE messages_fts MATCH ? AND rowid IN ({marks})",
                (opener, closer, match, *usable),
            ).fetchall()
            found.update({key: _ranges_from_marked(marked, opener, closer) for key, marked in marked_rows})
            pending = {key: text for key, text in pending.items() if key not in usable}
        if not pending:
            break
    return found


def _substring_ranges(text: str, needles: Sequence[str]) -> list[list[int]]:
    folded = ascii_lower(text[:INDEXED_TEXT_CHARS])  # search reads the same part of an oversize text
    ranges: list[list[int]] = []
    for needle in needles:
        position = folded.find(needle)
        while position >= 0:
            ranges.append([position, position + len(needle)])
            position = folded.find(needle, position + len(needle))
    return ranges


def _merged(ranges: list[list[int]]) -> list[list[int]]:
    """Sorted ranges, with overlapping ranges joined."""
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _page_keys(conn: sqlite3.Connection, where: Scope, order: str, limit: int, offset: int) -> tuple[int, list[int]]:
    """(number of all matches, message row ids of the page)."""
    found = conn.execute(
        f"SELECT m.id, COUNT(*) OVER () FROM messages m {where.joins} WHERE {where.where}"
        f" ORDER BY {order} LIMIT ? OFFSET ?",
        (*where.params, limit, offset),
    ).fetchall()
    if found:
        return int(found[0][1]), [int(row[0]) for row in found]
    total = conn.execute(f"SELECT COUNT(*) FROM messages m {where.joins} WHERE {where.where}", where.params)
    return int(total.fetchone()[0]), []


def _hits(conn: sqlite3.Connection, parsed: ParsedQuery, keys: Sequence[int]) -> list[dict[str, Any]]:
    marks = ", ".join("?" for _ in keys)
    found = conn.execute(f"{rows.MESSAGE_SELECT} WHERE m.id IN ({marks})", tuple(keys)).fetchall()
    by_key = {int(row[1]): row for row in found}
    texts = {key: str(row[8]) for key, row in by_key.items()}
    indexed = _index_ranges(conn, parsed.marked, texts) if parsed.marked is not None else {}
    hits = []
    for key in keys:
        if key not in by_key:
            continue
        ranges = indexed.get(key, []) + _substring_ranges(texts[key], parsed.substrings)
        hits.append({"message": rows.message(by_key[key]), "ranges": _merged(ranges)})
    return hits


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    conversations: object = None,
    since: str | None = None,
    until: str | None = None,
    from_me: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    order: str = "newest",
    include_system: bool = False,
    no_conversation: bool = False,
    kind: str | None = None,
    attachments: str | None = None,
    browse: bool = False,
) -> dict[str, Any]:
    """Find messages. `conn` must be inside one read transaction so total and hits agree.

    With `browse=True` a query with no words lists every message of the filters, with no ranges.
    """
    parsed = parse_query(query)
    lists = check_bool("browse", browse) and parsed.is_empty
    check_int("limit", limit, 0)
    check_int("offset", offset, 0)
    check_choice("order", order, ORDERS)
    where = scope(
        conn,
        conversations=conversations,
        since=since,
        until=until,
        from_me=from_me,
        text=None if lists else parsed,
        include_system=include_system,
        no_conversation=no_conversation,
        kind=kind,
        attachments=attachments,
    )
    if where.matches_nothing:
        return {"total": 0, "hits": []}
    order_sql = ORDER_SQL[order] if parsed.match is not None else ORDER_SQL["oldest" if order == "oldest" else "newest"]
    try:
        total, keys = _page_keys(conn, where, order_sql, limit, offset)
        return {"total": total, "hits": _hits(conn, parsed, keys) if keys else []}
    except sqlite3.OperationalError as exc:
        if "fts5: syntax error" not in str(exc).lower():
            raise  # a damaged index must not look like "no results"
        return {"total": 0, "hits": []}  # text typed by a user never raises
