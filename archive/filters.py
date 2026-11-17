"""Turn the filter arguments of the query API into SQL over `messages m`."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from archive.rows import VISIBLE
from archive.search_query import ParsedQuery
from archive.timeconv import NS_PER_SECOND, day_end, day_start

FTS_JOIN = "JOIN messages_fts ON messages_fts.rowid = m.id"


@dataclass(frozen=True)
class Scope:
    """Which messages a query looks at."""

    joins: str
    where: str
    params: tuple[object, ...]
    matches_nothing: bool = False
    searches_text: bool = False  # reads message text: the full-text index or a substring test
    in_conversations: bool = False  # limited to some conversations
    counts_system: bool = False  # group events are included


NOTHING = Scope(joins="", where="0", params=(), matches_nothing=True)


def check_int(name: str, value: object, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a whole number of {minimum} or more, got {value!r}")
    return value


def check_choice(name: str, value: object, choices: Iterable[str]) -> str:
    allowed = tuple(choices)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return str(value)


def check_flag(name: str, value: object) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be True, False or None, got {value!r}")
    return None if value is None else bool(value)


def as_id_list(name: str, value: object) -> list[str]:
    """One id or several ids as a list without repeats."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (bytes, dict)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an id or a list of ids")
    items = list(value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError(f"{name} must be an id or a list of ids")
    return list(dict.fromkeys(items))


def conversation_keys(conn: sqlite3.Connection, value: object) -> tuple[int, ...]:
    """Row ids of the conversations named by `value`. Unknown ids are left out."""
    identifiers = as_id_list("conversation ids", value)
    if not identifiers:
        return ()
    marks = ", ".join("?" for _ in identifiers)
    rows = conn.execute(
        f"SELECT chat_identifier, id FROM conversations WHERE chat_identifier IN ({marks})", identifiers
    )
    found = dict(rows.fetchall())
    return tuple(found[identifier] for identifier in identifiers if identifier in found)


def day_bounds(since: str | None, until: str | None) -> tuple[int | None, int | None]:
    """UTC nanosecond bounds [lower, upper) for local days. A range includes all of both days."""
    lower = None if since is None else day_start(since) * NS_PER_SECOND
    upper = None if until is None else day_end(until) * NS_PER_SECOND
    return lower, upper


def time_clauses(lower: int | None, upper: int | None) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if lower is not None:
        clauses.append("m.sent_ns >= ?")
        params.append(lower)
    if upper is not None:
        clauses.append("m.sent_ns < ?")
        params.append(upper)
    return clauses, params


def in_conversations(keys: tuple[int, ...]) -> str:
    """SQL that limits `messages m` to some conversations.

    One conversation is written with `=`: SQLite then knows that the index gives time order.
    """
    if len(keys) == 1:
        return f"m.conversation_id = {int(keys[0])}"
    return f"m.conversation_id IN ({', '.join(str(int(key)) for key in keys)})"


def substring_clause(needle: str) -> str:
    """SQL that tests for a needle made by search_query. lower() is skipped when the needle has no letter a-z."""
    haystack = "lower(m.text)" if any("a" <= char <= "z" for char in needle) else "m.text"
    return f"instr({haystack}, ?) > 0"


def scope(
    conn: sqlite3.Connection,
    *,
    conversations: object = None,
    since: str | None = None,
    until: str | None = None,
    from_me: bool | None = None,
    text: ParsedQuery | None = None,
    include_system: bool = False,
) -> Scope:
    """Build the FROM and WHERE parts for a filtered query."""
    lower, upper = day_bounds(since, until)
    direction = check_flag("from_me", from_me)
    if text is not None and text.is_empty:
        return NOTHING
    clauses: list[str] = []
    params: list[object] = []
    joins = ""
    if text is not None and text.match is not None:
        joins = FTS_JOIN
        clauses.append("messages_fts MATCH ?")
        params.append(text.match)
    if conversations is not None:
        keys = conversation_keys(conn, conversations)
        if not keys:
            return NOTHING
        clauses.append(in_conversations(keys))
    time_sql, time_params = time_clauses(lower, upper)
    clauses.extend(time_sql)
    params.extend(time_params)
    if direction is not None:
        clauses.append(f"m.is_from_me = {int(direction)}")
    if not include_system:
        clauses.append(VISIBLE)
    for needle in text.substrings if text is not None else ():
        clauses.append(substring_clause(needle))
        params.append(needle)
    return Scope(
        joins=joins,
        where=" AND ".join(clauses) or "1",
        params=tuple(params),
        searches_text=text is not None,
        in_conversations=conversations is not None,
        counts_system=include_system,
    )
