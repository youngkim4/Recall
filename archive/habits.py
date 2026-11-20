"""Habits of each conversation: when messages are sent, how long they are, who starts a talk, how fast replies come.

Two reads, each inside the caller's read transaction:

- The messages of the scope, counted by conversation and local day: how many,
  how many late at night, the local days, and the characters of texts.
- The talks: every message of the one-to-one conversations of the scope's
  conversations, kind and days, in time order. The words, the direction and the
  attachments of the scope are not applied here, because a talk is the whole
  exchange: with only the messages that hold a word, nobody replies to anybody.

A talk starts with the first message of a conversation, or with a message after
TALK_GAP of quiet. Its first message says who started it. A reply is the first
message from the other side in the talk; its time is from the start of the talk,
in elapsed seconds, so a change of the clocks does not count. A group has no talks:
the replies of many people to many people are not one person's reply.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from archive.filters import Scope
from archive.rows import CONVERSATION_INDEX, ORDER_OLDEST
from archive.timeconv import NS_PER_SECOND, SECONDS_PER_DAY, SECONDS_PER_HOUR, sql_local_seconds

TALK_GAP = 6 * SECONDS_PER_HOUR
# a message from 00:00 to 04:59 local is late at night
LATE_UNTIL_HOUR = 5
HOURS_PER_DAY = 24
ONE_TO_ONE = "m.conversation_id IN (SELECT id FROM conversations WHERE is_group = 0)"


@dataclass(frozen=True)
class Habits:
    """What one conversation's messages add up to. Parts of one person are added with `joined`."""

    messages: int = 0
    late: int = 0
    days: frozenset[int] = field(default_factory=frozenset)  # local day numbers since 1970-01-01
    sent_chars: int = 0
    sent_texts: int = 0
    received_chars: int = 0
    received_texts: int = 0
    you_started: int = 0
    they_started: int = 0
    you_replied: tuple[int, ...] = ()  # seconds from the start of their talk to your first message
    they_replied: tuple[int, ...] = ()


def joined(parts: Iterable[Habits]) -> Habits:
    """The habits of several conversations as one: counts added, days and replies put together."""
    found = list(parts)
    return Habits(
        messages=sum(part.messages for part in found),
        late=sum(part.late for part in found),
        days=frozenset().union(*(part.days for part in found)),
        sent_chars=sum(part.sent_chars for part in found),
        sent_texts=sum(part.sent_texts for part in found),
        received_chars=sum(part.received_chars for part in found),
        received_texts=sum(part.received_texts for part in found),
        you_started=sum(part.you_started for part in found),
        they_started=sum(part.they_started for part in found),
        you_replied=tuple(seconds for part in found for seconds in part.you_replied),
        they_replied=tuple(seconds for part in found for seconds in part.they_replied),
    )


# the counts of one conversation's day, in the order of the columns that _message_parts reads
_COUNTED = ("messages", "late", "sent_chars", "sent_texts", "received_chars", "received_texts")


def _message_parts(conn: sqlite3.Connection, where: Scope, lower: int | None, upper: int | None) -> dict[int, Habits]:
    local = sql_local_seconds("m.sent_ns", lower, upper)
    late = f"(({local}) / {SECONDS_PER_HOUR}) % {HOURS_PER_DAY} < {LATE_UNTIL_HOUR}"
    texts = "m.kind = 'text'"
    found = conn.execute(
        f"""SELECT m.conversation_id, ({local}) / {SECONDS_PER_DAY} AS day, COUNT(*), SUM({late}),
               SUM(CASE WHEN m.is_from_me = 1 AND {texts} THEN length(m.text) ELSE 0 END),
               SUM(CASE WHEN m.is_from_me = 1 AND {texts} THEN 1 ELSE 0 END),
               SUM(CASE WHEN m.is_from_me = 0 AND {texts} THEN length(m.text) ELSE 0 END),
               SUM(CASE WHEN m.is_from_me = 0 AND {texts} THEN 1 ELSE 0 END)
            FROM messages m {where.joins} WHERE {where.where} GROUP BY m.conversation_id, day""",
        where.params,
    )
    # added up in place while the rows are read: a frozen copy per day costs a copy of every day before it
    sums: dict[int, list[int]] = {}
    days: dict[int, set[int]] = {}
    for key, day, *counts in found:
        totals = sums.setdefault(key, [0] * len(_COUNTED))
        for index, count in enumerate(counts):
            totals[index] += int(count or 0)
        days.setdefault(key, set()).add(int(day))
    return {
        key: Habits(**dict(zip(_COUNTED, totals, strict=True)), days=frozenset(days[key]))
        for key, totals in sums.items()
    }


@dataclass
class _Talks:
    """The talks of one conversation as they are read, in time order."""

    you_started: int = 0
    they_started: int = 0
    you_replied: list[int] = field(default_factory=list)
    they_replied: list[int] = field(default_factory=list)


def _talk_parts(conn: sqlite3.Connection, where: Scope) -> dict[int, _Talks]:
    found = conn.execute(
        f"SELECT m.conversation_id, m.sent_ns / {NS_PER_SECOND}, m.is_from_me"
        f" FROM messages m INDEXED BY {CONVERSATION_INDEX} {where.joins}"
        f" WHERE {where.where} AND {ONE_TO_ONE} ORDER BY m.conversation_id, {ORDER_OLDEST}",
        where.params,
    )
    talks: dict[int, _Talks] = {}
    previous_key: int | None = None
    previous_at = 0
    started_at = 0
    opener = 0
    answered = True
    for key, at, from_me in found:
        # one object a conversation, not one a row: setdefault would build one for every message
        talk = talks.get(key)
        if talk is None:
            talk = talks[key] = _Talks()
        if key != previous_key or at - previous_at >= TALK_GAP:
            opener, started_at, answered = from_me, at, False
            if from_me:
                talk.you_started += 1
            else:
                talk.they_started += 1
        elif not answered and from_me != opener:
            (talk.you_replied if from_me else talk.they_replied).append(at - started_at)
            answered = True
        previous_key, previous_at = key, at
    return talks


def habits(
    conn: sqlite3.Connection, messages: Scope, talks: Scope, lower: int | None, upper: int | None
) -> dict[str, Habits]:
    """Habits by conversation id. `messages` is the whole scope; `talks` is the scope without
    words, direction or attachments. `lower` and `upper` are the UTC nanosecond bounds that both apply."""
    lower_unix = None if lower is None else lower // NS_PER_SECOND
    upper_unix = None if upper is None else upper // NS_PER_SECOND
    parts = {} if messages.matches_nothing else _message_parts(conn, messages, lower_unix, upper_unix)
    exchanges = {} if talks.matches_nothing else _talk_parts(conn, talks)
    if not parts and not exchanges:
        return {}
    names = dict(conn.execute("SELECT id, chat_identifier FROM conversations").fetchall())
    result: dict[str, Habits] = {}
    for key in sorted(set(parts) | set(exchanges)):
        exchange = exchanges.get(key, _Talks())
        result[str(names.get(key, ""))] = replace(
            parts.get(key, Habits()),
            you_started=exchange.you_started,
            they_started=exchange.they_started,
            you_replied=tuple(exchange.you_replied),
            they_replied=tuple(exchange.they_replied),
        )
    return result
