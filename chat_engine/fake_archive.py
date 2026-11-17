"""An in-memory archive that follows the `archive.queries.Archive` contract.

Used by the tests and by the evaluation set. It holds invented messages only.
Every call is recorded in `calls`, so a test can prove what was read.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .ports import ArchiveSummary, Conversation, CountRow, Message, MessagePage, SearchResult
from .textnorm import find_unit, query_units, token_spans

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_GROUP_KEY_LENGTH = {"day": 10, "month": 7, "year": 4}


def make_message(
    message_id: str,
    conversation: str,
    timestamp: str,
    text: str,
    *,
    is_from_me: bool = False,
    sender: str = "",
    kind: str = "text",
    attachments: Sequence[str] = (),
    reply_to: Optional[str] = None,
    edited: bool = False,
    unsent: bool = False,
) -> Message:
    moment = datetime.strptime(timestamp, _TIME_FORMAT)
    return {
        "id": str(message_id),
        "conversation": str(conversation),
        "timestamp": timestamp,
        "sent_at": int((moment - datetime(2001, 1, 1)).total_seconds()),
        "is_from_me": bool(is_from_me),
        "sender": "" if is_from_me else str(sender),
        "text": str(text),
        "kind": kind,
        "attachments": list(attachments),
        "reply_to": reply_to,
        "edited": bool(edited),
        "unsent": bool(unsent),
    }


def _copy(message: Message) -> Message:
    copied = dict(message)
    copied["attachments"] = list(message["attachments"])
    return copied  # type: ignore[return-value]


def _moment(message: Message) -> datetime:
    return datetime.strptime(message["timestamp"], _TIME_FORMAT)


class FakeArchive:
    def __init__(
        self,
        messages: Iterable[Message],
        conversations: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> None:
        ordered = sorted(
            enumerate(_copy(m) for m in messages),
            key=lambda pair: (pair[1]["timestamp"], pair[1]["sent_at"], pair[0]),
        )
        self._messages: list[Message] = [message for _, message in ordered]
        self._by_id = {m["id"]: m for m in self._messages}
        self._by_conversation: dict[str, list[Message]] = defaultdict(list)
        for message in self._messages:
            self._by_conversation[message["conversation"]].append(message)
        self._conversations = self._build_conversations(conversations)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # --- contract ---

    def summary(self) -> ArchiveSummary:
        self._record("summary")
        return {
            "messages": len(self._messages),
            "conversations": sum(1 for c in self._conversations.values() if c["message_count"]),
            "first_timestamp": self._messages[0]["timestamp"] if self._messages else None,
            "last_timestamp": self._messages[-1]["timestamp"] if self._messages else None,
        }

    def conversations(
        self, *, limit: Optional[int] = None, offset: int = 0, order: str = "count"
    ) -> list[Conversation]:
        self._record("conversations", limit=limit, offset=offset, order=order)
        rows = [c for c in self._conversations.values() if c["message_count"]]
        if order == "recent":
            rows.sort(key=lambda c: c["last_timestamp"] or "", reverse=True)
        else:
            rows.sort(key=lambda c: (-c["message_count"], c["id"]))
        end = None if limit is None else offset + limit
        return [_copy_conversation(c) for c in rows[offset:end]]

    def conversation(self, conversation_id: str) -> Optional[Conversation]:
        self._record("conversation", conversation_id=conversation_id)
        found = self._conversations.get(str(conversation_id))
        return _copy_conversation(found) if found else None

    def messages(
        self,
        conversation_id: str,
        *,
        around: Optional[str] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 50,
        include_system: bool = False,
    ) -> MessagePage:
        self._record(
            "messages", conversation_id=conversation_id, around=around, before=before, after=after, limit=limit
        )
        rows = _visible(self._by_conversation.get(str(conversation_id), []), include_system)
        start, end = _page_bounds(rows, around, before, after, max(1, limit))
        return {
            "messages": [_copy(m) for m in rows[start:end]],
            "has_older": 0 < start < end,
            "has_newer": start < end < len(rows),
        }

    def context(self, message_id: str, *, radius: int = 6, include_system: bool = False) -> list[Message]:
        self._record("context", message_id=message_id, radius=radius)
        anchor = self._by_id.get(str(message_id))
        rows = _visible(self._by_conversation[anchor["conversation"]], include_system) if anchor else []
        if anchor is None or anchor not in rows:
            return []
        index = rows.index(anchor)
        return [_copy(m) for m in rows[max(0, index - radius):index + radius + 1]]

    def search(
        self,
        query: str,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "newest",
        include_system: bool = False,
    ) -> SearchResult:
        self._record(
            "search", query=query, conversations=_listed(conversations), since=since, until=until,
            from_me=from_me, limit=limit, offset=offset, order=order,
        )
        units = query_units(query)
        if not units:
            return {"total": 0, "hits": []}
        found = []
        for message in self._filtered(conversations, since, until, from_me, include_system):
            ranges = _match(units, message["text"])
            if ranges:
                found.append((message, ranges))
        if order == "oldest":
            pass
        elif order == "relevance":
            found.sort(key=lambda pair: (-len(pair[1]), _reverse_time(pair[0])))
        else:
            found.reverse()
        page = found[offset:offset + max(0, limit)]
        return {
            "total": len(found),
            "hits": [{"message": _copy(m), "ranges": [list(r) for r in ranges]} for m, ranges in page],
        }

    def count(
        self,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        text: Optional[str] = None,
        group_by: Optional[str] = None,
        include_system: bool = False,
    ) -> Union[int, list[CountRow]]:
        self._record(
            "count", conversations=_listed(conversations), since=since, until=until,
            from_me=from_me, text=text, group_by=group_by,
        )
        rows = self._filtered(conversations, since, until, from_me, include_system)
        if text is not None:
            units = query_units(text)
            rows = [m for m in rows if units and _match(units, m["text"])]
        if group_by is None:
            return len(rows)
        return _grouped(rows, group_by)

    def stats(
        self,
        conversation_id: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        include_system: bool = False,
    ) -> dict[str, Any]:
        self._record("stats", conversation_id=conversation_id, since=since, until=until)
        rows = self._filtered([str(conversation_id)], since, until, None, include_system)
        return _stats(str(conversation_id), rows)

    # --- helpers ---

    def _record(self, name: str, **arguments: Any) -> None:
        self.calls.append((name, arguments))

    def _filtered(
        self,
        conversations: Optional[Sequence[str]],
        since: Optional[str],
        until: Optional[str],
        from_me: Optional[bool],
        include_system: bool = False,
    ) -> list[Message]:
        wanted = None if conversations is None else {str(c) for c in conversations}
        rows = []
        for message in _visible(self._messages, include_system):
            day = message["timestamp"][:10]
            if wanted is not None and message["conversation"] not in wanted:
                continue
            if (since and day < since) or (until and day > until):
                continue
            if from_me is not None and message["is_from_me"] != from_me:
                continue
            rows.append(message)
        return rows

    def _build_conversations(
        self, given: Optional[Iterable[Mapping[str, Any]]]
    ) -> dict[str, Conversation]:
        described = {str(row["id"]): row for row in (given or [])}
        ids = list(dict.fromkeys([*described, *self._by_conversation]))
        return {cid: self._describe(cid, described.get(cid, {})) for cid in ids}

    def _describe(self, conversation_id: str, given: Mapping[str, Any]) -> Conversation:
        rows = self._by_conversation.get(conversation_id, [])
        sent = sum(1 for m in rows if m["is_from_me"])
        senders = list(dict.fromkeys(m["sender"] for m in rows if m["sender"]))
        return {
            "id": conversation_id,
            "is_group": bool(given.get("is_group", False)),
            "name": str(given.get("name", "")),
            "message_count": len(rows),
            "sent_count": sent,
            "received_count": len(rows) - sent,
            "first_timestamp": rows[0]["timestamp"] if rows else None,
            "last_timestamp": rows[-1]["timestamp"] if rows else None,
            "members": list(given.get("members", senders)),
            "person_id": given.get("person_id"),
        }


def _visible(rows: list[Message], include_system: bool) -> list[Message]:
    return rows if include_system else [m for m in rows if m["kind"] != "system"]


def _listed(values: Optional[Sequence[str]]) -> Optional[list[str]]:
    return None if values is None else list(values)


def _copy_conversation(conversation: Conversation) -> Conversation:
    copied = dict(conversation)
    copied["members"] = list(conversation["members"])
    return copied  # type: ignore[return-value]


def _reverse_time(message: Message) -> tuple[float, int]:
    return (-_moment(message).timestamp(), -message["sent_at"])


def _match(units: list[str], text: str) -> list[tuple[int, int]]:
    """Ranges of all units, or [] unless every unit matches (AND)."""
    spans = token_spans(text)
    ranges: list[tuple[int, int]] = []
    for unit in units:
        found = find_unit(unit, spans)
        if not found:
            return []
        ranges.extend(found)
    return sorted(set(ranges))


def _page_bounds(
    rows: list[Message],
    around: Optional[str],
    before: Optional[str],
    after: Optional[str],
    limit: int,
) -> tuple[int, int]:
    """Slice of `rows` for one page. An unknown anchor gives an empty page."""
    given = [cursor for cursor in (around, before, after) if cursor is not None]
    if len(given) > 1:
        raise ValueError("give at most one of around, before, after")
    ids = [m["id"] for m in rows]
    if not given:
        return max(0, len(rows) - limit), len(rows)
    if given[0] not in ids:
        return 0, 0
    index = ids.index(given[0])
    if before is not None:
        return max(0, index - limit), index
    if after is not None:
        return index + 1, min(len(rows), index + 1 + limit)
    start = max(0, index - (limit - 1) // 2)
    end = min(len(rows), start + limit)
    return max(0, end - limit), end


def _group_key(message: Message, group_by: str) -> str:
    if group_by in _GROUP_KEY_LENGTH:
        return message["timestamp"][:_GROUP_KEY_LENGTH[group_by]]
    if group_by == "weekday_hour":
        moment = _moment(message)
        return f"{moment.weekday()}-{moment.hour:02d}"
    if group_by == "conversation":
        return message["conversation"]
    if group_by == "sender":
        return message["sender"]
    raise ValueError(f"unknown group_by: {group_by}")


def _grouped(rows: list[Message], group_by: str) -> list[CountRow]:
    totals: dict[str, CountRow] = {}
    for message in rows:
        key = _group_key(message, group_by)
        row = totals.setdefault(key, {"key": key, "total": 0, "sent": 0, "received": 0})
        row["total"] += 1
        row["sent" if message["is_from_me"] else "received"] += 1
    if group_by in ("conversation", "sender"):
        return sorted(totals.values(), key=lambda row: (-row["total"], row["key"]))
    return sorted(totals.values(), key=lambda row: row["key"])


def _stats(conversation_id: str, rows: list[Message]) -> dict[str, Any]:
    sent = sum(1 for m in rows if m["is_from_me"])
    per_day: dict[str, int] = defaultdict(int)
    for message in rows:
        per_day[message["timestamp"][:10]] += 1
    busiest = max(per_day.items(), key=lambda item: (item[1], _negative(item[0])), default=None)
    gap_days, gap_start, gap_end = 0.0, None, None
    for earlier, later in zip(rows, rows[1:]):
        days = (_moment(later) - _moment(earlier)).total_seconds() / 86400.0
        if days > gap_days:
            gap_days, gap_start, gap_end = days, earlier["timestamp"], later["timestamp"]
    return {
        "conversation": conversation_id,
        "total_messages": len(rows),
        "sent_count": sent,
        "received_count": len(rows) - sent,
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
        "active_days": len(per_day),
        "avg_messages_per_day": len(rows) / len(per_day) if per_day else 0.0,
        "busiest_day": busiest[0] if busiest else None,
        "busiest_day_count": busiest[1] if busiest else 0,
        "longest_gap_days": gap_days,
        "longest_gap_start": gap_start,
        "longest_gap_end": gap_end,
    }


def _negative(day: str) -> tuple[int, ...]:
    """Sort key that prefers the earliest day on a tie."""
    return tuple(-int(part) for part in day.split("-"))
