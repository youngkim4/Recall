"""count_messages, list_conversations and conversation_stats: exact numbers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .directory import Entry
from .ports import CountRow
from .render import OWNER, ResultWriter
from .tool_args import ToolInputError, bool_arg, choice_arg, day_arg, int_arg, list_arg, text_arg
from .tool_context import Prepared, ToolContext, ToolOutcome, plural

GROUPS = ("day", "month", "year", "weekday_hour", "conversation", "sender")
_GROUP_WORDS = {
    "day": "day",
    "month": "month",
    "year": "year",
    "weekday_hour": "weekday and hour",
    "conversation": "conversation",
    "sender": "sender",
}
_TABLE_ROWS_ALL = 60
_TABLE_ROWS_TOP = 40
LIST_LIMIT_DEFAULT = 10
LIST_LIMIT_MAX = 30
_MEMBERS_SHOWN = 4
_OWN_KEYS = ("", "me", "You")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_PLAIN_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_KNOWN_FIGURES = frozenset(
    {
        "conversation", "total_messages", "sent_count", "received_count", "first_timestamp",
        "last_timestamp", "active_days", "avg_messages_per_day", "busiest_day", "busiest_day_count",
        "longest_gap_days", "longest_gap_start", "longest_gap_end",
    }
)


def _number(value: Any) -> float:
    """A figure as a number. Anything that is not a number counts as 0."""
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


@dataclass(frozen=True)
class _Line:
    """One row of a count table, ready to show."""

    label: str
    total: int
    sent: int
    received: int


def _line(row: Mapping[str, Any], label: str) -> _Line:
    total, sent = int(_number(row.get("total"))), int(_number(row.get("sent")))
    received = row.get("received")
    return _Line(label, total, sent, total - sent if received is None else int(_number(received)))


# --- count_messages ---------------------------------------------------------


def prepare_count(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    text = text_arg(arguments, "text")
    group_by = choice_arg(arguments, "group_by", GROUPS, None)
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    from_me = bool_arg(arguments, "from_me")
    conversations = ctx.conversations(list_arg(arguments, "conversations"))
    since, until = ctx.days(since, until)
    filters: dict[str, Any] = {
        "conversations": conversations,
        "since": since,
        "until": until,
        "from_me": from_me,
        "text": text or None,
        "group_by": group_by,
    }
    status = "Counting messages"
    if text:
        status += f" that mention “{text}”"
    if group_by:
        status += f" by {_GROUP_WORDS[group_by]}"
    return Prepared(status, lambda budget: run_count(ctx, filters, budget))


def _subject(ctx: ToolContext, filters: Mapping[str, Any]) -> str:
    what = f'Messages that match "{filters["text"]}"' if filters["text"] else "Messages"
    place = ctx.where(filters["conversations"])
    return f"{what} in {place}{ctx.when(filters['since'], filters['until'])}{ctx.who(filters['from_me'])}"


def run_count(ctx: ToolContext, filters: Mapping[str, Any], budget: int) -> ToolOutcome:
    conversations = filters["conversations"]
    result = ctx.archive.count(
        conversations=None if conversations is None else list(conversations),
        since=filters["since"],
        until=filters["until"],
        from_me=filters["from_me"],
        text=filters["text"],
        group_by=filters["group_by"],
    )
    subject = _subject(ctx, filters)
    if filters["group_by"] is None:
        total = int(_number(result))
        return ToolOutcome(f"{subject}: {total:,}.", f"Counted messages: {total:,}")
    rows = [row for row in result if isinstance(row, dict)] if isinstance(result, list) else []
    return _count_table(ctx, subject, str(filters["group_by"]), rows, budget)


def _count_table(
    ctx: ToolContext, subject: str, group_by: str, rows: Sequence[CountRow], budget: int
) -> ToolOutcome:
    word = _GROUP_WORDS[group_by]
    lines = _table_lines(ctx, group_by, rows)
    total = sum(line.total for line in lines)
    sent, received = sum(line.sent for line in lines), sum(line.received for line in lines)
    split = "" if group_by == "sender" else _split(sent, received)
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(f"{subject} by {word}. Total {total:,}{split}.", force=True)
    shown = _shown_lines(lines, group_by)
    if len(shown) < len(lines):
        writer.add(f"Showing the {len(shown)} busiest of {len(lines):,} rows.", force=True)
    for line in shown:
        text = f"{line.label}: {line.total:,}" + ("" if group_by == "sender" else _split(line.sent, line.received))
        if not writer.add(text):
            writer.add("[Result cut to fit. Narrow the dates or the conversations.]", force=True)
            break
    return ToolOutcome(writer.text(), f"Counted messages by {word}: {plural(len(lines), 'row')}")


def _split(sent: int, received: int) -> str:
    return f" (you {sent:,}, others {received:,})"


def _table_lines(ctx: ToolContext, group_by: str, rows: Sequence[CountRow]) -> list[_Line]:
    """Rows with the label the model reads. One person with two handles is one sender row."""
    if group_by == "sender":
        return _by_person(ctx, rows)
    if group_by == "conversation":
        top = sorted(rows, key=lambda row: -_number(row.get("total")))
        named = [_line(row, ctx.book.title(str(row.get("key") or ""))) for row in top[:_TABLE_ROWS_TOP]]
        return named + [_line(row, "") for row in top[_TABLE_ROWS_TOP:]]
    label = weekday_hour if group_by == "weekday_hour" else str
    ordered = sorted(rows, key=lambda row: str(row.get("key") or ""))
    return [_line(row, label(str(row.get("key") or ""))) for row in ordered]


def _by_person(ctx: ToolContext, rows: Sequence[CountRow]) -> list[_Line]:
    merged: dict[str, _Line] = {}
    for row in rows:
        key = str(row.get("key") or "")
        one = _line(row, OWNER if key in _OWN_KEYS else ctx.labeler.sender(key))
        known = merged.get(one.label)
        merged[one.label] = one if known is None else _Line(
            one.label, known.total + one.total, known.sent + one.sent, known.received + one.received
        )
    return sorted(merged.values(), key=lambda line: -line.total)


def _shown_lines(lines: Sequence[_Line], group_by: str) -> list[_Line]:
    if group_by == "conversation":
        return list(lines[:_TABLE_ROWS_TOP])
    if len(lines) > _TABLE_ROWS_ALL:
        return sorted(lines, key=lambda line: -line.total)[:_TABLE_ROWS_TOP]
    return list(lines)


def weekday_hour(key: str) -> str:
    """ "0-09" is Monday 09:00. A key in another form is shown as it is."""
    day, _, hour = key.partition("-")
    if day.isdigit() and hour.isdigit() and int(day) < len(_WEEKDAYS) and int(hour) < 24:
        return f"{_WEEKDAYS[int(day)]} {int(hour):02d}:00"
    return key


# --- list_conversations -----------------------------------------------------


def prepare_list(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    name = text_arg(arguments, "name")
    limit = int_arg(arguments, "limit", default=LIST_LIMIT_DEFAULT, low=1, high=LIST_LIMIT_MAX)
    status = f"Looking up “{name}”" if name else "Listing conversations"
    return Prepared(status, lambda budget: run_list(ctx, name, limit, budget))


def run_list(ctx: ToolContext, name: str, limit: int, budget: int) -> ToolOutcome:
    directory = ctx.directory()
    found = directory.find(name) if name else directory.top(limit)
    return list_entries(ctx, found, limit, budget, name=name)


def list_entries(
    ctx: ToolContext, found: Sequence[Entry], limit: int, budget: int, *, name: str = ""
) -> ToolOutcome:
    directory = ctx.directory()
    shown = list(found[:limit])
    scoped = " in the scope the user chose" if ctx.scope.is_limited else ""
    if name and not found:
        return ToolOutcome(f'No conversation matches "{name}"{scoped}.', f"Looked up “{name}”: nothing found")
    if name:
        header = f'Conversations that match "{name}"{scoped}: {len(found)} of {len(directory)}.'
        header += f" Showing {len(shown)}." if len(shown) < len(found) else ""
        summary = f"Looked up “{name}”: {plural(len(found), 'conversation')}"
    else:
        header = f"{plural(len(shown), 'conversation')} with the most messages (of {len(directory)}){scoped}:"
        summary = f"Listed {plural(len(shown), 'conversation')}"
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(header, force=True)
    for entry in shown:
        if not writer.add(_entry_line(ctx, entry, directory.entries)):
            writer.add("[Result cut to fit. Ask for a name to list fewer.]", force=True)
            break
    return ToolOutcome(writer.text(), summary)


def _entry_line(ctx: ToolContext, entry: Entry, entries: Sequence[Entry]) -> str:
    conversation = entry.conversation
    other = "others" if conversation.get("is_group") else "them"
    parts = [ctx.book.alias(str(conversation["id"])), entry.label, _kind(ctx, entry)]
    if entry.counted is None:
        sent = int(_number(conversation.get("sent_count")))
        received = int(_number(conversation.get("received_count")))
        parts.append(f"{plural(entry.total, 'message')} (you {sent:,}, {other} {received:,})")
        first, last = conversation.get("first_timestamp"), conversation.get("last_timestamp")
        parts.append(f"{str(first)[:10]} to {str(last)[:10]}" if first and last else "no messages")
    else:
        counted = _line(entry.counted, "")
        counts = f"(you {counted.sent:,}, {other} {counted.received:,})"
        parts.append(f"{plural(counted.total, 'message')} in the scope dates {counts}")
    twin = _same_person(ctx, entry, entries)
    if twin:
        parts.append(f"same person as {twin}")
    return " · ".join(parts)


def _kind(ctx: ToolContext, entry: Entry) -> str:
    conversation = entry.conversation
    if not conversation.get("is_group"):
        return f"direct, {entry.via}" if entry.via else "direct"
    members = [ctx.labeler.sender(str(handle)) for handle in conversation.get("members") or []]
    shown = members[:_MEMBERS_SHOWN]
    more = f", +{len(members) - len(shown)} more" if len(members) > len(shown) else ""
    return f"group of {len(members)}: {', '.join(shown)}{more}" if members else "group"


def _same_person(ctx: ToolContext, entry: Entry, entries: Sequence[Entry]) -> str:
    person = entry.conversation.get("person_id")
    if person is None or entry.conversation.get("is_group"):
        return ""
    for other in entries:
        if other is entry:
            return ""
        if other.conversation.get("person_id") == person and not other.conversation.get("is_group"):
            return ctx.book.alias(str(other.conversation["id"]))
    return ""


# --- conversation_stats -----------------------------------------------------


def prepare_stats(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    value = text_arg(arguments, "conversation")
    if not value:
        raise ToolInputError("conversation needs an alias like c3.")
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    allowed = ctx.conversations([value])
    conversation_id = allowed[0] if allowed else ""
    since, until = ctx.days(since, until)
    status = f"Reading the figures for {ctx.book.label(conversation_id)}"
    return Prepared(status, lambda budget: run_stats(ctx, conversation_id, since, until, budget))


def run_stats(
    ctx: ToolContext, conversation_id: str, since: Optional[str], until: Optional[str], budget: int
) -> ToolOutcome:
    figures = ctx.archive.stats(conversation_id, since=since, until=until)
    span = ctx.when(since, until) or ", all time"
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(f"Figures for {ctx.book.title(conversation_id)}{span}:", force=True)
    for line in stats_lines(figures if isinstance(figures, Mapping) else {}):
        if not writer.add(line):
            break
    return ToolOutcome(writer.text(), f"Read the figures for {ctx.book.label(conversation_id)}")


def stats_lines(figures: Mapping[str, Any]) -> list[str]:
    """The figures in plain words. A value that could name a person is left out."""
    total = int(_number(figures.get("total_messages")))
    sent = int(_number(figures.get("sent_count")))
    received = figures.get("received_count")
    others = total - sent if received is None else int(_number(received))
    lines = [f"messages: {total:,} (you {sent:,}, others {others:,})"]
    if figures.get("first_timestamp"):
        lines.append(f"first message: {figures['first_timestamp']}")
    if figures.get("last_timestamp"):
        lines.append(f"last message: {figures['last_timestamp']}")
    active = int(_number(figures.get("active_days")))
    if active:
        average = _number(figures.get("avg_messages_per_day"))
        lines.append(f"active days: {active:,} ({average:.1f} messages on an active day)")
    if figures.get("busiest_day"):
        count = int(_number(figures.get("busiest_day_count")))
        lines.append(f"busiest day: {str(figures['busiest_day'])[:10]} ({plural(count, 'message')})")
    if figures.get("longest_gap_start") and figures.get("longest_gap_end"):
        days = _number(figures.get("longest_gap_days"))
        lines.append(
            f"longest gap: {days:.1f} days, from {figures['longest_gap_start']} to {figures['longest_gap_end']}"
        )
    lines.extend(f"{key}: {value}" for key, value in figures.items() if _extra(key, value))
    return lines


def _extra(key: object, value: Any) -> bool:
    """A field this module does not know: shown only when its name and its value are plain."""
    plain_key = isinstance(key, str) and key not in _KNOWN_FIGURES and bool(_PLAIN_KEY_RE.match(key))
    return plain_key and isinstance(value, (int, float)) and not isinstance(value, bool)
