"""count_messages, list_conversations and conversation_stats: exact numbers."""

from __future__ import annotations

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
_KNOWN_FIGURES = frozenset(
    {
        "conversation", "total_messages", "sent_count", "received_count", "first_timestamp",
        "last_timestamp", "active_days", "avg_messages_per_day", "busiest_day", "busiest_day_count",
        "longest_gap_days", "longest_gap_start", "longest_gap_end",
    }
)


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
        total = int(result) if isinstance(result, int) else 0
        return ToolOutcome(f"{subject}: {total:,}.", f"Counted messages: {total:,}")
    rows = [row for row in result if isinstance(row, dict)] if isinstance(result, list) else []
    return _count_table(ctx, subject, str(filters["group_by"]), rows, budget)


def _count_table(
    ctx: ToolContext, subject: str, group_by: str, rows: Sequence[CountRow], budget: int
) -> ToolOutcome:
    word = _GROUP_WORDS[group_by]
    total = sum(int(row.get("total") or 0) for row in rows)
    sent = sum(int(row.get("sent") or 0) for row in rows)
    split = "" if group_by == "sender" else f" (you {sent:,}, others {total - sent:,})"
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(f"{subject} by {word}. Total {total:,}{split}.", force=True)
    shown = _table_rows(rows, group_by)
    if len(shown) < len(rows):
        writer.add(f"Showing the {len(shown)} busiest of {len(rows):,} rows.", force=True)
    for row in shown:
        if not writer.add(_count_line(ctx, group_by, row)):
            writer.add("[Result cut to fit. Narrow the dates or the conversations.]", force=True)
            break
    return ToolOutcome(writer.text(), f"Counted messages by {word}: {plural(len(rows), 'row')}")


def _table_rows(rows: Sequence[CountRow], group_by: str) -> list[CountRow]:
    by_total = sorted(rows, key=lambda row: -int(row.get("total") or 0))
    if len(rows) > _TABLE_ROWS_ALL:
        return by_total[:_TABLE_ROWS_TOP]
    if group_by in ("conversation", "sender"):
        return by_total
    return sorted(rows, key=lambda row: str(row.get("key") or ""))


def _count_line(ctx: ToolContext, group_by: str, row: CountRow) -> str:
    key = str(row.get("key") or "")
    total = int(row.get("total") or 0)
    sent = int(row.get("sent") or 0)
    if group_by == "sender":
        label = OWNER if key in _OWN_KEYS else ctx.labeler.sender(key)
        return f"{label}: {total:,}"
    if group_by == "conversation":
        label = ctx.book.title(key)
    elif group_by == "weekday_hour":
        label = weekday_hour(key)
    else:
        label = key
    return f"{label}: {total:,} (you {sent:,}, others {total - sent:,})"


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
    scoped = " in the scope the user chose" if ctx.scope.conversations else ""
    if name and not found:
        return ToolOutcome(f'No conversation matches "{name}"{scoped}.', f"Looked up “{name}”: nothing found")
    if name:
        header = f'Conversations that match "{name}"{scoped}: {len(found)} of {len(directory)}.'
        header += f" Showing {len(shown)}." if len(shown) < len(found) else ""
        summary = f"Looked up “{name}”: {plural(len(found), 'conversation')}"
    else:
        header = f"{plural(len(shown), 'conversation')} with the most messages (of {len(directory)}){scoped}:"
        summary = f"Listed {plural(len(shown), 'conversation')}"
    ranged = _ranged_counts(ctx)
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(header, force=True)
    for entry in shown:
        if not writer.add(_entry_line(ctx, entry, directory.entries, ranged)):
            writer.add("[Result cut to fit. Ask for a name to list fewer.]", force=True)
            break
    return ToolOutcome(writer.text(), summary)


def _ranged_counts(ctx: ToolContext) -> Optional[dict[str, CountRow]]:
    """Counts inside the days of the scope, when the scope has days."""
    if not (ctx.scope.since or ctx.scope.until):
        return None
    rows = ctx.archive.count(
        conversations=list(ctx.scope.conversations) or None,
        since=ctx.scope.since,
        until=ctx.scope.until,
        from_me=None,
        text=None,
        group_by="conversation",
    )
    return {str(row["key"]): row for row in rows} if isinstance(rows, list) else {}


def _entry_line(
    ctx: ToolContext, entry: Entry, entries: Sequence[Entry], ranged: Optional[Mapping[str, CountRow]]
) -> str:
    conversation = entry.conversation
    conversation_id = str(conversation["id"])
    is_group = bool(conversation.get("is_group"))
    other = "others" if is_group else "them"
    parts = [ctx.book.alias(conversation_id), entry.label, _kind(ctx, entry)]
    if ranged is None:
        total = int(conversation.get("message_count") or 0)
        sent = int(conversation.get("sent_count") or 0)
        parts.append(f"{plural(total, 'message')} (you {sent:,}, {other} {total - sent:,})")
        first, last = conversation.get("first_timestamp"), conversation.get("last_timestamp")
        parts.append(f"{str(first)[:10]} to {str(last)[:10]}" if first and last else "no messages")
    else:
        row = ranged.get(conversation_id)
        total, sent = (int(row["total"]), int(row["sent"])) if row else (0, 0)
        parts.append(f"{plural(total, 'message')} in the scope dates (you {sent:,}, {other} {total - sent:,})")
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
    for line in stats_lines(figures or {}):
        if not writer.add(line):
            break
    return ToolOutcome(writer.text(), f"Read the figures for {ctx.book.label(conversation_id)}")


def stats_lines(figures: Mapping[str, Any]) -> list[str]:
    """The figures in plain words. A value that could name a person is left out."""
    total = int(figures.get("total_messages") or 0)
    sent = int(figures.get("sent_count") or 0)
    lines = [f"messages: {total:,} (you {sent:,}, others {total - sent:,})"]
    if figures.get("first_timestamp"):
        lines.append(f"first message: {figures['first_timestamp']}")
    if figures.get("last_timestamp"):
        lines.append(f"last message: {figures['last_timestamp']}")
    active = int(figures.get("active_days") or 0)
    if active:
        average = float(figures.get("avg_messages_per_day") or 0.0)
        lines.append(f"active days: {active:,} ({average:.1f} messages on an active day)")
    if figures.get("busiest_day"):
        count = int(figures.get("busiest_day_count") or 0)
        lines.append(f"busiest day: {str(figures['busiest_day'])[:10]} ({plural(count, 'message')})")
    if figures.get("longest_gap_start") and figures.get("longest_gap_end"):
        days = float(figures.get("longest_gap_days") or 0.0)
        lines.append(
            f"longest gap: {days:.1f} days, from {figures['longest_gap_start']} to {figures['longest_gap_end']}"
        )
    lines.extend(f"{key}: {value}" for key, value in figures.items() if _extra(key, value))
    return lines


def _extra(key: str, value: Any) -> bool:
    """A field this module does not know: show it only when it is a plain number."""
    return key not in _KNOWN_FIGURES and isinstance(value, (int, float)) and not isinstance(value, bool)
