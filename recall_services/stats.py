"""The /api/stats payload: counts over the whole archive, with the filters of search.

The query string takes the filters of `search_filters`, `query` and `prefix`
as /api/search reads them, with one difference: no `prefix` is false, so the
words are whole (`hi` never counts "hit"). `prefix=true` makes the last word a
prefix. Only the messages that match the words are counted. It also takes:

- `by`: day, week, month, year, weekday, hour, weekday_hour, person,
  conversation, or empty for the totals only.
- `limit`: rows of `person` and `conversation`, busiest first. 50 by default,
  held between 1 and 500.
- `habits`: true adds the habits of the filtered set (recall_services/habits.py):
  `"habits"` in the payload, and `"habits"` in each row of `person` and
  `conversation`. False by default: they read every message of the set.

The payload:

    {"by", "query", "prefix", "filters",
     "totals": {"total", "sent", "received"},   the whole filtered set
     "first", "last",                           its first and last local day, or null
     "days",                                    how many local days hold a message of it
     "rows": [{"key", "label", "total", "sent", "received"}],
     "rowsTotal"}                               the number of rows before the limit

- day, week, month, year: every bucket from `since` (else the first day of the
  set) to `until` (else its last day), empty ones as 0, in time order. The
  range stops at the first and the last day of the archive. A week is named by
  its Monday. The label is the key.
- weekday (0 is Monday), hour (00 to 23), weekday_hour (`W-HH`): every
  bucket, in order. The label names the day and the hour in English.
- conversation: one row for each conversation. person: the same, but the
  threads of one person (a phone number and an email address) are one row,
  as the archive joins them (`Person`). A group is no person and is left out:
  it is listed by `conversation` only. The key is
  the id of the busiest thread in the set; `conversations` lists every id.
  The label is the name the app shows (`conversation_label`): a saved name,
  a group name, "Number ending in 1234", or an address as it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from archive import Archive
from conversation import conversation_label
from recall_services import habits as habit_figures
from recall_services import inputs
from recall_services.names import Names
from recall_services.search import as_typed
from recall_services.search_filters import Filters, read_flag, read_filters, read_prefix
from recall_services.stats_time import COUNTS, RANGED, TIME_GROUPS, Span, time_rows

PEOPLE_GROUPS = ("person", "conversation")
GROUPINGS = (*TIME_GROUPS, *PEOPLE_GROUPS)
DEFAULT_LIMIT = 50
# no client counts as the reader types: a word is whole unless `prefix=true`
PREFIX_WHEN_NOT_GIVEN = False
MAX_LIMIT = 500
Row = dict[str, Any]


def read_by(params: Mapping[str, object]) -> str:
    value = inputs.text(params.get("by")).strip()
    if value and value not in GROUPINGS:
        raise ValueError(f"by must be {', '.join(GROUPINGS)} or empty.")
    return value


def read_limit(params: Mapping[str, object]) -> int:
    asked = inputs.whole("limit", params.get("limit"), DEFAULT_LIMIT) or DEFAULT_LIMIT
    return max(1, min(asked, MAX_LIMIT))


def _totals(rows: Sequence[Row]) -> dict[str, int]:
    """The three counts of rows added up, as the archive gives them."""
    return {name: sum(int(row[name]) for row in rows) for name in COUNTS}


def _archive_days(archive: Archive) -> Span | None:
    """The first and the last local day of the archive, or None when it holds no message."""
    found = archive.summary()
    first, last = found["first_timestamp"], found["last_timestamp"]
    if not first or not last:
        return None
    return Span(date.fromisoformat(first[:10]), date.fromisoformat(last[:10]))


def _span(filters: Filters, days: Sequence[Row], bounds: Span | None) -> Span | None:
    """The days a time grouping covers: the asked days, else those of the set, inside the archive."""
    start = filters.since or (days[0]["key"] if days else "")
    end = filters.until or (days[-1]["key"] if days else "")
    if not start or not end or bounds is None:
        return None
    return Span(max(date.fromisoformat(start), bounds.first), min(date.fromisoformat(end), bounds.last))


def _group_of(by: str, key: str, record: Mapping[str, Any] | None) -> object:
    """What a conversation row joins: its person for `person` (not a group), else itself."""
    person = record.get("person_id") if record and not record.get("is_group") else None
    return ("person", person) if by == "person" and person is not None else key


def _joined_row(names: Names, records: Mapping[str, Mapping[str, Any]], threads: Sequence[Row]) -> Row:
    """One row for threads of one person, busiest first: the first gives the key, any saved name the label."""
    keys = [str(thread["key"]) for thread in threads]
    named = (names.conversation(records[key]) for key in keys if key in records)
    return {
        "key": keys[0],
        "label": conversation_label(next((name for name in named if name), ""), keys[0]),
        **_totals(threads),
        "conversations": keys,
    }


def _people_rows(archive: Archive, names: Names, by: str, counted: Sequence[Row]) -> list[Row]:
    """One row per conversation, or per person with the threads of one person joined. Busiest first.
    A group chat is no person: `person` lists the one-to-one threads only, so it is not the
    list of conversations again (the two once matched row for row)."""
    records = {str(record["id"]): record for record in archive.conversations()}
    threads: dict[object, list[Row]] = {}
    for found in counted:  # the archive gives the busiest first
        key = str(found["key"])
        record = records.get(key)
        if by == "person" and record and record.get("is_group"):
            continue
        threads.setdefault(_group_of(by, key, record), []).append(found)
    rows = [_joined_row(names, records, joined) for joined in threads.values()]
    return sorted(rows, key=lambda row: (-row["total"], row["key"]))


def _rows(
    archive: Archive, names: Names, filters: Filters, by: str, scope: Mapping[str, Any], days: list[Row]
) -> list[Row]:
    """Every row of `by`, before the limit. `days` is the count of each day of the same scope."""
    if by in PEOPLE_GROUPS:
        return _people_rows(archive, names, by, archive.count(group_by="conversation", **scope))  # type: ignore[arg-type]
    if not by:
        return []
    counted = days if by == "day" else archive.count(group_by=by, **scope)
    span = _span(filters, days, _archive_days(archive)) if by in RANGED else None
    return time_rows(by, counted, span)  # type: ignore[arg-type]


def payload(
    archive: Archive,
    names: Names,
    filters: Filters,
    *,
    words: str = "",
    by: str = "",
    limit: int = DEFAULT_LIMIT,
    prefix: bool = False,
    habits: bool = False,
) -> dict[str, Any]:
    """Counts of the filtered set: the totals, its days, and the rows of `by`. One state of the archive."""
    text = (as_typed(words) if prefix else words.strip()) or None
    scope = {**filters.archive_args(), "text": text}
    with archive.snapshot():
        days: list[Row] = archive.count(group_by="day", **scope)  # type: ignore[assignment]
        rows = _rows(archive, names, filters, by, scope, days)
        parts = archive.habits(**scope) if habits else {}
    shown = rows[:limit] if by in PEOPLE_GROUPS else rows
    found = {
        "by": by,
        "query": words.strip(),
        "prefix": prefix,
        "filters": filters.payload(),
        "totals": _totals(days),
        "first": days[0]["key"] if days else None,
        "last": days[-1]["key"] if days else None,
        "days": len(days),
        "rows": [_with_habits(row, parts) for row in shown] if habits and by in PEOPLE_GROUPS else shown,
        "rowsTotal": len(rows),
    }
    return {**found, "habits": habit_figures.figures(parts.values())} if habits else found


def _with_habits(row: Row, parts: Mapping[str, Any]) -> Row:
    """A row of people or conversations with the habits of its threads."""
    return {**row, "habits": habit_figures.figures(parts[key] for key in row["conversations"] if key in parts)}


def answer(archive: Archive, names: Names, params: Mapping[str, object]) -> dict[str, Any]:
    """The payload for the query string of /api/stats. A value that is not allowed raises ValueError."""
    return payload(
        archive,
        names,
        read_filters(params),
        words=inputs.text(params.get("query")),
        by=read_by(params),
        limit=read_limit(params),
        prefix=read_prefix(params, default=PREFIX_WHEN_NOT_GIVEN),
        habits=read_flag(params, "habits", default=False),
    )
