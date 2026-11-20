"""Two insights read from dates and plain words. No model call.

- `unanswered`: the person asked you something, and you never wrote back.
- `birthday`: you wished them a happy birthday before, and that day is close.
"""

from __future__ import annotations

import calendar
import re
from collections import Counter
from datetime import date, datetime, timedelta

from insights.thread import Insight, Note, Thread

# unanswered
MIN_WAIT = timedelta(days=1)
MAX_WAIT_DAYS = 21
QUOTE_CHARS = 80

# birthday
BIRTHDAY_AHEAD_DAYS = 14
_WISH = re.compile(r"\b(?:happy\s+(?:belated\s+)?(?:birthday|bday|b-day)|hbd)\b", re.IGNORECASE)
_FOR_SOMEONE_ELSE = re.compile(r"\b(?:birthday|bday|b-day)\s+to\s+(?!you\b|u\b|ya\b)", re.IGNORECASE)


def _quote(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= QUOTE_CHARS else flat[: QUOTE_CHARS - 1].rstrip() + "…"


def unanswered(thread: Thread, now: datetime) -> Insight | None:
    """Their latest question after your last message, when it waited at least a day and at most three weeks."""
    notes = [note for note in thread.notes if note.when < now]
    trailing: list[Note] = []
    for note in reversed(notes):
        if note.from_me:
            break
        trailing.append(note)
    questions = [note for note in trailing if "?" in note.text]
    if not questions:
        return None
    asked = questions[0]  # the latest: `trailing` runs newest first
    waiting_days = (now.date() - asked.when.date()).days
    if now - asked.when < MIN_WAIT or waiting_days > MAX_WAIT_DAYS:
        return None
    days = "1 day" if waiting_days == 1 else f"{waiting_days} days"
    return Insight(
        kind="unanswered",
        thread=thread.key,
        name=thread.name,
        summary=f'{thread.name} asked you "{_quote(asked.text)}" {days} ago, and you have not answered.',
        when=asked.when,
        message_ids=(asked.id,),
        figures=(("waiting_days", float(waiting_days)),),
    )


def _wishes(thread: Thread, now: datetime) -> list[Note]:
    return [
        note
        for note in thread.notes
        if note.from_me and note.when < now and _WISH.search(note.text) and not _FOR_SOMEONE_ELSE.search(note.text)
    ]


def _on(year: int, month: int, day: int) -> date:
    if month == 2 and day == 29 and not calendar.isleap(year):
        return date(year, 2, 28)
    return date(year, month, day)


def _next(today: date, month: int, day: int) -> date:
    this_year = _on(today.year, month, day)
    return this_year if this_year >= today else _on(today.year + 1, month, day)


def birthday(thread: Thread, now: datetime) -> Insight | None:
    """Their birthday within `BIRTHDAY_AHEAD_DAYS`, from the day you wished it most often."""
    wishes = _wishes(thread, now)
    if not wishes:
        return None
    first_each_year: dict[int, Note] = {}
    for note in wishes:
        first_each_year.setdefault(note.when.year, note)
    days = Counter((note.when.month, note.when.day) for note in first_each_year.values())
    latest_day = {(note.when.month, note.when.day): year for year, note in first_each_year.items()}
    month, day = max(days, key=lambda key: (days[key], latest_day[key]))
    coming = _next(now.date(), month, day)
    days_until = (coming - now.date()).days
    if days_until > BIRTHDAY_AHEAD_DAYS:
        return None
    if wishes[-1].when.date() >= coming - timedelta(days=BIRTHDAY_AHEAD_DAYS):
        return None  # you already wished them for this one
    when_text = "today" if days_until == 0 else "tomorrow" if days_until == 1 else f"in {days_until} days"
    label = f"{calendar.month_name[month]} {day}"
    years = len(first_each_year)
    return Insight(
        kind="birthday",
        thread=thread.key,
        name=thread.name,
        summary=f"{thread.name}'s birthday looks like {label}, {when_text}. You wished them a happy birthday in {years} {'year' if years == 1 else 'years'}.",
        when=wishes[-1].when,
        message_ids=(wishes[-1].id,),
        figures=(("days_until", float(days_until)), ("years_seen", float(years))),
    )
