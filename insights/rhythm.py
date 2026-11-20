"""The rhythm of one relationship, counted from the messages alone. No model call.

- `drift`: you used to talk often, and now you have not talked for much
  longer than usual.
- `balance`: one side starts almost every conversation.

Every figure is counted from the notes before `now`.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from statistics import median

from insights.thread import Insight, Note, Thread

# drift
YEAR = timedelta(days=365)
MIN_ACTIVE_WEEKS = 12  # weeks with a message in the year before the silence
MIN_MESSAGES = 20  # messages in that year
MIN_SIDE_SHARE = 0.2  # each side wrote at least this share of them
MIN_SILENT_DAYS = 45
GAP_FACTOR = 4  # silent for 4 times the usual gap between two days of talk
MAX_SILENT_DAYS = 365  # longer than this is a past friendship, not a nudge

# balance
BALANCE_WINDOW = timedelta(days=180)
SESSION_GAP = timedelta(hours=8)  # a message after 8 quiet hours starts a new conversation
MIN_CONVERSATIONS = 10
LOPSIDED_SHARE = 0.8


def _before(thread: Thread, now: datetime) -> tuple[Note, ...]:
    return tuple(note for note in thread.notes if note.when < now)


def _gaps(days: list[date]) -> list[int]:
    return [(later - earlier).days for earlier, later in zip(days, days[1:])]


def _how_often(gap_days: float) -> str:
    if gap_days <= 2:
        return "most days"
    if gap_days <= 10:
        return "about once a week"
    if gap_days <= 20:
        return "every couple of weeks"
    return "about once a month"


def _how_long(days: int) -> str:
    if days < 60:
        return f"{days // 7} weeks"
    return f"{round(days / 30.4)} months"


def drift(thread: Thread, now: datetime) -> Insight | None:
    """Drift, or None. It needs a real rhythm in the year before the last message."""
    notes = _before(thread, now)
    if not notes:
        return None
    last = notes[-1]
    silent_days = (now.date() - last.when.date()).days
    if silent_days > MAX_SILENT_DAYS:
        return None
    year = [note for note in notes if note.when >= last.when - YEAR]
    days = sorted({note.when.date() for note in year})
    weeks = {day.isocalendar()[:2] for day in days}
    if len(weeks) < MIN_ACTIVE_WEEKS or len(year) < MIN_MESSAGES:
        return None
    mine = sum(1 for note in year if note.from_me) / len(year)
    if not MIN_SIDE_SHARE <= mine <= 1 - MIN_SIDE_SHARE:
        return None
    usual_gap = max(1.0, float(median(_gaps(days))))
    if silent_days < max(MIN_SILENT_DAYS, GAP_FACTOR * usual_gap):
        return None
    summary = (
        f"You and {thread.name} used to talk {_how_often(usual_gap)}. "
        f"It has been {_how_long(silent_days)} since you last talked."
    )
    return Insight(
        kind="drift",
        thread=thread.key,
        name=thread.name,
        summary=summary,
        when=last.when,
        message_ids=(last.id,),
        figures=(
            ("usual_gap_days", usual_gap),
            ("silent_days", float(silent_days)),
            ("active_weeks", float(len(weeks))),
            ("messages_in_year", float(len(year))),
        ),
    )


def _openers(notes: tuple[Note, ...]) -> list[Note]:
    """The first note of each conversation: a note after `SESSION_GAP` of quiet."""
    return [note for place, note in enumerate(notes) if place == 0 or note.when - notes[place - 1].when >= SESSION_GAP]


def balance(thread: Thread, now: datetime) -> Insight | None:
    """Who starts the conversations of the last half year, when one side starts almost all of them."""
    notes = tuple(note for note in _before(thread, now) if note.when >= now - BALANCE_WINDOW)
    openers = _openers(notes)
    if len(openers) < MIN_CONVERSATIONS:
        return None
    mine = sum(1 for note in openers if note.from_me)
    share = mine / len(openers)
    if 1 - LOPSIDED_SHARE < share < LOPSIDED_SHARE:
        return None
    total = len(openers)
    if mine >= total / 2:
        summary = f"You started {mine} of the last {total} conversations with {thread.name}."
    else:
        summary = f"{thread.name} started {total - mine} of the last {total} conversations with you."
    return Insight(
        kind="balance",
        thread=thread.key,
        name=thread.name,
        summary=summary,
        when=notes[-1].when,
        message_ids=(notes[-1].id,),
        figures=(("started_by_me", float(mine)), ("conversations", float(total)), ("share_by_me", share)),
    )
