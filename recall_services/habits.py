"""The habits of a set of conversations as the screen reads them (archive/habits.py counts them).

    {"messages",                         the messages of the scope
     "youReply", "theyReply",            the median seconds to a reply, or null with no reply
     "replies",                          how many replies the two medians are of
     "youStart",                         the share of talks you started, or null with no talk
     "talks",                            how many talks
     "streak": {"days", "first", "last"} the longest run of days in a row with a message, or null
     "lateNight",                        the share of messages from midnight to 5 a.m., or null
     "sentLength", "receivedLength"}     the mean characters of a text, or null with no text

Every figure is a number or null: a figure with nothing to count is null, never 0,
so the screen can tell "no reply" from "a reply at once". A tie for the longest
streak goes to the earliest.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from archive.habits import Habits, joined

EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def _median(seconds: Sequence[int]) -> int | None:
    return round(statistics.median(seconds)) if seconds else None


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _mean(total: int, count: int) -> float | None:
    return total / count if count else None


def _day(number: int) -> str:
    return date.fromordinal(EPOCH_ORDINAL + number).isoformat()


def longest_streak(days: Iterable[int]) -> dict[str, Any] | None:
    """The longest run of local days in a row, the earliest of equal runs, or None with no day."""
    ordered = sorted(days)
    if not ordered:
        return None
    best_start = best_end = start = ordered[0]
    for before, day in zip(ordered, ordered[1:], strict=False):
        if day != before + 1:
            start = day
        if day - start > best_end - best_start:
            best_start, best_end = start, day
    return {"days": best_end - best_start + 1, "first": _day(best_start), "last": _day(best_end)}


def figures(parts: Iterable[Habits]) -> dict[str, Any]:
    """The habits of the conversations of `parts` as one."""
    found = joined(parts)
    talks = found.you_started + found.they_started
    return {
        "messages": found.messages,
        "youReply": _median(found.you_replied),
        "theyReply": _median(found.they_replied),
        "replies": {"you": len(found.you_replied), "they": len(found.they_replied)},
        "youStart": _share(found.you_started, talks),
        "talks": talks,
        "streak": longest_streak(found.days),
        "lateNight": _share(found.late, found.messages),
        "sentLength": _mean(found.sent_chars, found.sent_texts),
        "receivedLength": _mean(found.received_chars, found.received_texts),
    }
