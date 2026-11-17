"""Exact figures for the reduce prompt, counted from the lines the model read.

The reduce step sees summaries, not messages. These numbers give it facts for
the dynamics section that no summary can carry. They use speaker names only.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from report_engine.dates import days_between, parse_day
from report_engine.transcript import OWN_LABEL, Line

QUIET_HOURS = 6
MAX_SPEAKERS = 8
MIN_SILENCE_DAYS = 2
MINUTES_PER_DAY = 1440


def _minute(line: Line, day_numbers: dict[str, int]) -> int:
    if line.day not in day_numbers:
        day_numbers[line.day] = parse_day(line.day).toordinal()
    return day_numbers[line.day] * MINUTES_PER_DAY + int(line.clock[:2]) * 60 + int(line.clock[3:5])


def _starts(lines: Sequence[Line]) -> Counter[str]:
    """Who sent the first message after each quiet stretch."""
    starts: Counter[str] = Counter()
    day_numbers: dict[str, int] = {}
    previous: int | None = None
    for line in lines:
        minute = _minute(line, day_numbers)
        if previous is None or minute - previous >= QUIET_HOURS * 60:
            starts[line.label] += 1
        previous = minute
    return starts


def _longest_silence(days: Sequence[str]) -> tuple[int, str, str]:
    best = (0, "", "")
    for earlier, later in zip(days, days[1:]):
        gap = days_between(earlier, later)
        if gap > best[0]:
            best = (gap, earlier, later)
    return best


def _listed(counts: Counter[str], order: Sequence[str]) -> str:
    shown = [f"{name} {counts[name]:,}" for name in order[:MAX_SPEAKERS] if counts[name]]
    rest = sum(counts[name] for name in order[MAX_SPEAKERS:])
    return ", ".join(shown + ([f"others {rest:,}"] if rest else []))


def conversation_figures(lines: Sequence[Line]) -> str:
    """A short block of facts. Raises ValueError for an empty conversation."""
    if not lines:
        raise ValueError("no lines to count")
    per_speaker = Counter(line.label for line in lines)
    order = sorted(per_speaker, key=lambda name: (-per_speaker[name], name))
    days = sorted({line.day for line in lines})
    per_year = Counter(line.day[:4] for line in lines)
    own_per_year = Counter(line.day[:4] for line in lines if line.label == OWN_LABEL)
    years = "; ".join(
        f"{year} {per_year[year]:,} ({OWN_LABEL} {own_per_year[year] / per_year[year]:.0%})"
        for year in sorted(per_year)
    )
    rows = [
        f"Messages with text: {len(lines):,} ({_listed(per_speaker, order)}).",
        f"First message: {days[0]}. Last message: {days[-1]}. Days with messages: {len(days):,}.",
        f"By year: {years}.",
        f"Conversation starts, counted as the first message after {QUIET_HOURS} quiet hours: "
        f"{_listed(_starts(lines), order)}.",
    ]
    gap, start, end = _longest_silence(days)
    if gap >= MIN_SILENCE_DAYS:
        rows.append(f"Longest silence: {gap:,} days, from {start} to {end}.")
    return "\n".join(rows)
