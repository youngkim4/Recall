"""Split a conversation into periods on calendar boundaries.

Three levels, one rule. Months are packed oldest first until the next month
does not fit the budget. A month that is larger than the budget stands alone
and is packed the same way by days. A day that is larger than the budget
stands alone and is packed by lines.

A unit only ever grows when new messages arrive, and a larger unit stands
alone, so a closed period never changes: new messages change only the last
period.

No line is larger than a period: `fit_lines` cuts the few that are, before
anything is packed. Every size is the size of a text with the line break that
follows it, and the header of a day is counted, not assumed. So the size of a
period is never less than the size of the text that is sent, and no period is
larger than the budget.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import groupby
from typing import TypeVar

from report_engine.tokens import TokenMeter
from report_engine.transcript import (
    COUNT_REF,
    ELISION_MARK,
    Line,
    cut_start,
    day_header,
    render_for_count,
    render_line,
    shield,
)

HEADER_TOKENS = 12
# a tokenizer gives one token for a byte at most, and a character has four bytes at most
MAX_TOKENS_PER_CHAR = 4
# the reference, the time and the name before the text of a line, and the header of a day
LINE_HEAD_CHARS = 64
HEADER_CHARS = 32
MAX_LINE_CUTS = 6
CUT_SAFETY = 0.9
MONTH_KEY_LENGTH = 7
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

Unit = TypeVar("Unit")
LineSizer = Callable[[int, int], Sequence[int]]
"""Token counts of the lines in [start, stop), one per line, each with its line break."""
HeaderSizer = Callable[[str], int]
"""Tokens of the header of one day, with its line break."""


@dataclass(frozen=True)
class DaySpan:
    """One calendar day: its lines are [start, stop) and its text weighs `tokens`."""

    day: str
    start: int
    stop: int
    tokens: int


@dataclass(frozen=True)
class Period:
    """One model call's worth of conversation: lines [start, stop)."""

    label: str
    first_day: str
    last_day: str
    start: int
    stop: int
    tokens: int


def _fixed_header(day: str) -> int:
    return HEADER_TOKENS


def header_sizer(meter: TokenMeter) -> HeaderSizer:
    """The size of the header of a day, counted one time for each day."""
    sizes: dict[str, int] = {}

    def size(day: str) -> int:
        if day not in sizes:
            sizes[day] = meter.count(f"{day_header(day)}\n")
        return sizes[day]

    return size


def _counted(line: Line) -> str:
    """A line as its size is taken: with a reference of full length and its line break."""
    return f"{render_line(COUNT_REF, line)}\n"


def _shown(line: Line) -> str:
    return line.flat or shield(line.text)


def _shrink(line: Line, size: int, ceiling: int, meter: TokenMeter) -> Line:
    """The line with less of its text, until the whole line is `ceiling` tokens or
    fewer. When no start of the text fits, the line shows the mark only."""
    shown = _shown(line).removesuffix(ELISION_MARK).rstrip() if line.cut else _shown(line)
    keep = len(shown)
    cut = line
    for _ in range(MAX_LINE_CUTS):
        if size <= ceiling or keep == 0:
            break
        keep = int(keep * ceiling / size * CUT_SAFETY)
        cut = replace(line, flat=cut_start(shown, keep), cut=True)
        size = meter.count(_counted(cut))
    return cut if size <= ceiling else replace(line, flat=ELISION_MARK, cut=True)


def fit_lines(
    lines: tuple[Line, ...],
    meter: TokenMeter,
    budget: int,
    header_tokens: HeaderSizer | None = None,
) -> tuple[Line, ...]:
    """The same lines, with each line that does not fit a period of `budget`
    tokens cut until it fits under the header of its day. The cap on a message
    counts characters, and text can hold a token in every character. Only
    long lines are counted: a short line can not be large."""
    headers = header_tokens or header_sizer(meter)
    reach = budget / (MAX_TOKENS_PER_CHAR * meter.margin) - LINE_HEAD_CHARS - HEADER_CHARS
    suspects = [index for index, line in enumerate(lines) if len(_shown(line)) > reach]
    if not suspects:
        return lines
    sizes = meter.count_many([_counted(lines[index]) for index in suspects])
    fitted: dict[int, Line] = {}
    for index, size in zip(suspects, sizes, strict=True):
        ceiling = budget - headers(lines[index].day)
        if size > ceiling:
            fitted[index] = _shrink(lines[index], size, ceiling, meter)
    if not fitted:
        return lines
    return tuple(fitted.get(index, line) for index, line in enumerate(lines))


def measure_days(lines: Sequence[Line], meter: TokenMeter) -> tuple[DaySpan, ...]:
    """Group ordered lines by day and weigh each day once, with the line break
    that parts it from the next day."""
    bounds: list[tuple[str, int, int]] = []
    cursor = 0
    for day, group in groupby(lines, key=lambda line: line.day):
        count = sum(1 for _ in group)
        bounds.append((day, cursor, cursor + count))
        cursor += count
    texts = [f"{render_for_count(lines[start:stop])}\n" for _, start, stop in bounds]
    sizes = meter.count_many(texts)
    return tuple(
        DaySpan(day=day, start=start, stop=stop, tokens=size)
        for (day, start, stop), size in zip(bounds, sizes, strict=True)
    )


def line_sizer(lines: Sequence[Line], meter: TokenMeter) -> LineSizer:
    """Per-line weights, counted only for the days that need them."""

    def sizes(start: int, stop: int) -> Sequence[int]:
        return meter.count_many([_counted(line) for line in lines[start:stop]])

    return sizes


def _pack(
    units: Sequence[Unit], budget: int, size: Callable[[Unit], int]
) -> list[tuple[Unit, ...]]:
    """Greedy, oldest first. A unit larger than the budget ends up alone."""
    groups: list[tuple[Unit, ...]] = []
    current: list[Unit] = []
    total = 0
    for unit in units:
        weight = size(unit)
        if current and total + weight > budget:
            groups.append(tuple(current))
            current, total = [], 0
        current.append(unit)
        total += weight
    if current:
        groups.append(tuple(current))
    return groups


def _tokens(spans: Sequence[DaySpan]) -> int:
    return sum(span.tokens for span in spans)


def _period(label: str, spans: Sequence[DaySpan]) -> Period:
    return Period(
        label=label,
        first_day=spans[0].day,
        last_day=spans[-1].day,
        start=spans[0].start,
        stop=spans[-1].stop,
        tokens=_tokens(spans),
    )


def _month_name(day: str) -> str:
    return f"{MONTHS[int(day[5:7]) - 1]} {day[:4]}"


def _months_label(spans: Sequence[DaySpan]) -> str:
    first, last = _month_name(spans[0].day), _month_name(spans[-1].day)
    return first if first == last else f"{first} – {last}"


def _days_label(spans: Sequence[DaySpan]) -> str:
    first, last = spans[0].day, spans[-1].day
    return first if first == last else f"{first} – {last}"


def _split_day(
    span: DaySpan, budget: int, line_tokens: LineSizer, header_tokens: HeaderSizer
) -> list[Period]:
    weights = list(line_tokens(span.start, span.stop))
    header = header_tokens(span.day)
    room = max(1, budget - header)
    offsets = range(len(weights))
    parts = _pack(offsets, room, lambda offset: weights[offset])
    return [
        Period(
            label=f"{span.day} part {number}",
            first_day=span.day,
            last_day=span.day,
            start=span.start + part[0],
            stop=span.start + part[-1] + 1,
            tokens=sum(weights[offset] for offset in part) + header,
        )
        for number, part in enumerate(parts, start=1)
    ]


def _split_month(
    spans: Sequence[DaySpan], budget: int, line_tokens: LineSizer, header_tokens: HeaderSizer
) -> list[Period]:
    periods: list[Period] = []
    for group in _pack(spans, budget, lambda span: span.tokens):
        if len(group) == 1 and group[0].tokens > budget:
            periods.extend(_split_day(group[0], budget, line_tokens, header_tokens))
        else:
            periods.append(_period(_days_label(group), group))
    return periods


def plan_periods(
    days: Sequence[DaySpan],
    budget: int,
    line_tokens: LineSizer,
    header_tokens: HeaderSizer = _fixed_header,
) -> tuple[Period, ...]:
    """Periods for ordered day spans. Every line lands in exactly one period."""
    if budget <= 0:
        raise ValueError("period token budget must be positive")
    months = [
        tuple(group) for _, group in groupby(days, key=lambda span: span.day[:MONTH_KEY_LENGTH])
    ]
    periods: list[Period] = []
    for group in _pack(months, budget, _tokens):
        if len(group) == 1 and _tokens(group[0]) > budget:
            periods.extend(_split_month(group[0], budget, line_tokens, header_tokens))
        else:
            spans = [span for month in group for span in month]
            periods.append(_period(_months_label(spans), spans))
    return tuple(periods)
