"""Make the reduce request fit the model.

The reduce request holds one line for each period and one for each moment.
When the lines do not fit, period lines are shortened first, the longest
first, down to a floor. Then moments are left out, the weakest first. The
caller is told what was cut: nothing is cut in silence, and nothing is paid
for a request that can not be sent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from report_engine.errors import ReduceError
from report_engine.models import Moment, PeriodSummary
from report_engine.plan import MIN_PERIOD_LINE_TOKENS
from report_engine.prompts import moment_line, period_line
from report_engine.tokens import TokenMeter
from report_engine.transcript import ELISION_MARK, shield

MAX_CUTS = 6
CUT_SAFETY = 0.9


@dataclass(frozen=True)
class Fitted:
    """The lines that go into the request, their size when each line is counted
    on its own, and how many were changed to fit."""

    periods: tuple[PeriodSummary, ...]
    moments: tuple[Moment, ...]
    tokens: int
    shortened: int = 0
    left_out: int = 0


def _allowance(sizes: Sequence[int], room: int) -> int:
    """The largest cap for which sum(min(size, cap)) fits the room."""
    ordered = sorted(sizes)
    remaining = room
    for index, size in enumerate(ordered):
        share = remaining // (len(ordered) - index)
        if size > share:
            return share
        remaining -= size
    return ordered[-1] if ordered else 0


def _bare(period: PeriodSummary) -> PeriodSummary:
    """The period with nothing of its summary but the mark. The line of a period
    that could not be analysed is fixed text: it stays as it is."""
    if period.error is not None:
        return period
    return period.model_copy(update={"summary": ELISION_MARK})


def _shorten(period: PeriodSummary, target: int, meter: TokenMeter) -> PeriodSummary:
    """The same period with its summary cut until its line is no larger than
    `target`. When no start of the summary fits, only the mark stays."""
    if period.error is not None:
        return period
    text = shield(period.summary)
    cut = period
    size = _line_sizes([period_line(cut)], meter)[0]
    for _ in range(MAX_CUTS):
        if size <= target or not text:
            break
        text = text[: int(len(text) * target / size * CUT_SAFETY)].rstrip()
        cut = period.model_copy(update={"summary": f"{text} {ELISION_MARK}"})
        size = _line_sizes([period_line(cut)], meter)[0]
    return cut if size <= target else _bare(period)


def _period_tokens(periods: Sequence[PeriodSummary], meter: TokenMeter) -> int:
    return sum(_line_sizes([period_line(period) for period in periods], meter))


def _squeeze(
    periods: Sequence[PeriodSummary], sizes: Sequence[int], room: int, meter: TokenMeter
) -> tuple[PeriodSummary, ...]:
    """Period lines for a room that holds no moment and not the usual length of
    a summary: each line as short as the room asks, down to the mark alone."""
    cap = _allowance(sizes, room)
    squeezed = tuple(
        _shorten(period, cap, meter) if size > cap else period
        for period, size in zip(periods, sizes, strict=True)
    )
    if _period_tokens(squeezed, meter) <= room:
        return squeezed
    return tuple(_bare(period) for period in periods)


def _line_sizes(lines: Sequence[str], meter: TokenMeter) -> list[int]:
    """Each line with its line break, as it stands in the request."""
    return meter.count_many([f"{line}\n" for line in lines])


def _weakest_first(moments: Sequence[Moment]) -> list[int]:
    return sorted(range(len(moments)), key=lambda index: (moments[index].score, -index))


def fit_reduce(
    periods: Sequence[PeriodSummary],
    moments: Sequence[Moment],
    room: int,
    meter: TokenMeter,
) -> Fitted:
    """Period and moment lines that fit `room` tokens. Raises ReduceError when
    even the shortest lines do not fit."""
    period_sizes = _line_sizes([period_line(period) for period in periods], meter)
    moment_sizes = _line_sizes([moment_line(moment) for moment in moments], meter)
    uncut = sum(period_sizes) + sum(moment_sizes)
    if uncut <= room:
        return Fitted(tuple(periods), tuple(moments), tokens=uncut)

    cap = max(MIN_PERIOD_LINE_TOKENS, _allowance(period_sizes, room - sum(moment_sizes)))
    fitted_periods = tuple(
        _shorten(period, cap, meter) if size > cap else period
        for period, size in zip(periods, period_sizes, strict=True)
    )
    total = _period_tokens(fitted_periods, meter) + sum(moment_sizes)

    left_out: set[int] = set()
    for index in _weakest_first(moments):
        if total <= room:
            break
        left_out.add(index)
        total -= moment_sizes[index]
    if total > room:
        # no moment is left to take out: the summaries go below their usual length
        fitted_periods = _squeeze(periods, period_sizes, room, meter)
        total = _period_tokens(fitted_periods, meter)
    if total > room:
        raise ReduceError(
            f"the reduce request is too large for the model: {total} tokens for {len(periods)} "
            f"periods after shortening, and room for {room}"
        )
    shortened = sum(new is not old for new, old in zip(fitted_periods, periods, strict=True))
    kept = tuple(moment for index, moment in enumerate(moments) if index not in left_out)
    return Fitted(fitted_periods, kept, tokens=total, shortened=shortened, left_out=len(left_out))
