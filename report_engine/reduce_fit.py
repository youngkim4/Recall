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
    """The lines that go into the request, and how many were changed to fit."""

    periods: tuple[PeriodSummary, ...]
    moments: tuple[Moment, ...]
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


def _shorten(period: PeriodSummary, target: int, meter: TokenMeter) -> PeriodSummary:
    """The same period with its summary cut until its line is no larger than `target`."""
    text = shield(period.summary)
    cut = period
    size = meter.count(period_line(cut))
    for _ in range(MAX_CUTS):
        if size <= target or not text:
            break
        text = text[: int(len(text) * target / size * CUT_SAFETY)].rstrip()
        cut = period.model_copy(update={"summary": f"{text} {ELISION_MARK}"})
        size = meter.count(period_line(cut))
    return cut


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
    period_sizes = meter.count_many([period_line(period) for period in periods])
    moment_sizes = meter.count_many([moment_line(moment) for moment in moments])
    if sum(period_sizes) + sum(moment_sizes) <= room:
        return Fitted(tuple(periods), tuple(moments))

    cap = max(MIN_PERIOD_LINE_TOKENS, _allowance(period_sizes, room - sum(moment_sizes)))
    fitted_periods = tuple(
        _shorten(period, cap, meter) if size > cap else period
        for period, size in zip(periods, period_sizes, strict=True)
    )
    shortened = sum(new is not old for new, old in zip(fitted_periods, periods, strict=True))
    total = sum(meter.count_many([period_line(period) for period in fitted_periods])) + sum(
        moment_sizes
    )

    left_out: set[int] = set()
    for index in _weakest_first(moments):
        if total <= room:
            break
        left_out.add(index)
        total -= moment_sizes[index]
    if total > room:
        raise ReduceError(
            f"the reduce request is too large for the model: {total} tokens for {len(periods)} "
            f"periods after shortening, and room for {room}"
        )
    kept = tuple(moment for index, moment in enumerate(moments) if index not in left_out)
    return Fitted(fitted_periods, kept, shortened=shortened, left_out=len(left_out))
