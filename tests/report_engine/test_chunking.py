"""Periods: calendar boundaries, a token budget, and boundaries that hold still."""

import random
from collections.abc import Sequence
from datetime import date, timedelta

import pytest

from report_engine.chunking import (
    HEADER_TOKENS,
    DaySpan,
    Period,
    line_sizer,
    measure_days,
    plan_periods,
)
from report_engine.tokens import TokenMeter
from report_engine.transcript import build_lines

from .synth import WordCounter, conversation, labeler

LINE = 5


def flat_lines(start: int, stop: int) -> list[int]:
    return [LINE] * (stop - start)


def days(*specs: tuple[str, int]) -> tuple[DaySpan, ...]:
    """Day spans from (day, number of lines). Every line weighs LINE tokens."""
    out: list[DaySpan] = []
    cursor = 0
    for day, count in specs:
        out.append(
            DaySpan(day=day, start=cursor, stop=cursor + count, tokens=count * LINE + HEADER_TOKENS)
        )
        cursor += count
    return tuple(out)


def labels(periods: Sequence[Period]) -> list[str]:
    return [period.label for period in periods]


def test_no_days_means_no_periods() -> None:
    assert plan_periods((), 100, flat_lines) == ()


def test_a_conversation_under_the_budget_is_one_period() -> None:
    spans = days(("2019-03-02", 4), ("2021-07-15", 4), ("2024-06-01", 4))
    periods = plan_periods(spans, 10_000, flat_lines)
    assert labels(periods) == ["Mar 2019 – Jun 2024"]
    assert (periods[0].first_day, periods[0].last_day) == ("2019-03-02", "2024-06-01")
    assert (periods[0].start, periods[0].stop) == (0, 12)
    assert periods[0].tokens == sum(span.tokens for span in spans)


def test_a_single_month_is_named_by_the_month() -> None:
    periods = plan_periods(days(("2024-06-01", 2), ("2024-06-20", 2)), 10_000, flat_lines)
    assert labels(periods) == ["Jun 2024"]


def test_months_pack_until_the_next_one_does_not_fit() -> None:
    spans = days(("2024-01-10", 10), ("2024-02-10", 10), ("2024-03-10", 10), ("2024-04-10", 10))
    month = 10 * LINE + HEADER_TOKENS
    periods = plan_periods(spans, 2 * month, flat_lines)
    assert labels(periods) == ["Jan 2024 – Feb 2024", "Mar 2024 – Apr 2024"]
    assert all(period.tokens <= 2 * month for period in periods)


def test_a_month_over_the_budget_is_split_by_days_and_never_joins_its_neighbours() -> None:
    spans = days(
        ("2024-01-10", 2),
        ("2024-02-01", 10),
        ("2024-02-02", 10),
        ("2024-02-03", 10),
        ("2024-03-10", 2),
    )
    day = 10 * LINE + HEADER_TOKENS
    periods = plan_periods(spans, 2 * day, flat_lines)
    assert labels(periods) == ["Jan 2024", "2024-02-01 – 2024-02-02", "2024-02-03", "Mar 2024"]


def test_a_day_over_the_budget_is_split_into_numbered_parts() -> None:
    spans = days(("2024-02-01", 2), ("2024-02-02", 25), ("2024-02-03", 2))
    budget = 10 * LINE + HEADER_TOKENS
    periods = plan_periods(spans, budget, flat_lines)
    assert labels(periods) == [
        "2024-02-01",
        "2024-02-02 part 1",
        "2024-02-02 part 2",
        "2024-02-02 part 3",
        "2024-02-03",
    ]
    assert [(p.start, p.stop) for p in periods] == [(0, 2), (2, 12), (12, 22), (22, 27), (27, 29)]
    assert all(period.tokens <= budget for period in periods)


def test_one_line_larger_than_the_budget_stands_alone() -> None:
    spans = (DaySpan(day="2024-02-02", start=0, stop=3, tokens=5 + 500 + 5 + HEADER_TOKENS),)
    periods = plan_periods(spans, 100, lambda start, stop: [5, 500, 5][start:stop])
    assert [(p.start, p.stop) for p in periods] == [(0, 1), (1, 2), (2, 3)]
    assert periods[1].tokens == 500 + HEADER_TOKENS


def test_periods_cover_every_line_once_and_in_order() -> None:
    rng = random.Random(3)
    spans = random_days(rng, months=14)
    periods = plan_periods(spans, 400, flat_lines)
    assert periods[0].start == 0
    assert periods[-1].stop == spans[-1].stop
    for earlier, later in zip(periods, periods[1:], strict=False):
        assert earlier.stop == later.start
        assert earlier.last_day <= later.first_day
    assert len(set(labels(periods))) == len(periods)


@pytest.mark.parametrize("budget", [0, -5])
def test_the_budget_must_be_positive(budget: int) -> None:
    with pytest.raises(ValueError):
        plan_periods(days(("2024-06-01", 1)), budget, flat_lines)


def random_days(rng: random.Random, months: int) -> tuple[DaySpan, ...]:
    specs: list[tuple[str, int]] = []
    cursor = date(2020, 1, 1)
    for _ in range(months * 6):
        cursor += timedelta(days=rng.randint(1, 9))
        specs.append((cursor.isoformat(), rng.choice([1, 2, 3, 5, 8, 40, 130])))
    return days(*specs)


def grow(rng: random.Random, spans: Sequence[DaySpan]) -> tuple[DaySpan, ...]:
    """New messages only: the last day gets longer, and later days may follow."""
    specs = [(span.day, span.stop - span.start) for span in spans]
    last_day, last_count = specs[-1]
    specs[-1] = (last_day, last_count + rng.randint(0, 150))
    cursor = date.fromisoformat(last_day)
    for _ in range(rng.randint(0, 4)):
        cursor += timedelta(days=rng.randint(1, 40))
        specs.append((cursor.isoformat(), rng.choice([1, 3, 20, 130])))
    return days(*specs)


@pytest.mark.parametrize("seed", range(40))
def test_new_messages_change_only_the_last_period(seed: int) -> None:
    rng = random.Random(seed)
    budget = rng.choice([150, 400, 900])
    before_days = random_days(rng, months=rng.randint(1, 10))
    before = plan_periods(before_days, budget, flat_lines)
    after = plan_periods(grow(rng, before_days), budget, flat_lines)
    assert after[: len(before) - 1] == before[:-1]


def test_measure_days_counts_each_day_with_its_header() -> None:
    lines = build_lines(conversation("2024-06-01", days=3, per_day=2, words=4), labeler)
    spans = measure_days(lines, TokenMeter(WordCounter()))
    assert [(span.day, span.start, span.stop) for span in spans] == [
        ("2024-06-01", 0, 2),
        ("2024-06-02", 2, 4),
        ("2024-06-03", 4, 6),
    ]
    # header: 4 words; line: reference, time, name, 4 words
    assert [span.tokens for span in spans] == [4 + 2 * 7] * 3


def test_line_sizer_counts_single_lines() -> None:
    lines = build_lines(conversation("2024-06-01", days=1, per_day=3, words=4), labeler)
    sizer = line_sizer(lines, TokenMeter(WordCounter()))
    assert list(sizer(0, 3)) == [7, 7, 7]
    assert list(sizer(1, 2)) == [7]


def test_real_lines_split_the_same_way_before_and_after_new_messages() -> None:
    meter = TokenMeter(WordCounter())
    old = conversation("2023-01-01", days=400, per_day=6, words=8)
    new = [*old, *conversation("2024-02-05", days=3, per_day=6, words=8, first_id=10_000)]

    def plan(messages: list) -> tuple[Period, ...]:
        lines = build_lines(messages, labeler)
        return plan_periods(measure_days(lines, meter), 4_000, line_sizer(lines, meter))

    before, after = plan(old), plan(new)
    assert len(before) > 5
    assert after[: len(before) - 1] == before[:-1]
