"""Which moments make the report: no duplicates, a floor, a spread, no filler."""

from typing import Any

from report_engine.models import Moment
from report_engine.selection import select_moments


def moment(date: str, title: str, score: float, **overrides: Any) -> Moment:
    values: dict[str, Any] = {
        "date": date,
        "title": title,
        "detail": "detail",
        "category": "milestone",
        "score": score,
    }
    values.update(overrides)
    return Moment(**values)


def titles(moments: tuple[Moment, ...]) -> list[str]:
    return [item.title for item in moments]


def pick(*periods: list[Moment], floor: float = 0.4, cap: int = 40) -> tuple[Moment, ...]:
    return select_moments(periods, score_floor=floor, max_moments=cap)


def test_nothing_in_means_nothing_out() -> None:
    assert pick() == ()
    assert pick([], []) == ()


def test_the_result_is_in_date_order() -> None:
    chosen = pick([moment("2024-03-01", "Later", 0.9)], [moment("2024-01-01", "Earlier", 0.5)])
    assert titles(chosen) == ["Earlier", "Later"]


def test_the_same_event_in_two_neighbouring_periods_is_kept_once() -> None:
    first = moment("2024-01-31", "Trip to the coast planned", 0.6)
    second = moment("2024-02-01", "Planned the trip to the coast", 0.8)
    chosen = pick([first], [second])
    assert chosen == (second,)


def test_two_moments_that_cite_the_same_message_are_one_event() -> None:
    low = moment("2024-01-10", "The call", 0.5, message_ids=("41", "42"))
    high = moment("2024-01-12", "An apology at last", 0.9, message_ids=("42",))
    assert pick([low], [high]) == (high,)


def test_between_equal_scores_the_one_with_a_verified_quote_stays() -> None:
    plain = moment("2024-01-10", "First kiss", 0.8)
    quoted = moment(
        "2024-01-10",
        "First kiss",
        0.8,
        quote="finally",
        quote_verified=True,
        quote_message_id="9",
        message_ids=("9",),
    )
    assert pick([plain], [quoted]) == (quoted,)


def test_like_titles_far_apart_are_two_events() -> None:
    chosen = pick(
        [moment("2024-01-10", "Big argument", 0.7)], [moment("2024-03-10", "Big argument", 0.7)]
    )
    assert len(chosen) == 2


def test_unlike_titles_on_the_same_day_are_two_events() -> None:
    chosen = pick(
        [moment("2024-01-10", "Big argument", 0.7), moment("2024-01-10", "New puppy", 0.7)]
    )
    assert len(chosen) == 2


def test_moments_under_the_floor_are_left_out() -> None:
    chosen = pick(
        [moment("2024-01-10", "Minor", 0.39), moment("2024-01-11", "Notable", 0.4)],
        floor=0.4,
    )
    assert titles(chosen) == ["Notable"]


def test_a_weak_duplicate_does_not_take_the_strong_one_with_it() -> None:
    weak = moment("2024-01-10", "The move", 0.2, message_ids=("7",))
    strong = moment("2024-01-11", "Moving in together", 0.9, message_ids=("7",))
    assert pick([weak], [strong], floor=0.4) == (strong,)


def test_a_quiet_period_adds_no_moment() -> None:
    busy = [moment("2024-01-10", "A", 0.9), moment("2024-01-12", "B", 0.8)]
    quiet = [moment("2024-02-10", "Small talk", 0.1)]
    chosen = pick(busy, quiet, [])
    assert titles(chosen) == ["A", "B"]


def test_the_cap_is_spread_over_the_periods() -> None:
    loud_titles = ["Harbor walk", "Lost keys", "Night train", "Kitchen fire", "Old letters"]
    loud = [moment(f"2024-01-{2 + 6 * n:02d}", title, 0.9) for n, title in enumerate(loud_titles)]
    soft_one = [moment("2024-03-05", "Soft one", 0.5)]
    soft_two = [moment("2024-05-05", "Second quiet thing", 0.45)]
    chosen = pick(loud, soft_one, soft_two, cap=4)
    assert len(chosen) == 4
    assert {"Soft one", "Second quiet thing"} <= set(titles(chosen))
    assert sum(title in loud_titles for title in titles(chosen)) == 2


def test_when_the_cap_is_smaller_than_the_periods_the_strongest_period_bests_win() -> None:
    periods = [[moment(f"2024-{month:02d}-10", f"M{month}", month / 10)] for month in range(4, 10)]
    chosen = pick(*periods, cap=3)
    assert titles(chosen) == ["M7", "M8", "M9"]


def test_no_filler_when_there_are_fewer_moments_than_the_cap() -> None:
    chosen = pick([moment("2024-01-10", "Only one", 0.9)], cap=40)
    assert titles(chosen) == ["Only one"]


def test_a_cap_of_zero_selects_nothing() -> None:
    assert pick([moment("2024-01-10", "A", 0.9)], cap=0) == ()


def test_inputs_are_not_changed() -> None:
    period = [moment("2024-03-01", "B", 0.5), moment("2024-01-01", "A", 0.9)]
    snapshot = list(period)
    pick(period)
    assert period == snapshot


def test_many_candidates_are_handled_quickly() -> None:
    periods = [
        [
            moment(
                f"20{10 + index // 300:02d}-{1 + (index // 25) % 12:02d}-{1 + index % 25:02d}",
                f"Event number {index}",
                0.5,
            )
        ]
        for index in range(3000)
    ]
    chosen = pick(*periods, cap=50)
    assert len(chosen) == 50
