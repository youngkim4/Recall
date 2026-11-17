"""Quotes and dates are checked against real messages. References map back to ids."""

from typing import Any

import pytest

from report_engine.digest import RawMoment
from report_engine.transcript import Line, build_lines
from report_engine.verify import MessageIndex, check_moments, normalise

from .synth import labeler, message

# special characters by code point, so no editor can change them
LDQUO, RDQUO, RSQUO, NBSP, DOTS, DASH = (chr(code) for code in (0x201C, 0x201D, 0x2019, 0xA0, 0x2026, 0x2014))
ACUTE, HEART, VARIATION = chr(0x301), chr(0x2764), chr(0xFE0F)

MESSAGES = [
    message("id-1", "2024-06-01 09:00:00", "Morning. Did you sleep at all?"),
    message("id-2", "2024-06-01 09:05:00", "Barely. I got the job at the observatory!", is_from_me=True),
    message(
        "id-3",
        "2024-06-01 09:06:00",
        f"No way {DASH} {LDQUO}the{RDQUO} observatory? I{RSQUO}m so proud of you",
    ),
    message("id-4", "2024-06-03 20:00:00", "Dinner on me   tonight"),
    message("id-5", "2024-06-04 08:00:00", "Thanks for last night", is_from_me=True),
    message("id-6", "2024-06-07 11:00:00", "Remember the harbor lights"),
    message("id-7", "2024-06-20 08:00:00", "ok"),
]


def lines() -> tuple[Line, ...]:
    return build_lines(MESSAGES, labeler)


def raw(**overrides: Any) -> RawMoment:
    values: dict[str, Any] = {
        "date": "2024-06-01",
        "title": "The job",
        "detail": "You got the observatory job.",
        "category": "milestone",
        "score": 0.9,
        "quote": "I got the job at the observatory!",
        "refs": ("m2",),
    }
    values.update(overrides)
    return RawMoment(**values)


def check(*moments: RawMoment, period: tuple[Line, ...] | None = None):
    every = lines()
    return check_moments(moments, period if period is not None else every, MessageIndex(every))


def test_normalise_evens_out_space_and_quote_marks() -> None:
    assert normalise(f"  {LDQUO}I{RSQUO}m{NBSP}here{RDQUO}{DOTS}  ") == "\"I'm here\"..."
    assert normalise("a\n\tb") == "a b"


def test_normalise_composes_accents_and_drops_variation_selectors() -> None:
    assert normalise("cafe" + ACUTE) == normalise("caf" + chr(0xE9))
    assert normalise(HEART + VARIATION) == normalise(HEART)


def test_normalise_keeps_letter_case() -> None:
    assert normalise("Hello") != normalise("hello")


def test_a_quote_from_a_cited_message_is_verified() -> None:
    result = check(raw())
    (moment,) = result.moments
    assert moment.quote == "I got the job at the observatory!"
    assert moment.quote_verified is True
    assert moment.message_ids == ("id-2",)
    assert (result.quotes_cleared, result.dates_repaired, result.dropped) == (0, 0, 0)


def test_references_map_back_to_message_ids_in_order_without_repeats() -> None:
    (moment,) = check(raw(refs=("m3", "m2", "m3", "m99", "junk", "m1"))).moments
    assert moment.message_ids == ("id-3", "id-2", "id-1")


def test_a_quote_may_differ_in_space_and_quote_marks_only() -> None:
    (moment,) = check(raw(quote='"the" observatory?  I\'m so proud', refs=("m3",))).moments
    assert moment.quote_verified is True
    (spaced,) = check(raw(date="2024-06-03", quote="Dinner on me tonight", refs=("m4",))).moments
    assert spaced.quote_verified is True


def test_wrapping_quote_marks_and_edge_dots_are_ignored() -> None:
    wrapped = f"{LDQUO}{DOTS}got the job at the observatory{DOTS}{RDQUO}"
    (moment,) = check(raw(quote=wrapped)).moments
    assert moment.quote_verified is True
    assert moment.quote == "got the job at the observatory"


def test_a_quote_within_three_days_is_verified_and_its_message_is_added() -> None:
    (moment,) = check(raw(quote="Dinner on me tonight", refs=("m1",))).moments
    assert moment.quote_verified is True
    assert moment.message_ids == ("id-1", "id-4")


def test_a_quote_four_days_away_is_cleared() -> None:
    result = check(raw(date="2024-06-03", quote="Remember the harbor lights", refs=("m4",)))
    (moment,) = result.moments
    assert moment.quote == ""
    assert moment.quote_verified is False
    assert moment.message_ids == ("id-4",)
    assert result.quotes_cleared == 1


def test_a_quote_exactly_three_days_away_still_counts() -> None:
    result = check(raw(date="2024-06-04", quote="Remember the harbor lights", refs=()))
    assert result.moments[0].quote_verified is True
    assert result.moments[0].message_ids == ("id-6",)


def test_a_quote_that_is_in_no_message_is_cleared_and_flagged() -> None:
    result = check(raw(quote="I never said this at all"))
    (moment,) = result.moments
    assert (moment.quote, moment.quote_verified) == ("", False)
    assert moment.message_ids == ("id-2",)
    assert result.quotes_cleared == 1


def test_a_quote_with_changed_letter_case_is_cleared() -> None:
    result = check(raw(quote="i got the job at the observatory!"))
    assert result.moments[0].quote_verified is False
    assert result.quotes_cleared == 1


def test_a_short_quote_counts_only_in_a_cited_message() -> None:
    near = check(raw(date="2024-06-20", quote="ok", refs=("m7",)))
    assert near.moments[0].quote_verified is True
    far = check(raw(date="2024-06-20", quote="ok", refs=()))
    assert far.moments[0].quote_verified is False
    assert far.quotes_cleared == 1


def test_an_empty_quote_is_not_verified_and_not_counted() -> None:
    result = check(raw(quote="   "))
    assert (result.moments[0].quote, result.moments[0].quote_verified) == ("", False)
    assert result.quotes_cleared == 0


def test_a_date_with_no_message_moves_to_the_earliest_cited_day() -> None:
    result = check(raw(date="2024-06-02", refs=("m4", "m2")))
    assert result.moments[0].date == "2024-06-01"
    assert result.dates_repaired == 1


@pytest.mark.parametrize("date", ["June 1st", "", "2024-02-30", "2031-01-01"])
def test_a_bad_date_with_references_is_repaired(date: str) -> None:
    result = check(raw(date=date))
    assert result.moments[0].date == "2024-06-01"
    assert result.dates_repaired == 1


@pytest.mark.parametrize("date", ["June 1st", "2024-06-02", "2019-01-01"])
def test_a_bad_date_without_references_drops_the_moment(date: str) -> None:
    result = check(raw(date=date, refs=("m99",)))
    assert result.moments == ()
    assert result.dropped == 1


def test_a_date_from_another_period_is_not_trusted() -> None:
    every = lines()
    june_first = tuple(line for line in every if line.day == "2024-06-01")
    result = check_moments([raw(date="2024-06-03", quote="", refs=())], june_first, MessageIndex(every))
    assert result.moments == ()
    assert result.dropped == 1


def test_a_good_date_without_references_keeps_the_moment() -> None:
    (moment,) = check(raw(quote="", refs=())).moments
    assert moment.date == "2024-06-01"
    assert moment.message_ids == ()


@pytest.mark.parametrize("field, value", [("category", "gossip"), ("title", "   "), ("category", "")])
def test_a_moment_that_is_not_usable_is_dropped(field: str, value: str) -> None:
    result = check(raw(**{field: value}))
    assert result.moments == ()
    assert result.dropped == 1


@pytest.mark.parametrize("score, expected", [(-2.0, 0.0), (3.5, 1.0), (0.4, 0.4)])
def test_scores_are_kept_between_zero_and_one(score: float, expected: float) -> None:
    assert check(raw(score=score)).moments[0].score == expected


def test_text_fields_are_put_on_one_line() -> None:
    (moment,) = check(raw(title="  The\njob ", detail="You  got\nit.")).moments
    assert (moment.title, moment.detail) == ("The job", "You got it.")


def test_counts_add_up_over_several_moments() -> None:
    result = check(
        raw(),
        raw(quote="made up words here"),
        raw(date="2024-06-02"),
        raw(category="gossip"),
    )
    assert len(result.moments) == 3
    assert (result.quotes_cleared, result.dates_repaired, result.dropped) == (1, 1, 1)


def test_index_yields_the_nearest_days_first() -> None:
    index = MessageIndex(lines())
    assert [line.id for line in index.near("2024-06-02", 1)] == ["id-1", "id-2", "id-3", "id-4"]
    assert [line.id for line in index.near("2024-06-05", 3)] == ["id-5", "id-4", "id-6"]
    assert list(index.near("2024-07-30", 3)) == []
