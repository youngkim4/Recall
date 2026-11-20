"""The time a question is about, in days, from its words and today's date."""

from datetime import datetime
from typing import Optional

import pytest

from chat_engine.labeler import MaskingLabeler
from chat_engine.question_time import Period, question_time, time_line

NOW = datetime(2026, 11, 16, 9, 30)  # a Monday
HELD = (2019, 2026)


def _days(question: str, now: datetime = NOW, held: tuple[int, int] = HELD) -> Optional[tuple[str, str]]:
    period = question_time(question, now, held, MaskingLabeler().redact)
    return None if period is None else (period.since, period.until)


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("what was 2021 like", ("2021-01-01", "2021-12-31")),
        ("who was my best friend from 2019-2021?", ("2019-01-01", "2021-12-31")),
        ("who did I text most between 2021 and 2019", ("2019-01-01", "2021-12-31")),
        ("who wrote most from 2020-06-01 to 2021-03-31", ("2020-01-01", "2021-12-31")),
    ],
)
def test_years_and_ranges_keep_their_whole_years(question: str, days: tuple[str, str]) -> None:
    assert _days(question) == days


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("what happened in March 2024?", ("2024-03-01", "2024-03-31")),
        ("anything from mar 2024", ("2024-03-01", "2024-03-31")),
        ("who wrote in Sept. 2023", ("2023-09-01", "2023-09-30")),
        ("february of 2024", ("2024-02-01", "2024-02-29")),
        ("from March 2024 to June 2024", ("2024-03-01", "2024-06-30")),
        ("what about May 2025 and 2026", ("2025-05-01", "2026-12-31")),
    ],
)
def test_a_month_with_a_year_is_that_month(question: str, days: tuple[str, str]) -> None:
    assert _days(question) == days


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("what did I do in fall 2022", ("2022-09-01", "2022-11-30")),
        ("autumn 2022", ("2022-09-01", "2022-11-30")),
        ("the summer of 2023", ("2023-06-01", "2023-08-31")),
        ("spring 2020", ("2020-03-01", "2020-05-31")),
        ("winter 2021", ("2021-12-01", "2022-02-28")),
        ("winter 2023", ("2023-12-01", "2024-02-29")),
        ("early 2023", ("2023-01-01", "2023-04-30")),
        ("late 2023", ("2023-09-01", "2023-12-31")),
    ],
)
def test_a_season_or_a_part_of_a_year(question: str, days: tuple[str, str]) -> None:
    assert _days(question) == days


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("who did I talk to most last year?", ("2025-01-01", "2025-12-31")),
        ("this year", ("2026-01-01", "2026-11-16")),
        ("last month", ("2026-10-01", "2026-10-31")),
        ("what did we cook last summer?", ("2026-06-01", "2026-08-31")),
        ("last winter", ("2025-12-01", "2026-02-28")),
        ("last fall", ("2025-09-01", "2025-11-30")),
        ("two years ago", ("2024-01-01", "2024-12-31")),
        ("3 years ago", ("2023-01-01", "2023-12-31")),
        ("a year ago", ("2025-01-01", "2025-12-31")),
        ("last week", ("2026-11-09", "2026-11-15")),
        ("who have I talked to recently?", ("2026-08-18", "2026-11-16")),
        ("anything lately", ("2026-08-18", "2026-11-16")),
        ("in the last 3 months", ("2026-08-16", "2026-11-16")),
        ("over the past six months", ("2026-05-16", "2026-11-16")),
    ],
)
def test_phrases_relative_to_today(question: str, days: tuple[str, str]) -> None:
    period = question_time(question, NOW, HELD)
    assert period is not None and period.relative
    assert (period.since, period.until) == days


@pytest.mark.parametrize(
    ("question", "days"),
    [
        ("what happened in march", ("2026-03-01", "2026-03-31")),
        ("who did I see during December?", ("2025-12-01", "2025-12-31")),
        ("plans in november", ("2026-11-01", "2026-11-30")),
        ("in may", ("2026-05-01", "2026-05-31")),
    ],
)
def test_a_month_alone_is_the_latest_such_month(question: str, days: tuple[str, str]) -> None:
    assert _days(question) == days


def test_last_summer_in_summer_is_the_summer_before() -> None:
    assert _days("last summer", now=datetime(2026, 7, 10)) == ("2025-06-01", "2025-08-31")


def test_a_named_period_wins_over_a_relative_one() -> None:
    period = question_time("last year, I mean in 2021", NOW, HELD)
    assert period == Period("2021-01-01", "2021-12-31", "2021", relative=False, words=("2021",))


@pytest.mark.parametrize(
    "question",
    [
        "",
        "Who do I text the most?",
        "I may text her",
        "march on",
        "summer plans?",
        "sent 3000 messages",
        "what about 20190 or 1989",
        "what happened in ２０２４?",
    ],
)
def test_no_time_gives_none(question: str) -> None:
    assert question_time(question, NOW, HELD) is None


@pytest.mark.parametrize(
    "question",
    [
        "who is +1 (720) 555-2019?",
        "who is +82 10 2019 2020?",
        "email jo2019@example.org",
        "what did mar.2024@example.org send",
        "texts from March 720-555-2024?",
        "who has 202-555-2021",
    ],
)
def test_digits_of_a_handle_never_become_a_date(question: str) -> None:
    assert question_time(question, NOW, HELD, MaskingLabeler().redact) is None


def test_a_masked_number_that_reads_as_two_years_is_no_period() -> None:
    masked = MaskingLabeler(known_handles=["+85220192020"])
    assert question_time("who is 2019 2020?", NOW, HELD, masked.redact) is None
    assert _days("who is 2019 2020?") == ("2019-01-01", "2020-12-31")


def test_a_handle_beside_a_relative_phrase_leaves_the_phrase() -> None:
    assert _days("what did +1 (720) 555-2019 send last month") == ("2026-10-01", "2026-10-31")


@pytest.mark.parametrize(
    ("question", "held"),
    [
        ("what happened in March 2030", HELD),
        ("last year", (2019, 2024)),
        ("recently", (2019, 2024)),
        ("winter 2017", HELD),
    ],
)
def test_a_period_outside_the_archive_gives_none(question: str, held: tuple[int, int]) -> None:
    assert question_time(question, NOW, held) is None


def test_winter_reaching_into_the_archive_counts() -> None:
    assert _days("winter 2018") == ("2018-12-01", "2019-02-28")
    assert _days("winter 2018", held=(2020, 2026)) is None


def test_the_words_of_the_phrase_are_kept_for_the_search() -> None:
    period = question_time("sourdough last summer?", NOW, HELD)
    assert period is not None and period.words == ("last", "summer")


def test_the_line_names_the_days_and_the_phrase() -> None:
    period = question_time("who did I talk to most last summer?", NOW, HELD)
    assert time_line(period).startswith("The question is about 2026-06-01 to 2026-08-31 (last summer).")
    assert time_line(None) == ""
