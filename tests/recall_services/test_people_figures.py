"""people_figures.py: the figures of a person, from the archive only, against hand counts."""

from __future__ import annotations

import dataclasses

import pytest

from recall_services.people_figures import Figures, figures_of, read_figures
from tests.recall_services.people_kit import (
    GROUP,
    GROUP_FIGURES,
    MINA,
    MINA_FIGURES,
    MINA_MAIL,
    RAVI,
    THEO,
    THEO_FIGURES,
    town,  # noqa: F401
)


def test_the_two_threads_of_one_person_are_counted_as_one(town) -> None:
    assert read_figures(town.archive, (MINA, MINA_MAIL)).as_dict() == MINA_FIGURES


def test_one_thread_and_a_group_against_hand_counts(town) -> None:
    assert read_figures(town.archive, (THEO,)).as_dict() == THEO_FIGURES
    assert read_figures(town.archive, (GROUP,)).as_dict() == GROUP_FIGURES


def test_the_figures_agree_with_the_figures_of_the_archive(town) -> None:
    found = read_figures(town.archive, (MINA, MINA_MAIL))
    stats = town.archive.stats([MINA, MINA_MAIL])
    assert (found.messages, found.sent, found.received) == (
        stats["total_messages"], stats["sent_count"], stats["received_count"],
    )  # fmt: skip
    assert found.active_days == stats["active_days"]
    assert found.longest_gap_days == round(stats["longest_gap_days"], 2)
    assert (found.first_day, found.last_day) == (stats["first_timestamp"][:10], stats["last_timestamp"][:10])


def test_a_message_with_no_text_is_a_message(town) -> None:
    found = read_figures(town.archive, (RAVI,))
    assert (found.messages, found.received, found.active_days, found.wrote_first_share) == (1, 1, 1, 0.0)


def test_no_message_gives_zeros_and_no_days() -> None:
    assert figures_of([]) == Figures(None, None, 0, 0, 0, 0, 0.0, None, 0.0)


def test_a_tie_of_months_goes_to_the_earliest() -> None:
    rows = [
        {"timestamp": "2023-05-01 10:00:00", "sent_at": 1_682_960_400, "is_from_me": True},
        {"timestamp": "2023-06-01 10:00:00", "sent_at": 1_685_638_800, "is_from_me": False},
    ]
    found = figures_of(rows)
    assert (found.busiest_month, found.wrote_first_share, found.longest_gap_days) == ("2023-05", 0.5, 31.0)


def test_the_figures_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        figures_of([]).messages = 3  # type: ignore[misc]
