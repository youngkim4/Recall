"""people_reply.py: a reply of the profile model, checked for shape and cleaned."""

from __future__ import annotations

from typing import Any

import pytest

from recall_services import people_prompt
from recall_services.people_reply import ProfileText, TurningPoint, clean_text, parse_reply
from recall_services.names import Names
from tests.recall_services.people_kit import MINA_RECORD, THEO, profile_reply

SPAN = ("2023-01-02", "2023-02-20")
REDACT = Names.from_records([MINA_RECORD]).chat_labeler([{"id": THEO, "members": [THEO]}]).redact


def parse(reply: Any, *, is_group: bool = False) -> ProfileText | None:
    return parse_reply(reply, span=SPAN, is_group=is_group, redact=REDACT)


def test_a_reply_in_the_right_shape() -> None:
    found = parse(
        profile_reply(
            kind="talking_stage",
            status="faded",
            summary="You and Mina flirted in January.",
            turning_points=[{"date": "2023-02-14", "what": "You sent a valentine."}],
        )
    )
    assert found == ProfileText(
        "talking_stage", "faded", "You and Mina flirted in January.", (TurningPoint("2023-02-14", "You sent a valentine."),)
    )


@pytest.mark.parametrize(
    "reply",
    [
        None,
        "not an object",
        {"kind": "friend"},
        profile_reply(summary=3),
        profile_reply(kind=None),
        profile_reply(turning_points="2023-02-14"),
        profile_reply(turning_points=[{"date": "2023-02-14"}]),
        profile_reply(turning_points=["2023-02-14"]),
        profile_reply(summary="   "),
        profile_reply(summary="(m12)"),
    ],
)
def test_a_reply_in_the_wrong_shape_is_none(reply: Any) -> None:
    assert parse(reply) is None


def test_an_unknown_kind_or_status_becomes_unclear() -> None:
    found = parse(profile_reply(kind="situationship", status="complicated"))
    assert found is not None and (found.kind, found.status) == ("unclear", "unclear")


def test_a_group_is_always_a_group_and_only_a_group_is_one() -> None:
    assert parse(profile_reply(kind="friend"), is_group=True).kind == "group"
    assert parse(profile_reply(kind="group")).kind == "unclear"


def test_a_turning_point_outside_the_span_or_with_no_day_is_dropped() -> None:
    points = [
        {"date": "2023-01-01", "what": "Before the first message."},
        {"date": "2023-02-21", "what": "After the last message."},
        {"date": "2023-02-30", "what": "No such day."},
        {"date": "Feb 2023", "what": "Not a day."},
        {"date": "2023-02-20", "what": "   "},
        {"date": "2023-01-02", "what": "The first day."},
        {"date": "2023-02-20", "what": "The last day."},
    ]
    found = parse(profile_reply(turning_points=points))
    assert found.turning_points == (TurningPoint("2023-01-02", "The first day."), TurningPoint("2023-02-20", "The last day."))


def test_at_most_five_turning_points_in_the_order_of_their_days() -> None:
    points = [{"date": f"2023-01-{day:02d}", "what": f"Day {day}."} for day in range(12, 2, -1)]
    found = parse(profile_reply(turning_points=points))
    assert [point.date for point in found.turning_points] == [f"2023-01-{day:02d}" for day in (8, 9, 10, 11, 12)]


def test_long_text_is_cut_at_a_word_inside_its_limit() -> None:
    found = parse(
        profile_reply(
            summary="word " * 200,
            turning_points=[{"date": "2023-01-05", "what": "step " * 60}],
        )
    )
    assert len(found.summary) <= people_prompt.SUMMARY_CHARS and found.summary.endswith("word…")
    assert len(found.turning_points[0].what) <= people_prompt.WHAT_CHARS


def test_engine_references_and_handles_are_taken_out_of_the_text() -> None:
    found = parse(
        profile_reply(
            summary="You met Mina (m12) and wrote to theo@example.com and to (202) 555-0143 in 2023-2024.",
            turning_points=[{"date": "2023-01-05", "what": "See m3–m7: Mina called +1 202 555 0171."}],
        )
    )
    assert found.summary == (
        "You met Mina and wrote to an unsaved email contact and to a number ending in 0143 in 2023-2024."
    )
    assert found.turning_points[0].what == "Mina called a number ending in 0171."


def test_clean_text_puts_the_text_on_one_line() -> None:
    assert clean_text("  two\n lines  ", REDACT) == "two lines"
