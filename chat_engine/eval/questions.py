"""The questions of the evaluation set, with what a right answer must hold.

The expected values come from the facts of the synthetic archive, so the set
stays right when the generator changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from .synthetic import SyntheticArchive


@dataclass(frozen=True)
class Expect:
    contains: tuple[str, ...] = ()  # every one must be in the answer
    any_of: tuple[tuple[str, ...], ...] = ()  # one of each group must be in the answer
    cites: tuple[str, ...] = ()  # one of these message ids must be cited
    absent: tuple[str, ...] = ()  # none may be in the answer
    not_found: bool = False  # the answer must say the archive does not hold it
    tools: tuple[str, ...] = ()  # tools a sound run calls (checked on the scripted run)


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    text: str
    expect: Expect
    scope: tuple[str, ...] = ()  # conversation keys of the synthetic archive
    follows: str = ""  # id of the question whose chat this one continues


def date_forms(day: str) -> tuple[str, ...]:
    """The ways an answer may write one day."""
    value = date.fromisoformat(day)
    month, short = value.strftime("%B"), value.strftime("%b")
    return (
        day,
        f"{month} {value.day}, {value.year}",
        f"{value.day} {month} {value.year}",
        f"{short} {value.day}, {value.year}",
        f"{value.day} {short} {value.year}",
        f"{month} {value.day}",
    )


def build_questions(synthetic: SyntheticArchive) -> list[Question]:
    facts = synthetic.facts
    planted = synthetic.planted
    return [
        *_recall(planted),
        *_counts(facts),
        *_first_and_last(facts, planted),
        *_people(facts, planted),
        *_relative_dates(facts, planted),
        *_follow_ups(facts, planted),
        *_not_in_the_archive(),
        *_scoped(planted),
        *_other_languages_and_long_text(planted),
        *_groups_and_threads(planted),
    ]


def _recall(planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "recall_starter", "recall", "What did Jonas name his sourdough starter?",
            Expect(contains=("Clint Yeastwood",), cites=(planted["starter_name"],)),
        ),
        Question(
            "recall_wifi", "recall", "What was the wifi password at the cabin?",
            Expect(contains=("pinecone-42-lantern",), cites=(planted["wifi"],)),
        ),
        Question(
            "recall_restaurant", "recall", "Which restaurant did Maya say we have to eat at in Lisbon?",
            Expect(contains=("Taberna do Largo",), cites=(planted["restaurant"],)),
        ),
    ]


def _counts(facts: Mapping[str, Any]) -> list[Question]:
    return [
        Question(
            "count_sourdough", "count", "How many messages mention sourdough?",
            Expect(contains=(str(facts["sourdough_total"]),), tools=("count_messages",)),
        ),
        Question(
            "count_priya_2024", "count", "How many messages did Priya and I exchange in 2024?",
            Expect(contains=(str(facts["priya_2024"]),), tools=("count_messages",)),
        ),
        Question(
            "count_top", "count", "Who do I text the most?",
            Expect(contains=("Priya",), tools=("count_messages",)),
        ),
        Question(
            "count_busiest_month", "count", "Which month of 2023 had the most messages with Maya?",
            Expect(any_of=(("September", "2023-09"),), tools=("count_messages",)),
        ),
        Question(
            "count_sent_received", "count",
            "In my thread with Maya on her phone number, how many messages did I send and how many did she send?",
            Expect(contains=(str(facts["maya_sent"]), str(facts["maya_received"]))),
        ),
    ]


def _first_and_last(facts: Mapping[str, Any], planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "first_sourdough", "first_last", "When did Jonas and I first talk about sourdough?",
            Expect(any_of=(date_forms(facts["sourdough_first_day"]),), cites=(planted["sourdough_first"],)),
        ),
        Question(
            "last_sourdough", "first_last", "When did sourdough last come up with Jonas?",
            Expect(any_of=(date_forms(facts["sourdough_last_day"]),), cites=(planted["sourdough_last"],)),
        ),
        Question(
            "first_message_maya", "first_last", "When did Maya and I first text each other?",
            Expect(any_of=(date_forms(facts["maya_first_day"]),)),
        ),
    ]


def _people(facts: Mapping[str, Any], planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "who_number", "who_is_this", "Who is the number ending in 0199?",
            Expect(contains=("Dana",), any_of=(("climbing", "gym"),), cites=(planted["dana_intro"],)),
        ),
        Question(
            "who_email", "who_is_this", "Who wrote to me about a tent?",
            Expect(
                contains=("Rafael",),
                any_of=(("Northwind",),),
                cites=(planted["rafael_intro"], planted["rafael_arrive"], planted["rafael_done"]),
            ),
        ),
        Question(
            "compare_volume", "compare", "Who do I text more, Maya or Jonas?",
            Expect(
                contains=("Maya",),
                any_of=((str(facts["maya_total"]), str(facts["maya_phone_total"])), (str(facts["jonas_total"]),)),
                tools=("count_messages",),
            ),
        ),
        Question(
            "compare_marathon", "compare", "Who talks about the marathon more, Maya or Priya?",
            Expect(
                contains=("Maya", str(facts["marathon_maya"]), str(facts["marathon_priya"])),
                tools=("count_messages",),
            ),
        ),
    ]


def _relative_dates(facts: Mapping[str, Any], planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "relative_summer", "relative_date", "What did Jonas bake last summer?",
            Expect(contains=("rye",), absent=("focaccia", "banana"), cites=(planted["baked_rye"],)),
        ),
        Question(
            "relative_two_years", "relative_date", "Where did Priya take the ferry to two years ago?",
            Expect(contains=("Bainbridge",), absent=("Vashon",), cites=(planted["ferry_bainbridge"],)),
        ),
        Question(
            "relative_last_month", "relative_date", "How many messages did Priya and I exchange last month?",
            Expect(contains=(str(facts["priya_last_month"]),), tools=("count_messages",)),
        ),
    ]


def _follow_ups(facts: Mapping[str, Any], planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "follow_when", "follow_up", "When was that?",
            Expect(
                any_of=(date_forms(facts["sourdough_first_day"]),),
                cites=(planted["starter_name"], planted["sourdough_first"]),
            ),
            follows="recall_starter",
        ),
        Question(
            "follow_count", "follow_up", "How many of those are in my thread with Jonas?",
            Expect(contains=(str(facts["sourdough_jonas"]),), tools=("count_messages",)),
            follows="count_sourdough",
        ),
    ]


def _not_in_the_archive() -> list[Question]:
    return [
        Question("absent_blood", "not_in_archive", "What is Maya's blood type?", Expect(not_found=True)),
        Question(
            "absent_zanzibar", "not_in_archive", "What did Jonas say about Zanzibar?", Expect(not_found=True)
        ),
    ]


def _scoped(planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "scoped_absent", "scoped", "What did we say about the lease?",
            Expect(not_found=True, absent=("signed", "landlord")),
            scope=("jonas",),
        ),
        Question(
            "scoped_lease", "scoped", "When was the lease signed?",
            Expect(any_of=(date_forms("2021-03-04"),), cites=(planted["lease_signed"],)),
            scope=("maya_phone",),
        ),
    ]


def _other_languages_and_long_text(planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "portuguese_question", "other_language", "O que a Inês disse sobre a cozinha do apartamento novo?",
            Expect(any_of=(("enorme",),), cites=(planted["kitchen"],)),
        ),
        Question(
            "portuguese_content", "other_language", "What did Inês think of the new apartment?",
            Expect(
                any_of=(("ótimo", "great", "enorme", "huge", "big", "wonderful", "excellent", "loved"),),
                cites=(planted["kitchen"],),
            ),
        ),
        Question(
            "long_message", "long_message", "Where did Tomás say the spare badge is?",
            Expect(
                any_of=(("grey cabinet", "gray cabinet"), ("drawer three", "drawer 3", "third drawer")),
                cites=(planted["handover"],),
            ),
        ),
    ]


def _groups_and_threads(planted: Mapping[str, str]) -> list[Question]:
    return [
        Question(
            "group_flights", "group", "Who booked the flights to Lisbon?",
            Expect(contains=("Maya",), cites=(planted["flights"],)),
        ),
        Question(
            "group_members", "group", "Who is in the Lisbon crew group chat?",
            Expect(contains=("Maya", "Jonas", "0142")),
        ),
        Question(
            "email_thread", "two_threads", "What did Maya send me by email about the lease?",
            Expect(any_of=(("scan",),), cites=(planted["lease_email"],)),
        ),
        Question(
            "threads_with_maya", "two_threads", "How many direct threads do I have with Maya, and how do they differ?",
            Expect(any_of=(("two", "2"), ("email",), ("phone",))),
        ),
    ]
