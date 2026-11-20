"""The question set of the eval, built from the world.

Kinds:
- fact: one fact.
- first_last: the first or the last time of something.
- count: an exact number. The number is counted from the world by code.
- pattern: how something changed over time. The figures are counted by code.
- relationship: who a person is, and whether the bond is ongoing or faded.
- multi_hop: two lookups put together.
- absent: the archive does not hold it. A right answer says so.
- follow_up: two turns. `prior` is asked first; `text` is scored.

The hard world (`--world hard`) adds six kinds; see `evals.hard_questions`.

Each question has a short gold answer, the key facts that a right answer
states, and the guids of the messages that support it.
"""

from __future__ import annotations

from collections.abc import Sequence

from evals.hard_questions import hard_questions
from evals.question_computed import computed_questions
from evals.question_static import STATIC
from evals.question_types import BASE_KINDS, HARD_KINDS, KINDS, CountQuery, Question, from_spec
from evals.world import World

__all__ = ["BASE_KINDS", "HARD_KINDS", "KINDS", "CountQuery", "Question", "QuestionError", "build_questions", "select"]


class QuestionError(ValueError):
    """A question set that can not be used."""


def _checked(questions: Sequence[Question]) -> tuple[Question, ...]:
    ids = [question.id for question in questions]
    twice = sorted({one for one in ids if ids.count(one) > 1})
    if twice:
        raise QuestionError(f"question ids used twice: {', '.join(twice)}")
    for question in questions:
        if question.kind not in KINDS:
            raise QuestionError(f"{question.id}: unknown kind {question.kind}")
        if not question.key_facts:
            raise QuestionError(f"{question.id}: no key facts")
        if (question.kind == "follow_up") != bool(question.prior):
            raise QuestionError(f"{question.id}: a follow_up, and only a follow_up, has a prior question")
        if (question.kind == "absent") == bool(question.gold_guids):
            raise QuestionError(f"{question.id}: an absent question, and only one, has no gold messages")
    return tuple(questions)


def build_questions(made: World) -> tuple[Question, ...]:
    """Every question of a world, kind by kind in the order of KINDS. The hard world adds its own."""
    every = [*(from_spec(made, spec) for spec in STATIC), *computed_questions(made)]
    if made.name == "hard":
        every = [*every, *hard_questions(made)]
    for question in every:
        if question.kind not in KINDS:
            raise QuestionError(f"{question.id}: unknown kind {question.kind}")
    return _checked(sorted(every, key=lambda question: KINDS.index(question.kind)))


def select(questions: Sequence[Question], kinds: Sequence[str] = (), limit: int | None = None) -> tuple[Question, ...]:
    """The questions of the chosen kinds, at most `limit`. No kinds means all."""
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise QuestionError(f"unknown kinds: {', '.join(unknown)}. Choose from: {', '.join(KINDS)}")
    chosen = [question for question in questions if not kinds or question.kind in kinds]
    return tuple(chosen if limit is None else chosen[: max(0, limit)])
