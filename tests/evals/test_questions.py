"""The question set: its size and kinds, gold messages that exist, and counts that agree with the archive."""

from __future__ import annotations

from collections import Counter

import pytest

from archive import Archive
from chat_engine.archive_adapter import ArchiveAdapter
from evals.questions import BASE_KINDS, QuestionError, build_questions, select
from evals.world import Built, world

QUESTIONS = build_questions(world())
COUNTED = [question for question in QUESTIONS if question.count_query is not None]


def test_about_sixty_questions_of_every_kind() -> None:
    kinds = Counter(question.kind for question in QUESTIONS)
    assert 55 <= len(QUESTIONS) <= 65
    assert set(kinds) == set(BASE_KINDS)
    assert all(kinds[kind] >= 5 for kind in BASE_KINDS)


def test_the_set_is_the_same_each_time() -> None:
    assert build_questions(world()) == QUESTIONS


def test_every_gold_message_is_in_the_archive(archive: Archive) -> None:
    for question in QUESTIONS:
        for guid in question.gold_guids:
            found = archive.message(guid)
            assert found is not None, (question.id, guid)
            assert found["id"] == guid


def test_only_absent_questions_have_no_gold_messages() -> None:
    for question in QUESTIONS:
        assert bool(question.gold_guids) == (question.kind != "absent"), question.id


def test_follow_ups_have_a_prior_question() -> None:
    for question in QUESTIONS:
        assert bool(question.prior) == (question.kind == "follow_up"), question.id


@pytest.mark.parametrize("question", COUNTED, ids=lambda question: question.id)
def test_each_computed_count_is_the_count_of_the_archive(question, archive: Archive) -> None:
    query = question.count_query
    counted = ArchiveAdapter(archive).count(
        conversations=query.conversations, from_me=query.from_me, since=query.since, until=query.until,
        text=query.text,
    )  # fmt: skip
    assert counted == question.expected_count
    assert f"{question.expected_count} " in question.gold


def test_every_count_question_has_a_computed_count() -> None:
    assert {question.id for question in QUESTIONS if question.kind == "count"} == {question.id for question in COUNTED}


def test_patterns_name_the_figures_of_the_world(built: Built) -> None:
    by_id = {question.id: question for question in QUESTIONS}
    assert "2024" in by_id["pattern_family_year"].gold.split(":")[0]
    juniper_2025 = len([line for line in built.world.of("juni") if line.day.startswith("2025")])
    assert f"{juniper_2025} in 2025" in by_id["pattern_juniper"].gold


def test_select_by_kind_and_limit() -> None:
    chosen = select(QUESTIONS, ("count", "fact"), 4)
    assert len(chosen) == 4 and {question.kind for question in chosen} <= {"count", "fact"}
    assert select(QUESTIONS) == QUESTIONS
    with pytest.raises(QuestionError):
        select(QUESTIONS, ("vibes",))
