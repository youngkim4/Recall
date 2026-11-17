"""How one result is scored."""

from typing import Any

import pytest

from chat_engine.eval.questions import Expect, Question, build_questions
from chat_engine.eval.scoring import holds, plain, says_day, score
from chat_engine.eval.synthetic import build_synthetic_archive

SYNTHETIC = build_synthetic_archive()
QUESTIONS = {question.id: question for question in build_questions(SYNTHETIC)}


def _result(answer: str, cited: tuple[str, ...] = (), **extra: Any) -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": [{"id": message_id} for message_id in cited],
        "unverified_quotes": [],
        "invalid_references": [],
        "cancelled": False,
        **extra,
    }


def _ask(**expect: Any) -> Question:
    return Question("x", "test", "?", Expect(**expect))


def test_score_names_each_failed_check() -> None:
    question = QUESTIONS["relative_summer"]
    good = _result("Jonas baked a rye loaf [1].", (SYNTHETIC.planted["baked_rye"],))
    assert all(score(good, question).values())
    bad = _result(
        "He baked focaccia, said +1 (720) 555-0199.", ("other",),
        unverified_quotes=["made up"], invalid_references=["m9"],
    )
    checks = score(bad, question, forbidden=["+17205550199"])
    assert [name for name, ok in checks.items() if not ok] == [
        "quotes_verified", "references_valid", "no_raw_handle", "contains", "cites", "absent",
    ]
    assert score({**good, "cancelled": True}, question)["answered"] is False


def test_a_result_without_a_key_of_the_contract_is_an_error_not_a_pass() -> None:
    for missing in ("answer", "citations", "unverified_quotes", "invalid_references", "cancelled"):
        result = _result("Jonas baked a rye loaf.")
        del result[missing]
        with pytest.raises(KeyError):
            score(result, QUESTIONS["relative_summer"])


def test_plain_ignores_case_accents_and_thousands_separators() -> None:
    assert plain("You sent 2,157 Messages — ÓTIMO") == "you sent 2157 messages — otimo"


@pytest.mark.parametrize(
    ("answer", "item", "found"),
    [
        ("You exchanged 12 messages.", "12", True),
        ("The total is 12.", "12", True),
        ("You exchanged 312 messages.", "12", False),
        ("You exchanged 120 messages.", "12", False),
        ("About 12.5 a day.", "12", False),
        ("A share of 0.12 of them.", "12", False),
        ("You sent 1,234 of them.", "1234", True),
        ("You sent 11,234 of them.", "1234", False),
        ("Maya's thread", "Maya", True),
        ("the Mayas", "Maya", True),
        ("a scanned copy", "scan", True),
        ("Samaya wrote", "Maya", False),
        ("in 2023-09-14", "2023-09", True),
        ("o apartamento é ÓTIMO", "ótimo", True),
    ],
)
def test_holds_matches_whole_numbers_and_the_start_of_words(answer: str, item: str, found: bool) -> None:
    assert holds(plain(answer), item) is found


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        ("It was signed on 2021-03-04.", True),
        ("Signed March 4, 2021.", True),
        ("Signed on Thursday, March 4th, 2021 in the evening.", True),
        ("Signed 4 March 2021.", True),
        ("Signed on the 4th of March, 2021.", True),
        ("Signed Mar 4 2021.", True),
        ("Signed Mar. 4, 2021.", True),
        ("Signed March 4.", False),
        ("Signed March 4, 2019.", False),
        ("Signed March 14, 2021.", False),
        ("Signed 14 March 2021.", False),
        ("Signed 2021-03-041.", False),
        ("Signed in March 2021.", False),
    ],
)
def test_a_day_counts_only_with_its_year(answer: str, said: bool) -> None:
    assert says_day(plain(answer), "2021-03-04") is said
    assert score(_result(answer), _ask(days=("2021-03-04",)))["days"] is said


def test_september_may_be_written_sept() -> None:
    assert says_day(plain("on Sept. 9, 2023"), "2023-09-09")
    assert says_day(plain("on 9 May 2024"), "2024-05-09")


@pytest.mark.parametrize(
    ("answer", "cited", "ok"),
    [
        ("I could not find any mention of that.", (), True),
        ("The archive does not hold this.", (), True),
        ("I found nothing about a lease in this conversation.", (), True),
        ("No messages mention Zanzibar.", (), True),
        ("He said it was lovely there.", (), False),
        ("Jonas never replied, but Maya said it was lovely.", (), False),
        ("I couldn't tell the exact day, but it was lovely.", (), False),
        ("I could not find it. But see this message [1].", ("p:other",), False),
    ],
)
def test_not_found_needs_a_plain_statement_and_no_citation(answer: str, cited: tuple, ok: bool) -> None:
    checks = score(_result(answer, cited), QUESTIONS["absent_zanzibar"])
    assert (checks["says_not_found"] and checks["cites_nothing"]) is ok


@pytest.mark.parametrize(
    "answer",
    [
        "Call +1 (720) 555-0199.",
        "Call 720-555-0199 now.",
        "Call 7205550199.",
        "Write to RK.Unsaved@Example.org today.",
    ],
)
def test_a_raw_handle_is_found_in_any_spelling(answer: str) -> None:
    forbidden = ["+17205550199", "rk.unsaved@example.org"]
    assert score(_result(answer), _ask(), forbidden)["no_raw_handle"] is False
    assert score(_result("Call a number ending in 0199."), _ask(), forbidden)["no_raw_handle"] is True


def test_the_expected_values_come_from_the_facts_of_the_archive() -> None:
    facts = SYNTHETIC.facts
    busiest = QUESTIONS["count_busiest_month"].expect
    assert busiest.any_of == (("September", facts["maya_busiest_month_2023"]),)
    top = QUESTIONS["count_top"].expect
    assert top.contains == ("Priya", str(facts["top_conversation_total"]))
    assert QUESTIONS["first_sourdough"].expect.days == (facts["sourdough_first_day"],)
