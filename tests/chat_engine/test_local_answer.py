from chat_engine.local_answer import local_answer
from chat_engine.scope import Scope
from tests.chat_engine.helpers import RAW_HANDLES
from tests.chat_engine.tiny import JONAS_PHONE, tiny_archive, tiny_labeler


def test_local_answer_lists_the_best_matches_with_citations() -> None:
    result = local_answer(tiny_archive(), "when was the lease signed?", labeler=tiny_labeler())
    assert result["mode"] == "local"
    assert result["answer"].startswith("No AI model was used.")
    assert 'Maya Okafor, 2021-03-04: "yes!! the lease got signed btw" [1]' in result["answer"]
    assert result["citations"][0]["id"] == "a2"
    assert result["citations"][0]["number"] == 1
    assert result["citations"][0]["conversation_label"] == "Maya Okafor"
    numbers = [citation["number"] for citation in result["citations"]]
    assert numbers == list(range(1, len(numbers) + 1))
    for number in numbers:
        assert f"[{number}]" in result["answer"]
    assert result["unverified_quotes"] == []
    assert result["usage"]["model_calls"] == 0
    assert result["cancelled"] is False


def test_local_answer_says_when_nothing_matches() -> None:
    result = local_answer(tiny_archive(), "anything about zanzibar?", labeler=tiny_labeler())
    assert "found no message" in result["answer"]
    assert result["citations"] == []


def test_local_answer_needs_a_word() -> None:
    result = local_answer(tiny_archive(), "??", labeler=tiny_labeler())
    assert result["citations"] == []
    assert "word" in result["answer"]


def test_local_answer_stays_inside_the_scope() -> None:
    archive = tiny_archive()
    scope = Scope(conversations=(JONAS_PHONE,))
    result = local_answer(archive, "lease or sourdough?", scope, labeler=tiny_labeler())
    assert {citation["conversation"] for citation in result["citations"]} == {JONAS_PHONE}
    assert "lease got signed" not in result["answer"]
    for name, arguments in archive.calls:
        if name in {"search", "count"}:
            assert arguments["conversations"] == [JONAS_PHONE]
        else:
            assert name == "conversation" and arguments["conversation_id"] == JONAS_PHONE


def test_local_answer_masks_unsaved_people_and_cuts_long_text() -> None:
    result = local_answer(tiny_archive(), "climbing gym line flowerpot", labeler=tiny_labeler(), limit=8)
    assert "a number ending in 0199" in result["answer"]
    for raw in RAW_HANDLES:
        assert raw not in result["answer"]
    assert "…" in result["answer"]


def test_local_answer_works_without_a_labeler() -> None:
    result = local_answer(tiny_archive(), "sourdough")
    assert "a number ending in 0124" in result["answer"]
