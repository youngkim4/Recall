"""A question reads a conversation itself through the route of the app, and starts no job.

A question reads at most `memory.DIGEST_PARTS_LIMIT` parts itself, and a
longer story is named in the payload. Mina's conversation has four parts
here. Each model is a fake.
"""
import pytest

import ui_server
from chat_engine.fake_llm import FakeLLM as ChatLLM
from chat_engine.fake_llm import say
from recall_services import chat as chat_service
from recall_services import memory
from recall_services import reports as report_service
from report_engine import chat_digest
from report_engine.fake_llm import FakeLLM
from tests.integration.test_chat_memory import digest_of, tool_results
from tests.recall_services.report_world import OPTIONS, PERIOD_LABELS
from tests.report_engine.synth import MINA, WordCounter
from tests.test_digest_job import desk  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_jobs")

QUESTION = "What is it like with Mina?"
NOT_READY = "too long to read inside one question"


@pytest.fixture
def world(desk, monkeypatch):  # noqa: F811
    """Four parts for Mina, for the job and for the question alike, and a model port of its own inside a question."""
    monkeypatch.setattr(chat_digest, "ReportOptions", lambda: OPTIONS)
    monkeypatch.setattr(report_service, "shared_counter", lambda _family: WordCounter())
    desk.in_question = FakeLLM()
    monkeypatch.setattr(memory, "question_llm", lambda _info: desk.in_question)
    monkeypatch.setattr(desk, "monkeypatch", monkeypatch, raising=False)
    memory.forget()
    yield desk
    memory.forget()


def question(world, *calls):
    """Ask through the route of the app. Gives the payload and what the digest tool told the model."""
    llm = ChatLLM([list(calls or [digest_of()]), say("From what was read.")])
    world.monkeypatch.setattr(chat_service, "default_llm", lambda _config: llm)
    status, body = ui_server.answer_question({"messagesPath": world.file, "question": QUESTION})
    assert status == 200, body
    return body, tool_results(llm.requests[1])


def digest_jobs():
    return sorted((job for job in ui_server.list_jobs() if job["action"] == "digest"), key=lambda job: job["createdAt"])


# --- a question reads what it can, and starts nothing ----------------------------


def test_a_question_reads_four_parts_itself_and_the_next_question_reads_them_from_the_notes(world):
    first, (story,) = question(world)
    assert all(f"== {label}" in story for label in PERIOD_LABELS) and NOT_READY not in story
    assert len(world.in_question.map_calls) == 4 and digest_jobs() == []
    assert first["usage"]["digest_usd"] > 0 and first["memory"] == {"notReady": []}
    second, (again,) = question(world)
    assert again == story and len(world.in_question.calls) == 4 and digest_jobs() == []
    assert second["usage"]["digest_usd"] == 0.0


def test_a_story_too_long_for_a_question_is_named_and_no_question_pays_or_starts_a_job(world):
    world.monkeypatch.setattr(memory, "DIGEST_PARTS_LIMIT", 2)
    for _number in range(3):
        payload, (told,) = question(world)
        assert NOT_READY in told and payload["memory"] == {"notReady": [MINA]}
    assert world.in_question.calls == [] and digest_jobs() == [] and world.llm.calls == []
    # narrower days fit inside the question
    _payload, (january,) = question(world, digest_of(until="2023-01-31"))
    assert "== Jan 2023" in january and len(world.in_question.calls) == 1 and digest_jobs() == []
