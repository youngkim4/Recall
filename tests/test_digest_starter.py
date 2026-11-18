"""A long conversation from the question to the job and back, and a reading that is not started again and again.

The chain of the app: a chat route, the digest tool, `prepare`, `digest_starter`,
the `digest` job, the notes, the next question. Mina's conversation has four
parts here, two more than a question reads. Each model is a fake.
"""
import pytest

import ui_server
from chat_engine.fake_llm import FakeLLM as ChatLLM
from chat_engine.fake_llm import say
from recall_services import chat as chat_service
from recall_services import memory
from recall_services import reports as report_service
from report_engine import chat_digest
from report_engine.fake_llm import FakeLLM, completed
from tests.archive_kit import ended
from tests.integration.test_chat_memory import digest_of, tool_results
from tests.recall_services.report_world import OPTIONS, PERIOD_LABELS
from tests.report_engine.synth import MINA, WordCounter
from tests.test_digest_job import desk, held_jobs, run  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_jobs")

QUESTION = "What is it like with Mina?"
STARTED = "Recall reads them now in the background"
NOT_STARTED = "one question reads at most 2"


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
    llm = ChatLLM([list(calls or [digest_of()]), say("Later.")])
    world.monkeypatch.setattr(chat_service, "default_llm", lambda _config: llm)
    status, body = ui_server.answer_question({"messagesPath": world.file, "question": QUESTION})
    assert status == 200, body
    return body, tool_results(llm.requests[1])


def digest_jobs():
    return sorted((job for job in ui_server.list_jobs() if job["action"] == "digest"), key=lambda job: job["createdAt"])


def wait():
    """Each digest job that the server started has ended. Gives them, oldest first."""
    return [ended(job["id"]) for job in digest_jobs()]


def refuses_from_february(call):
    if call.first_day >= "2023-02-01":
        return completed(call, None, text="", status="refused")  # paid, and no result
    return None


def refuses_all(call):
    return completed(call, None, text="", status="refused")


# --- the whole chain ---------------------------------------------------------------


def test_a_long_conversation_is_read_by_the_job_and_the_next_question_shows_its_story(world):
    first, (told,) = question(world)
    assert STARTED in told and world.in_question.calls == []
    (job,) = wait()
    assert (job["status"], job["contact"], job["since"], job["until"]) == ("completed", MINA, None, None)
    result = job["result"]
    assert (result["periods"], result["periodsBuilt"], result["periodsFailed"]) == (4, 4, 0)
    assert len(world.llm.map_calls) == 4 and world.models == [memory.DIGEST_MODEL]
    second, (story,) = question(world)
    assert all(f"== {label}" in story for label in PERIOD_LABELS) and STARTED not in story
    # the question read the story from the notes: no call, no cost, no second job
    assert world.in_question.calls == [] and len(world.llm.calls) == 4 and len(digest_jobs()) == 1
    assert second["usage"]["digest_usd"] == 0.0 and first["cancelled"] is False


# --- a reading that fails is not started again and again ---------------------------


def test_parts_that_fail_each_time_are_not_paid_for_at_each_question(world):
    """The first job reads one part and three fail. The second reads nothing new: no third job is started."""
    world.llm = FakeLLM(refuses_from_february)
    seen = []
    for _number in range(4):
        _payload, (told,) = question(world)
        wait()
        seen.append((STARTED in told, NOT_STARTED in told, len(digest_jobs()), len(world.llm.calls)))
    assert seen == [(True, False, 1, 4), (True, False, 2, 7), (False, True, 2, 7), (False, True, 2, 7)]
    first, second = wait()
    assert (first["result"]["periodsBuilt"], first["result"]["periodsFailed"]) == (1, 3)
    assert (second["result"]["periodsBuilt"], second["result"]["periodsFailed"]) == (0, 3)
    # narrower days show the part that is in the notes, with no call
    _payload, (january,) = question(world, digest_of(until="2023-01-31"))
    assert "== Jan 2023" in january and world.in_question.calls == [] and len(digest_jobs()) == 2


def test_a_job_that_failed_is_not_started_again(world):
    world.llm = FakeLLM(refuses_all)
    seen = []
    for _number in range(3):
        _payload, (told,) = question(world)
        wait()
        seen.append((STARTED in told, NOT_STARTED in told, len(digest_jobs()), len(world.llm.calls)))
    assert seen == [(True, False, 1, 4), (False, True, 1, 4), (False, True, 1, 4)]
    (job,) = wait()
    assert (job["status"], job["code"], job["error"]) == ("failed", "model_error", "The model could not read the conversation. Ask again.")


def test_the_starter_keeps_how_the_last_job_of_the_same_archive_conversation_and_days_ended(world):
    world.llm = FakeLLM(refuses_all)
    start = ui_server.digest_starter({"messagesPath": world.file})
    assert start(MINA, None, None) is True
    wait()
    assert start(MINA, None, None) is False and len(digest_jobs()) == 1
    # other days are another reading
    assert start(MINA, "2023-01-01", "2023-01-31") is True
    wait()
    assert start(MINA, "2023-01-01", "2023-01-31") is False and len(digest_jobs()) == 2


def test_a_job_that_read_new_parts_or_left_nothing_unread_can_be_started_again(world):
    start = ui_server.digest_starter({"messagesPath": world.file})
    assert start(MINA, None, None) is True
    (job,) = wait()
    assert job["result"]["periodsFailed"] == 0
    assert start(MINA, None, None) is True
    assert wait()[1]["result"]["periodsBuilt"] == 0 and len(world.llm.calls) == 4


@pytest.mark.parametrize(
    ("action", "result", "starts_again"),
    [("export", {}, True), ("setup_import", {}, True), ("bridge", {"status": "built"}, True), ("bridge", {"status": "current"}, False)],
)
def test_an_import_that_changed_the_archive_lets_the_reading_start_again(world, action, result, starts_again):
    world.llm = FakeLLM(refuses_all)
    start = ui_server.digest_starter({"messagesPath": world.file})
    start(MINA, None, None)
    wait()
    assert start(MINA, None, None) is False
    real = ui_server.do_job
    world.monkeypatch.setattr(ui_server, "do_job", lambda _job_id, _action, _payload: dict(result))
    run({"messagesPath": world.file}, action=action)
    world.monkeypatch.setattr(ui_server, "do_job", real)
    assert start(MINA, None, None) is starts_again
    wait()


def test_an_import_into_another_archive_changes_nothing_for_this_one(world, tmp_path):
    world.llm = FakeLLM(refuses_all)
    start = ui_server.digest_starter({"messagesPath": world.file})
    start(MINA, None, None)
    wait()
    world.monkeypatch.setattr(ui_server, "do_job", lambda _job_id, _action, _payload: {})
    run({"messagesPath": str(tmp_path / "another.sqlite3")}, action="export")
    assert start(MINA, None, None) is False
