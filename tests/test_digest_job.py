"""The `digest` job of the server, and the function that starts it for a question.

A long conversation is read for chat outside the question. The model is the
fake of the report engine: no request leaves this machine.
"""
import json
import threading

import pytest

import recall_paths
import ui_server
from chat_engine.fake_llm import FakeLLM as ChatLLM
from chat_engine.fake_llm import say
from recall_services import chat as chat_service
from recall_services import keys, memory, places
from recall_services import reports as report_service
from report_engine.fake_llm import FakeLLM
from tests.archive_kit import ended, listed
from tests.loopback import post
from tests.recall_services import report_world
from tests.report_engine.synth import MINA, THEO

pytestmark = pytest.mark.usefixtures("no_jobs")

RESULT_KEYS = {"conversation", "periods", "periodsBuilt", "periodsCached", "periodsFailed", "usd", "usage"}
IMPORTS = ["setup_import", "export", "bridge"]
DIGEST_IS_READ = "Recall is reading a conversation for chat. Wait until it ends."
IMPORT_FIRST = "An import is running. Wait until it ends, then ask again."
NO_KEY = "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then run the report again."


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """An archive with a long and a short conversation, a data folder of its own, and a fake model behind the server."""
    made = report_world.build_desk(tmp_path)
    cache = tmp_path / "contact_names.json"
    cache.write_text(json.dumps({"contacts": [report_world.MINA_CONTACT]}), encoding="utf-8")
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "saves")
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", cache)
    made.llm = FakeLLM()
    made.models = []
    monkeypatch.setattr(report_service, "default_llm", lambda info: made.models.append(info.id) or made.llm)
    made.file = str(made.pulled.archive_path)
    yield made
    report_world.close_desk(made)
    places.forget()


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that start and do not end until they are let go. Each one stays as it was made: queued."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


def run(payload):
    """Run one job to its end, the way its thread does, and give back how it ended."""
    job = ui_server.job_record("digest", payload)
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    try:
        ui_server.run_job(job["id"], "digest", payload)
        with ui_server.JOBS_LOCK:
            return ui_server.job_copy(ui_server.JOBS[job["id"]])
    finally:
        with ui_server.JOBS_LOCK:
            ui_server.JOBS.pop(job["id"], None)


def asked(desk, **more):
    return {"messagesPath": desk.file, "conversation": MINA, **more}


def lines(job):
    return [entry["message"] for entry in job["logs"]]


# --- the job -----------------------------------------------------------------------


def test_the_job_reads_the_conversation_with_the_digest_model_and_gives_the_counts_and_the_cost(desk):
    job = run(asked(desk))
    assert (job["status"], job["error"], job["code"], job["contact"]) == ("completed", None, None, MINA)
    result = job["result"]
    assert set(result) == RESULT_KEYS and result["conversation"] == MINA
    assert result["periods"] == result["periodsBuilt"] >= 1 and (result["periodsCached"], result["periodsFailed"]) == (0, 0)
    assert result["usd"] > 0 and result["usage"]["calls"] == result["periods"] == len(desk.llm.map_calls)
    assert desk.models == [memory.DIGEST_MODEL] and desk.llm.reduce_calls == []
    assert lines(job)[0] == "Reading the conversation"
    assert lines(job)[-1] == f"The conversation is read for chat: {result['periods']} part read now, 0 from earlier runs, 0 not read."
    assert job["progress"] == {"step": 2, "of": 2, "label": "Reading part 1 of 1"}


def test_the_lines_of_the_job_hold_no_handle_and_no_text_of_a_message(desk):
    job = run(asked(desk))
    told = "\n".join([*lines(job), str(job["progress"]), str(job["error"])])
    text = desk.archive.messages(MINA, limit=1)["messages"][0]["text"]
    assert MINA not in told and "2025550143" not in told and text not in told


def test_a_second_job_reads_no_part_again(desk):
    run(asked(desk))
    calls = len(desk.llm.calls)
    job = run(asked(desk))
    assert (job["result"]["periodsBuilt"], job["result"]["usd"]) == (0, 0.0) and len(desk.llm.calls) == calls
    assert any("was read in an earlier run" in line for line in lines(job))


def test_the_job_is_asked_for_as_every_job_is_and_takes_the_days(server, desk):
    status, answer = post(server, "/api/jobs", {**asked(desk), "action": "digest", "since": "2023-01-01", "until": "2023-01-31"})
    assert (status, answer["job"]["action"], answer["job"]["contact"]) == (202, "digest", MINA)
    done = ended(answer["job"]["id"])
    assert (done["status"], done["result"]["periods"]) == ("completed", 1)
    assert desk.llm.calls[0].first_day >= "2023-01-01"


@pytest.mark.parametrize("more", [{"since": "2023-02-30"}, {"since": "2023-03-01", "until": "2023-02-01"}])
def test_a_day_that_is_no_day_fails_the_job_in_one_sentence(desk, more):
    job = run(asked(desk, **more))
    assert (job["status"], job["code"]) == ("failed", "bad_request") and desk.llm.calls == []


def test_a_job_with_no_conversation_fails_in_plain_words(desk):
    job = run({"messagesPath": desk.file})
    assert (job["status"], job["code"], job["error"]) == ("failed", "bad_request", "A conversation is needed to read it for chat.")
    assert desk.llm.calls == []


@pytest.mark.parametrize("name", ["conversation", "since", "until"])
def test_a_request_with_a_value_that_is_no_text_makes_no_job(desk, name):
    with pytest.raises(ui_server.JobRefused, match=f"{name} must be text.") as refused:
        ui_server.create_job("digest", asked(desk, **{name: 5}))
    assert refused.value.status == 400 and listed() == []


def test_with_no_key_the_job_fails_in_one_plain_sentence(desk, tmp_path, monkeypatch):
    monkeypatch.undo()  # the real model port of the service
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", tmp_path / "no-key-file")
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "saves")
    job = run(asked(desk))
    assert (job["status"], job["error"], job["code"], job["result"]) == ("failed", NO_KEY, "no_key", None)


def test_a_job_on_a_place_with_no_archive_fails_in_plain_words(desk, tmp_path):
    job = run({"messagesPath": str(tmp_path / "nothing-here.sqlite3"), "conversation": MINA})
    assert (job["status"], job["code"], job["error"]) == ("failed", "no_archive", ui_server.NO_ARCHIVE)


# --- one at a time -----------------------------------------------------------------


def test_a_second_digest_job_is_refused_and_makes_no_job(server, held_jobs):
    ui_server.create_job("digest", {"conversation": MINA})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("digest", {"conversation": THEO})
    assert (str(refused.value), refused.value.status) == (DIGEST_IS_READ, 409)
    assert post(server, "/api/jobs", {"action": "digest", "conversation": MINA}) == (409, {"error": DIGEST_IS_READ})
    assert listed() == [("digest", "queued")]


@pytest.mark.parametrize("other", IMPORTS)
def test_a_digest_job_and_an_import_refuse_each_other(held_jobs, other):
    first = ui_server.create_job("digest", {"conversation": MINA})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job(other, {})
    assert str(refused.value) == DIGEST_IS_READ
    ui_server.update_job(first["id"], status="completed")
    ui_server.create_job(other, {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("digest", {"conversation": MINA})
    assert str(refused.value) == IMPORT_FIRST
    assert listed() == [("digest", "completed"), (other, "queued")]


def test_a_report_and_a_build_of_the_index_run_beside_a_digest_job(held_jobs):
    ui_server.create_job("digest", {"conversation": MINA})
    ui_server.create_job("analyze", {"contact": MINA})
    ui_server.create_job("meaning", {})
    assert listed() == [("digest", "queued"), ("analyze", "queued"), ("meaning", "queued")]


# --- the function that starts it for a question ------------------------------------


def test_the_starter_of_a_question_starts_the_job_for_the_archive_of_the_question(desk):
    start = ui_server.digest_starter({"messagesPath": desk.file})
    assert start(MINA, "2023-01-01", None) is True
    (job,) = [made for made in ui_server.list_jobs() if made["action"] == "digest"]
    done = ended(job["id"])
    assert (done["status"], done["contact"], done["result"]["conversation"]) == ("completed", MINA, MINA)
    assert desk.llm.calls and desk.llm.calls[0].first_day >= "2023-01-01"


def test_the_starter_says_yes_for_a_job_of_the_same_conversation_that_runs_and_no_for_any_other(held_jobs):
    start = ui_server.digest_starter({})
    assert start(MINA, None, None) is True
    assert start(MINA, None, None) is True  # it already runs: nothing more is started
    assert start(THEO, None, None) is False  # one digest job at a time
    assert listed() == [("digest", "queued")]


@pytest.mark.parametrize("other", IMPORTS)
def test_the_starter_says_no_while_an_import_runs(held_jobs, other):
    ui_server.create_job(other, {})
    assert ui_server.digest_starter({})(MINA, None, None) is False
    assert listed() == [(other, "queued")]


def test_the_starter_never_raises_into_a_question(monkeypatch, capfd):
    def broken(*_args, **_more):
        raise RuntimeError(f"a fault that names {MINA}")

    monkeypatch.setattr(ui_server, "create_job", broken)
    assert ui_server.digest_starter({"messagesPath": 5})(MINA, None, None) is False
    assert MINA not in "".join(capfd.readouterr())


@pytest.mark.parametrize("route", ["/api/ask", "/api/ask/stream"])
def test_each_chat_route_gives_the_question_a_starter_for_its_archive(server, desk, monkeypatch, route):
    given = []
    real = chat_service.stream

    def noted(*args, **more):
        given.append(more.get("start_digest"))
        return real(*args, **more)

    monkeypatch.setattr(chat_service, "stream", noted)
    monkeypatch.setattr(chat_service, "default_llm", lambda _config: ChatLLM([say("Nothing says so.")]))
    body = json.dumps({"messagesPath": desk.file, "question": "What is it like with Mina?"}).encode("utf-8")
    from tests.loopback import _ask

    status, _headers, _raw = _ask(server, "POST", route, body)
    assert status == 200 and callable(given[0])
    assert given[0](MINA, None, None) is True
    (job,) = [made for made in ui_server.list_jobs() if made["action"] == "digest"]
    assert ended(job["id"])["result"]["conversation"] == MINA


# --- the log of what a question spent ----------------------------------------------


def test_the_log_of_the_server_takes_the_line_of_what_a_digest_cost(monkeypatch):
    import io
    import logging

    behind = io.StringIO()
    root, spend = logging.getLogger(), logging.getLogger(memory.SPEND_LOG)
    before = (list(root.handlers), root.level, spend.level)
    monkeypatch.setattr(ui_server.sys, "__stderr__", behind)
    try:
        ui_server.log_to_the_server_log()
        spend.info("a digest inside a question: %d parts read now", 2)
        logging.getLogger("recall.test").info("a line that is no warning")
    finally:
        root.handlers, root.level = before[0], before[1]
        spend.setLevel(before[2])
    assert "a digest inside a question: 2 parts read now" in behind.getvalue()
    assert "a line that is no warning" not in behind.getvalue()
