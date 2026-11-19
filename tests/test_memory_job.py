"""The memory of chat behind the server: GET /api/memory, the `memory` job, and its start after an import.

The archive is an invented one, and the model is the fake of the report
engine: no request leaves this machine. A conversation of the memory needs
one message here (`PREPARE_MIN_MESSAGES`), so the small archives of the
import tests have one.
"""
import json
import threading
from urllib.parse import quote

import pytest

import ui_server
from recall_services import keys, memory, memory_prep, places
from recall_services import reports as report_service
from report_engine.fake_llm import FakeLLM
from tests.archive_kit import (
    data_folder,  # noqa: F401
    ended,
    jobs_end,  # noqa: F401
    listed,
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    pulled,
    run,
    said,
    write_database,
)
from tests.loopback import fetch, post

pytestmark = pytest.mark.usefixtures(
    "fresh_bundle_cache", "no_names", "own_archives", "no_jobs", "jobs_end", "data_folder", "messages_home"
)

MARK = "kingfisher"
FIRST = [f"the {MARK} dove at noon {number}" for number in range(20)]
MORE = [f"a later {MARK} dive {number}" for number in range(3)]
IMPORTS = ["setup_import", "export", "bridge"]
RESULT_KEYS = {
    "conversations", "conversationsRead", "conversationsFailed",
    "periodsBuilt", "periodsCached", "periodsFailed", "usd", "usage",
}  # fmt: skip
ANSWER_KEYS = {
    "state", "model", "minMessages", "conversations", "conversationsRead", "conversationsSkipped",
    "parts", "partsRead", "partsNotRead", "estimatedCost", "error", "code", "jobId",
}  # fmt: skip
MEMORY_IS_READ = "Recall is reading the memory of chat. Wait until it ends."
IMPORT_BEFORE_MEMORY = "An import is running. Wait until it ends, then read the memory."
MEMORY_BEFORE_DIGEST = "Recall is reading the memory of chat, and this conversation with it. Wait until it ends."
DIGEST_BEFORE_MEMORY = "Recall is reading a conversation for chat. Wait until it ends, then read the memory."


@pytest.fixture
def model(monkeypatch):
    """The model port of a report is the fake, and one message makes a conversation of the memory."""
    fake = FakeLLM()
    monkeypatch.setattr(report_service, "default_llm", lambda _info: fake)
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    return fake


@pytest.fixture
def archive_file(data_folder):  # noqa: F811
    return str(pulled(data_folder / "saves" / "archive.sqlite3", FIRST))


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that start and do not end until they are let go. Each one stays as it was made: queued."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


def memory_of(server, archive_file):
    status, _headers, body = fetch(server, f"/api/memory?messagesPath={quote(archive_file)}")
    return status, json.loads(body)


def read_memory(archive_file):
    return run("memory", {"messagesPath": archive_file})


def memory_jobs():
    with ui_server.JOBS_LOCK:
        return [ui_server.job_copy(job) for job in ui_server.JOBS.values() if job["action"] == "memory"]


def started():
    """The one `memory` job that the server started by itself, as it ended."""
    (job,) = memory_jobs()
    return ended(job["id"])


def lines(job):
    return [entry["message"] for entry in job["logs"]]


# --- GET /api/memory ---------------------------------------------------------------


def test_with_nothing_read_the_route_says_none_and_what_the_reading_would_cost(server, archive_file, model):
    status, answer = memory_of(server, archive_file)
    assert status == 200 and set(answer) == ANSWER_KEYS
    assert (answer["state"], answer["model"], answer["minMessages"], answer["jobId"]) == ("none", memory.DIGEST_MODEL, 1, None)
    assert (answer["conversations"], answer["parts"], answer["partsRead"], answer["partsNotRead"]) == (1, 1, 0, 1)
    assert answer["estimatedCost"] > 0 and (answer["error"], answer["code"]) == (None, None)
    assert model.calls == []


def test_after_the_job_the_route_says_ready(server, archive_file, model):
    assert read_memory(archive_file)["status"] == "completed"
    _status, answer = memory_of(server, archive_file)
    assert (answer["state"], answer["partsNotRead"], answer["estimatedCost"]) == ("ready", 0, 0.0)


def test_the_route_names_the_memory_job_that_runs(server, archive_file, model, held_jobs):
    job = ui_server.create_job("memory", {"messagesPath": archive_file})
    _status, answer = memory_of(server, archive_file)
    assert answer["jobId"] == job["id"]


def test_the_route_needs_an_archive_as_the_other_routes_do(server, tmp_path, model):
    status, _headers, body = fetch(server, f"/api/memory?messagesPath={quote(str(tmp_path / 'none.sqlite3'))}")
    assert status >= 400 and json.loads(body)["error"]


# --- the job -----------------------------------------------------------------------


def test_the_job_reads_the_memory_and_gives_the_counts_and_the_cost(archive_file, model):
    job = read_memory(archive_file)
    assert (job["status"], job["error"], job["code"], job["contact"]) == ("completed", None, None, "")
    result = job["result"]
    assert set(result) == RESULT_KEYS
    assert (result["conversations"], result["periodsBuilt"], result["periodsCached"]) == (1, 1, 0)
    assert result["usd"] > 0 and len(model.map_calls) == 1 and model.reduce_calls == []
    assert lines(job)[0] == "Looking at 1 conversation with 1 messages or more"
    assert lines(job)[-1] == "The memory is read: 1 part read now, 0 from earlier runs, 0 not read."
    assert job["progress"] == {"step": 2, "of": 2, "label": "Reading your conversations", "done": 1, "total": 1}
    assert MARK not in said(job)


def test_the_job_is_asked_for_as_every_job_is(server, archive_file, model):
    status, answer = post(server, "/api/jobs", {"action": "memory", "messagesPath": archive_file})
    assert (status, answer["job"]["action"], answer["job"]["status"]) == (202, "memory", "queued")
    assert ended(answer["job"]["id"])["status"] == "completed"


def test_with_no_key_the_job_fails_in_one_plain_sentence(archive_file, monkeypatch, tmp_path):
    import recall_paths

    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", tmp_path / "no-key-file")
    job = read_memory(archive_file)
    assert (job["status"], job["code"], job["result"]) == ("failed", "no_key", None)
    assert job["error"] == "No API key is set. Add one in Settings > Archive > API key, then read the memory again."


def test_a_job_on_a_place_with_no_archive_fails_in_plain_words(tmp_path, model):
    job = read_memory(str(tmp_path / "nothing-here.sqlite3"))
    assert (job["status"], job["code"], job["error"]) == ("failed", "no_archive", ui_server.NO_ARCHIVE)


# --- one at a time -----------------------------------------------------------------


def test_a_second_memory_job_is_refused_and_makes_no_job(server, held_jobs):
    ui_server.create_job("memory", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("memory", {})
    assert (str(refused.value), refused.value.status) == (MEMORY_IS_READ, 409)
    assert post(server, "/api/jobs", {"action": "memory"}) == (409, {"error": MEMORY_IS_READ})
    assert listed() == [("memory", "queued")]


@pytest.mark.parametrize("other", IMPORTS)
def test_a_memory_job_and_an_import_refuse_each_other(held_jobs, other):
    first = ui_server.create_job("memory", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job(other, {})
    assert str(refused.value) == MEMORY_IS_READ
    ui_server.update_job(first["id"], status="completed")
    ui_server.create_job(other, {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("memory", {})
    assert str(refused.value) == IMPORT_BEFORE_MEMORY
    assert listed() == [("memory", "completed"), (other, "queued")]


def test_a_report_and_an_index_build_run_beside_the_memory_job(held_jobs):
    ui_server.create_job("memory", {})
    ui_server.create_job("analyze", {"contact": "+12025550143"})
    ui_server.create_job("meaning", {})
    assert listed() == [("memory", "queued"), ("analyze", "queued"), ("meaning", "queued")]


def test_a_memory_job_and_a_digest_job_refuse_each_other(server, held_jobs):
    """Both read with the digest model and no limit of parts: side by side they would pay for each part twice."""
    first = ui_server.create_job("memory", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("digest", {"conversation": "+12025550143"})
    assert (str(refused.value), refused.value.status) == (MEMORY_BEFORE_DIGEST, 409)
    ui_server.update_job(first["id"], status="completed")
    ui_server.create_job("digest", {"conversation": "+12025550143"})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("memory", {})
    assert str(refused.value) == DIGEST_BEFORE_MEMORY
    assert post(server, "/api/jobs", {"action": "memory"}) == (409, {"error": DIGEST_BEFORE_MEMORY})
    assert listed() == [("memory", "completed"), ("digest", "queued")]


# --- after an import ---------------------------------------------------------------


def test_after_an_import_the_server_reads_the_rest_of_the_memory_when_it_is_small(archive_file, model, messages_home):
    assert read_memory(archive_file)["status"] == "completed"
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run("export", {"messagesPath": archive_file})
    assert job["status"] == "completed", job["error"]
    done = started()
    assert (done["status"], done["result"]["periodsBuilt"]) == ("completed", 1)
    assert len(model.map_calls) == 2


def test_after_an_import_the_server_waits_for_the_reader_when_the_rest_costs_more(
    archive_file, model, messages_home, monkeypatch, capfd
):
    assert read_memory(archive_file)["status"] == "completed"
    monkeypatch.setattr(memory_prep, "PREPARE_AUTO_USD", 0.0)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert memory_jobs() == [] and len(model.map_calls) == 1
    assert "the memory waits for the reader: 1 part" in capfd.readouterr().err


def test_after_an_import_with_nothing_read_the_server_starts_nothing_and_takes_no_estimate(
    archive_file, model, messages_home, monkeypatch
):
    asked = []
    monkeypatch.setattr(memory_prep, "payload", lambda *args, **more: asked.append(args) or pytest.fail("an estimate"))
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert memory_jobs() == [] and model.calls == [] and asked == []


def test_after_an_import_that_failed_the_server_starts_nothing(archive_file, model):
    assert read_memory(archive_file)["status"] == "completed"
    assert run("export", {"messagesPath": archive_file})["status"] == "failed"  # Messages keeps no database here
    assert memory_jobs() == []


def test_a_fault_while_the_server_looks_at_the_memory_does_not_change_how_the_import_ended(
    archive_file, model, messages_home, monkeypatch, capfd
):
    assert read_memory(archive_file)["status"] == "completed"

    def broken(*_args, **_more):
        raise RuntimeError(f"a fault that says {MARK}")

    monkeypatch.setattr(memory_prep, "payload", broken)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run("export", {"messagesPath": archive_file})
    assert (job["status"], job["error"]) == ("completed", None)
    assert memory_jobs() == [] and MARK not in said(job)
    assert MARK not in "".join(capfd.readouterr())


def test_the_job_that_the_server_starts_holds_up_the_next_import(archive_file, model, messages_home, monkeypatch):
    assert read_memory(archive_file)["status"] == "completed"
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    inside, let_go = threading.Event(), threading.Event()
    real = memory_prep.run

    def slow(*args, **more):
        inside.set()
        let_go.wait(timeout=5)
        return real(*args, **more)

    monkeypatch.setattr(memory_prep, "run", slow)
    try:
        assert run("export", {"messagesPath": archive_file})["status"] == "completed"
        assert inside.wait(timeout=5)
        with pytest.raises(ui_server.JobRefused) as refused:
            ui_server.create_job("export", {"messagesPath": archive_file})
        assert str(refused.value) == MEMORY_IS_READ
    finally:
        let_go.set()
    assert started()["status"] == "completed"


def test_a_reading_that_is_refused_after_the_import_changes_nothing(archive_file, model, messages_home, monkeypatch, capfd):
    """Another import or a reading started meanwhile: the server starts nothing and says nothing."""
    assert read_memory(archive_file)["status"] == "completed"
    refused = []

    def busy(action, payload):
        refused.append(action)
        raise ui_server.JobRefused(MEMORY_IS_READ)

    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    real = ui_server.create_job
    monkeypatch.setattr(ui_server, "create_job", lambda action, payload: busy(action, payload) if action == "memory" else real(action, payload))
    job = run("export", {"messagesPath": archive_file})
    assert (job["status"], job["error"]) == ("completed", None)
    assert refused == ["memory"] and memory_jobs() == []
    assert "[memory]" not in capfd.readouterr().err


# --- what a job spent, also when it failed ------------------------------------------


def test_a_job_that_ends_keeps_what_it_spent(archive_file, model):
    job = read_memory(archive_file)
    assert job["spent"] == {"usd": job["result"]["usd"], "usage": job["result"]["usage"]}


def test_a_job_that_fails_after_paid_parts_keeps_what_it_spent(archive_file, model, monkeypatch):
    usage = {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 100, "cost_usd": 0.0123, "calls": 3}

    def paid_then_refused(*_args, **_more):
        raise memory_prep.MemoryFailure("key_rejected", "The AI service refused the API key.", usage)

    monkeypatch.setattr(memory_prep, "run", paid_then_refused)
    job = read_memory(archive_file)
    assert (job["status"], job["code"], job["result"]) == ("failed", "key_rejected", None)
    assert job["spent"] == {"usd": 0.0123, "usage": usage}


def test_a_job_that_fails_before_any_call_spent_nothing(archive_file, model, monkeypatch):
    def refused(*_args, **_more):
        raise memory_prep.MemoryFailure("no_key", "No API key is set.")

    monkeypatch.setattr(memory_prep, "run", refused)
    job = read_memory(archive_file)
    assert job["status"] == "failed" and job["spent"] == {"usd": 0.0, "usage": None}


def test_a_job_that_failed_writes_no_marker(archive_file, model, monkeypatch):
    def refused(*_args, **_more):
        raise memory_prep.MemoryFailure("offline", "The AI service could not be reached.")

    monkeypatch.setattr(memory_prep, "run", refused)
    assert read_memory(archive_file)["status"] == "failed"
    assert memory_prep.completed_before(places.archive_path_for(archive_file)) is False


def test_a_job_that_ends_writes_the_marker_of_its_archive(archive_file, model):
    assert memory_prep.completed_before(places.archive_path_for(archive_file)) is False
    read_memory(archive_file)
    assert memory_prep.completed_before(places.archive_path_for(archive_file)) is True


def test_after_an_import_a_digest_read_by_a_question_starts_nothing(archive_file, model, messages_home):
    """A question or a report with the digest model put parts in the notes, but no memory job completed."""
    from recall_services import digests

    archive = places.state_for(archive_file).archive()
    conversation = archive.conversations()[0]["id"]
    digests.build(conversation, archive=archive, names=ui_server.saved_names(), model=memory.DIGEST_MODEL)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert memory_jobs() == [] and len(model.map_calls) == 1


# --- the answer of the route is kept -------------------------------------------------


@pytest.fixture
def counted_payloads(monkeypatch):
    calls = []
    real = memory_prep.payload

    def counted(*args, **more):
        calls.append(1)
        return real(*args, **more)

    monkeypatch.setattr(memory_prep, "payload", counted)
    return calls


def test_two_calls_in_a_row_work_the_answer_out_once(server, archive_file, model, counted_payloads):
    first = memory_of(server, archive_file)
    second = memory_of(server, archive_file)
    assert first == second and len(counted_payloads) == 1


def test_calls_that_arrive_together_wait_for_one_answer(server, archive_file, model, counted_payloads):
    answers = []
    threads = [threading.Thread(target=lambda: answers.append(memory_of(server, archive_file))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(answers) == 4 and len(counted_payloads) == 1


def test_after_a_part_is_saved_the_counts_move(server, archive_file, model, counted_payloads):
    from recall_services import digests

    assert memory_of(server, archive_file)[1]["partsRead"] == 0
    archive = places.state_for(archive_file).archive()
    conversation = archive.conversations()[0]["id"]
    digests.build(conversation, archive=archive, names=ui_server.saved_names(), model=memory.DIGEST_MODEL)
    _status, answer = memory_of(server, archive_file)
    assert (answer["state"], answer["partsRead"]) == ("ready", 1) and len(counted_payloads) == 2


def test_the_job_that_runs_is_read_fresh_beside_a_kept_answer(server, archive_file, model, held_jobs, counted_payloads):
    assert memory_of(server, archive_file)[1]["jobId"] is None
    job = ui_server.create_job("memory", {"messagesPath": archive_file})
    assert memory_of(server, archive_file)[1]["jobId"] == job["id"] and len(counted_payloads) == 1


def test_an_import_that_changed_the_archive_works_the_answer_out_again(server, archive_file, model, messages_home, counted_payloads):
    memory_of(server, archive_file)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    _status, answer = memory_of(server, archive_file)
    assert answer["parts"] == 1 and len(counted_payloads) == 2
