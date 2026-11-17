"""One import writes the archive at a time.

The choice is made once, when the job is made: the first import runs, and a
second one is refused with a plain sentence and makes no job. A request for an
import that cannot be made is refused in the same way.
"""
import json
import threading

import pytest

import ui_server
from tests.loopback import fetch, post, post_bytes

pytestmark = pytest.mark.usefixtures("no_jobs")

REFUSAL = "An import is already running. Wait until it ends."


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that start and do not end until they are let go. Each one stays as it was made: queued."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()


def listed():
    with ui_server.JOBS_LOCK:
        return [(job["action"], job["status"]) for job in ui_server.JOBS.values()]


@pytest.mark.parametrize("first, second", [
    ("export", "export"), ("export", "setup_import"), ("setup_import", "export"), ("setup_import", "setup_import"),
])
def test_a_second_import_is_refused_and_makes_no_job(held_jobs, first, second):
    made = ui_server.create_job(first, {})

    with pytest.raises(ui_server.JobRefused, match=REFUSAL) as refused:
        ui_server.create_job(second, {})

    assert refused.value.status == 409
    assert made["status"] == "queued"
    assert listed() == [(first, "queued")]


def test_of_many_imports_asked_for_at_once_one_runs(held_jobs):
    together = threading.Barrier(8)
    outcomes = []

    def ask():
        together.wait(timeout=5)
        try:
            outcomes.append(ui_server.create_job("setup_import", {})["id"])
        except ui_server.JobRefused as refused:
            outcomes.append(str(refused))

    askers = [threading.Thread(target=ask) for _ in range(8)]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(timeout=5)

    assert sorted(outcome == REFUSAL for outcome in outcomes) == [False] + [True] * 7
    assert listed() == [("setup_import", "queued")]


@pytest.mark.parametrize("ended", ["completed", "failed"])
def test_an_import_can_start_when_the_one_before_it_has_ended(held_jobs, ended):
    first = ui_server.create_job("export", {})
    ui_server.update_job(first["id"], status=ended)

    second = ui_server.create_job("setup_import", {})

    assert listed() == [("export", ended), ("setup_import", "queued")]
    assert second["id"] != first["id"]


def test_an_import_and_a_job_of_another_kind_run_together(held_jobs):
    ui_server.create_job("analyze", {"contact": "+15550100"})
    ui_server.create_job("export", {})
    ui_server.create_job("semantic", {})
    ui_server.create_job("analyze", {"contact": "+15550100"})

    assert listed() == [("analyze", "queued"), ("export", "queued"), ("semantic", "queued"), ("analyze", "queued")]


def test_a_second_build_of_the_index_is_refused_in_the_same_way(held_jobs):
    ui_server.create_job("semantic", {})

    with pytest.raises(ui_server.JobRefused, match="A semantic index build is already running.") as refused:
        ui_server.create_job("semantic", {})

    assert refused.value.status == 409
    assert listed() == [("semantic", "queued")]


def test_the_job_that_runs_does_not_stop_itself(messages_home, archive, tmp_path):
    # the import that was made first must not find itself in the list and stop
    from tests.archive_kit import messages_csv, run
    from tests.invented_db import messages_db

    messages_db(messages_home / "chat.db", ["one"])

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "completed", job["error"]


def test_a_job_that_cannot_start_is_failed_and_does_not_hold_up_the_next_import(held_jobs, monkeypatch):
    class NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(ui_server.threading, "Thread", NoThread)
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("export", {})
    monkeypatch.undo()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: held_jobs.wait(timeout=5))

    again = ui_server.create_job("export", {})

    assert (refused.value.status, str(refused.value)) == (500, "The job could not be started. Try again.")
    assert listed() == [("export", "failed"), ("export", "queued")]
    assert again["status"] == "queued"


# ---- a request that cannot be an import ----


@pytest.mark.parametrize("payload, sentence", [
    ({"dbPath": 7}, "The path of the copy must be text."),
    ({"dbPath": ["/somewhere/chat.db"]}, "The path of the copy must be text."),
    ({"dbPath": {"path": "/somewhere/chat.db"}}, "The path of the copy must be text."),
    ({"messagesPath": 7}, "The path of the archive must be text."),
    ({"messagesPath": ["/somewhere/messages.csv"]}, "The path of the archive must be text."),
])
def test_a_path_that_is_not_text_is_refused_and_makes_no_job(held_jobs, payload, sentence):
    with pytest.raises(ui_server.JobRefused, match=sentence) as refused:
        ui_server.create_job("setup_import", payload)

    assert refused.value.status == 400
    assert listed() == []


def test_an_update_takes_no_path_to_a_database_so_any_value_is_passed_over(held_jobs):
    ui_server.create_job("export", {"dbPath": 7})

    assert listed() == [("export", "queued")]


# ---- the route ----


def test_the_route_makes_the_first_import_and_refuses_the_second(server, held_jobs):
    first_status, first = post(server, "/api/jobs", {"action": "setup_import"})
    second_status, second = post(server, "/api/jobs", {"action": "export"})

    assert (first_status, first["job"]["action"], first["job"]["status"]) == (202, "setup_import", "queued")
    assert (second_status, second) == (409, {"error": REFUSAL})
    _, _, body = fetch(server, "/api/jobs")
    assert [job["id"] for job in json.loads(body)["jobs"]] == [first["job"]["id"]]


def test_the_route_refuses_a_path_that_is_not_text(server, held_jobs):
    status, answer = post(server, "/api/jobs", {"action": "setup_import", "dbPath": 7})

    assert (status, answer) == (400, {"error": "The path of the copy must be text."})
    assert listed() == []


@pytest.mark.parametrize("body", [[], ["setup_import"], "setup_import", 7, None])
def test_the_route_refuses_a_body_that_is_not_a_record(server, held_jobs, body):
    status, answer = post(server, "/api/jobs", body)

    assert (status, answer) == (400, {"error": "The request must be a JSON object."})
    assert listed() == []


@pytest.mark.parametrize("action", [["setup_import"], {"is": "export"}, 7, True, None])
def test_the_route_refuses_an_action_that_is_not_text(server, held_jobs, action):
    status, answer = post(server, "/api/jobs", {"action": action})

    assert (status, answer) == (400, {"error": "The action must be text."})
    assert listed() == []


def test_the_route_refuses_an_action_it_does_not_know_and_makes_no_job(server, held_jobs):
    status, answer = post(server, "/api/jobs", {"action": "erase"})

    assert (status, answer) == (400, {"error": "Unknown job action: erase"})
    assert listed() == []


@pytest.mark.parametrize("payload, sentence", [
    ({"action": "setup_import", "dbPath": "bad\u0000name.db"}, "The path of the copy cannot be used."),
    ({"action": "export", "messagesPath": "bad\u0000name.csv"}, "The path of the archive cannot be used."),
    ({"action": "analyze", "messagesPath": 7}, "The path of the archive must be text."),
    ({"action": "analyze", "outDir": ["out"]}, "The path of the reports folder must be text."),
    ({"action": "setup_import", "includeContacts": "false"}, "includeContacts must be true or false."),
])
def test_the_route_refuses_what_a_job_could_not_use(server, held_jobs, payload, sentence):
    status, answer = post(server, "/api/jobs", payload)

    assert (status, answer) == (400, {"error": sentence})
    assert listed() == []


@pytest.mark.parametrize("path", ["/api/jobs", "/api/setup/complete"])
@pytest.mark.parametrize("body", [b"{not json", b"\xff\xfe", b'{"action": '])
def test_a_body_that_is_not_json_is_refused(server, held_jobs, marker, path, body):
    status, answer = post_bytes(server, path, body)

    assert (status, answer) == (400, {"error": "The request body must be JSON."})
    assert listed() == [] and not marker.exists()


def test_the_answer_to_a_new_job_is_the_job_as_it_was_made(monkeypatch):
    # the job ends before the answer is written: the answer still says "queued", and it is not the job itself
    class RunsAtOnce:
        def __init__(self, target, args, daemon):
            self.run = lambda: target(*args)

        def start(self):
            self.run()

    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: ui_server.update_job(job_id, status="completed"))
    monkeypatch.setattr(ui_server.threading, "Thread", RunsAtOnce)

    made = ui_server.create_job("export", {})

    with ui_server.JOBS_LOCK:
        kept = ui_server.JOBS[made["id"]]
    assert (made["status"], kept["status"]) == ("queued", "completed")
    assert made is not kept


def test_the_answer_about_a_job_is_a_copy_of_it(server, held_jobs):
    made = ui_server.create_job("export", {})

    _, _, body = fetch(server, f"/api/jobs/{made['id']}")

    assert json.loads(body)["job"]["status"] == "queued"


# ---- a job always ends ----


def register(action):
    job = ui_server.job_record(action, {})
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    return job["id"]


def state_of(job_id):
    with ui_server.JOBS_LOCK:
        return ui_server.JOBS[job_id]["status"], ui_server.JOBS[job_id]["error"]


def test_a_job_that_ends_in_a_way_nothing_catches_is_failed_and_holds_up_no_import(monkeypatch, held_jobs):
    def leave(*_args):
        raise SystemExit("a library ended the thread")

    monkeypatch.undo()
    monkeypatch.setattr(ui_server, "export_when_free", leave)
    job_id = register("export")

    with pytest.raises(SystemExit):
        ui_server.run_job(job_id, "export", {})

    assert state_of(job_id) == ("failed", "The job stopped before it ended.")
    monkeypatch.setattr(ui_server, "run_job", lambda *_args: held_jobs.wait(timeout=5))
    assert ui_server.create_job("setup_import", {})["status"] == "queued"


def test_an_import_whose_trace_cannot_be_written_is_still_failed_with_its_sentence(monkeypatch, tmp_path):
    def closed(*_args, **_more):
        raise ValueError("I/O operation on closed file")

    def stop(*_args):
        raise ui_server.ImportStopped("The copy could not be read.") from RuntimeError("what led to it")

    monkeypatch.setattr(ui_server.traceback, "print_exception", closed)
    monkeypatch.setattr(ui_server, "export_when_free", stop)
    job_id = register("export")

    ui_server.run_job(job_id, "export", {"messagesPath": str(tmp_path / "messages.csv")})

    assert state_of(job_id) == ("failed", "The copy could not be read.")

