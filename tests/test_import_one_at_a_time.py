"""One import writes the archive at a time.

An import is a first setup or an update (`setup_import`, `export`) or the load
of an export of an earlier build (`bridge`). The choice is made once, when the
job is made: the first import runs, and a second one is refused with a plain
sentence and makes no job. A request for an import that cannot be made is
refused in the same way. However a job ends, it ends in the list.
"""
import json
import threading

import pytest

import ui_server
from tests.archive_kit import no_names, own_archives, run, write_database, write_export  # noqa: F401
from tests.loopback import fetch, post, post_bytes

pytestmark = pytest.mark.usefixtures("no_jobs")

REFUSAL = "An import is already running. Wait until it ends."
IMPORTS = ["setup_import", "export", "bridge"]


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


def test_the_imports_are_the_three_jobs_that_write_the_archive():
    assert ui_server.IMPORT_ACTIONS == frozenset(IMPORTS)


@pytest.mark.parametrize("second", IMPORTS)
@pytest.mark.parametrize("first", IMPORTS)
def test_a_second_import_is_refused_and_makes_no_job(held_jobs, first, second):
    made = ui_server.create_job(first, {})

    with pytest.raises(ui_server.JobRefused, match=REFUSAL) as refused:
        ui_server.create_job(second, {})

    assert refused.value.status == 409
    assert made["status"] == "queued"
    assert listed() == [(first, "queued")]


@pytest.mark.parametrize("second", IMPORTS)
@pytest.mark.parametrize("first", IMPORTS)
def test_a_second_import_is_refused_beside_one_that_is_at_work(held_jobs, first, second):
    made = ui_server.create_job(first, {})
    ui_server.update_job(made["id"], status="running")

    with pytest.raises(ui_server.JobRefused, match=REFUSAL) as refused:
        ui_server.create_job(second, {})

    assert refused.value.status == 409
    assert listed() == [(first, "running")]


@pytest.mark.parametrize("asked", [["setup_import"] * 8, ["export"] * 8, ["bridge"] * 8, [*IMPORTS, *IMPORTS, "export", "bridge"]])
def test_of_many_imports_asked_for_at_once_one_runs(held_jobs, asked):
    together = threading.Barrier(len(asked))
    outcomes = []

    def ask(action):
        together.wait(timeout=5)
        try:
            outcomes.append(ui_server.create_job(action, {})["id"])
        except ui_server.JobRefused as refused:
            outcomes.append(str(refused))

    askers = [threading.Thread(target=ask, args=(action,)) for action in asked]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(timeout=5)

    assert sorted(outcome == REFUSAL for outcome in outcomes) == [False] + [True] * (len(asked) - 1)
    assert [status for _action, status in listed()] == ["queued"]


@pytest.mark.parametrize("ended", ["completed", "failed"])
@pytest.mark.parametrize("first, second", [("export", "setup_import"), ("setup_import", "bridge"), ("bridge", "export")])
def test_an_import_can_start_when_the_one_before_it_has_ended(held_jobs, first, second, ended):
    made = ui_server.create_job(first, {})
    ui_server.update_job(made["id"], status=ended)

    after = ui_server.create_job(second, {})

    assert listed() == [(first, ended), (second, "queued")]
    assert after["id"] != made["id"]


@pytest.mark.parametrize("other", IMPORTS)
def test_a_report_runs_beside_an_import(held_jobs, other):
    ui_server.create_job("analyze", {"contact": "+15550100"})
    ui_server.create_job(other, {})
    ui_server.create_job("analyze", {"contact": "+15550100"})

    assert listed() == [("analyze", "queued"), (other, "queued"), ("analyze", "queued")]


def test_the_job_to_build_an_index_of_meaning_is_gone(held_jobs):
    with pytest.raises(ui_server.JobRefused, match="Unknown job action: semantic") as refused:
        ui_server.create_job("semantic", {})

    assert refused.value.status == 400
    assert listed() == []


@pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives")
@pytest.mark.parametrize("action", ["export", "setup_import"])
def test_the_import_that_runs_does_not_stop_itself(messages_home, tmp_path, action):
    # the import that was made first must not find itself in the list and stop
    write_database(messages_home / "chat.db", ["one"])

    job = run(action, {"messagesPath": str(tmp_path / "saves" / "archive.sqlite3"), "includeContacts": False})

    assert job["status"] == "completed", job["error"]


@pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives")
def test_the_load_that_runs_does_not_stop_itself(tmp_path):
    export = write_export(tmp_path / "messages.csv", ["one"])

    job = run("bridge", {"messagesPath": str(export)})

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


@pytest.mark.parametrize("asked", IMPORTS)
def test_the_route_makes_the_first_import_and_refuses_the_second(server, held_jobs, asked):
    first_status, first = post(server, "/api/jobs", {"action": "setup_import"})
    second_status, second = post(server, "/api/jobs", {"action": asked})

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


def test_the_answer_to_a_new_job_does_not_share_the_lines_of_the_job(held_jobs):
    made = ui_server.create_job("export", {})

    ui_server.append_log(made["id"], "a line that the job writes later")

    assert made["logs"] == []


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


WORK = {"export": "pull_into_archive", "setup_import": "pull_into_archive", "bridge": "load_export"}


@pytest.mark.parametrize("action", IMPORTS)
def test_a_job_that_ends_in_a_way_nothing_catches_is_failed_and_holds_up_no_import(monkeypatch, held_jobs, action):
    def leave(*_args):
        raise SystemExit("a library ended the thread")

    monkeypatch.undo()
    monkeypatch.setattr(ui_server, WORK[action], leave)
    job_id = register(action)

    with pytest.raises(SystemExit):
        ui_server.run_job(job_id, action, {"includeContacts": False})

    assert state_of(job_id) == ("failed", "The job stopped before it ended.")
    monkeypatch.setattr(ui_server, "run_job", lambda *_args: held_jobs.wait(timeout=5))
    assert ui_server.create_job("setup_import", {})["status"] == "queued"


@pytest.mark.parametrize("action", IMPORTS)
def test_a_job_whose_work_returns_without_a_word_is_failed(monkeypatch, action):
    # the work is over and the list still says that the job runs
    monkeypatch.setattr(ui_server, "work_job", lambda job_id, action, payload: ui_server.update_job(job_id, status="running"))
    job_id = register(action)

    ui_server.run_job(job_id, action, {})

    assert state_of(job_id) == ("failed", "The job stopped before it ended.")


@pytest.mark.parametrize("ending", ["completed", "failed"])
def test_a_job_that_ended_keeps_how_it_ended(monkeypatch, ending):
    monkeypatch.setattr(ui_server, "work_job", lambda job_id, action, payload: ui_server.update_job(job_id, status=ending, error=None))
    job_id = register("export")

    ui_server.run_job(job_id, "export", {})

    assert state_of(job_id) == (ending, None)


@pytest.mark.parametrize("action", IMPORTS)
def test_an_import_whose_trace_cannot_be_written_is_still_failed_with_its_sentence(monkeypatch, tmp_path, action):
    def closed(*_args, **_more):
        raise ValueError("I/O operation on closed file")

    def stop(*_args):
        raise ui_server.ImportStopped("The copy could not be read.") from RuntimeError("what led to it")

    monkeypatch.setattr(ui_server.traceback, "print_exception", closed)
    monkeypatch.setattr(ui_server, WORK[action], stop)
    job_id = register(action)

    ui_server.run_job(job_id, action, {"messagesPath": str(tmp_path / "archive.sqlite3"), "includeContacts": False})

    assert state_of(job_id) == ("failed", "The copy could not be read.")

