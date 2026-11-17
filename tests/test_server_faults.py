"""What the server says when something goes wrong that nobody planned for.

The reader gets one plain sentence. What was raised can hold a path or the
text of a query: it goes to the log of the server, and to no page.
"""
import io
import json
import threading

import pytest

import ui_server
from tests.export_kit import AVERY, data_folder, get, jobs_ended, loaded, row  # noqa: F401
from tests.loopback import _ask, post

SECRET = "SELECT text FROM message WHERE handle = '+15550000001' -- /Users/someone/Library/Messages/chat.db"
ROWS = [row(AVERY, "2024-03-01 09:00:00", "the ferry leaves at nine"), row(AVERY, "2024-03-01 09:05:00", "see you there", from_me=1)]


def run(action, payload):
    """Run one job to its end, the way its thread does, and give back how it ended."""
    job = ui_server.job_record(action, payload)
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    ui_server.run_job(job["id"], action, payload)
    with ui_server.JOBS_LOCK:
        return ui_server.job_copy(ui_server.JOBS[job["id"]])


def said(job):
    return "\n".join([str(job.get("error") or ""), *(str(entry.get("message", "")) for entry in job["logs"])])


def test_a_job_that_fails_in_a_way_nobody_planned_says_one_sentence_and_shows_no_text_of_the_fault(data_folder, monkeypatch):
    def fails(*_args, **_more):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(ui_server, "do_job", fails)
    job = run("export", {})
    assert (job["status"], job["error"], job["code"]) == ("failed", ui_server.JOB_FAILED, "job_failed")
    assert "SELECT" not in said(job) and "Traceback" not in said(job) and "RuntimeError" not in said(job)


def test_a_route_that_fails_in_a_way_nobody_planned_answers_one_sentence(data_folder, server, monkeypatch):
    def fails(*_args, **_more):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(ui_server, "memories_payload", fails)
    status, answer = get(server, "/api/memories")
    assert (status, answer) == (500, {"error": ui_server.SERVER_FAULT, "code": "server_error"})


@pytest.mark.parametrize("path", ["/api/overview", "/api/memories", "/api/first-words", "/api/search", "/api/day", "/api/preview", "/api/contacts", "/api/setup/status", "/api/analysis"])
def test_an_archive_file_that_no_system_can_hold_is_refused_on_each_read_route(data_folder, server, path):
    status, answer = get(server, path, messagesPath="odd\x00name.sqlite3", contact=AVERY, day="2024-03-01", query="ferry", reportPath="x")
    assert (status, answer) == (400, {"error": ui_server.BAD_ARCHIVE_FILE})


def test_an_archive_file_that_is_no_file_is_refused(data_folder, server):
    assert get(server, "/api/overview", messagesPath="/") == (400, {"error": ui_server.BAD_ARCHIVE_FILE})
    status, answer = post(server, "/api/ask", {"messagesPath": ["not", "text"], "question": "anything"})
    assert (status, answer) == (400, {"error": ui_server.BAD_ARCHIVE_FILE})


def test_a_preview_of_a_conversation_that_the_archive_does_not_hold_is_a_fault_of_the_request(data_folder, server):
    loaded(ui_server.DEFAULT_MESSAGES_CSV, ROWS)
    status, answer = get(server, "/api/preview", contact="+15550009999")
    assert (status, answer) == (400, {"error": "No messages found for this conversation in these dates."})
    status, answer = get(server, "/api/preview", contact=AVERY, since="2024-13-45")
    assert status == 400 and "2024-13-45" in answer["error"]


def test_a_question_stream_that_fails_in_a_way_nobody_planned_ends_with_one_error_event(data_folder, server, monkeypatch):
    loaded(ui_server.DEFAULT_MESSAGES_CSV, ROWS)

    def fails(*_args, **_more):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(ui_server.chat_service, "stream", fails)
    status, headers, raw = _ask(server, "POST", "/api/ask/stream", json.dumps({"question": "anything"}).encode())
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    frames = [json.loads(frame[len("data: "):]) for frame in raw.decode().split("\n\n") if frame.strip()]
    assert frames == [{"type": "error", "error": ui_server.SERVER_FAULT, "code": "server_error"}]


def test_a_question_with_no_archive_is_refused_before_the_stream_starts(data_folder, server):
    status, headers, raw = _ask(server, "POST", "/api/ask/stream", json.dumps({"question": "anything"}).encode())
    assert status == 400 and headers["Content-Type"].startswith("application/json")
    assert json.loads(raw) == {"error": ui_server.NO_ARCHIVE, "code": "no_archive"}


def test_only_a_report_file_is_opened_as_a_report(data_folder, server):
    loaded(ui_server.DEFAULT_MESSAGES_CSV, ROWS)
    key_file = data_folder / ".env"
    key_file.write_text("OPENAI_API_KEY=not-a-real-key\n## Summary\nanything\n", encoding="utf-8")
    status, answer = get(server, "/api/analysis", reportPath=str(key_file))
    assert (status, answer) == (404, {"error": "Report not found"})


def test_the_request_log_holds_the_path_and_no_query(data_folder, server, monkeypatch):
    lines = io.StringIO()
    monkeypatch.setattr(ui_server.sys, "__stderr__", lines)
    get(server, "/api/search", query="a private word", contact=AVERY)
    logged = lines.getvalue()
    assert "/api/search" in logged
    assert "private" not in logged and "15550000001" not in logged and "?" not in logged


def test_what_a_request_thread_prints_is_never_a_line_of_a_job(data_folder):
    """One job runs. A thread that answers a request prints: the words go to the stream of the server."""
    behind = io.StringIO()
    router = ui_server._JobStreamRouter(behind)
    job = ui_server.job_record("export", {})
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    router.register(ui_server.JobWriter(job["id"]))
    handler = ui_server.RecallHandler.__new__(ui_server.RecallHandler)

    def answers_a_request():
        with handler.answering():
            router.write("a traceback of a request\n")

    def works_for_the_job():
        router.write("a line of a worker of the job\n")

    for work in (answers_a_request, works_for_the_job):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    router.unregister()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()
    assert behind.getvalue() == "a traceback of a request\n"
    assert [entry["message"] for entry in job["logs"]] == ["a line of a worker of the job"]
    assert ui_server._JobStreamRouter.answering == set()


def test_the_list_of_jobs_shares_no_list_with_a_job(data_folder):
    job = ui_server.job_record("export", {})
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    listed = ui_server.list_jobs()
    ui_server.append_log(job["id"], "a later line")
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()
    assert listed[0]["logs"] == []


def test_a_body_that_is_too_large_or_nested_too_deep_is_refused(data_folder, server):
    status, _, raw = _ask(server, "POST", "/api/ask", b"[" * 100_000 + b"]" * 100_000)
    assert status == 400 and json.loads(raw) == {"error": "The request body must be JSON."}


def test_a_body_that_says_it_is_too_large_is_refused_before_it_is_read():
    class Asked:
        headers = {"Content-Length": str(ui_server.MAX_BODY_BYTES + 1)}

        @property
        def rfile(self):
            raise AssertionError("the body was read")

    with pytest.raises(ui_server.BadRequest, match="The request is too large."):
        ui_server.parse_body(Asked())


# ---- a load of an export that did not end well ----


def test_an_export_that_cannot_be_read_is_tried_one_time_and_not_at_each_request(data_folder, server):
    """A folder in the place of the export: the load fails, and the next request starts no second load."""
    ui_server.DEFAULT_MESSAGES_CSV.mkdir()
    first, answer = get(server, "/api/contacts")
    assert first == 503 and answer["code"] == "archive_building"
    assert jobs_ended()
    for _ in range(3):
        status, again = get(server, "/api/overview", messagesPath=str(ui_server.DEFAULT_MESSAGES_CSV))
        assert status == 500 and again["code"] == "archive_failed" and again["error"].endswith(".")
    assert [job["action"] for job in ui_server.list_jobs()] == ["bridge"]


def test_a_load_that_raises_is_not_started_again_for_the_same_export(data_folder, server, monkeypatch):
    from recall_services import archive_state

    ui_server.DEFAULT_MESSAGES_CSV.write_text("message_id,timestamp\n1,2024-01-01 09:00:00\n", encoding="utf-8")

    def fails(self, *_args, **_more):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(archive_state.ArchiveState, "ensure_from_export", fails)
    assert get(server, "/api/contacts")[0] == 503
    assert jobs_ended()
    status, answer = get(server, "/api/overview", messagesPath=str(ui_server.DEFAULT_MESSAGES_CSV))
    assert (status, answer) == (500, {"error": ui_server.EXPORT_NOT_LOADED, "code": "archive_failed"})
    assert len(ui_server.list_jobs()) == 1 and "SELECT" not in said(ui_server.list_jobs()[0])


def test_a_load_that_met_a_busy_archive_is_tried_again_after_a_moment(data_folder, monkeypatch):
    from recall_services import places

    export = ui_server.DEFAULT_MESSAGES_CSV
    export.write_text("message_id,timestamp\n1,2024-01-01 09:00:00\n", encoding="utf-8")
    state = places.state_for(export)
    clock = [1000.0]
    monkeypatch.setattr(ui_server.time, "monotonic", lambda: clock[0])
    ui_server.note_bridge_failure(state, export, "An import is already running.", ui_server.BRIDGE_RETRY_SECONDS)
    assert ui_server.bridge_failure(state, export) == "An import is already running."
    clock[0] += ui_server.BRIDGE_RETRY_SECONDS
    assert ui_server.bridge_failure(state, export) == ""
    # any other end holds until the export changes
    ui_server.note_bridge_failure(state, export, "Not an export.")
    clock[0] += 10_000
    assert ui_server.bridge_failure(state, export) == "Not an export."
    export.write_text("message_id,timestamp\n1,2024-01-01 09:00:00\n2,2024-01-01 09:01:00\n", encoding="utf-8")
    assert ui_server.bridge_failure(state, export) == ""


# ---- a report of a conversation that the archive no longer holds ----


def test_a_report_can_still_be_read_when_its_conversation_left_the_archive(data_folder, server):
    """A chat was deleted in Messages, and an update followed. The story is on disk and needs no archive."""
    loaded(ui_server.DEFAULT_MESSAGES_CSV, ROWS)
    report = ui_server.OUT_DIR / "analysis_+15550009999.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Conversation Analysis - +15550009999\n\n## Summary\nThe story of a summer.\n", encoding="utf-8")
    status, answer = get(server, "/api/analysis", reportPath=str(report))
    assert status == 200
    opened = answer["analysis"]
    assert "The story of a summer." in opened["summaryHtml"] and opened["contact"] == "+15550009999"
    assert opened["stats"]["totalMessages"] == 0 and opened["monthly"] == [] and opened["recentMessages"] == []
    assert opened["estimate"] is None and opened["estimateError"] == ui_server.NO_FIGURES


def test_setup_is_not_taken_for_done_while_a_first_import_fills_the_archive(data_folder, server):
    """The archive file is there as soon as a first import starts. It holds nothing until the import ends."""
    from recall_services import places
    from tests.archive_kit import emptied

    emptied(places.archive_path_for(None))
    assert places.state_for(None).exists()
    status, answer = get(server, "/api/defaults")
    assert status == 200 and answer["hasMessages"] is False


def test_what_the_code_logs_is_no_line_of_a_job(data_folder, monkeypatch):
    import logging

    behind = io.StringIO()
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    monkeypatch.setattr(ui_server.sys, "__stderr__", behind)
    ui_server.log_to_the_server_log()

    def logs_and_ends(*_args, **_more):
        logging.getLogger("recall.test").error("a line with %s", SECRET)
        return {}

    monkeypatch.setattr(ui_server, "do_job", logs_and_ends)
    try:
        job = run("export", {})
    finally:
        root.handlers, root.level = before[0], before[1]
    assert job["status"] == "completed" and "SELECT" not in said(job)
    assert "a line with SELECT" in behind.getvalue()


# ---- the steps of an import, as the server names them and as the guide lists them ----


def test_the_guide_lists_the_steps_of_an_import_in_the_order_of_the_server(data_folder, monkeypatch, tmp_path):
    """The guide marks the step whose number the job gives. The two lists must have one order."""
    import re
    from pathlib import Path

    from recall_services.outcomes import PullDone

    labels = []
    monkeypatch.setattr(ui_server, "names_from_contacts", lambda _job_id: True)
    monkeypatch.setattr(ui_server, "pull_into_archive", lambda *_args: PullDone(counts={}, messages=1, conversations=1))
    monkeypatch.setattr(ui_server, "counted", lambda *_args: {"messages": 1, "conversations": 1})
    real = ui_server.update_job

    def noted(job_id, **fields):
        if "progress" in fields:
            labels.append((fields["progress"]["step"], fields["progress"]["label"]))
        real(job_id, **fields)

    monkeypatch.setattr(ui_server, "update_job", noted)
    assert run("setup_import", {"messagesPath": str(tmp_path / "new.sqlite3")})["status"] == "completed"
    guide = (Path(ui_server.__file__).parent / "app/src/components/onboarding/ImportStep.tsx").read_text(encoding="utf-8")
    phases = re.findall(r"'([^']+)'", re.search(r"const PHASES = \[(.*?)\]", guide, re.S).group(1))
    plain = [label.replace(" your ", " the ") for _step, label in labels]
    assert [step for step, _label in labels] == [1, 2, 3]
    assert plain == phases
