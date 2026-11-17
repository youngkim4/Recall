"""The `analyze` job and /api/analysis on an invented archive, with a fake model.

A report is written by the report engine from the archive as it is. No
request leaves this machine.
"""
import json
from urllib.parse import quote

import pytest

import recall_paths
import ui_server
from recall_services import keys, places
from recall_services import reports as report_service
from report_engine.fake_llm import FakeLLM
from tests.loopback import fetch
from tests.recall_services import report_world
from tests.report_engine.synth import MINA

MODEL = report_world.MODEL
SAVED_NAME = report_world.MINA_CONTACT["name"]
NO_KEY = "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then run the report again."
RESULT_KEYS = {
    "reportPath", "eventsPath", "htmlPath", "jsonPath", "reportId", "status", "warnings", "shownPlain", "usage",
    "analysis", "reports",
}  # fmt: skip
MOMENT_KEYS = {"date", "title", "detail", "category", "score", "quote", "chatId", "day", "at", "pickedId"}


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """An archive with a long conversation, a data folder of its own, and a fake model behind the server."""
    made = report_world.build_desk(tmp_path)
    saves, out = tmp_path / "saves", tmp_path / "out"
    cache = tmp_path / "contact_names.json"
    cache.write_text(json.dumps({"contacts": [report_world.MINA_CONTACT]}), encoding="utf-8")
    monkeypatch.setattr(recall_paths, "SAVES_DIR", saves)
    monkeypatch.setattr(recall_paths, "OUT_DIR", out)
    for name, value in (("OUT_DIR", out), ("DATA_DIR", tmp_path), ("CONTACTS_CACHE_PATH", cache)):
        monkeypatch.setattr(ui_server, name, value)
    monkeypatch.setattr(ui_server, "get_cache_entry", lambda *_args, **_more: None)
    monkeypatch.setattr(ui_server, "set_cache_entry", lambda *_args, **_more: None)
    made.llm = FakeLLM()
    monkeypatch.setattr(report_service, "default_llm", lambda _info: made.llm)
    made.file = str(made.pulled.archive_path)
    yield made
    report_world.close_desk(made)
    places.forget()
    for cache_lock, held in ((ui_server.PREVIEW_CACHE_LOCK, ui_server.PREVIEW_CACHE), (ui_server.ANALYSIS_CACHE_LOCK, ui_server.ANALYSIS_CACHE)):
        with cache_lock:
            held.clear()


def run(payload):
    """Run one job to its end, the way its thread does, and give back how it ended."""
    job = ui_server.job_record("analyze", payload)
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    try:
        ui_server.run_job(job["id"], "analyze", payload)
        with ui_server.JOBS_LOCK:
            return ui_server.job_copy(ui_server.JOBS[job["id"]])
    finally:
        with ui_server.JOBS_LOCK:
            ui_server.JOBS.pop(job["id"], None)


def asked(desk, **more):
    return {"messagesPath": desk.file, "contact": MINA, "model": MODEL, **more}


def lines(job):
    return [entry["message"] for entry in job["logs"]]


def test_a_report_job_writes_the_report_and_gives_what_the_screens_read(desk):
    job = run(asked(desk))
    assert (job["status"], job["error"], job["code"]) == ("completed", None, None)
    result = job["result"]
    assert set(result) == RESULT_KEYS
    assert result["htmlPath"] is None and result["status"] == "complete" and result["shownPlain"] is False
    assert result["reportPath"].endswith(".md") and result["eventsPath"].endswith(".csv")
    assert [report["path"] for report in result["reports"]] == [result["reportPath"]]
    assert result["reports"][0]["contact"] == MINA and result["reports"][0]["displayName"] == SAVED_NAME
    assert job["progress"]["label"] == "Saving the report" and job["progress"]["step"] == job["progress"]["of"]
    assert "Reading the conversation" in lines(job) and "The report is saved." in lines(job)


def test_the_log_of_a_report_job_names_the_model_and_no_handle(desk):
    job = run(asked(desk))
    assert f"Writing a report with {MODEL}" in lines(job)
    assert not any(MINA in line for line in lines(job))


def test_a_moment_of_the_report_opens_its_day_at_the_quoted_message(desk):
    analysis = run(asked(desk))["result"]["analysis"]
    moments = analysis["events"]
    assert moments and all(set(moment) == MOMENT_KEYS for moment in moments)
    assert all(moment["chatId"] == MINA for moment in moments)
    quoted = [moment for moment in moments if moment["pickedId"]]
    assert quoted, "the fake model quotes a message of the conversation"
    for moment in quoted:
        message = desk.archive.message(moment["pickedId"])
        assert message["conversation"] == MINA and message["timestamp"] == moment["at"]
        assert moment["day"] == moment["at"][:10]
        day = ui_server.day_service.payload(desk.archive, ui_server.saved_names(), MINA, moment["day"], around=moment["at"])
        assert moment["pickedId"] in [row["messageId"] for row in day["messages"]]


def test_the_report_opens_again_from_its_file_with_the_same_moments(desk, server):
    result = run(asked(desk))["result"]
    path = f"/api/analysis?messagesPath={quote(desk.file)}&reportPath={quote(result['reportPath'])}&model={MODEL}"
    status, _, raw = fetch(server, path)
    opened = json.loads(raw)["analysis"]
    assert status == 200
    assert opened["events"] == result["analysis"]["events"]
    assert opened["shownPlain"] is False and opened["summaryHtml"].count("<h3>") == 4
    assert opened["contact"] == MINA and opened["contactDisplayName"] == SAVED_NAME
    assert [file["kind"] for file in opened["files"]] == ["md", "csv"]


def test_with_no_key_the_job_fails_in_one_plain_sentence(desk, tmp_path, monkeypatch):
    monkeypatch.undo()  # the real model port of the service
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", tmp_path / "no-key-file")
    monkeypatch.setattr(recall_paths, "SAVES_DIR", tmp_path / "saves")
    monkeypatch.setattr(recall_paths, "OUT_DIR", tmp_path / "out")
    job = run(asked(desk))
    assert (job["status"], job["error"], job["code"]) == ("failed", NO_KEY, "no_key")
    assert job["result"] is None
    assert not any("Traceback" in line or "Error" in line for line in lines(job))
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize(
    ("more", "code", "said"),
    [
        ({"model": "gpt-5.5"}, "model_not_offered", "The model gpt-5.5 is not offered."),
        ({"model": "no-such-model"}, "unknown_model", "Recall does not know the model no-such-model."),
    ],
)
def test_a_model_that_cannot_be_run_fails_the_job_before_any_call(desk, more, code, said):
    job = run(asked(desk, **more))
    assert job["status"] == "failed" and job["code"] == code and job["error"].startswith(said)
    assert desk.llm.calls == []


def test_a_job_with_no_conversation_fails_in_plain_words(desk):
    job = run({"messagesPath": desk.file, "model": MODEL})
    assert (job["status"], job["error"]) == ("failed", "Select a conversation before you start a report.")
    assert desk.llm.calls == []


def test_a_job_on_a_place_with_no_archive_fails_in_plain_words(desk, tmp_path):
    job = run({"messagesPath": str(tmp_path / "nothing-here.sqlite3"), "contact": MINA, "model": MODEL})
    assert (job["status"], job["code"]) == ("failed", "no_archive")
    assert job["error"] == ui_server.NO_ARCHIVE and desk.llm.calls == []


def test_a_second_run_reads_no_part_again_and_the_estimate_says_so(desk):
    before = ui_server.preview_payload(desk.file, MINA, MODEL)["estimate"]
    assert before["periods"] >= 1 and before["periods_cached"] == 0 and before["estimated_cost"] > 0
    run(asked(desk))
    first_calls = len(desk.llm.calls)
    after = ui_server.preview_payload(desk.file, MINA, MODEL)["estimate"]
    assert after["periods_cached"] == after["periods"] == before["periods"]
    assert after["estimated_cost"] < before["estimated_cost"]
    again = run(asked(desk))
    assert again["status"] == "completed"
    assert any("was read in an earlier run" in line for line in lines(again))
    # only the report itself is written again
    assert len(desk.llm.calls) - first_calls < first_calls


def test_a_preview_that_was_kept_is_not_served_after_a_run(desk, monkeypatch):
    """The key of a kept preview holds the file of what earlier runs read: the estimate follows a run at once."""
    key_before = ui_server.preview_cache_key(desk.file, MINA, MODEL)
    run(asked(desk))
    assert ui_server.preview_cache_key(desk.file, MINA, MODEL) != key_before


def test_the_estimate_of_a_model_that_cannot_be_run_is_a_sentence_and_no_number(desk):
    preview = ui_server.preview_payload(desk.file, MINA, "gpt-5.5")
    assert preview["estimate"] is None and preview["estimateError"].startswith("The model gpt-5.5 is not offered.")
    assert preview["stats"]["totalMessages"] == report_world.MINA_MESSAGES
