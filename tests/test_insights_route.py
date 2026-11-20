"""Insights behind the server: GET /api/insights and the `insights` job.

The archive is the invented insight world, pulled the way the app pulls.
"Today" is the world's day. The model is a fake: no request leaves this machine.
"""

import json
import threading
from datetime import datetime
from urllib.parse import quote

import pytest

import recall_paths
import ui_server
from evals.insight_world import NOW, insight_world
from evals.insight_world.cast import CONTACTS
from evals.world import build_world, zone
from recall_services import insights, keys, places
from recall_services import reports as report_service
from report_engine.ports import LLMResult, LLMUsage
from tests.archive_kit import data_folder, ended, jobs_end, own_archives, run, said  # noqa: F401
from tests.loopback import fetch, post

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "own_archives", "no_jobs", "jobs_end", "data_folder", "messages_home")

ANSWER_KEYS = {"state", "readAt", "model", "people", "toRead", "estimatedCost", "picks", "insights", "jobId"}
RESULT_KEYS = {"read", "failed", "found", "usd", "usage"}


class FakeLLM:
    """One piece of news for each thread, citing its newest message."""

    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def generate(self, messages, *, schema=None, max_output_tokens, reasoning=None) -> LLMResult:
        with self.lock:
            self.calls += 1
        last = [line.split(" ", 1)[0] for line in messages[-1]["content"].splitlines() if line.startswith("m")][-1]
        item = {"kind": "news", "summary": "Something big happened.", "refs": [last], "done": False}
        return LLMResult("{}", {"items": [item]}, LLMUsage(1000, 0, 100), "completed")


class Today(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture
def model(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(report_service, "default_llm", lambda _info: fake)
    monkeypatch.setattr(insights, "datetime", Today)
    return fake


@pytest.fixture
def archive_file(data_folder, tmp_path, monkeypatch):  # noqa: F811
    names = data_folder / "saves" / "contact_names.json"
    names.parent.mkdir(parents=True, exist_ok=True)
    names.write_text(json.dumps({"contacts": list(CONTACTS)}))
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", names)
    monkeypatch.setattr(ui_server, "export_contact_names", lambda *args, **more: None)
    with zone():
        built = build_world(tmp_path / "world", insight_world())
        archive = data_folder / "saves" / "archive.sqlite3"
        outcome = places.state_for(archive).pull_from_messages(built.source_path, None)
        assert outcome.status == "done", outcome.reason
        yield str(places.archive_path_for(archive))


@pytest.fixture
def held_jobs(monkeypatch):
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


def insights_of(server, archive_file):
    status, _headers, body = fetch(server, f"/api/insights?messagesPath={quote(archive_file)}")
    return status, json.loads(body)


def test_before_a_reading_the_route_gives_the_counted_insights_and_the_cost(server, archive_file, model):
    status, answer = insights_of(server, archive_file)
    assert status == 200 and set(answer) == ANSWER_KEYS
    assert (answer["state"], answer["readAt"], answer["jobId"]) == ("none", None, None)
    assert answer["people"] == 15 and answer["toRead"] == 12 and answer["estimatedCost"] > 0
    assert [pick["name"] for pick in answer["picks"]][:2] == ["Iris Calder", "Jonah Whitlow"]
    assert {item["kind"] for item in answer["insights"]} == {"drift", "balance", "unanswered", "birthday"}
    assert model.calls == 0


def test_the_job_reads_saves_and_the_route_shows_it(server, archive_file, model):
    status, made = post(server, "/api/jobs", {"action": "insights", "messagesPath": archive_file})
    assert (status, made["job"]["action"]) == (202, "insights")
    job = ended(made["job"]["id"])
    assert (job["status"], job["error"]) == ("completed", None)
    assert set(job["result"]) == RESULT_KEYS and job["result"]["read"] == 12 and model.calls == 12
    assert job["spent"]["usd"] == job["result"]["usd"] > 0
    assert "Felix" not in said(job) and "observatory" not in said(job)
    _status, answer = insights_of(server, archive_file)
    assert answer["state"] == "ready" and answer["readAt"] == NOW.isoformat(timespec="seconds")
    assert sum(item["kind"] == "news" for item in answer["insights"]) == 12


def test_the_route_names_the_insights_job_that_runs(server, archive_file, model, held_jobs):
    job = ui_server.create_job("insights", {"messagesPath": archive_file})
    assert insights_of(server, archive_file)[1]["jobId"] == job["id"]


def test_with_no_key_no_job_is_made(server, archive_file, model, monkeypatch, tmp_path):
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", tmp_path / "no-key-file")
    status, answer = post(server, "/api/jobs", {"action": "insights", "messagesPath": archive_file})
    assert (status, answer["error"]) == (409, insights.NO_KEY)
    assert answer["error"].endswith("then look for things to follow up on again.")
    with ui_server.JOBS_LOCK:
        assert not [job for job in ui_server.JOBS.values() if job["action"] == "insights"]


def test_one_reading_at_a_time_and_no_import_meanwhile(server, held_jobs):
    ui_server.create_job("insights", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("insights", {})
    assert (str(refused.value), refused.value.status) == (ui_server.INSIGHTS_ARE_READ, 409)
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("export", {})
    assert str(refused.value) == ui_server.INSIGHTS_ARE_READ


def test_a_reading_waits_for_an_import(server, held_jobs):
    ui_server.create_job("export", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("insights", {})
    assert str(refused.value) == ui_server.IMPORT_BEFORE_INSIGHTS


def test_a_job_on_a_place_with_no_archive_fails_in_plain_words(tmp_path, model):
    job = run("insights", {"messagesPath": str(tmp_path / "nothing-here.sqlite3")})
    assert (job["status"], job["code"], job["error"]) == ("failed", "no_archive", ui_server.NO_ARCHIVE)
