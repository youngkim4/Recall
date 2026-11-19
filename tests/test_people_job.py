"""The people index behind the server: GET /api/people, the `people` job, and its start after a job or an import.

The archive is an invented one; the models are the fakes of the report engine
and of the index of meanings: no request leaves this machine. A conversation
of the memory needs one message here (`PREPARE_MIN_MESSAGES`).
"""
import json
import threading
from urllib.parse import quote

import pytest

import ui_server
from meaning_index import FakeEmbedder
from recall_services import keys, meaning, memory_prep, people, places
from recall_services import reports as report_service
from recall_services.people_store import PeopleStore
from report_engine.fake_llm import FakeCall, FakeLLM
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

# the model port of the app, as it is before a fixture puts the fake in its place
APP_LLM = report_service.default_llm
MARK = "kingfisher"
FIRST = [f"the {MARK} dove at noon {number}" for number in range(20)]
MORE = [f"a later {MARK} dive {number}" for number in range(3)]
IMPORTS = ["setup_import", "export", "bridge"]
RESULT_KEYS = {"people", "built", "kept", "failed", "notRead", "removed", "embedded", "embedTokens", "usd", "usage"}
ANSWER_KEYS = {
    "state", "model", "people", "profiles", "current", "toWrite", "notRead",
    "estimatedCost", "builtAt", "error", "code", "jobId",
}  # fmt: skip
PEOPLE_IS_BUILT = "The people index is being built. Wait until it ends."
IMPORT_BEFORE_PEOPLE = "An import is running. Wait until it ends, then build the people index."
MEMORY_BEFORE_PEOPLE = "Recall is reading the memory of chat. Wait until it ends, then build the people index."


def reply(call: FakeCall):
    if call.kind == "map":
        return None
    return {"kind": "friend", "status": "ongoing", "summary": "You and this person talk about birds.", "turning_points": []}


@pytest.fixture
def model(monkeypatch):
    """The fake model answers the parts of the memory and the profiles; the fake embedder embeds."""
    fake = FakeLLM(reply)
    monkeypatch.setattr(report_service, "default_llm", lambda _info: fake)
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    monkeypatch.setattr(meaning, "embedder_for", lambda name, dimensions, **_more: FakeEmbedder(dimensions, name))
    monkeypatch.setattr(people, "_count_tokens", lambda: lambda text: len(text.split()))
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


def people_of(server, archive_file):
    status, _headers, body = fetch(server, f"/api/people?messagesPath={quote(archive_file)}")
    return status, json.loads(body)


def build_people(archive_file):
    return run("people", {"messagesPath": archive_file})


def read_memory(archive_file):
    return run("memory", {"messagesPath": archive_file})


def jobs_of(action):
    with ui_server.JOBS_LOCK:
        return [ui_server.job_copy(job) for job in ui_server.JOBS.values() if job["action"] == action]


def profile_calls(model):
    return [call for call in model.calls if call.kind != "map"]


def store_of(archive_file):
    return PeopleStore(people.people_folder(places.archive_path_for(archive_file)))


# --- GET /api/people ---------------------------------------------------------------


def test_with_nothing_read_the_route_says_none_and_who_waits(server, archive_file, model):
    status, answer = people_of(server, archive_file)
    assert status == 200 and set(answer) == ANSWER_KEYS
    assert (answer["state"], answer["people"], answer["notRead"], answer["toWrite"], answer["jobId"]) == ("none", 1, 1, 0, None)
    assert (answer["error"], answer["code"]) == (None, None) and model.calls == []


def test_after_the_memory_the_route_says_what_a_build_would_cost(server, archive_file, model):
    assert read_memory(archive_file)["status"] == "completed"
    _status, answer = people_of(server, archive_file)
    assert (answer["state"], answer["toWrite"], answer["notRead"]) == ("none", 1, 0) and answer["estimatedCost"] > 0
    assert profile_calls(model) == []


def test_the_route_names_the_people_job_that_runs(server, archive_file, model, held_jobs):
    job = ui_server.create_job("people", {"messagesPath": archive_file})
    _status, answer = people_of(server, archive_file)
    assert answer["jobId"] == job["id"]


def test_the_route_needs_an_archive_as_the_other_routes_do(server, tmp_path, model):
    status, _headers, body = fetch(server, f"/api/people?messagesPath={quote(str(tmp_path / 'none.sqlite3'))}")
    assert status >= 400 and json.loads(body)["error"]


def test_two_calls_in_a_row_work_the_answer_out_once_and_a_build_moves_it(server, archive_file, model, monkeypatch):
    calls = []
    real = people.payload
    monkeypatch.setattr(people, "payload", lambda *args, **more: calls.append(1) or real(*args, **more))
    read_memory(archive_file)
    first = people_of(server, archive_file)
    assert people_of(server, archive_file) == first and len(calls) == 1
    assert build_people(archive_file)["status"] == "completed"
    assert people_of(server, archive_file)[1]["state"] == "ready" and len(calls) == 2


# --- the job -----------------------------------------------------------------------


def test_the_job_writes_the_profiles_and_gives_the_counts_and_the_cost(archive_file, model):
    read_memory(archive_file)
    job = build_people(archive_file)
    assert (job["status"], job["error"], job["code"], job["contact"]) == ("completed", None, None, "")
    result = job["result"]
    assert set(result) == RESULT_KEYS and (result["people"], result["built"], result["embedded"]) == (1, 1, 1)
    assert len(profile_calls(model)) == 1 and result["usd"] > 0
    assert job["spent"] == {"usd": result["usd"], "usage": result["usage"]}
    assert job["progress"] == {"step": 3, "of": 3, "label": "Saving the people index", "done": 1, "total": 1}
    assert MARK not in said(job)
    assert len(store_of(archive_file).read().profiles) == 1


def test_the_job_is_asked_for_as_every_job_is(server, archive_file, model):
    status, answer = post(server, "/api/jobs", {"action": "people", "messagesPath": archive_file})
    assert (status, answer["job"]["action"], answer["job"]["status"]) == (202, "people", "queued")
    assert ended(answer["job"]["id"])["status"] == "completed"


def test_with_no_key_the_job_fails_in_one_plain_sentence(archive_file, model, monkeypatch, tmp_path):
    import recall_paths

    read_memory(archive_file)
    monkeypatch.setattr(report_service, "default_llm", APP_LLM)
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", tmp_path / "no-key-file")
    job = build_people(archive_file)
    assert (job["status"], job["code"], job["result"]) == ("failed", "no_key", None)
    assert job["error"] == (
        "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then build the people index again."
    )
    assert job["spent"] == {"usd": 0.0, "usage": None}


def test_a_job_on_a_place_with_no_archive_fails_in_plain_words(tmp_path, model):
    job = build_people(str(tmp_path / "nothing-here.sqlite3"))
    assert (job["status"], job["code"], job["error"]) == ("failed", "no_archive", ui_server.NO_ARCHIVE)


# --- one at a time -----------------------------------------------------------------


def test_a_second_people_job_is_refused_and_makes_no_job(server, held_jobs):
    ui_server.create_job("people", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("people", {})
    assert (str(refused.value), refused.value.status) == (PEOPLE_IS_BUILT, 409)
    assert post(server, "/api/jobs", {"action": "people"}) == (409, {"error": PEOPLE_IS_BUILT})
    assert listed() == [("people", "queued")]


@pytest.mark.parametrize("other", IMPORTS)
def test_a_people_job_and_an_import_refuse_each_other(held_jobs, other):
    first = ui_server.create_job("people", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job(other, {})
    assert str(refused.value) == PEOPLE_IS_BUILT
    ui_server.update_job(first["id"], status="completed")
    ui_server.create_job(other, {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("people", {})
    assert str(refused.value) == IMPORT_BEFORE_PEOPLE


def test_a_people_job_waits_for_the_memory_job_but_not_the_other_way(held_jobs):
    first = ui_server.create_job("memory", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("people", {})
    assert str(refused.value) == MEMORY_BEFORE_PEOPLE
    ui_server.update_job(first["id"], status="completed")
    ui_server.create_job("people", {})
    ui_server.create_job("memory", {})
    assert listed() == [("memory", "completed"), ("people", "queued"), ("memory", "queued")]


def test_a_report_an_index_build_and_a_digest_run_beside_the_people_job(held_jobs):
    ui_server.create_job("people", {})
    ui_server.create_job("analyze", {"contact": "+12025550143"})
    ui_server.create_job("meaning", {})
    ui_server.create_job("digest", {"conversation": "+12025550143"})
    assert [action for action, _status in listed()] == ["people", "analyze", "meaning", "digest"]


# --- after a job or an import --------------------------------------------------------


def test_after_a_memory_job_the_server_writes_the_profiles_that_changed(archive_file, model, messages_home):
    read_memory(archive_file)
    assert build_people(archive_file)["status"] == "completed"
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    (memory_job,) = jobs_of("memory")
    assert ended(memory_job["id"])["status"] == "completed"
    deadline = threading.Event()
    for _ in range(500):
        if jobs_of("people"):
            break
        deadline.wait(0.01)
    (job,) = jobs_of("people")
    done = ended(job["id"])
    assert (done["status"], done["result"]["built"]) == ("completed", 1)
    assert len(profile_calls(model)) == 2


def test_after_a_memory_job_with_no_people_job_before_the_server_starts_nothing(archive_file, model, monkeypatch):
    asked = []
    monkeypatch.setattr(people, "payload", lambda *args, **more: asked.append(1) or pytest.fail("an estimate"))
    assert read_memory(archive_file)["status"] == "completed"
    assert jobs_of("people") == [] and asked == []


def test_after_a_memory_job_the_server_waits_for_the_reader_when_it_costs_more(archive_file, model, monkeypatch, capfd):
    read_memory(archive_file)
    build_people(archive_file)
    monkeypatch.setattr(people, "PEOPLE_AUTO_USD", 0.0)
    monkeypatch.setattr(memory_prep, "PREPARE_MIN_MESSAGES", 1)
    archive = places.state_for(archive_file).archive()
    store = store_of(archive_file)
    held = store.read()
    from recall_services.people_store import Held

    store.write(Held(held.meta, tuple(p.__class__(**{**p.__dict__, "input_hash": "old"}) for p in held.profiles)))
    assert read_memory(archive_file)["status"] == "completed"
    assert jobs_of("people") == [] and archive is not None
    assert "the people index waits for the reader: 1 profile to write" in capfd.readouterr().err


def test_after_an_import_with_the_memory_not_read_the_server_starts_nothing(archive_file, model, messages_home):
    read_memory(archive_file)
    build_people(archive_file)
    memory_prep.marker_path(places.archive_path_for(archive_file)).unlink()
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert jobs_of("memory") == [] and jobs_of("people") == []


def test_a_fault_while_the_server_looks_at_the_people_does_not_change_how_the_job_ended(archive_file, model, monkeypatch, capfd):
    read_memory(archive_file)
    build_people(archive_file)

    def broken(*_args, **_more):
        raise RuntimeError(f"a fault that says {MARK}")

    monkeypatch.setattr(people, "payload", broken)
    job = read_memory(archive_file)
    assert (job["status"], job["error"]) == ("completed", None)
    assert jobs_of("people") == [] and MARK not in "".join(capfd.readouterr())


def test_a_failed_job_keeps_what_it_spent(archive_file, model, monkeypatch):
    usage = {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 100, "cost_usd": 0.0123, "calls": 3}

    def paid_then_refused(*_args, **_more):
        raise people.PeopleFailure("key_rejected", "OpenAI rejected the API key.", usage)

    monkeypatch.setattr(people, "build", paid_then_refused)
    job = build_people(archive_file)
    assert (job["status"], job["code"], job["error"]) == ("failed", "key_rejected", "OpenAI rejected the API key.")
    assert job["spent"] == {"usd": 0.0123, "usage": usage}
