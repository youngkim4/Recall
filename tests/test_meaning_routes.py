"""The index of meanings behind the server: GET /api/meaning and the `meaning` job.

The archive is an invented one. The embedder is a stand-in that is put in the
place of the one of the app: no request leaves this machine.
"""
import json
import logging
import threading
from urllib.parse import quote

import pytest

import ui_server
from meaning_index import EmbedderFacts, EmbeddingKeyError, FakeEmbedder, IndexStore, StoreError
from recall_services import meaning as meaning_service
from recall_services import places
from tests.archive_kit import (
    PERSON,
    bridged,
    data_folder,  # noqa: F401
    ended,
    import_from,
    jobs_end,  # noqa: F401
    listed,
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    pulled,
    run,
    said,
    write_database,
    write_export,
)
from tests.loopback import fetch, post

pytestmark = pytest.mark.usefixtures(
    "fresh_bundle_cache", "no_names", "own_archives", "no_jobs", "jobs_end", "data_folder", "messages_home"
)

# a word that no line of a job and no line of the log may hold
MARK = "zebrafinch"
FIRST = [f"the {MARK} sang at dawn {number}" for number in range(20)]
MORE = [f"a later {MARK} line {number}" for number in range(3)]
FAKE_FACTS = EmbedderFacts(
    model="fake-embedding-1", dimensions=64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="words"
)
STATUS_KEYS = {"state", "segments", "model", "dimensions", "builtAt", "coveredUntil"}
ESTIMATE_KEYS = {"segments", "newSegments", "staleSegments", "tokens", "estimatedCost", "upperBound"}
BUILD_KEYS = {"segments", "embedded", "reused", "removed", "tokens"}
IMPORTS = ["setup_import", "export", "bridge"]
INDEX_IS_BUILT = "The index for the search by meaning is being built. Wait until it ends."
IMPORT_FIRST = "An import is running. Wait until it ends, then build the index."
NO_KEY = "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then build the index again."
HANDLE_PARTS = (PERSON, "5550190001", "555019")


@pytest.fixture
def embedder(monkeypatch):
    """The embedder of the app is a fixed one, and the app builds with its model."""
    fake = FakeEmbedder()
    monkeypatch.setattr(meaning_service, "default_facts", lambda: FAKE_FACTS)
    monkeypatch.setattr(meaning_service, "embedder_for", lambda _model, _dimensions, **_more: fake)
    return fake


@pytest.fixture
def archive_file(data_folder):  # noqa: F811
    """An archive that a pull filled with the messages of FIRST."""
    return str(pulled(data_folder / "saves" / "archive.sqlite3", FIRST))


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that start and do not end until they are let go. Each one stays as it was made: queued."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()
    # a job that was held never ends in the list: it must not hold up the end of the test
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


def meaning_of(server, archive_file):
    status, _headers, body = fetch(server, f"/api/meaning?messagesPath={quote(archive_file)}")
    return status, json.loads(body)


def build(archive_file):
    return run("meaning", {"messagesPath": archive_file})


def pull_more(archive_file, texts):
    """Pull a database with more messages into the archive, with no job."""
    source = write_database(places.archive_path_for(archive_file).parent / "more" / "chat.db", texts)
    outcome = places.state_for(archive_file).pull_from_messages(source, None)
    assert outcome.status == "done", outcome.reason


def meaning_jobs():
    with ui_server.JOBS_LOCK:
        return [ui_server.job_copy(job) for job in ui_server.JOBS.values() if job["action"] == "meaning"]


# --- GET /api/meaning --------------------------------------------------------------


def test_with_no_index_the_route_says_none_and_what_a_build_would_cost(server, archive_file, embedder):
    status, answer = meaning_of(server, archive_file)
    assert status == 200 and set(answer) == {"status", "estimate", "error", "code"}
    assert answer["status"] == {
        "state": "none", "segments": 0, "model": None, "dimensions": None, "builtAt": None, "coveredUntil": None,
    }  # fmt: skip
    assert set(answer["estimate"]) == ESTIMATE_KEYS
    assert answer["estimate"]["segments"] == answer["estimate"]["newSegments"] == 2
    assert answer["estimate"]["tokens"] > 0 and answer["estimate"]["estimatedCost"] > 0
    assert (answer["error"], answer["code"]) == (None, None)
    assert embedder.calls == [] and not meaning_service.index_store(archive_file).folder.exists()


def test_with_a_ready_index_the_route_says_ready_and_a_build_would_send_nothing(server, archive_file, embedder):
    assert build(archive_file)["status"] == "completed"
    status, answer = meaning_of(server, archive_file)
    assert status == 200 and set(answer["status"]) == STATUS_KEYS
    assert (answer["status"]["state"], answer["status"]["segments"]) == ("ready", 2)
    assert (answer["status"]["model"], answer["status"]["dimensions"]) == ("fake-embedding-1", 64)
    assert answer["status"]["coveredUntil"] == "2024-07-01 09:19:00" and answer["status"]["builtAt"]
    assert (answer["estimate"]["newSegments"], answer["estimate"]["tokens"], answer["estimate"]["estimatedCost"]) == (0, 0, 0.0)


def test_with_later_messages_in_the_archive_the_route_says_behind(server, archive_file, embedder):
    build(archive_file)
    pull_more(archive_file, [*FIRST, *MORE])
    status, answer = meaning_of(server, archive_file)
    assert (status, answer["status"]["state"], answer["status"]["segments"]) == (200, "behind", 2)
    assert answer["estimate"]["newSegments"] == 1 and answer["estimate"]["tokens"] > 0


def test_with_no_archive_the_route_answers_as_every_route_that_reads_the_archive(server, data_folder):  # noqa: F811
    status, answer = meaning_of(server, str(data_folder / "saves" / "archive.sqlite3"))
    assert (status, answer["code"]) == (400, "no_archive")


def test_with_no_key_the_route_still_gives_the_status_and_the_estimate(server, archive_file, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    status, answer = meaning_of(server, archive_file)
    assert (status, answer["status"]["state"], answer["error"]) == (200, "none", None)
    assert answer["estimate"]["newSegments"] == 2


def test_an_index_that_this_build_can_not_read_is_told_in_one_sentence(server, archive_file, embedder):
    folder = meaning_service.index_store(archive_file).folder
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text('{"format_version": 99}', encoding="utf-8")
    status, answer = meaning_of(server, archive_file)
    assert (status, answer["status"]["state"], answer["estimate"], answer["code"]) == (200, "none", None, "index_newer")
    assert answer["error"] == "A newer version of Recall made this index. Update Recall, then try again."


# --- the job -----------------------------------------------------------------------


def test_the_job_builds_the_index_and_gives_the_counts(archive_file, embedder):
    job = build(archive_file)
    assert (job["status"], job["error"], job["code"], job["contact"]) == ("completed", None, None, "")
    assert job["result"] == {"segments": 2, "embedded": 2, "reused": 0, "removed": 0, "tokens": job["result"]["tokens"]}
    assert set(job["result"]) == BUILD_KEYS and job["result"]["tokens"] > 0
    assert job["progress"] == {"step": 3, "of": 3, "label": "Saving the index", "done": 2, "total": 2}
    assert [entry["message"] for entry in job["logs"]] == [
        "Reading 1 conversation",
        "Reading the meaning of 2 exchanges",
        "Saving 2 exchanges",
        "The index holds 2 exchanges: 2 new, 0 kept, 0 removed.",
    ]
    assert len(embedder.texts) == 2


def test_the_job_is_asked_for_as_every_job_is(server, archive_file, embedder):
    status, answer = post(server, "/api/jobs", {"action": "meaning", "messagesPath": archive_file})
    assert (status, answer["job"]["action"]) == (202, "meaning")
    done = ended(answer["job"]["id"])
    assert (done["status"], done["result"]["segments"]) == ("completed", 2)


def test_a_second_build_brings_the_index_up_to_date_and_pays_for_the_new_part_only(archive_file, embedder):
    build(archive_file)
    pull_more(archive_file, [*FIRST, *MORE])
    job = build(archive_file)
    assert (job["result"]["segments"], job["result"]["embedded"], job["result"]["reused"]) == (2, 1, 1)
    assert len(embedder.texts) == 3


def test_the_lines_of_the_job_and_the_log_hold_counts_only(archive_file, embedder, caplog, capfd):
    caplog.set_level(logging.DEBUG)
    job = build(archive_file)
    pull_more(archive_file, [*FIRST, *MORE])
    again = build(archive_file)
    told = "\n".join([said(job), said(again), str(job["progress"]), str(job["result"]), caplog.text, *capfd.readouterr()])
    assert job["status"] == again["status"] == "completed"
    assert MARK not in told and "sang at dawn" not in told
    assert not [part for part in HANDLE_PARTS if part in told]


def test_with_no_key_the_job_fails_with_the_plain_sentence(archive_file, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    job = build(archive_file)
    assert (job["status"], job["error"], job["code"], job["result"]) == ("failed", NO_KEY, "no_key", None)
    assert meaning_service.built(archive_file) is False


def test_a_key_that_the_provider_refuses_fails_the_job_with_one_sentence(archive_file, monkeypatch, capfd):
    class Refused(FakeEmbedder):
        def embed(self, texts):
            raise EmbeddingKeyError(f"the provider said {MARK}", status_code=401)

    monkeypatch.setattr(meaning_service, "default_facts", lambda: FAKE_FACTS)
    monkeypatch.setattr(meaning_service, "embedder_for", lambda _model, _dimensions, **_more: Refused())
    job = build(archive_file)
    assert (job["status"], job["code"]) == ("failed", "key_rejected")
    assert job["error"] == (
        "OpenAI rejected the API key. Check OPENAI_API_KEY in the file .env in Recall's data folder, "
        "then build the index again."
    )
    assert MARK not in said(job)


def test_the_job_of_an_archive_that_is_not_there_says_so(data_folder, embedder):  # noqa: F811
    job = build(str(data_folder / "saves" / "archive.sqlite3"))
    assert (job["status"], job["code"]) == ("failed", "no_archive")
    assert job["error"] == "There is no archive yet. Open the setup guide to read your messages."


# --- one at a time -----------------------------------------------------------------


def test_a_second_build_of_the_index_is_refused_and_makes_no_job(server, held_jobs):
    made = ui_server.create_job("meaning", {})
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("meaning", {})
    assert (str(refused.value), refused.value.status) == (INDEX_IS_BUILT, 409)
    assert post(server, "/api/jobs", {"action": "meaning"}) == (409, {"error": INDEX_IS_BUILT})
    assert listed() == [("meaning", "queued")] and made["status"] == "queued"


@pytest.mark.parametrize("state", ["queued", "running"])
@pytest.mark.parametrize("other", IMPORTS)
def test_an_import_is_refused_while_the_index_is_built(held_jobs, other, state):
    made = ui_server.create_job("meaning", {})
    ui_server.update_job(made["id"], status=state)
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job(other, {})
    assert (str(refused.value), refused.value.status) == (INDEX_IS_BUILT, 409)
    assert listed() == [("meaning", state)]


@pytest.mark.parametrize("state", ["queued", "running"])
@pytest.mark.parametrize("other", IMPORTS)
def test_a_build_of_the_index_is_refused_while_an_import_runs(held_jobs, other, state):
    made = ui_server.create_job(other, {})
    ui_server.update_job(made["id"], status=state)
    with pytest.raises(ui_server.JobRefused) as refused:
        ui_server.create_job("meaning", {})
    assert (str(refused.value), refused.value.status) == (IMPORT_FIRST, 409)
    assert listed() == [(other, state)]


@pytest.mark.parametrize("how", ["completed", "failed"])
def test_each_can_start_when_the_other_has_ended_and_a_report_runs_beside_a_build(held_jobs, how):
    first = ui_server.create_job("meaning", {})
    ui_server.create_job("analyze", {"contact": "+15550100"})
    ui_server.update_job(first["id"], status=how)
    second = ui_server.create_job("export", {})
    ui_server.update_job(second["id"], status=how)
    ui_server.create_job("meaning", {})
    assert listed() == [("meaning", how), ("analyze", "queued"), ("export", how), ("meaning", "queued")]


# --- after an import ---------------------------------------------------------------


def refreshed():
    """The one job that the server started for the index, as it ended."""
    (job,) = meaning_jobs()
    return ended(job["id"])


@pytest.mark.parametrize("action", ["setup_import", "export"])
def test_after_an_import_that_changed_the_archive_the_server_brings_the_index_up_to_date(
    archive_file, embedder, messages_home, action
):
    build(archive_file)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run(action, {"messagesPath": archive_file, "includeContacts": False})
    assert job["status"] == "completed", job["error"]
    done = refreshed()
    assert (done["status"], done["error"]) == ("completed", None)
    assert (done["result"]["embedded"], done["result"]["reused"]) == (1, 1)
    status = meaning_service.status(places.state_for(archive_file).archive(), ui_server.saved_names())
    assert (status["state"], status["coveredUntil"]) == ("ready", "2024-07-01 09:22:00")


def test_after_an_import_with_no_index_the_server_starts_nothing(archive_file, embedder, tmp_path):
    job = import_from(write_database(tmp_path / "next" / "chat.db", [*FIRST, *MORE]), archive_file)
    assert job["status"] == "completed", job["error"]
    assert meaning_jobs() == [] and embedder.calls == []
    assert not meaning_service.index_store(archive_file).folder.exists()


def test_after_an_import_that_failed_the_server_starts_nothing(archive_file, embedder, messages_home):
    build(archive_file)
    job = run("export", {"messagesPath": archive_file})  # Messages keeps no database here
    assert job["status"] == "failed"
    assert meaning_jobs() == []


def test_after_an_export_is_loaded_again_the_server_brings_the_index_up_to_date(embedder, tmp_path):
    export = bridged(tmp_path / "earlier" / "messages.csv", ["one", "two", "three"])
    assert build(str(export))["status"] == "completed"
    write_export(export, ["one", "two", "three", "four"])
    job = run("bridge", {"messagesPath": str(export)})
    assert (job["status"], job["result"]["status"]) == ("completed", "built")
    assert refreshed()["status"] == "completed"


def test_after_an_export_is_loaded_into_an_archive_with_no_index_of_its_own_nothing_is_started(
    archive_file, embedder, tmp_path
):
    build(archive_file)  # another archive has an index
    export = write_export(tmp_path / "earlier" / "messages.csv", ["one", "two", "three"])
    job = run("bridge", {"messagesPath": str(export)})
    assert (job["status"], job["result"]["status"]) == ("completed", "built")
    assert meaning_jobs() == []


def test_after_a_load_that_changed_nothing_the_server_starts_nothing(archive_file, embedder, tmp_path):
    build(archive_file)
    export = bridged(tmp_path / "earlier" / "messages.csv", ["one", "two", "three"])
    job = run("bridge", {"messagesPath": str(export)})
    assert (job["status"], job["result"]["status"]) == ("completed", "current")
    assert meaning_jobs() == []


def test_a_report_job_starts_no_build_of_the_index(archive_file, embedder):
    build(archive_file)
    run("analyze", {"messagesPath": archive_file})  # it fails: no conversation is named
    assert meaning_jobs() == []


def test_the_job_that_the_server_starts_holds_up_the_next_import(archive_file, embedder, messages_home, monkeypatch):
    build(archive_file)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    started, let_go = threading.Event(), threading.Event()
    real = meaning_service.build

    def slow(*args, **more):
        started.set()
        let_go.wait(timeout=5)
        return real(*args, **more)

    monkeypatch.setattr(meaning_service, "build", slow)
    try:
        assert run("export", {"messagesPath": archive_file})["status"] == "completed"
        assert started.wait(timeout=5)
        with pytest.raises(ui_server.JobRefused) as refused:
            ui_server.create_job("export", {"messagesPath": archive_file})
        assert str(refused.value) == INDEX_IS_BUILT
    finally:
        let_go.set()
    assert refreshed()["status"] == "completed"


@pytest.mark.parametrize("fault", [StoreError("the folder says nothing"), OSError("the disk is gone")])
def test_an_index_folder_that_can_not_be_looked_at_does_not_fail_the_import_and_starts_nothing(
    archive_file, embedder, messages_home, monkeypatch, fault
):
    build(archive_file)

    def broken(_store):
        raise fault

    monkeypatch.setattr(IndexStore, "status", broken)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run("export", {"messagesPath": archive_file})
    assert (job["status"], job["error"], job["code"]) == ("completed", None, None)
    assert meaning_jobs() == []


def test_the_server_looks_at_the_index_after_the_import_has_ended_in_the_list(
    archive_file, embedder, messages_home, monkeypatch
):
    """A slow look at the index folder delays no import: the job says completed before the look starts."""
    seen = []

    def noted(*_args, **_more):
        with ui_server.JOBS_LOCK:
            seen.extend((job["action"], job["status"], job["result"] is not None) for job in ui_server.JOBS.values())
        return False

    monkeypatch.setattr(meaning_service, "built", noted)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert seen == [("export", "completed", True)]


def test_with_no_index_folder_the_server_asks_no_database_about_the_index(archive_file, messages_home, monkeypatch):
    """With no index there is nothing to read: an import opens no file of the index."""
    asked = []
    monkeypatch.setattr(IndexStore, "status", lambda store: asked.append(store) or pytest.fail("the store was asked"))
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert asked == [] and meaning_service.built(archive_file) is False
    assert not meaning_service.index_store(archive_file).folder.exists()


def test_a_fault_while_the_server_looks_at_the_index_does_not_change_how_the_import_ended(
    archive_file, embedder, messages_home, monkeypatch, capfd
):
    def broken(*_args, **_more):
        raise RuntimeError(f"a fault that says {MARK}")

    monkeypatch.setattr(meaning_service, "built", broken)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run("export", {"messagesPath": archive_file})
    assert (job["status"], job["error"]) == ("completed", None)
    assert meaning_jobs() == [] and MARK not in said(job)
    assert MARK not in "".join(capfd.readouterr())
