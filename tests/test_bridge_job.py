"""The load of an export of an earlier build: the `bridge` job, and what a request is told while it runs.

An earlier build kept the messages in `messages.csv`. When the app names
such an export and its archive is not there yet, the first request starts one
job that loads the export, and each request is told to wait for that job. An
export that cannot be loaded is told in a plain sentence, and it is not tried
again until its file changes. An archive that a pull filled is never replaced
by an export.
"""
import json
import threading
from urllib.parse import quote

import pytest

import ui_server
from tests.archive_kit import (
    KEPT,
    PERSON,
    beside,
    bridged,
    data_folder,  # noqa: F401
    ended,
    held,
    jobs_end,  # noqa: F401
    listed,
    no_names,  # noqa: F401
    not_an_archive,
    own_archives,  # noqa: F401
    picture,
    pulled,
    run,
    said,
    unchanged,
    write_export,
)
from tests.loopback import fetch, post

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives", "no_jobs", "jobs_end")

BUILDING = "Recall is getting your archive ready. This takes a moment."
NOT_AN_EXPORT = "The file messages.csv is not an export of messages that Recall can read."
TEXTS = ["one", "two", "three"]
NOT_PLAIN = ("SELECT", "Traceback", "Error", "Exception", "sqlite3", ".py")


@pytest.fixture
def export(tmp_path):
    """An export of an earlier build with three messages. Its archive is not made yet."""
    return write_export(tmp_path / "earlier" / "messages.csv", TEXTS)


@pytest.fixture
def held_load(monkeypatch, jobs_end):  # noqa: F811
    """A load that starts and waits until it is let go. Gives the switch that lets it go."""
    let_go = threading.Event()
    real = ui_server.load_export

    def wait_then_load(job_id, payload):
        let_go.wait(timeout=10)
        return real(job_id, payload)

    monkeypatch.setattr(ui_server, "load_export", wait_then_load)
    yield let_go
    let_go.set()


def not_ready(messages_path):
    """What a request that needs the archive is told when there is none to read."""
    with pytest.raises(ui_server.ArchiveNotReady) as waiting:
        ui_server.archive_for(messages_path)
    return waiting.value


def conversations(server, messages_path):
    status, _, body = fetch(server, f"/api/contacts?messagesPath={quote(str(messages_path))}")
    return status, json.loads(body)


def write_what_is_no_export(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not,an,export\n1,2,3\n", encoding="utf-8")
    return path


# ---- the first request starts the load ----


def test_a_request_for_an_export_with_no_archive_starts_one_load_and_is_told_to_wait(export):
    waiting = not_ready(export)

    assert (waiting.status, waiting.code, str(waiting)) == (503, "archive_building", BUILDING)
    assert waiting.body() == {"error": BUILDING, "code": "archive_building", "job": waiting.job}
    assert (waiting.job["action"], waiting.job["status"]) == ("bridge", "queued")
    assert [action for action, _status in listed()] == ["bridge"]
    assert ended(waiting.job["id"])["status"] == "completed"


def test_when_the_load_has_ended_the_archive_is_read(export):
    job = ended(not_ready(export).job["id"])

    _state, archive = ui_server.archive_for(export)

    assert job["status"] == "completed", job["error"]
    assert archive.summary()["messages"] == 3
    assert held(export) == {"legacy:1": "one", "legacy:2": "two", "legacy:3": "three"}
    # and no second load was started for it
    assert listed() == [("bridge", "completed")]


def test_a_second_request_while_the_load_runs_gets_the_same_job(export, held_load):
    first = not_ready(export)
    second = not_ready(export)
    third = not_ready(export.with_suffix(".sqlite3"))

    assert second.job["id"] == first.job["id"]
    assert (third.status, third.code) == (503, "archive_building")
    assert len(listed()) == 1
    held_load.set()
    assert ended(first.job["id"])["status"] == "completed"


def test_many_requests_at_the_same_moment_start_one_load(export, held_load):
    together = threading.Barrier(8)
    told = []

    def ask():
        together.wait(timeout=5)
        try:
            ui_server.archive_for(export)
        except ui_server.ArchiveNotReady as waiting:
            told.append((waiting.status, waiting.code, waiting.job["id"]))

    askers = [threading.Thread(target=ask) for _ in range(8)]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join(timeout=5)

    assert len(told) == 8
    assert len(set(told)) == 1 and told[0][:2] == (503, "archive_building")
    assert len(listed()) == 1


def test_the_route_answers_503_with_the_job_and_serves_the_list_when_it_has_ended(server, export, held_load):
    first_status, first = conversations(server, export)
    second_status, second = conversations(server, export)

    assert (first_status, second_status) == (503, 503)
    assert first == {"error": BUILDING, "code": "archive_building", "job": first["job"]}
    assert (first["job"]["action"], second["job"]["id"]) == ("bridge", first["job"]["id"])
    assert len(listed()) == 1

    held_load.set()
    ended(first["job"]["id"])
    status, answer = conversations(server, export)

    assert status == 200
    assert [(row["chat_id"], row["message_count"]) for row in answer["contacts"]] == [(PERSON, 3)]
    # the answer names the archive file itself, as /api/defaults does
    assert (answer["source"], answer["messagesPath"]) == ("messages", str(export.with_suffix(".sqlite3")))
    assert answer["archive"]["messages"] == 3


@pytest.mark.parametrize("path", [
    "/api/overview?",
    "/api/memories?",
    "/api/first-words?",
    "/api/search?query=one&",
    f"/api/day?contact={quote(PERSON)}&day=2024-07-01&",
    f"/api/preview?contact={quote(PERSON)}&",
])
def test_each_route_that_reads_the_archive_waits_for_the_same_load(server, export, held_load, path):
    _, told = conversations(server, export)

    status, _, body = fetch(server, f"{path}messagesPath={quote(str(export))}")

    answer = json.loads(body)
    assert status == 503
    assert (answer["code"], answer["error"], answer["job"]["id"]) == ("archive_building", BUILDING, told["job"]["id"])
    assert len(listed()) == 1


def test_the_job_can_be_followed_to_its_end_through_the_route(server, export):
    _, told = conversations(server, export)
    ended(told["job"]["id"])

    status, _, body = fetch(server, f"/api/jobs/{told['job']['id']}")

    job = json.loads(body)["job"]
    assert (status, job["action"], job["status"], job["error"], job["code"]) == (200, "bridge", "completed", None, None)
    assert job["result"]["messages"] == 3


# ---- what the job tells ----


def test_the_result_of_a_load_has_the_counts_and_the_years(export):
    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 3, "conversations": 1, "firstYear": 2024, "lastYear": 2024, "status": "built"}
    assert "Loading the earlier export into the archive" in said(job)


def test_the_archive_stands_beside_the_export_with_its_note(export):
    run("bridge", {"messagesPath": str(export)})

    archive_file = export.with_suffix(".sqlite3")
    note = json.loads((export.parent / "messages.sqlite3.source.json").read_text(encoding="utf-8"))
    assert archive_file.is_file()
    assert beside(archive_file) == ["messages.csv"]
    assert note["source"] == "export" and note["export"]["path"] == str(export.resolve())


def test_a_load_of_an_export_that_the_archive_holds_changes_nothing(export):
    run("bridge", {"messagesPath": str(export)})
    before = picture(export.with_suffix(".sqlite3"))

    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "completed", job["error"]
    assert job["result"]["status"] == "current"
    assert unchanged(export.with_suffix(".sqlite3"), before)


def test_an_export_that_changed_is_loaded_again(export):
    run("bridge", {"messagesPath": str(export)})
    write_export(export, [*TEXTS, "four"])

    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "completed", job["error"]
    assert (job["result"]["status"], job["result"]["messages"]) == ("built", 4)
    assert sorted(held(export).values()) == sorted([*TEXTS, "four"])


STORES = [
    (ui_server.PREVIEW_CACHE_NAMESPACE, "PREVIEW_CACHE", "PREVIEW_CACHE_LOCK"),
    (ui_server.ANALYSIS_CACHE_NAMESPACE, "ANALYSIS_CACHE", "ANALYSIS_CACHE_LOCK"),
]


def keep_one_in_each_store():
    for namespace, held_name, lock_name in STORES:
        ui_server.set_cache_entry(namespace, "of-the-earlier-archive", {"stats": {}})
        with getattr(ui_server, lock_name):
            getattr(ui_server, held_name)["of-the-earlier-archive"] = {"stats": {}}


def kept_in_the_stores():
    stored = [ui_server.get_cache_entry(namespace, "of-the-earlier-archive") is not None for namespace, _held, _lock in STORES]
    in_memory = ["of-the-earlier-archive" in getattr(ui_server, held_name) for _namespace, held_name, _lock in STORES]
    return stored + in_memory


@pytest.fixture
def stores_cleared():
    yield
    ui_server.forget_previews()


@pytest.mark.usefixtures("stores_cleared")
def test_a_load_drops_the_previews_and_a_load_that_changes_nothing_keeps_them(export):
    keep_one_in_each_store()

    run("bridge", {"messagesPath": str(export)})
    dropped = kept_in_the_stores()
    keep_one_in_each_store()
    run("bridge", {"messagesPath": str(export)})

    assert dropped == [False] * 4
    assert kept_in_the_stores() == [True] * 4


def test_the_figures_follow_the_export_that_was_loaded_again(export):
    run("bridge", {"messagesPath": str(export)})
    assert sorted(ui_server.cached_messages_bundle(export)[0]["text"]) == sorted(TEXTS)
    write_export(export, [*TEXTS, "four"])

    run("bridge", {"messagesPath": str(export)})

    assert sorted(ui_server.cached_messages_bundle(export)[0]["text"]) == sorted([*TEXTS, "four"])


# ---- an export that cannot be loaded ----


def test_an_export_that_cannot_be_loaded_gives_a_failed_job_with_a_plain_sentence(tmp_path):
    export = write_what_is_no_export(tmp_path / "earlier" / "messages.csv")

    job = ended(not_ready(export).job["id"])

    assert job["status"] == "failed"
    assert job["error"].startswith(NOT_AN_EXPORT)
    assert job["result"] is None and job["code"] is None
    assert not any(word in said(job) for word in NOT_PLAIN)
    assert [path.name for path in export.parent.iterdir()] == ["messages.csv"]


def test_a_load_that_fails_with_no_archive_says_nothing_of_an_archive(tmp_path):
    export = write_what_is_no_export(tmp_path / "earlier" / "messages.csv")

    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "failed"
    assert job["error"] == NOT_AN_EXPORT


def test_the_request_after_a_load_that_failed_is_told_why_and_starts_no_job(tmp_path):
    export = write_what_is_no_export(tmp_path / "earlier" / "messages.csv")
    failed = ended(not_ready(export).job["id"])

    again = not_ready(export)
    once_more = not_ready(export)

    assert (again.status, again.code, str(again)) == (500, "archive_failed", failed["error"])
    assert again.body() == {"error": failed["error"], "code": "archive_failed"}
    assert once_more.body() == again.body()
    assert listed() == [("bridge", "failed")]


def test_the_route_answers_500_after_a_load_that_failed(server, tmp_path):
    export = write_what_is_no_export(tmp_path / "earlier" / "messages.csv")
    first_status, first = conversations(server, export)
    failed = ended(first["job"]["id"])

    status, answer = conversations(server, export)

    assert first_status == 503
    assert (status, answer) == (500, {"error": failed["error"], "code": "archive_failed"})
    assert listed() == [("bridge", "failed")]


def test_an_export_that_failed_is_loaded_when_its_file_has_changed(tmp_path):
    export = write_what_is_no_export(tmp_path / "earlier" / "messages.csv")
    ended(not_ready(export).job["id"])
    write_export(export, TEXTS)

    waiting = not_ready(export)
    job = ended(waiting.job["id"])

    assert (waiting.status, waiting.code) == (503, "archive_building")
    assert job["status"] == "completed", job["error"]
    assert listed() == [("bridge", "failed"), ("bridge", "completed")]
    assert sorted(held(export).values()) == sorted(TEXTS)
    # the failure is forgotten: the archive is served
    assert ui_server.archive_for(export)[1].summary()["messages"] == 3


def test_an_export_that_changed_and_still_cannot_be_loaded_fails_once_more(tmp_path):
    export = write_what_is_no_export(tmp_path / "earlier" / "messages.csv")
    ended(not_ready(export).job["id"])
    export.write_text("still,not,an,export\n1,2,3,4\n", encoding="utf-8")

    second = ended(not_ready(export).job["id"])
    after = not_ready(export)

    assert second["status"] == "failed"
    assert (after.status, after.code) == (500, "archive_failed")
    assert listed() == [("bridge", "failed"), ("bridge", "failed")]


def test_a_load_that_stops_in_a_way_nobody_planned_for_is_not_started_again(export, monkeypatch, capfd):
    def stop(*_args, **_more):
        raise RuntimeError("a fault with SELECT text FROM messages in it")

    monkeypatch.setattr(ui_server.ArchiveState, "ensure_from_export", stop)

    job = ended(not_ready(export).job["id"])
    after = not_ready(export)

    assert job["status"] == "failed"
    assert not any(word in said(job) for word in NOT_PLAIN)
    assert (after.status, after.code) == (500, "archive_failed")
    assert not any(word in str(after) for word in NOT_PLAIN)
    assert listed() == [("bridge", "failed")]
    # what was raised is in the log of the server
    assert "SELECT text FROM messages" in capfd.readouterr().err


def test_a_load_that_fails_leaves_the_archive_that_an_earlier_load_made(export):
    run("bridge", {"messagesPath": str(export)})
    before = picture(export.with_suffix(".sqlite3"))
    write_what_is_no_export(export)

    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "failed"
    assert job["error"] == f"{NOT_AN_EXPORT} {KEPT}"
    assert unchanged(export.with_suffix(".sqlite3"), before)
    # the archive is there, so a request reads it
    assert ui_server.archive_for(export)[1].summary()["messages"] == 3


# ---- a place with no export ----


def test_a_load_asked_for_an_archive_file_has_no_export_to_load(tmp_path):
    job = run("bridge", {"messagesPath": str(tmp_path / "saves" / "archive.sqlite3")})

    assert (job["status"], job["error"]) == ("failed", "This archive has no earlier export to load.")
    assert not (tmp_path / "saves").exists()


def test_a_load_asked_for_an_export_that_is_not_there_says_so(tmp_path):
    missing = tmp_path / "earlier" / "messages.csv"

    job = run("bridge", {"messagesPath": str(missing)})

    assert (job["status"], job["error"]) == ("failed", f"There is no export at {missing.resolve()}.")
    assert not any(word in said(job) for word in NOT_PLAIN)
    assert not missing.parent.exists()


def test_a_request_with_no_archive_and_no_export_is_told_that_there_is_no_archive(tmp_path):
    for nothing in (tmp_path / "earlier" / "messages.csv", tmp_path / "saves" / "archive.sqlite3"):
        waiting = not_ready(nothing)

        assert (waiting.status, waiting.code) == (400, "no_archive")
        assert waiting.body() == {"error": "There is no archive yet. Open the setup guide to read your messages.", "code": "no_archive"}
    assert listed() == []


def test_an_export_with_no_message_starts_no_load(tmp_path):
    export = write_export(tmp_path / "earlier" / "messages.csv", [])

    waiting = not_ready(export)

    assert (waiting.status, waiting.code) == (400, "no_archive")
    assert listed() == []


def test_a_file_that_is_no_archive_is_told_as_one_that_failed_and_starts_no_load(tmp_path):
    export = write_export(tmp_path / "earlier" / "messages.csv", TEXTS)
    wrong = not_an_archive(export.with_suffix(".sqlite3"))
    kept = wrong.read_bytes()

    waiting = not_ready(export)

    assert (waiting.status, waiting.code) == (500, "archive_failed")
    assert waiting.body() == {"error": "The archive file could not be opened. Open the setup guide and update the archive.", "code": "archive_failed"}
    assert listed() == []
    assert wrong.read_bytes() == kept


# ---- beside an import ----


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that start and do not end until they are let go. Each one stays as it was made: queued."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield let_go
    let_go.set()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


@pytest.mark.parametrize("running", ["setup_import", "export"])
def test_a_request_beside_an_import_is_told_to_wait_for_that_import_and_starts_no_load(export, held_jobs, running):
    importing = ui_server.create_job(running, {})

    waiting = not_ready(export)

    assert (waiting.status, waiting.code) == (503, "archive_building")
    assert waiting.job["id"] == importing["id"]
    assert listed() == [(running, "queued")]


@pytest.mark.parametrize("running", ["setup_import", "export"])
def test_with_no_export_a_request_beside_a_first_import_is_told_to_wait(tmp_path, held_jobs, running):
    importing = ui_server.create_job(running, {})

    waiting = not_ready(tmp_path / "saves" / "archive.sqlite3")

    assert (waiting.status, waiting.code, waiting.job["id"]) == (503, "archive_building", importing["id"])
    assert listed() == [(running, "queued")]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "product fault: a first pull writes into the archive file itself, so a request during it reads an archive "
        "with no message (archive_for gives it, /api/contacts answers 200 with an empty list) and is not told to "
        "wait with 503 archive_building; only the moment before the file is made is covered (why_no_archive)"
    ),
)
def test_a_request_during_a_first_import_is_told_to_wait_for_it(messages_home, tmp_path, monkeypatch):
    from archive import importer, source
    from tests.archive_kit import write_database

    target = tmp_path / "saves" / "archive.sqlite3"
    write_database(messages_home / "chat.db", ["one", "two", "three"])
    monkeypatch.setattr(importer, "BATCH_ROWS", 1)
    real = source.SourceReader.messages_after
    told = []

    def ask_after_each_batch(reader, rowid, size):
        for batch in real(reader, rowid, size):
            yield batch
            try:
                told.append(("read", ui_server.archive_for(target)[1].summary()["messages"]))
            except ui_server.ArchiveNotReady as waiting:
                told.append((waiting.status, waiting.code))

    monkeypatch.setattr(source.SourceReader, "messages_after", ask_after_each_batch)

    job = run("setup_import", {"messagesPath": str(target), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert told == [(503, "archive_building")] * 3


def test_the_route_refuses_an_import_while_the_load_runs(server, export, held_load):
    _, told = conversations(server, export)

    status, answer = post(server, "/api/jobs", {"action": "setup_import", "messagesPath": str(export), "includeContacts": False})

    assert (status, answer) == (409, {"error": "An import is already running. Wait until it ends."})
    assert [action for action, _status in listed()] == ["bridge"]
    held_load.set()
    assert ended(told["job"]["id"])["status"] == "completed"


# ---- an archive that a pull filled ----


def test_an_archive_that_a_pull_filled_is_never_replaced_by_an_export(export):
    archive_file = pulled(export.with_suffix(".sqlite3"), ["from messages"])
    before = picture(archive_file)

    job = run("bridge", {"messagesPath": str(export)})

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 1, "conversations": 1, "firstYear": 2024, "lastYear": 2024, "status": "kept"}
    assert unchanged(archive_file, before)
    assert held(export) == {"guid-from messages": "from messages"}


def test_an_export_that_changes_later_does_not_replace_what_a_pull_filled(export):
    archive_file = pulled(export.with_suffix(".sqlite3"), ["from messages"])
    write_export(export, [*TEXTS, "written later"])

    _state, archive = ui_server.archive_for(export)
    job = run("bridge", {"messagesPath": str(export)})

    assert archive.summary()["messages"] == 1
    assert job["result"]["status"] == "kept"
    assert held(archive_file) == {"guid-from messages": "from messages"}
    assert listed() == []


# ---- when the server starts ----


def test_at_the_start_an_export_that_waits_is_loaded_by_one_job(data_folder):  # noqa: F811
    write_export(data_folder / "messages.csv", TEXTS)

    ui_server.load_export_at_start()

    assert [action for action, _status in listed()] == ["bridge"]
    with ui_server.JOBS_LOCK:
        job_id = next(iter(ui_server.JOBS))
    assert ended(job_id)["status"] == "completed"
    assert (data_folder / "saves" / "archive.sqlite3").is_file()
    assert sorted(held(None).values()) == sorted(TEXTS)
    # the next start finds nothing that waits
    ui_server.load_export_at_start()
    assert listed() == [("bridge", "completed")]


def nothing_there(folder):
    return None


def an_export_with_only_its_header(folder):
    write_export(folder / "messages.csv", [])


def an_export_that_the_archive_holds(folder):
    bridged(folder / "messages.csv", TEXTS)


def an_archive_that_a_pull_filled(folder):
    pulled(folder / "saves" / "archive.sqlite3", ["from messages"])
    write_export(folder / "messages.csv", TEXTS)


def an_archive_file_that_cannot_be_opened(folder):
    not_an_archive(folder / "saves" / "archive.sqlite3")
    write_export(folder / "messages.csv", TEXTS)


NOTHING_WAITS = [
    nothing_there,
    an_export_with_only_its_header,
    an_export_that_the_archive_holds,
    an_archive_that_a_pull_filled,
    an_archive_file_that_cannot_be_opened,
]


@pytest.mark.parametrize("there", NOTHING_WAITS, ids=lambda there: there.__name__)
def test_at_the_start_no_job_is_made_when_no_export_waits(data_folder, there):  # noqa: F811
    there(data_folder)
    files = {path: path.read_bytes() for path in data_folder.rglob("*") if path.suffix in (".csv", ".sqlite3")}

    ui_server.load_export_at_start()

    assert listed() == []
    assert {path: path.read_bytes() for path in files} == files


def test_at_the_start_an_export_that_changed_since_its_load_is_loaded_again(data_folder):  # noqa: F811
    bridged(data_folder / "messages.csv", TEXTS)
    write_export(data_folder / "messages.csv", [*TEXTS, "four"])

    ui_server.load_export_at_start()

    with ui_server.JOBS_LOCK:
        job_id = next(iter(ui_server.JOBS))
    assert ended(job_id)["result"]["status"] == "built"
    assert sorted(held(None).values()) == sorted([*TEXTS, "four"])


def test_at_the_start_no_load_is_made_beside_an_import(data_folder, held_jobs):  # noqa: F811
    write_export(data_folder / "messages.csv", TEXTS)
    ui_server.create_job("setup_import", {})

    ui_server.load_export_at_start()

    assert listed() == [("setup_import", "queued")]


def test_a_start_with_an_archive_that_cannot_be_looked_at_goes_on(data_folder, monkeypatch):  # noqa: F811
    write_export(data_folder / "messages.csv", TEXTS)

    def fail(*_args, **_more):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(ui_server.ArchiveState, "export_waits", fail)

    ui_server.load_export_at_start()

    assert listed() == []


def test_the_default_place_is_served_from_its_export_by_the_route(server, data_folder):  # noqa: F811
    write_export(data_folder / "messages.csv", TEXTS)

    status, _, body = fetch(server, "/api/contacts")
    told = json.loads(body)
    ended(told["job"]["id"])
    after, _, listed_body = fetch(server, "/api/contacts")

    assert (status, told["code"]) == (503, "archive_building")
    assert after == 200
    assert json.loads(listed_body)["archive"]["messages"] == 3
