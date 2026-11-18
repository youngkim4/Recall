"""The index of meanings behind the server, after the review of the whole change.

Each archive has an index of its own. A build sends no handle that is not
saved. The server brings an index up to date by itself only when little is
new, and the status says `behind` when the ids of the index are dead.
The embedder is a stand-in: no request leaves this machine.
"""
import threading

import pytest

import ui_server
from meaning_index import IndexStore, MeaningIndex, StoreError
from recall_services import meaning as meaning_service
from recall_services import memory, places
from tests.archive_kit import (
    PERSON,
    bridged,
    data_folder,  # noqa: F401
    ended,
    import_from,
    jobs_end,  # noqa: F401
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    pulled,
    run,
    said,
    write_database,
    write_export,
)
from tests.test_meaning_routes import (
    FIRST,
    INDEX_IS_BUILT,
    MARK,
    MORE,
    archive_file,  # noqa: F401
    build,
    embedder,  # noqa: F401
    meaning_jobs,
    meaning_of,
)

pytestmark = pytest.mark.usefixtures(
    "fresh_bundle_cache", "no_names", "own_archives", "no_jobs", "jobs_end", "data_folder", "messages_home"
)

ORCHARD = [f"apple orchard walk {number}" for number in range(20)]
HARBOR = [f"winter harbor ferry {number}" for number in range(40)]
PIER = [f"the lantern on the pier {number}" for number in range(30)]


def held_ids(archive_file):  # noqa: F811
    """The first message id of each segment of the index of this archive."""
    index = MeaningIndex.load(meaning_service.index_store(places.archive_path_for(archive_file)))
    assert index is not None
    return [hit.segment.first_message_id for hit in index.search([1.0] * index.dimensions, limit=10_000)]


# --- one index for one archive -----------------------------------------------------


def test_a_build_for_a_second_archive_leaves_the_index_of_the_first(server, data_folder, embedder, tmp_path):  # noqa: F811
    first = str(pulled(data_folder / "saves" / "archive.sqlite3", ORCHARD))
    second = str(pulled(tmp_path / "second" / "other.sqlite3", HARBOR))
    built_first, built_second = build(first), build(second)
    assert (built_first["result"]["segments"], built_second["result"]["segments"]) == (2, 3)
    assert built_second["result"]["removed"] == 0
    _status, of_first = meaning_of(server, first)
    _status, of_second = meaning_of(server, second)
    assert (of_first["status"]["state"], of_first["status"]["segments"], of_first["estimate"]["newSegments"]) == ("ready", 2, 0)
    assert (of_second["status"]["state"], of_second["status"]["segments"]) == ("ready", 3)
    archive = places.state_for(first).archive()
    assert all(archive.message(message_id) is not None for message_id in held_ids(first))
    # a question on the first archive gets hits of the first archive only
    memory.forget()
    port = memory.ports(archive, ui_server.saved_names(), embedder_factory=lambda _model, _dimensions: embedder)["meaning"]
    hits = port.search("apple orchard", conversations=None, since=None, until=None, limit=5)
    memory.forget()
    assert hits and all(archive.message(hit["first_message_id"]) is not None for hit in hits)


def test_an_archive_with_no_index_of_its_own_says_none_beside_one_that_has(server, data_folder, embedder, tmp_path):  # noqa: F811
    first = str(pulled(data_folder / "saves" / "archive.sqlite3", ORCHARD))
    second = str(pulled(tmp_path / "second" / "other.sqlite3", HARBOR))
    build(first)
    _status, answer = meaning_of(server, second)
    assert (answer["status"]["state"], answer["estimate"]["newSegments"]) == ("none", 3)


# --- what a build sends ------------------------------------------------------------


def test_the_build_of_the_server_masks_a_handle_that_is_not_saved(data_folder, embedder):  # noqa: F811
    typed = f"call me at {PERSON} tonight"
    archive = str(pulled(data_folder / "saves" / "archive.sqlite3", ["first line", typed]))
    assert build(archive)["status"] == "completed"
    sent = "\n".join(embedder.texts)
    assert "call me at" in sent and PERSON not in sent and "5550190001" not in sent


# --- ids that changed, and a refresh that would cost -------------------------------


def test_after_the_first_pull_over_a_loaded_export_the_status_says_behind_until_the_refresh(
    server, data_folder, embedder, tmp_path, monkeypatch  # noqa: F811
):
    export = str(bridged(data_folder / "messages.csv", PIER))
    assert build(export)["result"]["segments"] == 2
    before = held_ids(export)
    real = ui_server.refresh_meaning_index
    monkeypatch.setattr(ui_server, "refresh_meaning_index", lambda *_args, **_more: None)  # look before the refresh
    job = import_from(write_database(tmp_path / "live" / "chat.db", PIER), export)
    assert job["status"] == "completed", job["error"]
    archive = places.state_for(export).archive()
    assert not [message_id for message_id in before if archive.message(message_id) is not None]
    _status, answer = meaning_of(server, export)
    assert answer["status"]["state"] == "behind"
    assert (answer["estimate"]["staleSegments"], answer["estimate"]["newSegments"]) == (2, 0)
    assert answer["estimate"]["estimatedCost"] == 0.0
    # at the next update the refresh of the server runs: it costs nothing here
    monkeypatch.setattr(ui_server, "refresh_meaning_index", real)
    import_from(write_database(tmp_path / "live" / "chat.db", PIER), export)
    done = refreshed()
    assert (done["result"]["embedded"], done["result"]["reused"]) == (0, 2)
    _status, answer = meaning_of(server, export)
    assert (answer["status"]["state"], answer["estimate"]["staleSegments"]) == ("ready", 0)
    assert all(places.state_for(export).archive().message(message_id) is not None for message_id in held_ids(export))


def test_the_server_does_not_start_a_refresh_that_would_embed_much(
    server, archive_file, embedder, messages_home, monkeypatch, capfd  # noqa: F811
):
    """Above the limit nothing is started: the status says behind, and the reader sees the cost first."""
    build(archive_file)
    monkeypatch.setattr(meaning_service, "REFRESH_SEGMENTS", 0)  # 1 new segment of 2 is more than 10 percent
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run("export", {"messagesPath": archive_file})
    assert job["status"] == "completed", job["error"]
    assert meaning_jobs() == [] and len(embedder.texts) == 2
    _status, answer = meaning_of(server, archive_file)
    assert (answer["status"]["state"], answer["estimate"]["newSegments"]) == ("behind", 1)
    assert answer["estimate"]["estimatedCost"] > 0
    assert "1 of 2 segments" in capfd.readouterr().err
    # the reader starts the build: it always builds
    done = build(archive_file)
    assert (done["status"], done["result"]["embedded"]) == ("completed", 1)
    assert meaning_of(server, archive_file)[1]["status"]["state"] == "ready"


def test_the_server_starts_a_refresh_at_the_limit(archive_file, embedder, messages_home, monkeypatch):  # noqa: F811
    build(archive_file)
    monkeypatch.setattr(meaning_service, "REFRESH_SEGMENTS", 1)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    assert run("export", {"messagesPath": archive_file})["status"] == "completed"
    assert refreshed()["result"]["embedded"] == 1


def test_an_estimate_that_fails_starts_nothing_and_does_not_fail_the_import(
    archive_file, embedder, messages_home, monkeypatch  # noqa: F811
):
    build(archive_file)

    def broken(*_args, **_more):
        raise meaning_service.MeaningFailure("archive_error", "The message archive could not be read.")

    monkeypatch.setattr(meaning_service, "estimate", broken)
    write_database(messages_home / "chat.db", [*FIRST, *MORE])
    job = run("export", {"messagesPath": archive_file})
    assert (job["status"], job["error"]) == ("completed", None) and meaning_jobs() == []


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
