"""An update reads Messages itself and nothing else.

The archive is filled from one database. For an update that database is the
one Messages keeps on this Mac: a path the client sends, a path picked in an
earlier setup and a copy in the data folder do not change that. A copy is read
only when the setup guide names it for a first setup, and for that import
only. One test says whether an import is an update: `holds_messages`.
"""
import builtins
import json
import os
from pathlib import Path
from urllib.parse import quote, unquote

import pytest

import local_store
import recall_paths
import ui_server
from archive import source
from archive.importer import work_dir
from meaning_index import EmbedderFacts, FakeEmbedder
from recall_services import archive_state, places
from recall_services import meaning as meaning_service
from setup_status import DbProbe
from tests.archive.builders import ChatDbBuilder
from tests.archive_kit import (
    KEPT,
    PERSON,
    bridged,
    data_folder,  # noqa: F401
    databases_opened,
    emptied,
    ended,
    held,
    jobs_end,  # noqa: F401
    listed,
    no_names,  # noqa: F401
    not_an_archive,
    own_archives,  # noqa: F401
    picture,
    pulled,
    put_in_messages,
    run,
    unchanged,
    write_database,
    write_export,
)
from tests.loopback import fetch, post
from tests.recall_services.state_data import OTHER, START

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives", "no_jobs")

FIRST = ["one", "two", "three", "four", "five"]
FROM_MESSAGES = [*FIRST, "from messages"]
COPY_REFUSED = f"An update reads Messages on this Mac. It does not read a copy. {KEPT}"
GROUP = "chat190000"


@pytest.fixture
def filled(tmp_path):
    """An archive that a pull filled with the messages of FIRST."""
    return pulled(tmp_path / "saves" / "archive.sqlite3", FIRST)


def texts(archive_file):
    return sorted(held(archive_file).values())


# ---- which database is read ----


def test_an_update_reads_messages_itself_whatever_path_the_client_sends(messages_home, filled, tmp_path):
    write_database(messages_home / "chat.db", FROM_MESSAGES)
    copy = write_database(tmp_path / "old.db", [*FIRST, "from an old copy"])

    job = run("export", {"dbPath": str(copy), "messagesPath": str(filled)})

    assert job["status"] == "completed", job["error"]
    assert texts(filled) == sorted(FROM_MESSAGES)


@pytest.mark.parametrize("there", ["nothing", "an archive"])
def test_the_setup_guide_reads_messages_itself_when_it_names_no_copy(messages_home, tmp_path, there):
    write_database(messages_home / "chat.db", FROM_MESSAGES)
    target = tmp_path / "saves" / "archive.sqlite3"
    if there == "an archive":
        pulled(target, FIRST)

    job = run("setup_import", {"messagesPath": str(target), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts(target) == sorted(FROM_MESSAGES)


def no_file(folder):
    return folder / "new.csv"


def an_export_with_nothing_in_it(folder):
    (folder / "new.csv").write_text("", encoding="utf-8")
    return folder / "new.csv"


def an_export_with_only_its_header(folder):
    return write_export(folder / "new.csv", [])


def no_archive_file(folder):
    return folder / "new" / "archive.sqlite3"


def an_archive_that_holds_no_message(folder):
    return emptied(folder / "new" / "archive.sqlite3")


NO_MESSAGE_YET = [
    no_file,
    an_export_with_nothing_in_it,
    an_export_with_only_its_header,
    no_archive_file,
    an_archive_that_holds_no_message,
]


@pytest.mark.parametrize("there", NO_MESSAGE_YET, ids=lambda there: there.__name__)
def test_a_first_setup_reads_the_copy_it_names(messages_home, tmp_path, there):
    # nothing is held yet at the place that the app names
    write_database(messages_home / "chat.db", ["from messages"])
    copy = write_database(tmp_path / "copy.db", ["from the copy"])
    target = there(tmp_path)

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(target), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts(target) == ["from the copy"]


def test_a_first_setup_from_a_copy_runs_when_the_only_row_of_the_archive_is_an_event_of_a_group(messages_home, tmp_path):
    from tests.recall_services.state_data import write_database_of_events

    target = pulled(tmp_path / "new" / "archive.sqlite3", ["once here"])
    events = write_database_of_events(tmp_path / "events.db")
    assert places.state_for(target).pull_from_messages(events, None, max_removed_share=1.0).status == "done"
    assert ui_server.holds_messages(target) is False
    copy = write_database(tmp_path / "copy.db", ["from the copy"])

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(target), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts(target) == ["from the copy"]


def test_the_copy_is_read_for_that_import_only(messages_home, tmp_path):
    write_database(messages_home / "chat.db", ["from the copy", "from messages"])
    copy = write_database(tmp_path / "copy.db", ["from the copy"])
    target = tmp_path / "saves" / "archive.sqlite3"
    asked = {"messagesPath": str(target), "includeContacts": False}
    run("setup_import", {**asked, "dbPath": str(copy)})

    update = run("setup_import", asked)

    # the next import names nothing, and reads Messages: no path of the copy was kept
    assert update["status"] == "completed", update["error"]
    assert texts(target) == ["from messages", "from the copy"]
    with pytest.raises(ui_server.JobRefused, match="It does not read a copy"):
        ui_server.create_job("setup_import", {**asked, "dbPath": str(copy)})


def test_an_archive_that_holds_messages_is_never_updated_from_a_copy(messages_home, filled, tmp_path, monkeypatch):
    live = write_database(messages_home / "chat.db", FROM_MESSAGES)
    copy = write_database(tmp_path / "copy.db", [*FIRST, "from an old copy"])
    before = picture(filled)
    opened = databases_opened(monkeypatch)

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(filled), "includeContacts": False})

    assert (job["status"], job["error"]) == ("failed", COPY_REFUSED)
    assert job["code"] is None
    assert unchanged(filled, before)
    # the archive was looked at; no database of messages was opened, by any reader
    assert not any(str(copy) in target or str(live) in target for target in opened)


def a_pulled_archive(folder):
    return pulled(folder / "saves" / "archive.sqlite3", FIRST)


def a_bridged_archive(folder):
    return bridged(folder / "messages.csv", FIRST)


def an_export_that_waits_to_be_loaded(folder):
    return write_export(folder / "messages.csv", FIRST)


def an_archive_file_that_cannot_be_opened(folder):
    return not_an_archive(folder / "saves" / "archive.sqlite3")


HOLDS_MESSAGES = [a_pulled_archive, a_bridged_archive, an_export_that_waits_to_be_loaded, an_archive_file_that_cannot_be_opened]


@pytest.mark.parametrize("there", HOLDS_MESSAGES, ids=lambda there: there.__name__)
def test_the_route_refuses_a_copy_for_a_place_that_holds_messages(server, messages_home, tmp_path, there):
    write_database(messages_home / "chat.db", FROM_MESSAGES)
    copy = write_database(tmp_path / "copy.db", ["from an old copy"])
    target = there(tmp_path)
    files = {path: path.read_bytes() for path in target.parent.iterdir() if path.suffix in (".csv", ".sqlite3")}

    status, answer = post(server, "/api/jobs", {"action": "setup_import", "dbPath": str(copy), "messagesPath": str(target)})

    assert (status, answer) == (409, {"error": COPY_REFUSED})
    # no job is made: not the import, and no load of the export either
    assert listed() == []
    assert {path: path.read_bytes() for path in files} == files
    assert not any(path.name.endswith(".sqlite3") for path in target.parent.iterdir() if path not in files)


@pytest.mark.parametrize("there", HOLDS_MESSAGES, ids=lambda there: there.__name__)
def test_a_place_that_holds_messages_is_updated_from_messages(messages_home, tmp_path, there):
    assert ui_server.holds_messages(there(tmp_path)) is True


@pytest.mark.parametrize("there", NO_MESSAGE_YET, ids=lambda there: there.__name__)
def test_a_place_that_holds_no_message_is_a_first_setup(tmp_path, there):
    assert ui_server.holds_messages(there(tmp_path)) is False


def test_a_file_that_cannot_be_read_is_not_taken_for_an_archive_with_no_message(tmp_path):
    # a folder stands where the export would be
    assert ui_server.holds_messages(tmp_path) is True


def test_the_default_place_holds_messages_when_its_export_waits(data_folder):  # noqa: F811
    assert ui_server.holds_messages(None) is False

    write_export(data_folder / "messages.csv", ["one"])

    assert ui_server.holds_messages(None) is True
    assert ui_server.holds_messages(data_folder / "messages.csv") is True
    assert ui_server.holds_messages(data_folder / "saves" / "archive.sqlite3") is True


def test_only_the_setup_guide_can_name_a_database(messages_home, tmp_path):
    live = messages_home / "chat.db"
    copy = tmp_path / "copy.db"
    named = {"dbPath": str(copy), "extractFirst": True, "messagesPath": str(tmp_path / "new" / "archive.sqlite3")}

    assert ui_server.database_for("export", named) == live
    assert ui_server.database_for("setup_import", {}) == live
    assert ui_server.database_for("setup_import", named) == copy.resolve()
    assert ui_server.database_for("analyze", named) is None
    assert ui_server.database_for("bridge", named) is None


@pytest.mark.parametrize("action", ["export", "setup_import"])
def test_a_job_reads_the_database_that_database_for_names(messages_home, filled, tmp_path, monkeypatch, action):
    # no other code chooses a database for a job
    write_database(messages_home / "chat.db", FROM_MESSAGES)
    chosen = write_database(tmp_path / "chosen.db", [*FIRST, "from the one that was chosen"])
    monkeypatch.setattr(ui_server, "database_for", lambda _action, _payload: chosen)

    job = run(action, {"messagesPath": str(filled), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts(filled) == sorted([*FIRST, "from the one that was chosen"])


def test_a_report_does_not_read_a_database_first(messages_home, filled, tmp_path, monkeypatch):
    live = write_database(messages_home / "chat.db", FROM_MESSAGES)
    copy = write_database(tmp_path / "old.db", [*FIRST, "from an old copy"])
    before = picture(filled)
    opened = databases_opened(monkeypatch)

    # the job stops for want of a conversation; before that it must not have read a database
    job = run("analyze", {"dbPath": str(copy), "extractFirst": True, "messagesPath": str(filled)})

    assert job["status"] == "failed"
    assert unchanged(filled, before)
    assert not any(str(copy) in target or str(live) in target for target in opened)


# ---- every database that an import opens ----


def file_of(target):
    """The file that sqlite3 was asked to open: the address without its scheme and its options."""
    text = str(target)
    if text.startswith("file:"):
        text = unquote(text.removeprefix("file://").removeprefix("file:").split("?", 1)[0])
    return Path(text).resolve()


class Reads:
    """Every database that is read from now on: by sqlite3, and by a plain read of the file."""

    def __init__(self, monkeypatch, databases):
        self._watched = {Path(path).resolve() for path in databases}
        self.by_sqlite = databases_opened(monkeypatch)
        self.by_file = []
        real_open = builtins.open
        real_copy = source.copy_source

        def note_and_open(file, *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and Path(file).resolve() in self._watched:
                self.by_file.append(Path(file).resolve())
            return real_open(file, *args, **kwargs)

        def note_and_copy(path, *args, **kwargs):
            self.by_file.append(Path(path).resolve())
            return real_copy(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", note_and_open)
        monkeypatch.setattr(source, "copy_source", note_and_copy)

    def kinds(self, live, archive_file):
        """What each database that sqlite3 opened is to this import. A file that is none of its own is named.

        A file in the folder of the index of meanings is "index": after an
        import the server looks whether an index is there, and brings it up to
        date. That folder holds ids and vectors, and no database of Messages.
        """
        work = work_dir(Path(archive_file)).resolve()
        index = meaning_service.index_store().folder.resolve()
        own = {Path(live).resolve(): "messages", Path(archive_file).resolve(): "archive", Path(local_store.DEFAULT_STORE_PATH).resolve(): "store"}
        folders = {work: "work", index: "index"}
        found = set()
        for target in self.by_sqlite:
            opened = file_of(target)
            found.add(folders.get(opened.parent) or own.get(opened, f"another database: {opened}"))
        return found


@pytest.fixture
def second_database(tmp_path, monkeypatch):
    """A copy where an earlier build kept one: it holds a message that Messages no longer has."""
    second = write_database(tmp_path / "data" / "chat.db", [*FIRST, "gone from messages"])
    monkeypatch.setattr(recall_paths, "SNAPSHOT_DB", second)
    return second


@pytest.mark.parametrize("action", ["export", "setup_import"])
def test_no_second_source_is_read_for_an_update(messages_home, filled, second_database, monkeypatch, action):
    live = write_database(messages_home / "chat.db", [*FIRST, "still in messages"])
    reads = Reads(monkeypatch, [live, second_database])

    job = run(action, {"messagesPath": str(filled), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts(filled) == sorted([*FIRST, "still in messages"])
    assert "recovered" not in job["result"]
    # sqlite3 opened Messages for the check, the work file of the importer, the archive and the store of the previews
    assert reads.kinds(live, filled) == {"messages", "work", "archive", "store"}
    # and the bytes of one database were read: those of Messages
    assert set(reads.by_file) == {live.resolve()}


def test_no_second_source_is_read_for_a_first_setup_from_a_copy(messages_home, second_database, tmp_path, monkeypatch):
    live = write_database(messages_home / "chat.db", ["in messages"])
    copy = write_database(tmp_path / "copy.db", ["from the copy"])
    target = tmp_path / "new" / "archive.sqlite3"
    reads = Reads(monkeypatch, [live, copy, second_database])

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(target), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts(target) == ["from the copy"]
    assert reads.kinds(copy, target) == {"messages", "work", "archive", "store"}
    assert set(reads.by_file) == {copy.resolve()}


def test_the_record_of_reads_names_a_second_database(messages_home, filled, second_database, monkeypatch):
    # what the two tests above lean on: a second database that is read does not pass unseen
    live = write_database(messages_home / "chat.db", [*FIRST, "still in messages"])
    reads = Reads(monkeypatch, [live, second_database])
    real = ui_server.pull_into_archive

    def also_reads_a_second(action, payload):
        import sqlite3

        sqlite3.connect(f"file:{quote(str(second_database))}?mode=ro", uri=True).close()
        with open(second_database, "rb") as copy:
            copy.read(16)
        return real(action, payload)

    monkeypatch.setattr(ui_server, "pull_into_archive", also_reads_a_second)

    job = run("export", {"messagesPath": str(filled)})

    assert job["status"] == "completed", job["error"]
    assert reads.kinds(live, filled) == {"messages", "work", "archive", "store", f"another database: {second_database.resolve()}"}
    assert set(reads.by_file) == {live.resolve(), second_database.resolve()}


def test_with_an_index_an_update_reads_the_index_too_and_still_no_second_source(
    messages_home, data_folder, second_database, jobs_end, tmp_path, monkeypatch  # noqa: F811
):
    """The server brings an index that is there up to date after the update. It reads no other database for it."""
    filled = pulled(tmp_path / "saves" / "archive.sqlite3", FIRST)
    facts = EmbedderFacts("fake-embedding-1", 64, usd_per_mtok=0.13, max_input_tokens=8192, tokenizer_family="words")
    fake = FakeEmbedder()
    monkeypatch.setattr(meaning_service, "default_facts", lambda: facts)
    monkeypatch.setattr(meaning_service, "embedder_for", lambda _model, _dimensions, **_more: fake)
    assert run("meaning", {"messagesPath": str(filled)})["status"] == "completed"
    live = write_database(messages_home / "chat.db", [*FIRST, "still in messages"])
    reads = Reads(monkeypatch, [live, second_database])

    job = run("export", {"messagesPath": str(filled)})

    assert job["status"] == "completed", job["error"]
    with ui_server.JOBS_LOCK:
        (started,) = [made["id"] for made in ui_server.JOBS.values() if made["action"] == "meaning"]
    assert ended(started)["status"] == "completed"
    assert texts(filled) == sorted([*FIRST, "still in messages"])
    assert reads.kinds(live, filled) == {"messages", "work", "archive", "store", "index"}
    assert set(reads.by_file) == {live.resolve()}


def test_the_record_of_reads_names_a_database_beside_the_index_folder(messages_home, filled, monkeypatch):
    # a file is "index" only inside the folder of the index: a database next to that folder is still named
    live = write_database(messages_home / "chat.db", [*FIRST, "still in messages"])
    beside = write_database(meaning_service.index_store().folder.parent / "not-the-index" / "segments.sqlite3", ["a copy"])
    reads = Reads(monkeypatch, [live])
    real = ui_server.pull_into_archive

    def also_reads_it(action, payload):
        import sqlite3

        sqlite3.connect(f"file:{quote(str(beside))}?mode=ro", uri=True).close()
        return real(action, payload)

    monkeypatch.setattr(ui_server, "pull_into_archive", also_reads_it)
    try:
        job = run("export", {"messagesPath": str(filled)})
    finally:
        beside.unlink()
        beside.parent.rmdir()

    assert job["status"] == "completed", job["error"]
    assert reads.kinds(live, filled) == {"messages", "work", "archive", "store", f"another database: {beside.resolve()}"}


# ---- when the database cannot be read ----


def test_an_update_that_macos_hides_the_database_from_says_so_and_keeps_the_archive(messages_home, filled, monkeypatch):
    # without Full Disk Access macOS can show the file as missing; the check tells the two apart
    hidden = lambda path=None, deep=False: DbProbe(status="fda_blocked", kind="live", path=str(path))  # noqa: E731
    monkeypatch.setattr(archive_state, "probe_database", hidden)
    before = picture(filled)

    job = run("export", {"messagesPath": str(filled)})

    assert job["status"] == "failed"
    assert "Full Disk Access" in job["error"]
    assert job["error"].endswith(KEPT)
    assert unchanged(filled, before)


def test_an_update_that_is_refused_the_database_says_so_and_keeps_the_archive(messages_home, filled):
    live = write_database(messages_home / "chat.db", FROM_MESSAGES)
    before = picture(filled)
    live.chmod(0)
    try:
        if os.access(live, os.R_OK):
            pytest.skip("this user reads any file")
        job = run("export", {"messagesPath": str(filled)})
    finally:
        live.chmod(0o600)

    assert job["status"] == "failed"
    assert "Full Disk Access" in job["error"]
    assert unchanged(filled, before)


def test_an_update_on_a_mac_with_no_messages_database_says_so(messages_home, filled):
    before = picture(filled)

    job = run("export", {"messagesPath": str(filled)})

    assert job["status"] == "failed"
    assert job["error"] == f"There is no Messages database on this Mac. {KEPT}"
    assert unchanged(filled, before)


def test_a_copy_that_is_not_there_is_named_in_what_went_wrong(messages_home, tmp_path):
    write_database(messages_home / "chat.db", ["from messages"])
    missing = tmp_path / "gone.db"
    target = tmp_path / "new" / "archive.sqlite3"

    job = run("setup_import", {"dbPath": str(missing), "messagesPath": str(target), "includeContacts": False})

    # a copy that was named and cannot be read stops the import: Messages is not read in its place
    assert job["status"] == "failed"
    assert job["error"] == f"There is no file at {missing.resolve()}."
    assert not target.exists()


# ---- what is remembered ----


def test_a_path_picked_in_an_earlier_setup_is_not_kept(marker):
    marker.parent.mkdir()
    marker.write_text(json.dumps({"completed": True, "pickedDbPath": "/somewhere/chat.db"}), encoding="utf-8")

    assert ui_server.read_setup_marker() == {"completed": True}
    assert ui_server.write_setup_marker({"firstWordsShown": True}) == {"completed": True, "firstWordsShown": True}
    assert "pickedDbPath" not in json.loads(marker.read_text(encoding="utf-8"))


def test_setup_records_how_far_it_got_and_no_path():
    said = {"completed": True, "skipped": False, "pickedDbPath": "/somewhere/chat.db", "other": True}

    assert ui_server.setup_marker_updates(said) == {"completed": True, "skipped": False}


def test_the_defaults_name_no_database(server, data_folder):  # noqa: F811
    status, _, body = fetch(server, "/api/defaults")

    defaults = json.loads(body)
    assert status == 200
    assert "dbPath" not in defaults
    assert "hasDb" not in defaults
    assert defaults["messagesPath"] == str(data_folder / "saves" / "archive.sqlite3")
    assert defaults["hasMessages"] is False


def test_no_path_of_a_database_is_written_beside_the_archive(messages_home, tmp_path):
    copy = write_database(tmp_path / "copy.db", ["from the copy"])
    target = tmp_path / "new" / "archive.sqlite3"

    run("setup_import", {"dbPath": str(copy), "messagesPath": str(target), "includeContacts": False})

    note = (target.parent / "archive.sqlite3.source.json").read_text(encoding="utf-8")
    assert json.loads(note) == {"source": "messages"}
    assert "copy.db" not in note


# ---- the list of conversations ----


def group_database(path, title):
    """A database with one group of two people, and the title that Messages has for it."""
    builder = ChatDbBuilder(path)
    people = [builder.handle(PERSON), builder.handle(OTHER)]
    group = builder.chat(GROUP, display_name=title, members=people)
    builder.message(group, when=START, text="who is driving", handle=people[0], guid="guid-group")
    builder.close()
    return path


def names_of_conversations(server, archive_file, db=None):
    query = f"messagesPath={quote(str(archive_file))}" + (f"&dbPath={quote(str(db))}" if db else "")
    status, _, body = fetch(server, f"/api/contacts?{query}")
    assert status == 200, body
    return {contact["chat_id"]: contact.get("display_name") for contact in json.loads(body)["contacts"]}


def test_the_list_of_conversations_takes_no_database_from_the_client(server, messages_home, tmp_path, monkeypatch):
    target = tmp_path / "saves" / "archive.sqlite3"
    put_in_messages(group_database(tmp_path / "chat.db", ""))
    run("setup_import", {"messagesPath": str(target), "includeContacts": False})
    copy = group_database(tmp_path / "old.db", "Named In A Copy")
    opened = databases_opened(monkeypatch)

    names = names_of_conversations(server, target, db=copy)

    assert names[GROUP] != "Named In A Copy"
    assert not any(str(copy) in address for address in opened)


def test_with_an_archive_the_list_does_not_read_messages(server, messages_home, tmp_path, monkeypatch):
    # the title of a group is the one that the archive holds; Messages is read by an update, and by nothing else
    target = tmp_path / "saves" / "archive.sqlite3"
    put_in_messages(group_database(tmp_path / "chat.db", "Project Crew"))
    run("setup_import", {"messagesPath": str(target), "includeContacts": False})
    live = put_in_messages(group_database(tmp_path / "later.db", "Renamed Since"))
    opened = databases_opened(monkeypatch)

    names = names_of_conversations(server, target)

    assert names[GROUP] == "Project Crew"
    assert not any(str(live) in address for address in opened)


# ---- what the guide is told ----


def setup_status(server, target):
    status, _, body = fetch(server, f"/api/setup/status?messagesPath={quote(str(target))}")
    assert status == 200, body
    return json.loads(body)


THERE = [
    (no_file, False, False),
    (an_export_with_nothing_in_it, False, False),
    (an_export_with_only_its_header, False, False),
    (no_archive_file, False, False),
    (an_export_that_waits_to_be_loaded, True, True),
    (a_bridged_archive, True, True),
    (a_pulled_archive, True, True),
    (an_archive_that_holds_no_message, True, False),
    (an_archive_file_that_cannot_be_opened, True, True),
]


@pytest.mark.parametrize("there, exists, holds", THERE, ids=lambda value: value.__name__ if callable(value) else None)
def test_the_state_of_setup_says_what_is_there_and_whether_it_holds_messages(server, messages_home, tmp_path, there, exists, holds):
    # the guide and the server use one test for "is this an update"
    target = there(tmp_path)

    told = setup_status(server, target)

    assert told["export"] == {"exists": exists, "path": str(target), "holdsMessages": holds}
    assert told["export"]["holdsMessages"] is ui_server.holds_messages(target)


def test_the_check_of_setup_loads_no_export_and_makes_no_archive(server, messages_home, tmp_path):
    export = write_export(tmp_path / "earlier" / "messages.csv", FIRST)

    told = setup_status(server, export)

    assert told["export"]["holdsMessages"] is True
    assert listed() == []
    assert [path.name for path in export.parent.iterdir()] == ["messages.csv"]


@pytest.mark.parametrize("there, state", [(no_archive_file, "needs_export"), (an_export_that_waits_to_be_loaded, "ready"), (a_pulled_archive, "ready")], ids=lambda value: value.__name__ if callable(value) else None)
def test_setup_is_ready_when_there_is_something_to_read(server, messages_home, tmp_path, there, state):
    write_database(messages_home / "chat.db", ["in messages"])

    assert setup_status(server, there(tmp_path))["state"] == state
