"""Run a job to its end, fill an archive from an invented database, and look at what the archive holds.

The archive is one SQLite file. A test names it the way the app does
(`messagesPath`): a file that ends in `.sqlite3` is the archive, and any other
file is an export of an earlier build with its archive beside it.

The fixtures `messages_home`, `no_jobs` and `fresh_bundle_cache` are in
conftest.py. The fixtures here are imported by name into a test file:

    from tests.archive_kit import no_names, own_archives  # noqa: F401
"""
import shutil
import sqlite3
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import pytest

import recall_paths
import ui_server
from archive import Archive
from recall_services import archive_state, places
from tests import guards
from tests.archive.builders import ChatDbBuilder
from tests.recall_services.state_data import PERSON, START, write_database, write_export

KEPT = "The archive was not changed."
NOTE_ENDING = ".source.json"
# the files that SQLite keeps beside a database that is open
SIDE_ENDINGS = ("-wal", "-shm")
# every table of the archive but the log of its imports and the index of its words
NOT_CONTENT = ("import_runs", "messages_fts", "sqlite_")

__all__ = ["KEPT", "PERSON", "write_database", "write_export"]


# ---- fixtures ----


@pytest.fixture
def no_names(tmp_path, monkeypatch):
    """No name is saved: the Contacts cache is a file that is not there."""
    cache = tmp_path / "no-names" / "contact_names.json"
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", cache)
    return cache


@pytest.fixture
def own_archives():
    """The archives of a test are its own: no handle and no failed load is kept for the next test."""
    ui_server.BRIDGE_FAILED.clear()
    yield
    places.forget()
    archive_state.shared().close()
    ui_server.BRIDGE_FAILED.clear()


@pytest.fixture
def data_folder(tmp_path, monkeypatch):
    """A data folder of the app with nothing in it: no export, no archive, no names."""
    folder = (tmp_path / "data").resolve()
    folder.mkdir()
    monkeypatch.setattr(recall_paths, "DATA_DIR", folder)
    for name, leaf in guards.DATA_FILES.items():
        monkeypatch.setattr(recall_paths, name, folder / leaf)
    monkeypatch.setattr(ui_server, "DEFAULT_MESSAGES_CSV", folder / "messages.csv")
    monkeypatch.setattr(ui_server, "SAVES_DIR", folder / "saves")
    monkeypatch.setattr(ui_server, "OUT_DIR", folder / "out")
    monkeypatch.setattr(ui_server, "SETUP_MARKER_PATH", folder / "saves" / "setup.json")
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", folder / "saves" / "contact_names.json")
    archive_state.shared().close()
    yield folder
    archive_state.shared().close()


# ---- jobs ----


def run(action, payload):
    """Run one job to its end, the way its thread does, and give back how it ended."""
    job = ui_server.job_record(action, payload)
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    try:
        ui_server.run_job(job["id"], action, payload)
        with ui_server.JOBS_LOCK:
            return ui_server.job_copy(ui_server.JOBS[job["id"]])
    finally:
        with ui_server.JOBS_LOCK:
            ui_server.JOBS.pop(job["id"], None)


def ended(job_id, wait=10.0):
    """Wait until a job that the server started has ended, and give it back as it ended."""
    until = time.monotonic() + wait
    while time.monotonic() < until:
        with ui_server.JOBS_LOCK:
            job = ui_server.JOBS.get(job_id)
            if job is not None and job["status"] not in ui_server.UNFINISHED:
                return ui_server.job_copy(job)
        time.sleep(0.01)
    raise AssertionError(f"the job {job_id} did not end in {wait} seconds")


def listed():
    """The jobs in the list, oldest first: what each does and how far it is."""
    with ui_server.JOBS_LOCK:
        return [(job["action"], job["status"]) for job in ui_server.JOBS.values()]


def said(job):
    """All that a job tells the reader: why it stopped, and each line it wrote."""
    lines = [str(entry.get("message", "")) for entry in job.get("logs", [])]
    return "\n".join([str(job.get("error") or ""), *lines])


# ---- invented databases ----


def write_rows(path, rows, chat=PERSON):
    """A Messages database with one received message for each `(guid, text)`. A second call makes a new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    builder = ChatDbBuilder(path)
    handle = builder.handle(chat)
    thread = builder.chat(chat, members=[handle])
    for place, (guid, text) in enumerate(rows):
        builder.message(thread, when=START + timedelta(minutes=place), text=text, handle=handle, guid=guid)
    builder.close()
    return path


def put_in_messages(db):
    """Make `db` the database that Messages keeps (the `messages_home` fixture). No `db`: Messages has none."""
    live = Path(ui_server.LIVE_CHAT_DB)
    for ending in ("", *SIDE_ENDINGS):
        live.with_name(live.name + ending).unlink(missing_ok=True)
    if Path(db).exists():
        shutil.copy(db, live)
    return live


def import_from(db, archive_file, contacts=False, **more):
    """Import the database `db` into the archive at `archive_file`, the way the app can.

    An archive that holds messages is updated from Messages itself, so `db` is
    put where Messages keeps its own (the `messages_home` fixture). A `db` that
    is not there leaves no database of Messages. A first import reads `db` as
    a copy that the setup guide named.
    """
    named = {"dbPath": str(db)}
    if ui_server.holds_messages(Path(archive_file)):
        put_in_messages(db)
        named = {}
    return run("setup_import", {**named, "messagesPath": str(archive_file), "includeContacts": contacts, **more})


def pulled(archive_file, texts, folder=None):
    """An archive that a pull filled with one message for each text. Gives the archive file."""
    archive_file = Path(archive_file)
    source = write_database((folder or archive_file.parent) / "filled-from" / "chat.db", texts)
    outcome = places.state_for(archive_file).pull_from_messages(source, None)
    assert outcome.status == "done", outcome.reason
    shutil.rmtree(source.parent)
    return places.archive_path_for(archive_file)


def bridged(export, texts):
    """An export of an earlier build with one message for each text, loaded into its archive. Gives the export."""
    export = write_export(Path(export), texts)
    answer = places.state_for(export).ensure_from_export(export)
    assert answer.status == "built", answer.reason
    return export


# ---- what an archive holds ----


def held(archive_file):
    """Each message of the archive, by its id: the text. Read through a handle of its own."""
    with Archive.open(places.archive_path_for(archive_file)) as archive:
        chats = [chat["id"] for chat in archive.conversations(include_empty=True)]
        filed = [message for chat in chats for message in archive.iter_messages(chat)]
        loose = list(archive.iter_messages(None, no_conversation=True))
        return {message["id"]: message["text"] for message in [*filed, *loose]}


def texts_in(archive_file):
    """The texts of the archive, in order of the alphabet."""
    return sorted(held(archive_file).values())


def picture(archive_file):
    """Every row of every table that holds content, read from the file itself. The log of imports is not content."""
    address = f"file:{quote(str(Path(archive_file).resolve()))}?mode=ro"
    conn = sqlite3.connect(address, uri=True)
    try:
        names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        tables = [name for name in names if not name.startswith(NOT_CONTENT)]
        return {name: conn.execute(f"SELECT * FROM {name} ORDER BY 1, 2").fetchall() for name in tables}
    finally:
        conn.close()


def unchanged(archive_file, before):
    """True when the archive holds each row that it held at `before` (a `picture`), and it still opens."""
    with Archive.open(archive_file) as archive:
        archive.summary()
    return picture(archive_file) == before


def beside(archive_file):
    """What is in the folder of the archive beyond the archive, its note and the side files of SQLite."""
    archive_file = Path(archive_file)
    own = {archive_file.name, archive_file.name + NOTE_ENDING, *(archive_file.name + ending for ending in SIDE_ENDINGS)}
    return sorted(path.name for path in archive_file.parent.iterdir() if path.name not in own)


# ---- which databases are opened ----


def databases_opened(monkeypatch):
    """Every database that sqlite3 is asked to open from now on, as the text it was asked with."""
    opened = []
    connect = sqlite3.connect

    def note_and_connect(target, *args, **kwargs):
        opened.append(str(target))
        return connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", note_and_connect)
    return opened
