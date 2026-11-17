"""An invented export of an earlier build, and the archive that is loaded from it.

The server reads one archive. A test of what the server reads writes an
export, loads it into the archive that stands beside it (`loaded`), and saves
names in a Contacts cache file (`save_names`). Every name, number and text
here is invented.

A test file imports the fixtures by name:

    from tests.export_kit import archives, contacts_cache  # noqa: F401

- `archives`: a test reads its own archive. Nothing that the server held for
  another test is read, and the archives of the test are let go after it.
- `contacts_cache`: the Contacts cache of the server is a temp file that is
  not there yet, so nobody has a saved name until `save_names` is called.
- `data_folder`: the data folder of the app is a temp folder of this test,
  for a request that names no file. It holds no job when the test starts.
"""
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest

import local_store
import recall_paths
import ui_server
from recall_services import archive_state, places
from tests.loopback import fetch

# every column that parse_imessage.py writes: the bridge refuses an export without one of them
COLUMNS = [
    "message_id",
    "timestamp",
    "sender",
    "text",
    "chat_id",
    "chat_display_name",
    "is_from_me",
    "service",
    "attachment_types",
    "attachment_files",
]
AVERY = "+15550000001"
BLAKE = "+15550000002"
UNSAVED = "+15550000009"
GROUP = "chat77"
NAMES = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}
PLACEHOLDER = chr(0xFFFC)
# an offered model: recall_services/models.json
MODEL = "gpt-6-luna"
JOB_WAIT_SECONDS = 20


def row(chat_id, timestamp, text, from_me=0, sender=None, group_name="", attachments=""):
    """One row of an export. `attachments` is the `|` list of types, as the exporter wrote it."""
    files = "|".join(f"~/Library/Messages/Attachments/invented-{place}.bin" for place, _ in enumerate(attachments.split("|"))) if attachments else ""
    writer = sender if sender is not None else chat_id
    return (0, timestamp, writer, text, chat_id, group_name, from_me, "iMessage", attachments, files)


def thread(chat_id, last, count, text="ok"):
    """`count` messages, one a minute, the last one at `last`."""
    return [row(chat_id, str(stamp), text) for stamp in pd.date_range(end=last, periods=count, freq="min")]


def write_export(path, rows):
    """Write the rows as an export. The rows are numbered from 1 in their order."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.assign(message_id=range(1, len(frame) + 1)).to_csv(path, index=False)
    return Path(path)


def load(path):
    """Load an export into the archive that stands beside it, as the `bridge` job does.

    The job counts what the archive holds when the load ends (`load_export`).
    That first read is done here too: it makes the file in which SQLite keeps
    the newest writes, and that file is part of `file_mark`.
    """
    state = places.state_for(path)
    answer = state.ensure_from_export(path)
    assert answer.status == "built", answer
    state.archive().summary()
    return Path(path)


def loaded(path, rows):
    """An export with these rows, and its archive."""
    return load(write_export(path, rows))


def pulled(archive_file, db_path, names=None):
    """Pull an invented Messages database into the archive at `archive_file`, as an update does.

    `archive_file` ends in `.sqlite3`: the server takes such a file for the archive itself.
    """
    state = places.state_for(archive_file)
    outcome = state.pull_from_messages(db_path, _records(names or {}))
    assert outcome.status == "done", outcome
    return Path(archive_file)


def get(server, path, **query):
    """Ask the test server (the `server` fixture) for a route. Gives the status and the answer, read as JSON."""
    status, _headers, body = fetch(server, f"{path}?{urlencode(query)}")
    return status, json.loads(body)


def _records(names):
    people = {}
    for handle, name in names.items():
        kind = "emails" if "@" in handle else "phones"
        held = people.get(name, {"name": name, "phones": [], "emails": []})
        people = {**people, name: {**held, kind: [*held[kind], handle]}}
    return list(people.values())


def save_names(names, cache_path=None):
    """Write the Contacts cache of the server: these names and no other. `names` is {handle: name}."""
    path = Path(cache_path or ui_server.CONTACTS_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"contacts": _records(names)}), encoding="utf-8")
    return path


def _forget_what_the_server_holds():
    blank = {"sig": None, "df": None, "catalog": None, "names": None}
    with ui_server._ASK_CACHE_LOCK:
        ui_server._ASK_CACHE.update(blank)
    with ui_server._PATTERNS_LOCK:
        ui_server._PATTERNS_CACHE.update({"sig": None, "value": None})
    with ui_server.PREVIEW_CACHE_LOCK:
        ui_server.PREVIEW_CACHE.clear()
    with ui_server.ANALYSIS_CACHE_LOCK:
        ui_server.ANALYSIS_CACHE.clear()
    ui_server.BRIDGE_FAILED.clear()


@pytest.fixture
def archives(tmp_path, monkeypatch):
    """A test reads its own archive, and what it stored is in its own temp folder."""
    monkeypatch.setattr(local_store, "DEFAULT_STORE_PATH", tmp_path / "store" / "recall_store.sqlite3")
    _forget_what_the_server_holds()
    yield
    _forget_what_the_server_holds()
    places.forget()


@pytest.fixture
def contacts_cache(tmp_path, monkeypatch):
    """The Contacts cache is a temp file that is not there: nobody has a saved name until `save_names`."""
    path = tmp_path / "contacts" / "contact_names.json"
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", path)
    return path


def jobs_ended(wait=JOB_WAIT_SECONDS):
    """Wait until no job runs. False when one still runs after `wait` seconds."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        with ui_server.JOBS_LOCK:
            running = [job for job in ui_server.JOBS.values() if job["status"] in ui_server.UNFINISHED]
        if not running:
            return True
        time.sleep(0.01)
    return False


def _no_jobs():
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


@pytest.fixture
def data_folder(tmp_path, monkeypatch):
    """The data folder of the app is a temp folder of this test. Gives the folder.

    A request that names no file reads the archive at `saves/archive.sqlite3`
    in it, and the export `messages.csv` in it. The reports folder, the
    Contacts cache, the setup record and the stored previews are in it too.
    No job is listed at the start, and no job of the test runs after it.
    """
    folder = (tmp_path / "data").resolve()
    saves, out = folder / "saves", folder / "out"
    saves.mkdir(parents=True)
    out.mkdir()
    # the archive place and the report places are read from recall_paths at each call
    for module in (recall_paths, ui_server):
        monkeypatch.setattr(module, "DATA_DIR", folder)
        monkeypatch.setattr(module, "SAVES_DIR", saves)
        monkeypatch.setattr(module, "OUT_DIR", out)
        monkeypatch.setattr(module, "DEFAULT_MESSAGES_CSV", folder / "messages.csv")
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", saves / "contact_names.json")
    monkeypatch.setattr(ui_server, "SETUP_MARKER_PATH", saves / "setup.json")
    monkeypatch.setattr(local_store, "DEFAULT_STORE_PATH", saves / "recall_store.sqlite3")
    _no_jobs()
    _forget_what_the_server_holds()
    yield folder
    # a job that still runs would write into the folder of the next test: the paths go back after this
    ended = jobs_ended()
    _no_jobs()
    _forget_what_the_server_holds()
    archive_state.shared().close()
    places.forget()
    assert ended, "a job of this test did not end"
