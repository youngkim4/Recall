"""Run a job to its end and look at the archive it left. The fixtures that go
with these helpers (`messages_home`, `archive`, `marker`) are in conftest.py."""
import shutil
from pathlib import Path

import pandas as pd

import ui_server


def run(action, payload):
    """Run one job to its end, the way its thread does, and give back how it ended."""
    job = ui_server.job_record(action, payload)
    with ui_server.JOBS_LOCK:
        ui_server.JOBS[job["id"]] = job
    try:
        ui_server.run_job(job["id"], action, payload)
        with ui_server.JOBS_LOCK:
            return dict(ui_server.JOBS[job["id"]])
    finally:
        with ui_server.JOBS_LOCK:
            ui_server.JOBS.pop(job["id"], None)


def texts_in(path):
    return set(pd.read_csv(path)["text"].dropna())


def messages_csv(archive):
    return next(path for path in archive if path.name == "messages.csv")


def unchanged(archive):
    """True when each file of the archive holds what it held before."""
    return all(path.exists() and path.read_text(encoding="utf-8") == text for path, text in archive.items())


def left_behind(archive):
    """Files in the folder of the archive that are not the archive."""
    folder = messages_csv(archive).parent
    return sorted(path.name for path in folder.iterdir() if path not in archive)


def said(job):
    """All that a job tells the reader: why it stopped, and each line it wrote."""
    lines = [str(entry.get("message", "")) for entry in job.get("logs", [])]
    return "\n".join([str(job.get("error") or ""), *lines])


def import_from(db, archive_file, contacts=False):
    """Import the database `db` into the archive at `archive_file`, the way the app can.

    An archive that holds messages is updated from Messages itself, so `db` is
    put where Messages keeps its own (the `messages_home` fixture). A `db` that
    is not there leaves no database of Messages. A first import reads `db` as
    a copy that the setup guide named.
    """
    named = {"dbPath": str(db)}
    if ui_server.holds_messages(Path(archive_file)):
        live = ui_server.LIVE_CHAT_DB
        live.unlink(missing_ok=True)
        if Path(db).exists():
            shutil.copy(db, live)
        named = {}
    return run("setup_import", {**named, "messagesPath": str(archive_file), "includeContacts": contacts})
