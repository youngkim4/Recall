"""Run a job to its end and look at the archive it left. The fixtures that go
with these helpers (`messages_home`, `archive`, `marker`) are in conftest.py."""
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
