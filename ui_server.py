#!/usr/bin/env python3
"""Local web UI server for Recall."""

import contextlib
import io
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LoopbackHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        # skip HTTPServer.server_bind's socket.getfqdn() reverse-DNS lookup:
        # on the frozen build it dead-ends in an mDNS query that hangs the
        # boot for 30s+, and a loopback-only server never needs its fqdn
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from openai import OpenAI

from ai_config import DEFAULT_MODEL, PLANNER_MODEL, UI_MODEL_CHOICES
from analysis import run_cli
from cli import estimate_cost
from contact_names import (
    CONTACTS_CACHE_PATH,
    contacts_cache_summary,
    decorate_contacts_frame,
    export_contact_names,
    resolve_contact_names,
)
from local_store import clear_cache_namespace, get_cache_entry, set_cache_entry, stable_cache_key
from conversation import (
    activity_series,
    compute_stats,
    contact_label,
    conversation_label,
    day_totals,
    direction_word,
    fill_months,
    filter_conversation,
    in_conversation,
    group_chat_mask,
    hour_by_weekday,
    is_group_chat,
    load_attachments,
    load_messages,
    load_reactions,
    month_points,
    parse_boolish,
    progression_series,
    read_times,
    received_mask,
    sanitize_filename,
    sent_mask,
    within_dates,
)
from day_view import DAY_LIMIT, day_window
from group_names import derive_group_names
from group_people import group_speakers
from message_text import ATTACHMENT_PLACEHOLDER, attachment_label, clip_text, display_text
from openai_client import _call_openai, _call_openai_stream
from parse_imessage import extract_attachments, extract_messages, extract_reactions, list_contacts
from retrieval_planner import build_catalog, plan_retrieval
from safe_html import story_html
import semantic_index as sem
from recall_paths import (
    DATA_DIR,
    SAVES_DIR,
    OUT_DIR,
    DEFAULT_MESSAGES_CSV,
    ensure_data_dirs,
)
from setup_status import LIVE_CHAT_DB, DbProbe, derive_state, open_read_only, probe_database

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
REACT_DIST_DIR = ROOT / "app" / "dist"
JOBS = {}
JOBS_LOCK = threading.Lock()
ANALYSIS_CACHE = OrderedDict()
ANALYSIS_CACHE_LOCK = threading.Lock()
ANALYSIS_CACHE_LIMIT = 32
ANALYSIS_CACHE_NAMESPACE = "analysis_payload"
ANALYSIS_STORE_LIMIT = 128
PREVIEW_CACHE = OrderedDict()
PREVIEW_CACHE_LOCK = threading.Lock()
PREVIEW_CACHE_LIMIT = 64
PREVIEW_CACHE_NAMESPACE = "preview_payload"
PREVIEW_STORE_LIMIT = 256


class JobWriter(io.TextIOBase):
    """Routes worker stdout into the job log. tqdm redraws its progress line
    with \r-terminated frames; instead of buffering them into one unreadable
    mega-line, emit a log line whenever the frame's stable part (the stage
    description) changes -- live progress without percentage spam."""

    _FRAME_TAIL = re.compile(r":?\s*\d+%\|.*$")

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._buffer = ""
        self._last_frame_key = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while True:
            newline = self._buffer.find("\n")
            carriage = self._buffer.find("\r")
            if newline == -1 and carriage == -1:
                break
            if newline != -1 and (carriage == -1 or newline < carriage):
                line, self._buffer = self._buffer[:newline], self._buffer[newline + 1:]
                if line.strip():
                    append_log(self.job_id, line)
            else:
                frame, self._buffer = self._buffer[:carriage], self._buffer[carriage + 1:]
                self._emit_frame(frame)
        return len(text)

    def _emit_frame(self, frame: str):
        key = self._FRAME_TAIL.sub("", frame).strip()
        if key and key != self._last_frame_key:
            self._last_frame_key = key
            append_log(self.job_id, key)

    def flush(self):
        if self._buffer.strip():
            self._emit_frame(self._buffer)
        self._buffer = ""


def json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def send_json(handler, status: int, payload: dict):
    # no CORS headers on purpose: the UI is same-origin (served by this server,
    # or behind the Vite /api proxy in dev), so no cross-origin page may read these
    data = json.dumps(payload, default=json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


def _header_hostname(value: str) -> str:
    try:
        return urlparse(f"//{value}").hostname or ""
    except ValueError:
        return ""


def is_local_api_request(handler) -> bool:
    """Block cross-site access to the archive: a non-local Host means DNS
    rebinding, and browsers attach Origin to cross-origin requests. curl and
    the same-origin app (incl. the Vite /api dev proxy) pass both checks."""
    if _header_hostname(str(handler.headers.get("Host") or "")) not in LOCAL_HOSTNAMES:
        return False
    origin = str(handler.headers.get("Origin") or "").strip()
    if not origin:
        return True
    try:
        parsed = urlparse(origin)
        return parsed.scheme == "http" and (parsed.hostname or "") in LOCAL_HOSTNAMES
    except ValueError:
        return False


def parse_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def safe_path(value: str, default: Path = None) -> Path:
    if not value:
        return default
    return Path(os.path.expanduser(value)).resolve()


def ensure_under_root(path: Path) -> bool:
    # user data may live under DATA_DIR (packaged app) or the repo (dev)
    return ensure_under(path, ROOT) or ensure_under(path, DATA_DIR)


def ensure_under(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def path_exists(path: Path) -> bool:
    return bool(path and path.exists())


def selected_static_root() -> Path:
    """Choose the frontend bundle to serve from the local API server."""
    mode = os.environ.get("RECALL_UI_FRONTEND", "auto").strip().lower()
    react_ready = (REACT_DIST_DIR / "index.html").exists()
    if mode in {"legacy", "static"}:
        return UI_DIR
    if mode in {"react", "vite"} and react_ready:
        return REACT_DIST_DIR
    if mode in {"react", "vite"}:
        return UI_DIR
    return REACT_DIST_DIR if react_ready else UI_DIR


def append_log(job_id: str, message: str):
    message = message.strip("\r")
    if not message:
        return
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["logs"].append({"time": datetime.now().isoformat(timespec="seconds"), "message": message})
        job["logs"] = job["logs"][-300:]


def update_job(job_id: str, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)
            JOBS[job_id]["updatedAt"] = datetime.now().isoformat(timespec="seconds")


_REPORT_HEADING = re.compile(r"^# Conversation Analysis - (.+)$", re.MULTILINE)


def report_contact(path: Path) -> str:
    """The conversation a saved report is about. A file name cannot hold an
    email address, so the id is read from the report's own heading; the file
    name stands in when there is none."""
    markdown_path = path.with_suffix(".md")
    try:
        with markdown_path.open(encoding="utf-8") as report:
            match = _REPORT_HEADING.search(report.readline())
    except OSError:
        match = None
    return match.group(1).strip() if match else path.stem.removeprefix("analysis_")


def list_report_files():
    reports = []
    out_dir = OUT_DIR
    if not out_dir.exists():
        return reports
    for path in sorted(out_dir.glob("analysis_*.*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.suffix.lower() not in {".html", ".md"}:
            continue
        contact = report_contact(path)
        reports.append({
            "name": path.name,
            "path": str(path),
            "kind": path.suffix.lower().lstrip("."),
            "contact": contact,
            "updatedAt": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "size": path.stat().st_size,
        })
    name_map = resolve_contact_names(report["contact"] for report in reports)
    for report in reports:
        report["displayName"] = name_map.get(report["contact"], "")
    return reports


def companion_report_paths(report_path: Path, contact: str) -> tuple[Path | None, Path | None]:
    stem = sanitize_filename(contact) if contact else report_path.stem.removeprefix("analysis_")
    out_dir = report_path.parent
    events_path = out_dir / f"events_timeline_{stem}.csv"
    html_path = out_dir / f"analysis_{stem}.html"
    return (
        events_path if events_path.exists() else None,
        html_path if html_path.exists() else None,
    )


ARCHIVE_KEPT = "The archive was not changed."
# how long an import waits before it tries a busy database once more
BUSY_WAIT_SECONDS = 2
# what sqlite3 says, in small letters, when a database is held, closed to this app, or not one of Messages
BUSY_SIGNS = ("locked", "busy")
CLOSED_SIGNS = ("unable to open", "permission", "not permitted", "authoriz")
NOT_MESSAGES_SIGNS = ("no such table", "no such column", "not a database", "file is encrypted")


class ImportStopped(Exception):
    """An import that stops for a reason told to the reader in one plain sentence."""


class DatabaseBusy(ImportStopped):
    """Another program holds the database. The import is tried once more."""


class ArchiveInParts(ImportStopped):
    """An import stopped part way, and an earlier file could not be put back."""


def archive_note(archive_there: bool) -> str:
    """What a failed import says of the archive: it is as it was, when there was one."""
    return f" {ARCHIVE_KEPT}" if archive_there else ""


def database_for(action: str, payload: dict) -> Path | None:
    """The database a job reads, if it reads one.

    An update reads the database Messages keeps on this Mac. A path the client
    sends does not change that, and neither does a copy in the data folder.
    Only the setup guide can name a copy, and only for the import it starts.
    No other job reads a database.
    """
    if action == "export":
        return LIVE_CHAT_DB
    if action == "setup_import":
        named = payload.get("dbPath")
        if named and not isinstance(named, str):
            raise ValueError("The path of the copy must be text.")
        return safe_path(named, LIVE_CHAT_DB)
    return None


def closed_to_recall(probe: DbProbe) -> str:
    """What to do about a database macOS will not let this app open."""
    if probe.kind == "live":
        return (
            "Recall cannot read Messages. Give it Full Disk Access in System Settings > "
            "Privacy & Security > Full Disk Access, then open Recall again."
        )
    return (
        f"Recall cannot read the copy at {probe.path}: macOS keeps that folder closed to it. "
        "Move the copy to your home folder and choose it again."
    )


def not_there(probe: DbProbe) -> str:
    """What to say where no database is."""
    if probe.kind == "live":
        return "There is no Messages database on this Mac."
    return f"There is no file at {probe.path}."


def no_messages(probe: DbProbe) -> str:
    """What to say of a database that holds no message to keep."""
    if probe.kind == "live":
        return "Messages holds no messages on this Mac."
    return f"The copy at {probe.path} holds no messages."


def not_messages(probe: DbProbe, said: str = "") -> str:
    """What to say of a file that is not a database of Messages. `said` is what a check found, in brackets."""
    what = "The file that Messages keeps on this Mac" if probe.kind == "live" else f"The copy at {probe.path}"
    return f"{what} is not a Messages database{said}."


def not_read(probe: DbProbe, said: str = "") -> str:
    """What to say of a database that could not be read, when nothing more is known."""
    what = "The Messages database on this Mac" if probe.kind == "live" else f"The copy at {probe.path}"
    return f"{what} could not be read{said}."


def held(probe: DbProbe) -> str:
    """What to say of a database that another program holds."""
    if probe.kind == "live":
        return "The Messages database on this Mac is busy. Quit Messages or wait a moment, then try again."
    return f"The copy at {probe.path} is busy. Close the app that has it open, then try again."


def why_unreadable(probe: DbProbe, archive_there: bool) -> ImportStopped | None:
    """What stops an import before it starts, or None when the database can be tried.

    The check tells a database macOS keeps closed from one that is not there:
    without Full Disk Access macOS can show the file as missing. The reason
    speaks of the archive only when there is one.
    """
    # a busy database is tried: Messages may only be in the middle of a write
    if probe.status in ("readable", "locked"):
        return None
    said = f" ({probe.detail})" if probe.detail else ""
    if probe.status == "fda_blocked":
        reason = closed_to_recall(probe)
    elif probe.status == "missing":
        reason = not_there(probe)
    elif probe.status == "empty":
        reason = no_messages(probe)
    elif probe.status == "invalid":
        reason = not_messages(probe, said)
    else:
        # the file could not be looked at: it is not said to be another kind of file
        reason = not_read(probe, said)
    return ImportStopped(f"{reason}{archive_note(archive_there)}")


def reading_stopped(exc: Exception, probe: DbProbe, archive_there: bool) -> ImportStopped:
    """The plain sentence for what a reader raised.

    What the reader raised is not put into the sentence: it can hold the text
    of a query. Only an error of sqlite3 is read for signs of its kind.
    """
    signs = str(exc).lower() if isinstance(exc, sqlite3.Error) else ""
    note = archive_note(archive_there)
    if any(sign in signs for sign in BUSY_SIGNS):
        return DatabaseBusy(f"{held(probe)}{note}")
    if any(sign in signs for sign in CLOSED_SIGNS):
        return ImportStopped(f"{closed_to_recall(probe)}{note}")
    if any(sign in signs for sign in NOT_MESSAGES_SIGNS):
        return ImportStopped(f"{not_messages(probe)}{note}")
    return ImportStopped(f"{not_read(probe)}{note}")


def writing_stopped(exc: Exception, messages_path: Path, archive_there: bool) -> ImportStopped:
    """The plain sentence for what stopped the writing of the archive. The files are as they were."""
    reason = f": {exc.strerror}" if isinstance(exc, OSError) and exc.strerror else ""
    return ImportStopped(
        f"The archive could not be written to {messages_path.parent}{reason}.{archive_note(archive_there)}"
    )


def archive_files(messages_path: Path) -> tuple[Path, Path, Path]:
    """The three files of an archive, in the order an import moves them in.

    The messages go in last. The archive is known by its messages file, so
    when that one is new the other two already are.
    """
    return (
        companion_data_path(messages_path, "_attachments.csv"),
        companion_data_path(messages_path, "_reactions.csv"),
        messages_path,
    )


def staging_path(target: Path) -> Path:
    """Where a new file is written before it moves to its place."""
    return target.with_name(f"{target.stem}.tmp-export{target.suffix}")


def earlier_path(target: Path) -> Path:
    """Where the file that was at a place is kept while the new files move in."""
    return target.with_name(f"{target.stem}.before-update{target.suffix}")


def discard(paths) -> None:
    """Remove files that an import made for its own use. One that cannot be removed is left."""
    for path in paths:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def keep_earlier(target: Path) -> bool:
    """Keep the file at `target` under its earlier name. False when no file is there.

    The earlier name is a second name for the same file where the volume can
    do that, and a copy where it cannot.
    """
    earlier = earlier_path(target)
    discard([earlier])
    try:
        os.link(target, earlier)
    except FileNotFoundError:
        return False
    except OSError:
        shutil.copy2(target, earlier)
    return True


def restored(target: Path, had_earlier: bool) -> bool:
    """Put one place back as it was before the import. False when that fails."""
    earlier = earlier_path(target)
    try:
        if had_earlier:
            os.replace(earlier, target)
            # two names of one file: the move leaves the earlier name where it was
            earlier.unlink(missing_ok=True)
        else:
            target.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def unfinished(targets: tuple[Path, ...]) -> bool:
    """True while an import has not finished its moves: the earlier messages file is still kept.

    That file is kept first and given up last, so it stands for the whole
    earlier generation.
    """
    return os.path.lexists(earlier_path(targets[-1]))


def in_parts(stuck: list[tuple[Path, bool]], targets: tuple[Path, ...]) -> ArchiveInParts:
    """What to say when a place could not be put back as it was. `stuck` holds each such place, and whether it had an earlier file."""
    folder = targets[-1].parent
    kept = [earlier_path(target).name for target, had_earlier in stuck if had_earlier]
    left = [target.name for target, had_earlier in stuck if not had_earlier]
    said = ["The import stopped part way, and the archive could not be put back as it was."]
    if kept:
        said.append(f"The earlier files are kept in {folder} as {', '.join(kept)}.")
    if kept and unfinished(targets):
        said.append("The next import puts them back.")
    if left:
        said.append(f"A new file could not be removed from {folder}: {', '.join(left)}.")
    return ArchiveInParts(" ".join(said))


def settle_earlier(targets: tuple[Path, ...]) -> None:
    """Deal with the earlier files that an import before this one left beside the archive.

    While the earlier messages file is there, that import did not finish: the
    process ended during the moves, or a file could not be put back. Each
    earlier file is put back now, the messages last, so the archive is again
    as it was before that import. With no earlier messages file, that import
    finished, and what is left of its earlier files is removed.
    """
    others, messages = targets[:-1], targets[-1]
    if not unfinished(targets):
        discard(earlier_path(target) for target in others)
        return
    stuck = [
        (target, True)
        for target in others
        if os.path.lexists(earlier_path(target)) and not restored(target, True)
    ]
    # The earlier messages file goes last, and only when the others are back:
    # while it is there, a later import still puts the others back.
    if not stuck and not restored(messages, True):
        stuck = [(messages, True)]
    if stuck:
        raise in_parts(stuck, targets)


def move_in(targets: tuple[Path, ...]) -> None:
    """Move each new file to its place, or leave every place as it was.

    1. The files that are there are kept beside their places, as one earlier
       generation, the messages first.
    2. Each new file takes its place in one step, the messages last. A place
       never stands empty, and a request served meanwhile reads a whole file.
    3. The earlier messages file is given up. That finishes the import: until
       then, the next import puts the earlier files back (settle_earlier).
       The other earlier files are then removed.

    When a step fails, the files moved so far are put back from the earlier
    generation, the last one first.
    """
    messages = targets[-1]
    had_earlier: dict[Path, bool] = {}
    moved: list[Path] = []
    stuck: list[tuple[Path, bool]] = []
    try:
        for target in reversed(targets):
            had_earlier[target] = keep_earlier(target)
        for target in targets:
            os.replace(staging_path(target), target)
            moved.append(target)
        earlier_path(messages).unlink(missing_ok=True)
    except BaseException as stopped:
        stuck = [
            (target, had_earlier[target])
            for target in reversed(moved)
            if not restored(target, had_earlier[target])
        ]
        if stuck:
            # the messages file is as it was: it is kept again, to say that the import did not finish
            with contextlib.suppress(OSError):
                if messages not in dict(stuck) and not unfinished(targets):
                    keep_earlier(messages)
            raise in_parts(stuck, targets) from stopped
        raise
    finally:
        # An earlier file that could not be put back is the only copy of what
        # it holds. It stays, and so does the earlier messages file.
        held = set(dict(stuck)) | ({messages} if stuck else set())
        discard(earlier_path(target) for target in targets if target not in held)


class ArchiveFrames(NamedTuple):
    """What an archive holds, in the order that its files move in (archive_files)."""

    attachments: pd.DataFrame
    reactions: pd.DataFrame
    messages: pd.DataFrame


def read_database(db_path: Path) -> ArchiveFrames:
    """All that an archive holds, read from one database before a file is written."""
    path = str(db_path)
    messages = extract_messages(path)
    return ArchiveFrames(
        attachments=extract_attachments(path),
        reactions=extract_reactions(path),
        messages=messages,
    )


def write_archive(frames: ArchiveFrames, messages_path: Path) -> None:
    """Write the files of the archive, all of them or none."""
    targets = archive_files(messages_path)
    try:
        messages_path.parent.mkdir(parents=True, exist_ok=True)
        for target, frame in zip(targets, frames):
            frame.to_csv(staging_path(target), index=False)
        move_in(targets)
    finally:
        discard(staging_path(target) for target in targets)


def export_database(db_path: Path, messages_path: Path) -> dict:
    """Write the archive from one database: messages, attachments, reactions.

    Nothing is merged in from a second source, so the archive holds what the
    database holds. All three are read before a file is written. The files are
    then written beside their places and moved in (move_in): a failure at any
    step leaves the three files as they were. What an import before this one
    could not finish is put back first (settle_earlier).

    All three files are written each time. A database with no attachments or
    no reactions gives a file with only its header, so no figure of an earlier
    archive stays beside new messages. A database with no message to keep is
    not imported: the archive would be empty.
    """
    # what an import before this one could not finish is put back first
    settle_earlier(archive_files(messages_path))
    archive_there = path_exists(messages_path)
    probe = probe_database(db_path)
    stopped = why_unreadable(probe, archive_there)
    if stopped:
        raise stopped
    try:
        frames = read_database(db_path)
    except Exception as exc:  # noqa: BLE001 - whatever a reader raises is told in a plain sentence
        raise reading_stopped(exc, probe, archive_there) from exc
    if frames.messages.empty:
        raise ImportStopped(f"{no_messages(probe)}{archive_note(archive_there)}")
    try:
        write_archive(frames, messages_path)
    except ImportStopped:
        raise
    except Exception as exc:  # noqa: BLE001 - the files are as they were, and the reader is told so
        raise writing_stopped(exc, messages_path, archive_there) from exc

    return {
        "messages": len(frames.messages),
        "attachments": len(frames.attachments),
        "reactions": len(frames.reactions),
        "messagesPath": str(messages_path),
    }


def export_when_free(job_id: str, db_path: Path, messages_path: Path) -> dict:
    """Export, and try once more when the database is busy: Messages may be in the middle of a write."""
    try:
        return export_database(db_path, messages_path)
    except DatabaseBusy:
        append_log(job_id, "The database is busy. Trying again in a moment")
        time.sleep(BUSY_WAIT_SECONDS)
        return export_database(db_path, messages_path)


IMPORT_ACTIONS = frozenset({"setup_import", "export"})
UNFINISHED = frozenset({"queued", "running"})
# Jobs of one group do not run together: each import writes the archive, and
# each build writes the index. The second one is refused when it is asked for.
ONE_AT_A_TIME = (
    (IMPORT_ACTIONS, "An import is already running. Wait until it ends."),
    (frozenset({"semantic"}), "A semantic index build is already running."),
)


class JobRefused(Exception):
    """A request for a job that makes no job. `status` is the answer to the request."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def check_import(action: str, payload: dict) -> None:
    """Refuse an import whose request cannot be one. A job of another kind is not looked at."""
    if action not in IMPORT_ACTIONS:
        return
    named = payload.get("messagesPath")
    if named and not isinstance(named, str):
        raise JobRefused("The path of the archive must be text.", 400)
    try:
        database_for(action, payload)
    except ValueError as exc:
        raise JobRefused(str(exc), 400) from exc


def refuse_a_second(action: str) -> None:
    """Refuse a job while another of its group is not ended. The caller holds JOBS_LOCK."""
    for group, sentence in ONE_AT_A_TIME:
        if action in group and any(
            job["action"] in group and job["status"] in UNFINISHED for job in JOBS.values()
        ):
            raise JobRefused(sentence)


def job_record(action: str, payload: dict) -> dict:
    """A job as it is listed: what it does, for which conversation if any, and how far along it is."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": uuid.uuid4().hex[:12],
        "action": action,
        # the conversation a report is written about; "" for work on the whole archive
        "contact": str(payload.get("contact") or ""),
        "status": "queued",
        "createdAt": now,
        "updatedAt": now,
        "logs": [],
        "result": None,
        "error": None,
    }


def create_job(action: str, payload: dict) -> dict:
    """Make a job and start it. Raises JobRefused when no job is made.

    Whether a second import can run is decided here, once, with the list of
    jobs held. A job is in the list as queued from this moment, so two
    imports asked for together cannot each find the other and both stop.
    """
    check_import(action, payload)
    job = job_record(action, payload)
    job_id = job["id"]
    with JOBS_LOCK:
        refuse_a_second(action)
        JOBS[job_id] = job
        # prune finished jobs so a long session doesn't grow memory unbounded
        if len(JOBS) > 60:
            for old_id in [
                j["id"]
                for j in sorted(JOBS.values(), key=lambda j: str(j.get("createdAt", "")))
                if j["status"] in {"completed", "failed"}
            ][: len(JOBS) - 60]:
                JOBS.pop(old_id, None)

    thread = threading.Thread(target=run_job, args=(job_id, action, payload), daemon=True)
    try:
        thread.start()
    except RuntimeError as exc:
        # a job that never starts would stay queued and hold up every later job of its group
        update_job(job_id, status="failed", error="The job could not be started.")
        raise JobRefused("The job could not be started. Try again.", 500) from exc
    return job


class _JobStreamRouter(io.TextIOBase):
    """Per-thread stdout/stderr routing for concurrent jobs. A global
    redirect_stdout(writer) would interleave two jobs' logs; this routes each
    job thread to its own JobWriter. Worker threads a job spawns (parallel
    chunk extraction) fall through to the only active job when there is
    exactly one, else to the real stream."""

    def __init__(self, fallback):
        self._fallback = fallback
        self._writers: dict[int, JobWriter] = {}
        self._lock = threading.Lock()

    def register(self, writer: JobWriter):
        with self._lock:
            self._writers[threading.get_ident()] = writer

    def unregister(self):
        with self._lock:
            self._writers.pop(threading.get_ident(), None)

    def _target(self):
        with self._lock:
            writer = self._writers.get(threading.get_ident())
            if writer is None and len(self._writers) == 1:
                writer = next(iter(self._writers.values()))
        return writer or self._fallback

    def write(self, text):
        return self._target().write(text)

    def flush(self):
        target = self._target()
        # at the end of the process the stream can be closed before this is
        if not getattr(target, "closed", False):
            target.flush()


_JOB_STDOUT = _JobStreamRouter(sys.stdout)
_JOB_STDERR = _JobStreamRouter(sys.stderr)


SETUP_MARKER_PATH = SAVES_DIR / "setup.json"
SETUP_MARKER_KEYS = ("completed", "skipped", "firstWordsShown")
# the record is read, changed and written by one request at a time
SETUP_MARKER_LOCK = threading.Lock()


def read_setup_marker() -> dict:
    """How far setup got. A path picked in an earlier setup is not kept: a
    pick is for the one import it was made for."""
    try:
        marker = json.loads(SETUP_MARKER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(marker, dict):
        return {}
    return {key: value for key, value in marker.items() if key != "pickedDbPath"}


def setup_marker_updates(payload) -> dict:
    """What the client may record about setup: how far it got, and no path.

    Only true and false are taken. Text such as "false" says nothing, and a
    body that is not a record records nothing.
    """
    if not isinstance(payload, dict):
        return {}
    return {key: payload[key] for key in SETUP_MARKER_KEYS if isinstance(payload.get(key), bool)}


def write_setup_marker(updates: dict) -> dict:
    """Persist onboarding state server-side: WKWebView localStorage can be
    cleared while saves/ survives. Atomic so a crash never corrupts it. With
    nothing to record, nothing is written."""
    with SETUP_MARKER_LOCK:
        marker = {**read_setup_marker(), **updates}
        if not updates:
            return marker
        SAVES_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETUP_MARKER_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(marker, indent=1), encoding="utf-8")
        os.replace(tmp, SETUP_MARKER_PATH)
        return marker


def setup_status_payload(db_path: Path | None, messages_path: Path, deep: bool = False) -> dict:
    try:
        probe = probe_database(db_path, deep=deep)
    except Exception as exc:
        # the wizard derives every screen from this -- it must never 500
        probe = DbProbe(status="error", kind="copy", path=str(db_path or ""), detail=str(exc))
    export_exists = path_exists(messages_path)
    return {
        "state": derive_state(probe, export_exists),
        "db": {
            "status": probe.status,
            "kind": probe.kind,
            "path": probe.path,
            "detail": probe.detail,
            "approxMessages": probe.approx_messages,
            "conversations": probe.conversations,
            "firstYear": probe.first_year,
            "lastYear": probe.last_year,
        },
        "export": {"exists": export_exists, "path": str(messages_path)},
        "setup": read_setup_marker(),
    }


def fail_unfinished(job_id: str) -> None:
    """Mark a job as failed when its work is over and the list still says queued or running.

    However a job ends, it must not stay unfinished: it would hold up every
    later job of its group.
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job and job["status"] in UNFINISHED:
            job.update(
                status="failed",
                error="The job stopped before it ended.",
                updatedAt=datetime.now().isoformat(timespec="seconds"),
            )


def run_job(job_id: str, action: str, payload: dict):
    """Do the work of a job in its thread. When this returns or raises, the job is ended in the list."""
    try:
        work_job(job_id, action, payload)
    finally:
        fail_unfinished(job_id)


def work_job(job_id: str, action: str, payload: dict):
    update_job(job_id, status="running")
    writer = JobWriter(job_id)
    _JOB_STDOUT.register(writer)
    _JOB_STDERR.register(writer)
    try:
        with contextlib.redirect_stdout(_JOB_STDOUT), contextlib.redirect_stderr(_JOB_STDERR):
            if action == "export":
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                append_log(job_id, "Reading Messages")
                result = export_when_free(job_id, database_for(action, payload), messages_path)
            elif action == "analyze":
                # a report is written from the archive as it is: it reads no database
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                if not path_exists(messages_path):
                    raise FileNotFoundError(f"Messages CSV not found: {messages_path}")
                contact = payload.get("contact")
                if not contact:
                    raise ValueError("Select a contact before starting analysis")
                since = parse_optional_date(payload.get("since"))
                until = parse_optional_date(payload.get("until"))
                out_dir = str(safe_path(payload.get("outDir"), OUT_DIR))
                model = payload.get("model") or DEFAULT_MODEL
                html = bool(payload.get("html", True))
                append_log(job_id, f"Analyzing {contact} with {model}")
                _, report_path, events_path, html_path = run_cli(
                    str(messages_path),
                    contact,
                    out_dir,
                    since=since,
                    until=until,
                    html=html,
                    model=model,
                )
                analysis, _ = cached_analysis_payload(
                    messages_path,
                    contact,
                    model,
                    since=payload.get("since"),
                    until=payload.get("until"),
                    report_path=report_path,
                    events_path=events_path,
                    html_path=html_path,
                )
                result = {
                    "reportPath": report_path,
                    "eventsPath": events_path,
                    "htmlPath": html_path,
                    "analysis": analysis,
                    "reports": list_report_files(),
                }
            elif action == "setup_import":
                db_path = database_for(action, payload)
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                include_contacts = bool(payload.get("includeContacts", True))

                update_job(job_id, progress={"step": 1, "of": 3, "label": "Reading your Messages database"})
                append_log(job_id, "Reading Messages" if db_path == LIVE_CHAT_DB else f"Reading the copy at {db_path}")
                export_when_free(job_id, db_path, messages_path)

                contacts_skipped = False
                if include_contacts:
                    update_job(job_id, progress={"step": 2, "of": 3, "label": "Matching names from Contacts"})
                    append_log(job_id, "Matching names from Contacts")
                    try:
                        export_contact_names()
                    except Exception as exc:
                        # names are a nicety, never a blocker -- numbers still work
                        contacts_skipped = True
                        append_log(job_id, f"Contacts skipped: {exc}")

                update_job(job_id, progress={"step": 3, "of": 3, "label": "Getting your archive ready"})
                append_log(job_id, "Warming the archive")
                df, catalog, _names = cached_messages_bundle(messages_path)
                years = read_times(df["timestamp"]).dt.year.dropna()
                result = {
                    "messages": int(len(df)),
                    "conversations": int(len(catalog)),
                    "firstYear": int(years.min()) if len(years) else None,
                    "lastYear": int(years.max()) if len(years) else None,
                    "contactsSkipped": contacts_skipped,
                }
            elif action == "semantic":
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                if not path_exists(messages_path):
                    raise FileNotFoundError(f"Messages CSV not found: {messages_path}")
                append_log(job_id, "Building semantic index")
                df, _, _ = cached_messages_bundle(messages_path)
                summary = sem.build_index(
                    df,
                    file_cache_signature(messages_path),
                    SEMANTIC_DIR,
                    progress=lambda message: append_log(job_id, message),
                )
                invalidate_semantic_cache()
                result = {"semantic": summary}
            else:
                raise ValueError(f"Unknown job action: {action}")
        writer.flush()
        update_job(job_id, status="completed", result=result)
    except ImportStopped as exc:
        update_job(job_id, status="failed", error=str(exc))
        # What led to the sentence is for the log of the server, not for the
        # page. A log that cannot be written does not change how the job ended.
        with contextlib.suppress(Exception):
            writer.flush()
            if exc.__cause__ is not None:
                traceback.print_exception(exc, file=sys.__stderr__)
    except Exception as exc:
        writer.flush()
        append_log(job_id, traceback.format_exc())
        update_job(job_id, status="failed", error=str(exc))
    finally:
        _JOB_STDOUT.unregister()
        _JOB_STDERR.unregister()


def parse_optional_date(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def compute_dynamics(conv: pd.DataFrame) -> dict:
    """Relationship dynamics for the Analyze preview: balance drift, who texts
    first, volume trend, and per-speaker shares for group chats."""
    out: dict = {}
    if conv.empty or "timestamp" not in conv.columns:
        return out
    frame = conv.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return out
    last_ts = frame["timestamp"].max()
    cutoff = last_ts - pd.Timedelta(days=90)

    def sent_share(rows: pd.DataFrame):
        if rows.empty or "is_from_me" not in rows.columns:
            return None
        direction = rows["is_from_me"].apply(parse_boolish).dropna()
        if not len(direction):
            return None
        return round(float((direction.astype(int) == 1).mean()), 3)

    out["balanceLifetime"] = sent_share(frame)
    out["balanceRecent"] = sent_share(frame[frame["timestamp"] >= cutoff])

    # who opens the day: the first message of each active day
    daily_first = frame.groupby(frame["timestamp"].dt.date, as_index=False).head(1)
    out["initiationLifetime"] = sent_share(daily_first)
    out["initiationRecent"] = sent_share(daily_first[daily_first["timestamp"] >= cutoff])

    monthly_counts = frame.groupby(frame["timestamp"].dt.to_period("M")).size()
    if len(monthly_counts) >= 2 and float(monthly_counts.mean()) > 0:
        lifetime_avg = float(monthly_counts.mean())
        recent_avg = float(monthly_counts.tail(3).mean())
        out["volumeTrendPct"] = int(round((recent_avg - lifetime_avg) / lifetime_avg * 100))

    if "sender" in frame.columns:
        # A row you sent can carry the handle it went to, so speakers are counted
        # from received rows. A row that does not say could be one of those too.
        received = frame[received_mask(frame["is_from_me"])] if "is_from_me" in frame.columns else frame
        senders = received["sender"].fillna("").astype(str).str.strip()
        senders = senders[senders.ne("")]
        if senders.nunique() > 1:
            saved = dict(resolve_contact_names(sorted(senders.unique())))
            # everyone is listed, so a place in the list is a true place; a name is
            # the saved one or "", and the interface labels the rest
            speaker_count, speakers = group_speakers(senders, saved)
            if speaker_count > 1:
                out["speakerCount"] = speaker_count
                out["speakers"] = speakers
    return out


RECENT_MESSAGE_COUNT = 18
RECENT_MESSAGE_LIMIT = 240
# Bump when the preview changes in shape or in what a figure counts, so that a
# stored one is not served as it was. Version 7: a row that does not say who
# sent it is no longer counted as received, and names follow Contacts.
PREVIEW_PAYLOAD_VERSION = 7


def sent_by_me(value) -> bool | None:
    """Who sent a message, the way every payload says it: True for you, False
    for someone else, None when the export does not say. One reading of
    is_from_me (parse_boolish) for everything the interface is sent."""
    from_me = parse_boolish(value)
    return None if pd.isna(from_me) else int(from_me) == 1


def recent_message_payload(row, in_group: bool, sender_names: dict[str, str]) -> dict:
    mine = sent_by_me(row.get("is_from_me"))
    # who wrote it matters in a group; with one person the page already says
    sender = _row_str(row, "sender") if in_group and mine is False else ""
    return {
        "timestamp": row["timestamp"],
        "text": clip_text(display_text(row.get("text")), RECENT_MESSAGE_LIMIT),
        "attachment": attachment_label(row.get("attachment_types")),
        "isFromMe": mine,
        "sender": sender,
        "senderName": str(sender_names.get(sender) or ""),
    }


def preview_payload(messages_path: Path, contact: str, model: str, since=None, until=None):
    # the warm bundle replaces what used to be TWO full CSV parses per preview
    # (one here, one inside estimate_cost) -- ~4s of dead 'Calculating...' time
    full_df, _, _ = cached_messages_bundle(messages_path)
    since_dt = parse_optional_date(since)
    until_dt = parse_optional_date(until)
    df = within_dates(full_df, since_dt, until_dt)
    conv = filter_conversation(df, contact)
    if conv.empty:
        raise ValueError(f"No messages found for contact '{contact}'")

    attachments_df = load_attachments(str(messages_path), since_dt, until_dt)
    reactions_df = load_reactions(str(messages_path), since_dt, until_dt)
    stats = compute_stats(conv, attachments_df, reactions_df)
    monthly = fill_months(progression_series(conv))
    # the dates are chosen from the whole conversation, so this does not narrow with them
    daily = day_totals(conv if df is full_df else filter_conversation(full_df, contact))
    estimate = estimate_cost(
        str(messages_path), contact, model=model,
        since=since_dt, until=until_dt, frame=df,
    )
    recent = conv.tail(RECENT_MESSAGE_COUNT)
    in_group = is_group_chat(str(contact))
    recent_senders = sorted({_row_str(row, "sender") for _, row in recent.iterrows()} - {""}) if in_group else []
    sender_names = dict(resolve_contact_names(recent_senders)) if recent_senders else {}

    return {
        "stats": {
            "chatId": stats.chat_id,
            "totalMessages": stats.total_messages,
            "sentCount": stats.sent_count,
            "receivedCount": stats.received_count,
            "unknownDirectionCount": stats.unknown_direction_count,
            "firstTimestamp": stats.first_timestamp,
            "lastTimestamp": stats.last_timestamp,
            "activeDays": stats.active_days,
            "avgMessagesPerDay": stats.avg_messages_per_day,
            "busiestDay": stats.busiest_day,
            "busiestDayCount": stats.busiest_day_count,
            "longestGapDays": stats.longest_gap_days,
            "attachments": stats.attachments.__dict__,
            "reactions": stats.reactions.__dict__,
        },
        "estimate": estimate,
        "dynamics": compute_dynamics(conv),
        "monthly": [
            {
                "month": row["month"].strftime("%Y-%m"),
                "total": int(row["total"]),
                "sent": int(row["sent"]),
                "received": int(row["received"]),
                "sentRatio": float(row["sent_ratio"]),
            }
            for _, row in monthly.iterrows()
        ],
        "activity": activity_series(conv),
        "daily": daily,
        "recentMessages": [recent_message_payload(row, in_group, sender_names) for _, row in recent.iterrows()],
    }


def companion_data_path(messages_path: Path, suffix: str) -> Path:
    return messages_path.with_name(messages_path.stem + suffix)


def preview_cache_key(messages_path: Path, contact: str, model: str, since=None, until=None) -> tuple:
    return (
        PREVIEW_PAYLOAD_VERSION,
        file_cache_signature(messages_path),
        file_cache_signature(companion_data_path(messages_path, "_attachments.csv")),
        file_cache_signature(companion_data_path(messages_path, "_reactions.csv")),
        # a preview holds names: the people in a group, who wrote a recent message
        file_cache_signature(CONTACTS_CACHE_PATH),
        str(contact or ""),
        str(model or ""),
        str(since or ""),
        str(until or ""),
    )


def preview_cache_metadata(messages_path: Path, contact: str, model: str, since=None, until=None) -> dict:
    return {
        "messagesPath": str(messages_path),
        "contact": str(contact or ""),
        "model": str(model or ""),
        "since": str(since or ""),
        "until": str(until or ""),
    }


def cached_preview_payload(messages_path: Path, contact: str, model: str, since=None, until=None) -> tuple[dict, bool]:
    key_hash = stable_cache_key(preview_cache_key(messages_path, contact, model, since=since, until=until))
    with PREVIEW_CACHE_LOCK:
        cached = PREVIEW_CACHE.get(key_hash)
        if cached is not None:
            PREVIEW_CACHE.move_to_end(key_hash)
            return cached, True

    stored = get_cache_entry(PREVIEW_CACHE_NAMESPACE, key_hash)
    if stored is not None:
        payload = stored["value"]
        with PREVIEW_CACHE_LOCK:
            PREVIEW_CACHE[key_hash] = payload
            PREVIEW_CACHE.move_to_end(key_hash)
            while len(PREVIEW_CACHE) > PREVIEW_CACHE_LIMIT:
                PREVIEW_CACHE.popitem(last=False)
        return payload, True

    payload = preview_payload(messages_path, contact, model, since=since, until=until)
    with PREVIEW_CACHE_LOCK:
        PREVIEW_CACHE[key_hash] = payload
        PREVIEW_CACHE.move_to_end(key_hash)
        while len(PREVIEW_CACHE) > PREVIEW_CACHE_LIMIT:
            PREVIEW_CACHE.popitem(last=False)
    set_cache_entry(
        PREVIEW_CACHE_NAMESPACE,
        key_hash,
        payload,
        metadata=preview_cache_metadata(messages_path, contact, model, since=since, until=until),
        limit=PREVIEW_STORE_LIMIT,
    )
    return payload, False


def markdown_section(text: str, heading: str, stop_at_next_heading: bool = True) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    start = text.find("\n", start)
    if start < 0:
        return ""
    next_heading = text.find("\n## ", start + 1) if stop_at_next_heading else -1
    section = text[start:next_heading if next_heading >= 0 else len(text)]
    return section.strip()


def markdown_to_html(text: str) -> str:
    # The story is written by a model from other people's words, and the page
    # puts it in as HTML. safe_html decides what that HTML may hold, for this
    # page and for the saved report alike.
    return story_html(text)


_REPORT_DAYS = re.compile(r"^- First / Last: (\d{4}-\d{2}-\d{2})[^/\n]* / (\d{4}-\d{2}-\d{2})", re.MULTILINE)
_REPORT_SCOPE = re.compile(r"^- Scope: (.*)$", re.MULTILINE)
_DAY = r"\d{4}-\d{2}-\d{2}"
_SCOPE_BETWEEN = re.compile(rf"({_DAY}) to ({_DAY})")
_SCOPE_FROM = re.compile(rf"from ({_DAY})")
_SCOPE_THROUGH = re.compile(rf"through ({_DAY})")


def _stated_scope(overview: str) -> tuple[str | None, str | None] | None:
    """The dates a report says it was asked for (scope_text in reports wrote
    the line), or None when it does not say in words that can be read."""
    line = _REPORT_SCOPE.search(overview)
    stated = line.group(1).strip() if line else ""
    if stated == "all dates":
        return (None, None)
    if between := _SCOPE_BETWEEN.fullmatch(stated):
        return (between.group(1), between.group(2))
    if since := _SCOPE_FROM.fullmatch(stated):
        return (since.group(1), None)
    if until := _SCOPE_THROUGH.fullmatch(stated):
        return (None, until.group(1))
    return None


def report_scope(text: str) -> tuple[str | None, str | None]:
    """The days a saved report covers: (since, until), either side open.

    A report states the dates it was asked for, and those are used: reactions
    and attachments that fell inside them after the last message are then
    counted, as they were when the report was written. A report written before
    it stated them is taken to cover the days of its first and last message. A
    report that says neither covers the whole conversation.

    Only the Overview is read. The story and the events are a model's words
    and may hold a line that looks like either."""
    overview = markdown_section(text or "", "Overview")
    stated = _stated_scope(overview)
    if stated is not None:
        return stated
    match = _REPORT_DAYS.search(overview)
    return (match.group(1), match.group(2)) if match else (None, None)


def report_file(label: str, path: str, kind: str) -> dict:
    if not path:
        return {}
    file_path = safe_path(path)
    if not file_path or not file_path.exists():
        return {}
    return {
        "label": label,
        "path": str(file_path),
        "name": file_path.name,
        "kind": kind,
        "size": file_path.stat().st_size,
    }


_REPORT_FILE_NAME = re.compile(r"(?:analysis_.+\.(?:md|html)|events_timeline_.+\.csv)", re.IGNORECASE)
_REPORT_FILE_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
}


def served_report_file(raw_path: str) -> Path | None:
    """The saved report file a link may open, or None. Links to report files
    are the only use of the route, so it serves nothing else in the data
    folder: not the messages, not the key file. A symbolic link is followed
    first, and the file it leads to is the one judged."""
    try:
        file_path = safe_path(raw_path)
        if not file_path or not file_path.is_file() or not ensure_under_root(file_path):
            return None
    except (OSError, ValueError):
        return None
    return file_path if _REPORT_FILE_NAME.fullmatch(file_path.name) else None


# A saved report page: an origin of its own, no script, nothing fetched. Its
# styles are in the page, so those are allowed and nothing else is.
_REPORT_PAGE_POLICY = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:"


def report_file_headers(file_path: Path) -> dict[str, str]:
    """A report is made from a model's words about other people's messages.
    The browser is told to take the file for the type it is given. An HTML
    report opens in an origin of its own, where no script runs and nothing is
    fetched: a report saved by an older build can hold a handler in its text,
    and it must not be able to run or to send the page anywhere. The other
    report files are text and run nothing."""
    suffix = file_path.suffix.lower()
    headers = {
        "Content-Type": _REPORT_FILE_TYPES.get(suffix, "text/plain; charset=utf-8"),
        "X-Content-Type-Options": "nosniff",
    }
    if suffix == ".html":
        return {**headers, "Content-Security-Policy": _REPORT_PAGE_POLICY}
    return headers


def file_cache_signature(path: Path | str | None) -> tuple | None:
    if not path:
        return None
    file_path = Path(path).resolve()
    try:
        stat = file_path.stat()
    except OSError:
        return (str(file_path), None, None)
    return (str(file_path), stat.st_mtime_ns, stat.st_size)


# Bump when an opened report changes in shape or in how its story is rendered,
# so that a stored one is not served as it was. A change to story_html in
# safe_html is such a change. Version 5: a story is cut at a shorter length,
# and one that is slow to render is shown as plain paragraphs.
ANALYSIS_PAYLOAD_VERSION = 5


def analysis_cache_key(
    messages_path: Path,
    contact: str,
    model: str,
    since=None,
    until=None,
    report_path: str = None,
    events_path: str = None,
    html_path: str = None,
) -> tuple:
    return (
        ANALYSIS_PAYLOAD_VERSION,
        # An opened report holds the preview of its conversation. Whatever
        # makes a stored preview stale makes a stored report stale: the
        # export, its attachments and reactions, Contacts, and the version.
        preview_cache_key(messages_path, contact, model, since=since, until=until),
        file_cache_signature(report_path),
        file_cache_signature(events_path),
        file_cache_signature(html_path),
    )


def analysis_cache_metadata(
    messages_path: Path,
    contact: str,
    model: str,
    since=None,
    until=None,
    report_path: str = None,
    events_path: str = None,
    html_path: str = None,
) -> dict:
    return {
        "messagesPath": str(messages_path),
        "contact": str(contact or ""),
        "model": str(model or ""),
        "since": str(since or ""),
        "until": str(until or ""),
        "reportPath": str(report_path or ""),
        "eventsPath": str(events_path or ""),
        "htmlPath": str(html_path or ""),
    }


def cached_analysis_payload(
    messages_path: Path,
    contact: str,
    model: str,
    since=None,
    until=None,
    report_path: str = None,
    events_path: str = None,
    html_path: str = None,
) -> tuple[dict, bool]:
    key = analysis_cache_key(
        messages_path,
        contact,
        model,
        since=since,
        until=until,
        report_path=report_path,
        events_path=events_path,
        html_path=html_path,
    )
    key_hash = stable_cache_key(key)
    with ANALYSIS_CACHE_LOCK:
        cached = ANALYSIS_CACHE.get(key_hash)
        if cached is not None:
            ANALYSIS_CACHE.move_to_end(key_hash)
            return cached, True

    stored = get_cache_entry(ANALYSIS_CACHE_NAMESPACE, key_hash)
    if stored is not None:
        payload = stored["value"]
        with ANALYSIS_CACHE_LOCK:
            ANALYSIS_CACHE[key_hash] = payload
            ANALYSIS_CACHE.move_to_end(key_hash)
            while len(ANALYSIS_CACHE) > ANALYSIS_CACHE_LIMIT:
                ANALYSIS_CACHE.popitem(last=False)
        return payload, True

    payload = build_analysis_payload(
        messages_path,
        contact,
        model,
        since=since,
        until=until,
        report_path=report_path,
        events_path=events_path,
        html_path=html_path,
    )
    with ANALYSIS_CACHE_LOCK:
        ANALYSIS_CACHE[key_hash] = payload
        ANALYSIS_CACHE.move_to_end(key_hash)
        while len(ANALYSIS_CACHE) > ANALYSIS_CACHE_LIMIT:
            ANALYSIS_CACHE.popitem(last=False)
    set_cache_entry(
        ANALYSIS_CACHE_NAMESPACE,
        key_hash,
        payload,
        metadata=analysis_cache_metadata(
            messages_path,
            contact,
            model,
            since=since,
            until=until,
            report_path=report_path,
            events_path=events_path,
            html_path=html_path,
        ),
        limit=ANALYSIS_STORE_LIMIT,
    )
    return payload, False


def build_analysis_payload(
    messages_path: Path,
    contact: str,
    model: str,
    since=None,
    until=None,
    report_path: str = None,
    events_path: str = None,
    html_path: str = None,
) -> dict:
    summary = ""
    if report_path:
        report_file_path = safe_path(report_path)
        if report_file_path and report_file_path.exists():
            report_text = report_file_path.read_text(encoding="utf-8")
            summary = markdown_section(report_text, "Summary", stop_at_next_heading=False)
            if not (since or until):
                # the figures beside a story are for the days the story was written about
                since, until = report_scope(report_text)

    try:
        preview, _ = cached_preview_payload(messages_path, contact, model, since=since, until=until)
    except ValueError:
        # the archive no longer holds those days; the whole conversation is the closest truth
        preview, _ = cached_preview_payload(messages_path, contact, model)
    stats = preview["stats"]

    events = []
    if events_path:
        events_file_path = safe_path(events_path)
        if events_file_path and events_file_path.exists():
            events_df = pd.read_csv(events_file_path).fillna("")
            for _, row in events_df.iterrows():
                score = row.get("score", "")
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = None
                events.append({
                    "date": str(row.get("date", "")),
                    "title": str(row.get("title", "")),
                    "detail": str(row.get("detail", "")),
                    "category": str(row.get("category", "")),
                    "score": score,
                    "quote": str(row.get("quote", "")),
                })

    files = [
        file
        for file in [
            report_file("Markdown report", report_path, "md"),
            report_file("Events timeline", events_path, "csv"),
            report_file("HTML report", html_path, "html"),
        ]
        if file
    ]

    attachments = stats.get("attachments") or {}
    reactions = stats.get("reactions") or {}
    # the names the conversation list uses: saved, given to a group, or made from its people
    _df, _catalog, names = cached_messages_bundle(messages_path)
    return {
        # the same chart and figures as the conversation view, from the same fields
        **preview,
        "contact": contact,
        "contactDisplayName": str(names.get(str(contact)) or ""),
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "summaryHtml": markdown_to_html(summary),
        "events": events,
        "patterns": [
            {
                "label": "Busiest day",
                "value": str(stats.get("busiestDay") or "--"),
                "detail": f"{stats.get('busiestDayCount', 0):,} messages",
            },
            {
                "label": "Longest gap",
                "value": f"{float(stats.get('longestGapDays') or 0):.1f} days",
                "detail": "Longest quiet period in the selected scope.",
            },
            {
                "label": "Average pace",
                "value": f"{float(stats.get('avgMessagesPerDay') or 0):.1f}/day",
                "detail": f"{stats.get('activeDays', 0):,} active days.",
            },
            {
                "label": "Direction split",
                "value": f"{stats.get('sentCount', 0):,} / {stats.get('receivedCount', 0):,}",
                "detail": "Sent versus received messages.",
            },
        ],
        "media": {
            "attachments": attachments,
            "reactions": reactions,
        },
        "files": files,
    }


def first_nonempty(values) -> str:
    """Return the first non-empty string from a pandas aggregation group."""
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def group_display_names_from_database(db_path: Path | None, chat_ids) -> dict[str, str]:
    """Read custom group-chat display names from chat.db when available.
    Best-effort: a blocked/locked database (no FDA grant yet) must never
    break the conversation list -- the CSV archive still has everything."""
    if not path_exists(db_path):
        return {}

    wanted = sorted({
        str(chat_id)
        for chat_id in chat_ids
        if is_group_chat(str(chat_id or ""))
    })
    if not wanted:
        return {}

    try:
        conn = open_read_only(db_path, timeout=2.0)
    except sqlite3.Error:
        return {}
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chat)").fetchall()}
        if "display_name" not in columns:
            return {}
        placeholders = ",".join("?" for _ in wanted)
        rows = conn.execute(
            f"""
            SELECT chat_identifier, NULLIF(TRIM(display_name), '') AS display_name
            FROM chat
            WHERE chat_identifier IN ({placeholders})
              AND NULLIF(TRIM(display_name), '') IS NOT NULL
            """,
            wanted,
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()

    return {str(chat_id): str(display_name) for chat_id, display_name in rows if display_name}


def fill_group_display_names(df: pd.DataFrame, db_path: Path | None = None) -> pd.DataFrame:
    """Fill missing display names for group chats from chat.db."""
    result = df.copy()
    if result.empty or "chat_id" not in result.columns:
        if "display_name" not in result.columns:
            result["display_name"] = ""
        return result

    if "display_name" not in result.columns:
        result["display_name"] = ""

    display_names = result["display_name"].fillna("").astype(str).str.strip()
    chat_ids = result["chat_id"].fillna("").astype(str)
    missing_group_names = display_names.eq("") & group_chat_mask(chat_ids)
    if not missing_group_names.any():
        result["display_name"] = display_names
        return result

    names = group_display_names_from_database(db_path, chat_ids[missing_group_names])
    if not names:
        result["display_name"] = display_names
        return result

    resolved = chat_ids.map(names).fillna("")
    result["display_name"] = display_names.where(display_names.str.len().gt(0), resolved)
    return result


def name_groups_after_their_people(df: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """Give group chats that still have no display name the one derived from
    the people in them (see group_names)."""
    if df.empty:
        return df
    chat_ids = df["chat_id"].astype(str)
    current = df["display_name"].fillna("").astype(str).str.strip()
    unnamed_group = current.eq("") & group_chat_mask(chat_ids)
    derived = chat_ids.map(lambda chat_id: str(names.get(chat_id) or ""))
    return df.assign(display_name=current.where(~unnamed_group, derived))


def contacts_from_messages(messages_path: Path, limit: int = 40, name_lookup=None, db_path: Path | None = None) -> pd.DataFrame:
    """List conversations from the same messages CSV used for preview/reporting."""
    df, catalog, names = cached_messages_bundle(messages_path)
    if "chat_id" not in df.columns:
        raise ValueError("messages CSV must contain a 'chat_id' column")

    grouped = catalog.assign(is_group=group_chat_mask(catalog["chat_id"]).astype(int))
    grouped = grouped.sort_values(["message_count", "last_msg"], ascending=[False, False])
    grouped = fill_group_display_names(grouped.head(limit), db_path)
    grouped = name_groups_after_their_people(grouped, names)
    return decorate_contacts_frame(grouped, name_lookup)


def archive_summary(messages_path: Path) -> dict:
    """Size and span of the whole archive. The conversation list is capped, so
    totals come from here and never from adding up that list."""
    empty = {"messages": 0, "conversations": 0, "firstMessage": None, "lastMessage": None}
    if not path_exists(messages_path):
        return empty
    _df, catalog, _names = cached_messages_bundle(messages_path)
    if catalog.empty:
        return empty
    first, last = catalog["first_msg"].min(), catalog["last_msg"].max()
    return {
        "messages": int(catalog["message_count"].sum()),
        "conversations": int(len(catalog)),
        "firstMessage": None if pd.isna(first) else first.isoformat(),
        "lastMessage": None if pd.isna(last) else last.isoformat(),
    }


_PATTERNS_CACHE: dict = {"sig": None, "value": None}
_PATTERNS_LOCK = threading.Lock()


def archive_patterns(messages_path: Path) -> dict:
    """The shape of the whole archive, for the Overview: messages by month in
    both voices, by day, and by hour of each weekday. Worked out once per
    export."""
    if not path_exists(messages_path):
        nothing = pd.DataFrame({"timestamp": pd.to_datetime([])})
        return {"monthly": [], "daily": day_totals(nothing), "rhythm": hour_by_weekday(nothing)}
    sig = file_cache_signature(messages_path)
    # held while it is worked out: requests that arrive together wait for one answer
    with _PATTERNS_LOCK:
        if sig and _PATTERNS_CACHE["sig"] == sig:
            return _PATTERNS_CACHE["value"]
        df, _catalog, _names = cached_messages_bundle(messages_path)
        patterns = {"monthly": month_points(df), "daily": day_totals(df), "rhythm": hour_by_weekday(df)}
        _PATTERNS_CACHE.update({"sig": sig, "value": patterns})
        return patterns


def count_display_names(df: pd.DataFrame) -> int:
    if df.empty or "display_name" not in df.columns:
        return 0
    return int(df["display_name"].fillna("").astype(str).str.len().gt(0).sum())


def list_jobs() -> list[dict]:
    with JOBS_LOCK:
        jobs = [dict(job) for job in JOBS.values()]
    return sorted(jobs, key=lambda job: str(job.get("updatedAt", "")), reverse=True)


def _row_str(row, key: str, default: str = "") -> str:
    """NA/NaN-safe string extraction from a DataFrame row.

    The messages CSV loads chat_id/sender as pandas "string" dtype, so missing
    values are pd.NA. Using `value or ""` on pd.NA raises "boolean value of NA
    is ambiguous" (outbound messages have no sender), so guard with pd.isna.
    """
    value = row.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


MESSAGE_TEXT_LIMIT = 600


def message_id_text(row) -> str:
    """The id of a message, written one way wherever a payload names one."""
    return _row_str(row, "message_id")


def message_row_payload(
    row,
    name_map: dict[str, str] | None = None,
    sender_map: dict[str, str] | None = None,
) -> dict:
    chat_id = _row_str(row, "chat_id")
    text = _row_str(row, "text")
    sender = _row_str(row, "sender")
    display_name = (name_map or {}).get(chat_id, "")
    mine = sent_by_me(row.get("is_from_me"))
    # who actually said it (matters in group chats); blank for my own messages
    # the saved name or "": the interface labels a sender who is not in Contacts
    sender_name = (sender_map or {}).get(sender, "") if mine is not True and sender else ""
    return {
        "messageId": message_id_text(row),
        "chatId": chat_id,
        "displayName": display_name,
        "senderName": sender_name,
        "timestamp": _row_str(row, "timestamp") or None,
        "sender": sender,
        "text": clip_text(display_text(text), MESSAGE_TEXT_LIMIT),
        "attachment": attachment_label(row.get("attachment_types")),
        "isFromMe": mine,
    }


def day_payload(
    messages_path: Path,
    contact: str,
    day: str,
    around: str | None = None,
    offset: int | None = None,
    limit: int = DAY_LIMIT,
) -> dict:
    """One day of one conversation, as the messages it was, with the nearest
    days that hold messages on either side. A very long day is given in part."""
    if not path_exists(messages_path):
        raise FileNotFoundError(f"Messages CSV not found: {messages_path}")
    df, _, names = cached_messages_bundle(messages_path)
    # this conversation and no other: an id is never matched in part here
    conv = df[in_conversation(df, contact)]
    window = day_window(conv, day, around=around, limit=limit, offset=offset)
    rows = window.rows
    # in a group, who wrote each message
    in_group = is_group_chat(contact) and "sender" in rows.columns
    senders = sorted(set(rows["sender"].dropna().astype(str)) - {""}) if in_group else []
    sender_map = dict(resolve_contact_names(senders)) if senders else {}
    return {
        "contact": str(contact),
        "day": window.day,
        "total": window.total,
        "offset": window.offset,
        "previousDay": window.previous_day,
        "nextDay": window.next_day,
        "messages": [message_row_payload(row, names, sender_map) for _, row in rows.iterrows()],
    }


def search_messages_payload(messages_path: Path, query: str = "", contact: str = "", limit: int = 60) -> dict:
    if not path_exists(messages_path):
        raise FileNotFoundError(f"Messages CSV not found: {messages_path}")

    limit = max(1, min(int(limit or 60), 200))
    # the names the conversation list uses: saved, given to a group, or made from its people
    df, _, names = cached_messages_bundle(messages_path)
    if contact:
        df = filter_conversation(df, contact)

    needle = str(query or "").strip()
    if needle:
        text = df["text"].fillna("").astype(str)
        df = df[text.str.contains(needle, case=False, na=False, regex=False)]

    # everything that matches is counted, so the list can say when it is not showing it all
    total = int(len(df))
    if df.empty:
        return {"query": needle, "contact": contact, "results": [], "count": 0, "total": 0}

    df = df.sort_values("timestamp", ascending=False).head(limit)
    # in a group, who wrote each match
    in_groups = df[group_chat_mask(df["chat_id"])]
    senders = sorted(set(in_groups["sender"].dropna().astype(str)) - {""}) if "sender" in df.columns else []
    sender_map = dict(resolve_contact_names(senders)) if senders else {}
    return {
        "query": needle,
        "contact": contact,
        "count": int(len(df)),
        "total": total,
        "results": [message_row_payload(row, names, sender_map) for _, row in df.iterrows()],
    }


ASK_STOPWORDS = {
    "about", "after", "again", "all", "and", "any", "are", "back", "been", "before",
    "being", "between", "but", "can", "could", "did", "does", "ever", "every", "for",
    "from", "get", "going", "got", "had", "has", "have", "her", "him", "his", "how",
    "into", "just", "like", "made", "make", "messages", "more", "most", "much", "not",
    "off", "only", "other", "our", "out", "over", "really", "said", "say", "says",
    "she", "should", "some", "talk", "talked", "tell", "text", "texted", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "thing",
    "things", "this", "those", "thread", "time", "told", "very", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your",
    # contraction fragments and filler that word-boundary-match everywhere
    "its", "hed", "hes", "shes", "thats", "whats", "youre", "theyre", "weve",
    "gonna", "wanna", "kinda", "yeah", "let", "lets",
}


def query_terms(question: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-z0-9']+", str(question or "").lower()):
        if len(term) < 3 or term in ASK_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    # longer words carry the topic; keep them when capping
    return sorted(terms, key=len, reverse=True)[:8]


def _term_variants(term: str) -> set[str]:
    """Light morphological variants so 'taxes' matches 'tax' and 'talked'
    matches 'talk' -- texting never agrees with the question's inflection.
    Truncations are guarded: '-es' only strips after a sibilant (the actual
    English rule), bare '-ed'/'-ing' strips need a long-enough base, and any
    derived form landing on a stopword is dropped -- otherwise 'notes' leaks
    'not' and matches half the archive, defeating the burst-candidate prune."""
    base = term.lower()
    variants = {base}
    if not re.fullmatch(r"[a-z']+", base):
        return variants
    derived: set[str] = set()
    if base.endswith("ies") and len(base) > 4:
        derived.add(base[:-3] + "y")
    if re.search(r"(?:[xsz]|ch|sh)es$", base) and len(base) > 4:
        derived.add(base[:-2])
    if base.endswith("s") and len(base) > 3 and not base.endswith("ss") and base != "news":
        derived.add(base[:-1])
    if base.endswith("ing") and len(base) > 5:
        derived.add(base[:-3] + "e")
        if len(base) > 6:
            derived.add(base[:-3])
    if base.endswith("ed") and len(base) > 4:
        derived.add(base[:-1])
        if len(base) > 5:
            derived.add(base[:-2])
    if not base.endswith("s"):
        derived.add(base + "s")
        if re.search(r"(?:[xz]|ch|sh)$", base):
            derived.add(base + "es")
    variants |= {v for v in derived if len(v) >= 3 and v not in ASK_STOPWORDS}
    return variants


def _term_pattern(term: str) -> str:
    if re.fullmatch(r"[\w']+", term):
        options = sorted(_term_variants(term), key=len, reverse=True)
        return r"\b(?:" + "|".join(re.escape(v) for v in options) + r")\b"
    return re.escape(term)


def _burst_scores(df: pd.DataFrame, terms: list[str], window: int = 5) -> pd.Series:
    """Score each message by the conversation BURST around it, not the lone text.
    People text in fragments -- 'front' and '10k' land in adjacent messages --
    so distinct terms co-occurring within a few messages is the strongest
    signal a moment matches the question. Score = 2x distinct terms present in
    the surrounding window + the row's own hits (so the exact line outranks
    its neighbors).

    One combined regex pass prunes the archive to candidate neighborhoods;
    per-term masks then run only on that small subset (an unscoped 760k-row
    ask used to spend ~8s here in per-group Python rolling)."""
    if df.empty or not terms:
        return pd.Series(0, index=df.index)
    import numpy as np

    ordered = df.sort_values(["chat_id", "timestamp"], kind="stable")
    text = ordered["text"].fillna("").astype(str)
    chats = ordered["chat_id"].astype(str).values

    combined = "|".join(_term_pattern(t) for t in terms)
    hit = text.str.contains(combined, case=False, na=False, regex=True).values
    if not hit.any():
        return pd.Series(0, index=df.index)

    # candidate set = every hit row plus its in-chat neighbors within the window
    half = window // 2
    near = hit.copy()
    for k in range(1, half + 1):
        same = chats[k:] == chats[:-k]
        near[k:] |= hit[:-k] & same
        near[:-k] |= hit[k:] & same

    idx = np.where(near)[0]
    pos = idx
    chat_sub = chats[idx]
    sub_text = text.iloc[idx]
    masks = np.stack(
        [sub_text.str.contains(_term_pattern(t), case=False, na=False, regex=True).values for t in terms]
    )  # shape: (terms, candidates)
    own = masks.sum(axis=0).astype(np.int32)
    distinct = masks.astype(np.float32).copy()
    for k in range(1, half + 1):
        # candidate neighborhoods are contiguous runs, so an original-order
        # neighbor at distance k is a subset neighbor at offset k when the
        # position delta and chat both line up
        ok = (pos[k:] - pos[:-k] == k) & (chat_sub[k:] == chat_sub[:-k])
        distinct[:, k:] = np.maximum(distinct[:, k:], masks[:, :-k] * ok)
        distinct[:, :-k] = np.maximum(distinct[:, :-k], masks[:, k:] * ok)
    score_sub = distinct.sum(axis=0) * 2 + own

    scores = pd.Series(0.0, index=ordered.index)
    scores.iloc[idx] = score_sub
    return scores.reindex(df.index)


def citation_context(citations: list[dict]) -> str:
    lines = []
    for idx, citation in enumerate(citations, start=1):
        outbound = sent_by_me(citation.get("isFromMe")) is True
        convo = contact_label(citation)
        in_group = is_group_chat(str(citation.get("chatId") or ""))
        sender = str(citation.get("sender") or "")
        # in a group, someone who is not in Contacts is still told apart from the rest
        unsaved = contact_label({"chatId": sender}) if in_group and sender else convo
        speaker = "You" if outbound else (citation.get("senderName") or unsaved)
        timestamp = citation.get("timestamp") or "unknown time"
        text = str(citation.get("text") or "").replace("\n", " ").strip()
        # name the conversation only when it adds info (a group, or You speaking)
        location = f" in {convo}" if (outbound or (convo and convo != speaker)) else ""
        lines.append(f"[{idx}] {speaker}{location} ({timestamp}): {text}")
    return "\n".join(lines)


def local_ask_answer(terms: list[str], citations: list[dict]) -> str:
    if not citations:
        return "I could not find messages that match that scope yet."

    lead = f"I found {len(citations)} relevant messages"
    if terms:
        lead += f" around {', '.join(terms[:4])}"
    lead += "."

    highlights = []
    for citation in citations[:3]:
        # the user reads this, so a conversation goes by the name the interface gives it
        name = conversation_label(citation.get("displayName"), citation.get("chatId"))
        text = str(citation.get("text") or "").replace("\n", " ").strip()
        if text:
            highlights.append(f"{name}: {text[:180]}")

    if not highlights:
        return lead
    return lead + "\n\n" + "\n".join(f"- {highlight}" for highlight in highlights)


def build_context_windows(df: pd.DataFrame, matches: pd.DataFrame, radius: int = 6, cap: int = 30) -> list[dict]:
    """For each hit, grab the surrounding conversation (radius messages each side)
    so the model sees the situation and nuance, not an isolated keyword match.
    Hits that fall inside an existing window merge into it (multiple highlights)
    instead of duplicating overlapping snippets."""
    if matches is None or matches.empty or "chat_id" not in df.columns:
        return []
    top = matches.head(cap)
    convos: dict[str, pd.DataFrame] = {}
    for cid in top["chat_id"].dropna().astype(str).unique():
        convos[cid] = (
            df[df["chat_id"].astype(str) == cid]
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
    windows: list[dict] = []
    for index, (_, hit) in enumerate(top.iterrows(), start=1):
        cid = _row_str(hit, "chat_id")
        convo = convos.get(cid)
        if convo is None or convo.empty:
            continue
        mid = _row_str(hit, "message_id")
        pos = convo.index[convo["message_id"].astype(str) == mid].tolist()
        if not pos:
            ts = _row_str(hit, "timestamp")
            pos = convo.index[convo["timestamp"].astype(str) == ts].tolist()
        if not pos:
            continue
        center = int(pos[0])
        merged = False
        for window in windows:
            if window["chat_id"] == cid and window["lo"] <= center < window["hi"]:
                window["hits"].setdefault(center - window["lo"], []).append(index)
                merged = True
                break
        if merged:
            continue
        lo = max(0, center - radius)
        hi = min(len(convo), center + radius + 1)
        windows.append(
            {
                "ns": [index],
                "chat_id": cid,
                "lo": lo,
                "hi": hi,
                "rows": convo.iloc[lo:hi],
                "hits": {center - lo: [index]},
                "timestamp": _row_str(hit, "timestamp"),
            }
        )
    return windows


def format_context_windows(windows: list[dict], name_map=None, sender_map=None) -> str:
    """Render each window as a dated conversation snippet with matched lines marked."""
    blocks = []
    for window in windows:
        cid = window["chat_id"]
        is_group = is_group_chat(cid)
        label = (name_map or {}).get(cid) or contact_label({"chatId": cid})
        date = str(window.get("timestamp") or "")[:10]
        all_ns = sorted({n for ns in window["hits"].values() for n in ns})
        moment = "".join(f"[{n}]" for n in all_ns)
        header = f"=== Moment {moment} -- with {label}" + (f", around {date}" if date else "") + " ==="
        lines = [header]
        for offset, (_, row) in enumerate(window["rows"].iterrows()):
            text = _row_str(row, "text").replace("\n", " ").strip()[:280]
            if not text:
                continue
            outbound = sent_by_me(row.get("is_from_me")) is True
            if outbound:
                who = "You"
            else:
                sender = _row_str(row, "sender")
                who = (sender_map or {}).get(sender, "")
                if not who:
                    # in a group, an unresolved sender must NOT collapse into the
                    # group label -- that misattributes quotes to the wrong person
                    who = contact_label({"chatId": sender}) if (is_group and sender) else label
            ts = _row_str(row, "timestamp")[:10]
            marker = ""
            if offset in window["hits"]:
                marker = "   <<< highlighted " + "".join(f"[{n}]" for n in window["hits"][offset])
            lines.append(f"[{ts}] {who}: {text}{marker}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ai_ask_answer(
    question: str,
    citations: list[dict],
    scope_label: str,
    model: str,
    intent: str = "general",
    context: str = "",
    history: list[dict] | None = None,
    on_delta=None,
) -> str:
    if not os.environ.get("OPENAI_API_KEY") or not citations:
        return ""

    system = (
        "You are Recall, a private iMessage archive assistant. "
        "Answer like a sharp, helpful chatbot: reason over the message excerpts to actually answer the "
        "question -- infer what they imply (who people are, relationships, what happened), don't just restate them. "
        "Ground every claim in the excerpts and separate what is directly stated from what you are inferring. "
        "Each highlighted line is shown inside the surrounding conversation -- read the whole exchange to "
        "understand the situation (what led to it, who was involved, the nuance) and explain the moment, "
        "not just the highlighted line. "
        "When the question is trying to RECALL something specific that was said -- 'what was the thing we "
        "talked about', 'what did they call it' -- the answer is the exact word or phrase from the "
        "conversation: find it in the excerpts, lead with it, and quote the exchange. Only reach for "
        "general knowledge if the excerpts genuinely don't contain it, and say so explicitly instead of "
        "presenting a guess as the memory. "
        "The archive belongs to the person you are talking to: refer to them as 'you', never as 'me' or by a "
        "name -- messages labeled 'You' are theirs. Refer to everyone else by the name shown before each excerpt. "
        "Never output a raw phone number or email address as a person's identity; "
        "if someone has no saved contact name, call them an unsaved contact. "
        "Cite evidence inline with individual bracket numbers like [1] or [2][5] -- never ranges like [2-5] "
        "or lists like [1, 3]. Keep the answer concise unless the user asks for detail."
    )
    if intent == "identity":
        system += (
            " This is a 'who is this' question. Work out who the person most likely is by reasoning over the "
            "clues in the thread: names they or others mention, people / places / events referenced, relationships, "
            "how they are addressed, school or work, and what they talk about. Give your best-supported conclusion "
            "about who they are and the reasoning behind it; if you can only narrow it down, say what you can tell "
            "about them. Do not invent a name the messages don't support, but don't refuse just because no one "
            "states a name outright -- infer from context."
        )
    body = context or citation_context(citations)
    intro = (
        "Conversations around each moment (the highlighted line is the matched message):\n"
        if context
        else "Message excerpts:\n"
    )
    user = (
        f"Question: {question}\n"
        f"Scope: {scope_label}\n\n"
        f"{intro}{body}"
    )
    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-6:]:
        role = "assistant" if str(turn.get("role")) == "assistant" else "user"
        content = str(turn.get("content") or "").strip()[:600]
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    effort = "medium" if (intent == "identity" or context) else "low"
    if on_delta is not None:
        return _call_openai_stream(
            OpenAI(),
            messages,
            on_delta,
            model=model or DEFAULT_MODEL,
            reasoning_effort=effort,
            verbosity="low",
        )
    return _call_openai(
        OpenAI(),
        messages,
        model=model or DEFAULT_MODEL,
        max_retries=1,
        reasoning_effort=effort,
        verbosity="low",
    )


CATALOG_COLUMNS = [
    "chat_id", "message_count", "sent_count", "received_count", "first_msg", "last_msg", "display_name",
]


def _conversation_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation rollup (CATALOG_COLUMNS). Built once per export and
    cached with it, so nothing that lists conversations walks the messages again."""
    convo = df[df["chat_id"].notna()].copy() if "chat_id" in df.columns else pd.DataFrame()
    if convo.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    convo["chat_id"] = convo["chat_id"].astype(str)
    # a row that does not say who sent it is in the total and in neither count
    convo["sent"] = sent_mask(convo["is_from_me"]) if "is_from_me" in convo.columns else False
    convo["received"] = received_mask(convo["is_from_me"]) if "is_from_me" in convo.columns else False
    aggregations = {
        "message_count": ("chat_id", "size"),
        "sent_count": ("sent", "sum"),
        "received_count": ("received", "sum"),
        "first_msg": ("timestamp", "min"),
        "last_msg": ("timestamp", "max"),
    }
    if "chat_display_name" in convo.columns:
        convo["chat_display_name"] = convo["chat_display_name"].fillna("").astype(str).str.strip()
        aggregations["display_name"] = ("chat_display_name", first_nonempty)
    grouped = convo.groupby("chat_id").agg(**aggregations).reset_index()
    if "display_name" not in grouped.columns:
        grouped["display_name"] = ""
    return grouped


# Ask-pipeline cache: the messages CSV is ~75MB / 760k rows; re-parsing it (plus the
# catalog groupby and full contact-name resolution) on every question costs seconds
# before any model call. Keyed on the file signature so refreshes invalidate it.
_ASK_CACHE_LOCK = threading.Lock()
_ASK_CACHE: dict = {"sig": None, "df": None, "catalog": None, "names": None}

SEMANTIC_DIR = SAVES_DIR / "semantic"
_SEM_CACHE_LOCK = threading.Lock()
_SEM_CACHE: dict = {"sig": None, "index": None}


def get_semantic_index(messages_path: Path):
    """The semantic index for the CURRENT messages file, or None (missing/stale)."""
    sig = file_cache_signature(messages_path)
    with _SEM_CACHE_LOCK:
        if sig and _SEM_CACHE["sig"] == sig:
            return _SEM_CACHE["index"]
    index = sem.load_index(SEMANTIC_DIR, sig)
    with _SEM_CACHE_LOCK:
        _SEM_CACHE.update({"sig": sig, "index": index})
    return index


def invalidate_semantic_cache():
    with _SEM_CACHE_LOCK:
        _SEM_CACHE.update({"sig": None, "index": None})


FIRST_WORDS_QUOTE_LIMIT = 280
# a quoted memory reads best as a sentence or two
MEMORY_QUOTE_MIN = 25
MEMORY_QUOTE_LIMIT = 130


_FW_URL_ONLY_RE = re.compile(r"^https?://\S+$", re.I)
_FW_LETTER_RE = re.compile(r"[^\W\d_]")
_FW_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿❤♥]")


def _first_real_texts(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows whose text reads like words someone typed: not attachment-only,
    not a bare link, carries at least one letter or an emoji."""
    texts = rows["text"].fillna("").astype(str).str.strip()
    ok = (
        texts.ne("")
        & ~texts.str.contains(ATTACHMENT_PLACEHOLDER, regex=False)
        & ~texts.str.lower().str.startswith("[attachment")
        & ~texts.str.match(_FW_URL_ONLY_RE)
        & (texts.str.contains(_FW_LETTER_RE) | texts.str.contains(_FW_EMOJI_RE))
    )
    return rows[ok]


def first_words_payload(messages_path: Path, limit: int = 5) -> dict:
    """The first text ever exchanged with the people you text most -- the
    post-import reveal. 1:1 chats only (the first line in a group chat is
    often a third party); a person's phone and email handles merge by
    resolved contact name, and named contacts win unless almost nothing is
    named yet (fresh machine before the Contacts step)."""
    empty = {"entries": [], "signature": "", "totals": {"messages": 0, "people": 0}}
    if not path_exists(messages_path):
        return empty
    raw_sig = file_cache_signature(messages_path)
    sig = "|".join(str(part) for part in raw_sig) if isinstance(raw_sig, tuple) else str(raw_sig or "")
    df, catalog, names = cached_messages_bundle(messages_path)
    if df.empty or catalog.empty or "timestamp" not in df.columns:
        return empty

    direct = catalog[~group_chat_mask(catalog["chat_id"])].copy()
    if direct.empty:
        return {**empty, "totals": {"messages": int(len(df)), "people": 0}}
    direct["chat_id"] = direct["chat_id"].astype(str)
    resolved = direct["chat_id"].map(lambda cid: str(names.get(cid) or "").strip())
    direct["person_key"] = [
        name.lower() if name else cid for name, cid in zip(resolved, direct["chat_id"])
    ]
    direct["is_named"] = resolved.ne("")

    people = (
        direct.groupby("person_key")
        .agg(
            count=("message_count", "sum"),
            handles=("chat_id", list),
            is_named=("is_named", "any"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    named_people = people[people["is_named"]]
    pool = named_people if len(named_people) >= 3 else people
    top = pool.head(limit)

    cids_all = df["chat_id"].astype(str)
    entries: list[dict] = []
    for _, person in top.iterrows():
        handles = {str(h) for h in person["handles"]}
        rows = df[cids_all.isin(handles)].sort_values("timestamp").head(80)
        real = _first_real_texts(rows)
        if real.empty:
            continue
        real = real.assign(direction=real["is_from_me"].map(direction_word))
        first = real.iloc[0]
        first_dir = str(first["direction"])
        reply = None
        # the reply is the first thing the other side said back; when the export
        # does not say who sent the first message there is no telling which side that is
        later = real.iloc[1:]
        answers = later[later["direction"].ne("") & later["direction"].ne(first_dir)] if first_dir else later.iloc[0:0]
        if not answers.empty:
            row = answers.iloc[0]
            reply = {
                "text": clip_text(display_text(row["text"]), FIRST_WORDS_QUOTE_LIMIT),
                "direction": str(row["direction"]),
                "timestamp": str(row["timestamp"]),
            }
        cid = str(first["chat_id"])
        first_ts = pd.Timestamp(first["timestamp"])
        entries.append({
            # the saved name or "": the interface labels someone who is not in Contacts
            "person": str(names.get(cid) or ""),
            "chatId": cid,
            "timestamp": str(first["timestamp"]),
            "yearsAgo": max(0, int((pd.Timestamp.now() - first_ts).days // 365)),
            "direction": first_dir,
            "text": clip_text(display_text(first["text"]), FIRST_WORDS_QUOTE_LIMIT),
            "reply": reply,
            "messageCount": int(person["count"]),
        })

    years = df["timestamp"].dt.year.dropna()
    return {
        "entries": entries,
        "signature": sig,
        "totals": {
            "messages": int(len(df)),
            "people": len(entries),
            "firstYear": int(years.min()) if len(years) else None,
            "lastYear": int(years.max()) if len(years) else None,
        },
    }


# an anniversary is shown from this many days before it
ANNIVERSARY_DAYS_AHEAD = 7


def _next_anniversary(first: pd.Timestamp, today: pd.Timestamp) -> pd.Timestamp | None:
    """The next day, today or later, with the month and day of `first`. It can
    be in the year after this one: 2 January is days away on 28 December.
    None for a 29 February when neither year has one."""
    for year in (today.year, today.year + 1):
        try:
            candidate = first.replace(year=year).normalize()
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


def memories_payload(messages_path: Path, now: pd.Timestamp | None = None) -> dict:
    """Proactive memories computed from the cached dataframe -- no AI calls.
    On-this-day moments from past years, upcoming first-message anniversaries,
    and reconnect nudges for high-volume conversations gone quiet.

    name is the saved name or "": the interface labels unsaved conversations,
    so the same conversation reads the same everywhere."""
    empty = {"onThisDay": [], "anniversaries": [], "reconnect": []}
    if not path_exists(messages_path):
        return empty
    df, _catalog, names = cached_messages_bundle(messages_path)
    if df.empty or "timestamp" not in df.columns:
        return empty

    now = now if now is not None else pd.Timestamp.now()
    # An archive that has not been updated is not a conversation gone quiet:
    # silence is measured up to the last message the archive holds.
    # a message that no conversation holds is in no moment: such rows come
    # from different threads and have no conversation to open
    df = df[df["chat_id"].notna()]
    archive_end = df["timestamp"].max()
    if pd.isna(archive_end):
        # no time in the export could be read, so nothing can be placed on a day
        return empty
    cids = df["chat_id"].astype(str)

    def saved_name(cid: str) -> str:
        return str(names.get(cid) or "")

    def memory_quote(cid: str, day_rows: pd.DataFrame) -> dict:
        """A quote that reads like a memory: conversational, never a link or a
        paste. Says who sent it, so it is never shown as someone else's words."""
        blank = {"preview": "", "previewFromMe": None, "previewSender": "", "previewAt": None, "previewId": ""}
        texts = day_rows["text"].map(display_text)
        clean = texts[
            ~texts.str.contains("http", case=False, na=False)
            & texts.str.contains(r"[a-zA-Z]{3}", na=False)
        ]
        if clean.empty:
            return blank
        lengths = clean.str.len()
        sweet = lengths[(lengths >= MEMORY_QUOTE_MIN) & (lengths <= MEMORY_QUOTE_LIMIT)]
        chosen = day_rows.loc[(sweet if not sweet.empty else lengths).idxmax()]
        mine = sent_by_me(chosen.get("is_from_me"))
        sender = _row_str(chosen, "sender")
        # in a group the title is the group, so the quote needs its own name
        in_group = is_group_chat(cid) and mine is False and bool(sender)
        return {
            "preview": clip_text(texts.loc[chosen.name], MEMORY_QUOTE_LIMIT),
            "previewFromMe": mine,
            "previewSender": str(resolve_contact_names([sender]).get(sender) or "") if in_group else "",
            # when it was sent, so the day can be opened at this message
            "previewAt": _row_str(chosen, "timestamp") or None,
            # and which message it is: two sent in the same second share a time
            "previewId": message_id_text(chosen),
        }

    # --- on this day, in past years ---
    on_this_day: list[dict] = []
    ts = df["timestamp"]
    past_today = df[(ts.dt.month == now.month) & (ts.dt.day == now.day) & (ts.dt.year < now.year)]
    if not past_today.empty:
        day_years = past_today["timestamp"].dt.year
        day_cids = cids[past_today.index]
        worded = past_today["text"].fillna("").astype(str).str.strip().ne("")
        # A day is counted in full, the way the day view and the calendar count
        # it. It is a moment only when someone wrote words on it.
        per_day = (
            pd.DataFrame({"year": day_years, "cid": day_cids, "worded": worded})
            .groupby(["year", "cid"])["worded"]
            .agg(count="size", worded="sum")
            .reset_index()
        )
        busiest = per_day[per_day["worded"] > 0].sort_values("count", ascending=False, kind="stable").head(6)
        for _, row in busiest.iterrows():
            year, cid, count = int(row["year"]), str(row["cid"]), int(row["count"])
            # the quote is picked from the messages that hold words
            day_rows = past_today[(day_years == year) & (day_cids == cid) & worded]
            on_this_day.append(
                {
                    "chatId": cid,
                    "name": saved_name(cid),
                    "year": year,
                    "day": f"{year:04d}-{now.month:02d}-{now.day:02d}",
                    "yearsAgo": int(now.year - year),
                    "count": count,
                    **memory_quote(cid, day_rows),
                }
            )

    firsts = df.groupby(cids)["timestamp"].min()
    lasts = df.groupby(cids)["timestamp"].max()
    counts = df.groupby(cids).size()

    # --- first-message anniversaries in the next week (named contacts only) ---
    anniversaries: list[dict] = []
    today = now.normalize()
    # a conversation with no time that can be read has no first day
    for cid, first in firsts.dropna().items():
        if int(counts.get(cid, 0)) < 300 or not names.get(str(cid)):
            continue
        upcoming = _next_anniversary(first, today)
        if upcoming is None:
            continue
        years = int(upcoming.year - first.year)
        in_days = int((upcoming - today).days)
        if years >= 1 and in_days <= ANNIVERSARY_DAYS_AHEAD:
            anniversaries.append(
                {
                    "chatId": str(cid),
                    "name": names[str(cid)],
                    "years": years,
                    "date": str(first)[:10],
                    "inDays": in_days,
                    "count": int(counts[cid]),
                }
            )
    anniversaries.sort(key=lambda item: (item["inDays"], -item["count"]))

    # --- reconnect: big 1:1 threads gone quiet ---
    reconnect: list[dict] = []
    for cid, last in lasts.dropna().items():
        cid = str(cid)
        if is_group_chat(cid) or int(counts.get(cid, 0)) < 1000:
            continue
        quiet_days = int((archive_end - last).days)
        if quiet_days < 90:
            continue
        reconnect.append(
            {
                "chatId": cid,
                "name": saved_name(cid),
                "count": int(counts[cid]),
                "quietDays": quiet_days,
                "lastDate": str(last)[:10],
            }
        )
    reconnect.sort(key=lambda item: -(item["count"] * min(item["quietDays"], 365)))

    return {
        "onThisDay": on_this_day,
        "anniversaries": anniversaries[:4],
        "reconnect": reconnect[:5],
    }


def _parse_export(messages_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_messages(str(messages_path))
    if "text" not in df.columns:
        df = df.assign(text="")
    return df, _conversation_catalog(df)


def _conversation_names(df: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, str]:
    """The name each conversation goes by: the one saved in Contacts, the one a
    group was given in Messages, or for a group nobody named, the people in it."""
    if catalog.empty:
        return {}
    saved = dict(resolve_contact_names(catalog["chat_id"].tolist()))
    # groups have no Contacts entry; their iMessage display name is the label
    given = {
        str(row["chat_id"]): str(row.get("display_name") or "").strip()
        for _, row in catalog.iterrows()
    }
    names = {**saved, **{cid: name for cid, name in given.items() if name and not saved.get(cid)}}
    unnamed_groups = [
        cid for cid in catalog["chat_id"].astype(str)
        if is_group_chat(cid) and not names.get(cid)
    ]
    return {
        **names,
        **derive_group_names(df, unnamed_groups, lambda handles: dict(resolve_contact_names(handles))),
    }


def bundle_signature(messages_path: Path) -> tuple:
    """What the cached archive is made from: the export, and for its names, Contacts."""
    return (file_cache_signature(messages_path), file_cache_signature(CONTACTS_CACHE_PATH))


def cached_messages_bundle(messages_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(messages df, conversation catalog, display-name map), kept until the
    export or Contacts changes. When only Contacts changed, the export is not
    read again: the names are worked out over the frame already held.
    Concurrent cold callers wait for the first loader instead of each parsing
    the 75MB CSV in parallel (app launch fires four endpoints at once)."""
    sig = bundle_signature(messages_path)
    export_sig = sig[0]
    while True:
        with _ASK_CACHE_LOCK:
            held = _ASK_CACHE["sig"]
            has_export = _ASK_CACHE["df"] is not None
            if export_sig and held == sig and has_export:
                return _ASK_CACHE["df"], _ASK_CACHE["catalog"], _ASK_CACHE["names"]
            loading = _ASK_CACHE.get("loading")
            if loading is None or loading.is_set():
                loading = threading.Event()
                _ASK_CACHE["loading"] = loading
                same_export = bool(export_sig) and has_export and held is not None and held[0] == export_sig
                parsed = (_ASK_CACHE["df"], _ASK_CACHE["catalog"]) if same_export else None
                break  # this thread builds; everyone else waits below
        loading.wait(timeout=180)

    try:
        df, catalog = parsed if parsed is not None else _parse_export(messages_path)
        names = _conversation_names(df, catalog)
        with _ASK_CACHE_LOCK:
            _ASK_CACHE.update({"sig": sig, "df": df, "catalog": catalog, "names": names})
        return df, catalog, names
    finally:
        loading.set()


def _plan_retrieval_safe(
    catalog: pd.DataFrame,
    names: dict,
    question: str,
    history_text: str = "",
) -> dict | None:
    """Ask the planner over a prebuilt catalog. Never raises."""
    try:
        selected = build_catalog(catalog, question, name_lookup=lambda _ids: names)
        return plan_retrieval(
            str(question or "").strip(),
            selected,
            model=PLANNER_MODEL,
            history_text=history_text,
        )
    except Exception as exc:
        print(f"[ask] planner failed: {exc}", file=sys.__stderr__)
        return None


def _representative_messages(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """A useful spread (recent-weighted with some earliest) of real-text messages."""
    real = df[df["text"].fillna("").astype(str).str.strip() != ""]
    if real.empty:
        real = df
    ordered = real.sort_values("timestamp")
    if len(ordered) <= cap:
        return ordered.sort_values("timestamp", ascending=False)
    recent = ordered.tail(max(1, cap - cap // 3))
    earliest = ordered.head(cap // 3)
    return (
        pd.concat([recent, earliest])
        .drop_duplicates()
        .sort_values("timestamp", ascending=False)
        .head(cap)
    )


def _moment_positions(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-row chat id + position in that chat's timeline, for moment dedupe."""
    ordered = df.sort_values(["chat_id", "timestamp"], kind="stable")
    pos = pd.Series(range(len(ordered)), index=ordered.index)
    return ordered["chat_id"].astype(str), pos


def _dedupe_moments(
    rows: pd.DataFrame,
    chat: pd.Series,
    pos: pd.Series,
    per_moment: int = 2,
    radius: int = 12,
    seed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cap rows per MOMENT (same chat, within `radius` messages): the context
    window around one hit already covers its burst-mates, so extra rows from
    the same exchange eat cap slots that other moments need. Rows must arrive
    in signal order; seed rows count as already-kept coverage."""
    if rows.empty:
        return rows
    kept: dict[str, list[int]] = {}
    for i in (seed.index if seed is not None else []):
        c, p = chat.get(i), pos.get(i)
        if c is not None and p is not None:
            kept.setdefault(c, []).append(int(p))
    kept_idx = []
    for i in rows.index:
        c, p = chat.get(i), pos.get(i)
        if c is None or p is None:
            kept_idx.append(i)
            continue
        if sum(1 for x in kept.get(c, ()) if abs(x - int(p)) <= radius) >= per_moment:
            continue
        kept.setdefault(c, []).append(int(p))
        kept_idx.append(i)
    return rows.loc[kept_idx]


def _select_messages(df: pd.DataFrame, terms: list[str], cap: int, prefer_recent: bool = True) -> pd.DataFrame:
    """Score by conversation bursts of (meaning-expanded) terms; fall back to a
    representative spread. prefer_recent=False surfaces the EARLIEST matches."""
    if df.empty:
        return df
    if terms:
        scores = _burst_scores(df, terms)
        # >= 3 keeps rows that themselves hit a term; pure neighbors of a
        # single hit (score 2) are recovered later by the context windows
        hits = df[scores >= 3].copy()
        if not hits.empty:
            hits["score"] = scores[scores >= 3]
            return hits.sort_values(["score", "timestamp"], ascending=[False, not prefer_recent]).head(cap)
    return _representative_messages(df, cap)


IDENTITY_HINT_RE = re.compile(
    r"\b(?:i['’]?m|i am|my name|name['’]?s|this is|it['’]?s me|roommate|girlfriend|boyfriend|"
    r"friend|brother|sister|mom|dad|mother|father|cousin|wife|husband|partner|"
    r"we met|met (?:you|at)|works?|working|school|class|major|dorm|team|club|lives?)\b",
    re.I,
)
NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _identity_messages(df: pd.DataFrame, cap: int, terms: list[str] | None = None) -> pd.DataFrame:
    """For 'who is this' questions: pull a richer context, weighted toward messages
    carrying identity signals (intro / relationship words, mentioned names) and,
    when the question has a topic, messages matching it."""
    real = df[df["text"].fillna("").astype(str).str.strip() != ""].copy()
    if real.empty:
        return _representative_messages(df, cap)
    text = real["text"].astype(str)
    real["_sig"] = (
        text.str.contains(IDENTITY_HINT_RE).astype(int) * 2
        + text.str.contains(NAME_TOKEN_RE).astype(int)
    )
    if terms:
        real["_sig"] = real["_sig"] + _burst_scores(real, terms)
    strong = real[real["_sig"] > 0].sort_values(["_sig", "timestamp"], ascending=[False, False]).head(cap)
    spread = _representative_messages(real.drop(columns=["_sig"]), max(4, cap // 3))
    combined = pd.concat([strong.drop(columns=["_sig"]), spread])
    if "message_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["message_id"])
    else:
        combined = combined.drop_duplicates()
    # keep SIGNAL order: downstream trims (semantic blend head()) must cut the
    # weakest rows, not the oldest -- the earliest intro lines are often the
    # strongest identity evidence
    return combined.head(cap)


def _canon_handle(value: str) -> str:
    """Canonical key for a handle/sender: last-10 digits for phones, lowercased email."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "@" in text:
        return text.lower()
    digits = re.sub(r"\D", "", text)
    return digits[-10:] if len(digits) >= 10 else digits


def _canon_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    email = s.str.contains("@", regex=False)
    digits = s.str.replace(r"\D", "", regex=True)
    last10 = digits.str[-10:].where(digits.str.len() >= 10, digits)
    return last10.where(~email, s.str.strip().str.lower())


def _person_scope(df: pd.DataFrame, chat_ids: list[str], handles: list[str]) -> pd.DataFrame:
    """Every chat the person appears in: the planner's picks plus any group chat
    where the person sends. Group context is where identity clues often live."""
    scope = {str(cid) for cid in chat_ids}
    person_keys = {key for key in (_canon_handle(h) for h in handles) if key}
    cids = df["chat_id"].astype(str)
    if person_keys and "sender" in df.columns:
        sent = _canon_series(df["sender"]).isin(person_keys)
        scope |= set(cids[sent & group_chat_mask(cids)].unique())
    return df[cids.isin(scope)]


def _person_messages(
    df: pd.DataFrame,
    person_keys: set[str],
    name_tokens: list[str],
    cap: int,
    terms: list[str] | None = None,
) -> pd.DataFrame:
    """Within the person's chats, prefer messages about the question's TOPIC,
    then messages they sent / that name them / identity-signal lines; fill with
    a representative spread. Without the topic component, 'what did X say about
    Y' degenerates into X's most recent messages."""
    real = df[df["text"].fillna("").astype(str).str.strip() != ""].copy()
    if real.empty:
        return _representative_messages(df, cap)
    text = real["text"].astype(str)
    sent_by = _canon_series(real["sender"]).isin(person_keys).astype(int) if "sender" in real.columns else 0
    name_hit = 0
    if name_tokens:
        pattern = r"\b(?:" + "|".join(re.escape(tok) for tok in name_tokens) + r")\b"
        name_hit = text.str.contains(pattern, case=False, na=False, regex=True).astype(int)
    hint = text.str.contains(IDENTITY_HINT_RE, na=False).astype(int)
    real["_sig"] = sent_by * 2 + name_hit * 2 + hint
    if terms:
        real["_sig"] = real["_sig"] + _burst_scores(real, terms)
    strong = real[real["_sig"] > 0].sort_values(["_sig", "timestamp"], ascending=[False, False]).head(cap)
    spread = _representative_messages(real.drop(columns=["_sig"]), max(4, cap // 3))
    combined = pd.concat([strong.drop(columns=["_sig"]), spread])
    if "message_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["message_id"])
    else:
        combined = combined.drop_duplicates()
    # keep SIGNAL order: downstream trims (semantic blend head()) must cut the
    # weakest rows, not the oldest -- the earliest intro lines are often the
    # strongest identity evidence
    return combined.head(cap)


def _expand_citation_markers(answer: str) -> str:
    """Normalize range/list markers the model may emit despite instructions:
    [2-5] -> [2][3][4][5], [1, 3] -> [1][3]. Keeps renumbering exact."""

    def expand_range(match: re.Match) -> str:
        start, end = int(match.group(1)), int(match.group(2))
        if 0 < start <= end and end - start <= 40:
            return "".join(f"[{n}]" for n in range(start, end + 1))
        return match.group(0)

    out = re.sub(r"\[(\d+)\s*[-–—]\s*(\d+)\]", expand_range, answer)
    out = re.sub(
        r"\[(\d+(?:\s*,\s*\d+)+)\]",
        lambda m: "".join(f"[{n}]" for n in re.split(r"\s*,\s*", m.group(1))),
        out,
    )
    return out


def _renumber_citations(answer: str, citations: list[dict]) -> tuple[str, list[dict]]:
    """Keep only the citations the answer actually cites, renumbered [1..k] in
    first-cited order, and rewrite the answer's bracket markers to match. Stops
    a broad identity retrieval from showing a wall of unrelated messages."""
    if not answer or not citations:
        return answer, citations
    answer = _expand_citation_markers(answer)
    order: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        n = int(match.group(1))
        if 1 <= n <= len(citations) and n not in order:
            order.append(n)
    if not order:
        # the model cited nothing concrete -- show a small sample, not the
        # whole retrieval set
        return answer, citations[:6]
    mapping = {old: idx + 1 for idx, old in enumerate(order)}
    rewritten = re.sub(
        r"\[(\d+)\]",
        lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else m.group(0),
        answer,
    )
    kept = [citations[old - 1] for old in order]
    return rewritten, kept


def ask_messages_payload(
    messages_path: Path,
    question: str,
    contact: str = "",
    limit: int = 8,
    model: str = DEFAULT_MODEL,
    use_ai: bool = False,
    history: list[dict] | None = None,
    on_event=None,
) -> dict:
    if not path_exists(messages_path):
        raise FileNotFoundError(f"Messages CSV not found: {messages_path}")

    def emit(kind: str, text: str) -> None:
        if on_event is not None:
            on_event(kind, text)

    emit("status", "Reading your archive…")
    full_df, catalog, catalog_names = cached_messages_bundle(messages_path)
    df = filter_conversation(full_df, contact) if contact else full_df

    history = [turn for turn in (history or []) if isinstance(turn, dict)][-6:]
    history_text = "\n".join(
        f"{'assistant' if str(t.get('role')) == 'assistant' else 'user'}: "
        f"{str(t.get('content') or '').strip()[:200]}"
        for t in history[-4:]
        if str(t.get("content") or "").strip()
    )

    # Retrieve by MEANING, not literal query words: when AI is on, let the planner
    # route to the right conversation(s) and expand the query into related terms.
    if use_ai:
        emit("status", "Working out where to look…")
    plan = _plan_retrieval_safe(catalog, catalog_names, question, history_text) if use_ai else None
    plan_ids = (plan or {}).get("chat_ids") or []
    intent = (plan or {}).get("intent", "general")
    person_focus = bool((plan or {}).get("person_focus")) or intent == "identity"

    if use_ai and person_focus and not plan_ids and not contact:
        # backstop: the planner resolved a person but didn't map them to a
        # conversation id. Recover it by WORD-BOUNDARY matching contact names
        # against the question and the latest user turn only (assistant
        # answers name other people and would pollute the match). Ambiguous
        # matches are skipped -- guessing the wrong person is worse than a
        # broad search.
        last_user = next(
            (str(t.get("content") or "") for t in reversed(history) if str(t.get("role")) != "assistant"),
            "",
        )
        blob = f"{question} {last_user}".lower()
        exact, partial = [], []
        for cid, nm in catalog_names.items():
            label = str(nm or "").strip().lower()
            if len(label) < 4:
                continue
            if re.search(r"\b" + re.escape(label) + r"\b", blob):
                exact.append((len(label), cid))
                continue
            tokens = [tok for tok in re.findall(r"[a-z]{4,}", label) if tok not in ASK_STOPWORDS]
            if tokens and any(re.search(r"\b" + re.escape(tok) + r"\b", blob) for tok in tokens):
                partial.append((len(label), cid))
        if exact:
            plan_ids = [max(exact)[1]]
        elif len(partial) == 1:
            plan_ids = [partial[0][1]]

    if contact:
        # the user's scope selection is authoritative: the planner contributes
        # terms/dates/intent, never a different conversation -- intersecting
        # its picks with the scope silently empties retrieval
        plan_ids = []

    handles = [cid for cid in plan_ids if cid and not is_group_chat(str(cid))]

    if plan_ids:
        if person_focus and handles:
            # the person across every chat they're in, not just one conversation
            df = _person_scope(df, plan_ids, handles)
        else:
            df = df[df["chat_id"].astype(str).isin(set(plan_ids))]

    # time-anchored questions ("last summer", "when did we first...") narrow the
    # range; prefer_recent=false surfaces the earliest matches first
    date_from = (plan or {}).get("date_from") or ""
    date_to = (plan or {}).get("date_to") or ""
    try:
        df = within_dates(df, date_from, date_to)
    except (ValueError, OverflowError):
        # the planner can emit impossible calendar dates; a bad date must
        # degrade to an unfiltered search, never crash the question
        date_from = date_to = ""
    prefer_recent = bool((plan or {}).get("prefer_recent", True))

    emit("status", "Searching your messages…")
    # the question's own distinctive words are the highest-precision matches
    # ("front", "10k"); the planner's expansions widen recall around them.
    # Union both -- expansions must never REPLACE the literals.
    literal_terms = query_terms(question)
    plan_terms = (plan or {}).get("search_terms") or []
    seen_terms = {t.lower() for t in literal_terms}
    terms = literal_terms + [t for t in plan_terms if t.lower() not in seen_terms]
    terms = terms[:14]
    scoped = bool(contact or plan_ids)
    if person_focus and handles:
        person_keys = {key for key in (_canon_handle(h) for h in handles) if key}
        person_names = resolve_contact_names(handles)
        name_tokens = sorted(
            {tok for nm in person_names.values() for tok in re.findall(r"[a-z]{4,}", str(nm).lower())}
        )
        # the deepest pool: person questions are the heavy recall class, and
        # the target moment routinely sits in the mid-tier of a noisy scope
        cap = 30
        raw_matches = _person_messages(df, person_keys, name_tokens, cap * 2, terms=terms)
    elif person_focus and scoped:
        cap = max(1, min(max(int(limit or 8), 24), 30))
        raw_matches = _identity_messages(df, cap * 2, terms=terms)
    else:
        # an identity question with no scoped conversation must not harvest
        # "I'm/my name is" lines from the whole archive -- fall back to terms.
        # AI asks get a deeper pool: the answer model reads context windows,
        # and recall questions live or die on the right moment making the cut
        floor = 18 if use_ai else 0
        cap = max(1, min(max(int(limit or 8), floor), 20))
        raw_matches = _select_messages(df, terms, cap * 2, prefer_recent=prefer_recent)
    # one long exchange must not hog the cap: overfetch above, then keep at
    # most two rows per moment so distinct moments fill the freed slots
    moment_chat, moment_pos = _moment_positions(df)
    matches = _dedupe_moments(raw_matches, moment_chat, moment_pos).head(cap)

    # Semantic recall: blend in windows that match the question's MEANING, so
    # paraphrases surface even when no keyword hits ("falling out" finds the
    # fight nobody called a fight). Skips silently when the index is missing,
    # stale, or the embeddings call fails.
    if use_ai and not df.empty:
        try:
            index = get_semantic_index(messages_path)
        except Exception:
            index = None
        if index is not None:
            emit("status", "Searching by meaning…")
            try:
                query_text = " ".join([str(question or "")] + list(terms or [])[:6])
                query_vec = sem.embed_query(query_text)
                scope_ids = (
                    set(df["chat_id"].dropna().astype(str).unique())
                    if (contact or plan_ids)
                    else None
                )
                hits = index.search(
                    query_vec, k=24, chat_ids=scope_ids,
                    date_from=date_from, date_to=date_to,
                )
                anchor_ids = [h["anchor_id"] for h in hits if h["anchor_id"]]
                if anchor_ids:
                    sem_rows = df[df["message_id"].astype(str).isin(anchor_ids)].copy()
                    if not sem_rows.empty:
                        # keep the index's score order -- df row order would
                        # arbitrarily discard the best-ranked windows when
                        # trimming below
                        rank = {aid: i for i, aid in enumerate(anchor_ids)}
                        sem_rows["_rank"] = sem_rows["message_id"].astype(str).map(rank)
                        sem_rows = sem_rows.sort_values("_rank").drop(columns=["_rank"])
                        # semantic earns its slots by covering moments the
                        # keyword side MISSED -- its top ranks usually re-cite
                        # exchanges the bursts already found. Novel windows
                        # APPEND to the keyword picks instead of evicting
                        # their tail: the freshest mid-tier keyword moments
                        # are exactly where recall targets live, and a few
                        # extra context windows cost tokens, not correctness
                        novel = _dedupe_moments(
                            sem_rows, moment_chat, moment_pos,
                            per_moment=1, seed=matches,
                        ).head(8)
                        if not novel.empty:
                            matches = pd.concat([matches, novel]).drop_duplicates(
                                subset=["message_id"]
                            )
            except Exception as exc:
                print(f"[ask] semantic search failed: {exc}", file=sys.__stderr__)

    name_map = resolve_contact_names(matches["chat_id"].dropna().astype(str).unique()) if not matches.empty else {}
    for cid in set(matches["chat_id"].dropna().astype(str)) if not matches.empty else set():
        if not name_map.get(cid) and catalog_names.get(cid):
            name_map[cid] = catalog_names[cid]
    windows = (
        # semantic blend may append past cap -- window every match we kept
        build_context_windows(df, matches, cap=max(cap, len(matches)))
        if (use_ai and not matches.empty)
        else []
    )

    # resolve names for everyone who speaks in the matches or the surrounding windows
    sender_handles: set[str] = set()
    if not matches.empty and "sender" in matches.columns:
        sender_handles.update(s for s in matches["sender"].dropna().astype(str) if s)
    for window in windows:
        if "sender" in window["rows"].columns:
            sender_handles.update(s for s in window["rows"]["sender"].dropna().astype(str) if s)
    sender_map = resolve_contact_names(list(sender_handles)) if sender_handles else {}

    citations = [message_row_payload(row, name_map, sender_map) for _, row in matches.iterrows()]
    window_context = format_context_windows(windows, name_map, sender_map) if windows else ""
    scope_label = (
        contact_label({"chatId": contact, "displayName": name_map.get(contact, "") or catalog_names.get(contact, "")})
        if contact
        else "All conversations"
    )
    answer_mode = "local"
    answer = ""
    if use_ai:
        try:
            if windows:
                emit("status", f"Reading the conversation around {len(windows)} moments…")
            answer = ai_ask_answer(
                str(question or "").strip(),
                citations,
                scope_label,
                model,
                intent=intent,
                context=window_context,
                history=history,
                on_delta=(lambda text: emit("delta", text)) if on_event is not None else None,
            )
            if answer:
                answer_mode = "ai"
                answer, citations = _renumber_citations(answer, citations)
        except Exception as exc:
            print(f"[ask] AI answer failed: {exc}", file=sys.__stderr__)
            answer = ""
    if not answer:
        answer = local_ask_answer(terms, citations)

    return {
        "question": str(question or "").strip(),
        "contact": contact,
        "terms": terms,
        "answer": answer,
        "mode": answer_mode,
        "citations": citations,
    }


class RecallHandler(BaseHTTPRequestHandler):
    server_version = "RecallUI/1.0"

    def log_message(self, format, *args):
        sys.__stderr__.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )

    def do_OPTIONS(self):
        # no CORS grant: the UI is same-origin, so a preflight can only come
        # from a cross-origin page probing the archive
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                if not is_local_api_request(self):
                    send_json(self, 403, {"error": "Forbidden: non-local origin"})
                    return
                self.handle_api_get(parsed)
            else:
                self.serve_static(parsed.path)
        except Exception as exc:
            send_json(self, 500, {"error": str(exc)})

    def do_HEAD(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                send_json(self, 405, {"error": "Method not allowed"})
            else:
                self.serve_static(parsed.path, head_only=True)
        except Exception as exc:
            send_json(self, 500, {"error": str(exc)})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if not is_local_api_request(self):
                send_json(self, 403, {"error": "Forbidden: non-local origin"})
                return
            if parsed.path == "/api/jobs":
                payload = parse_body(self)
                if not isinstance(payload, dict):
                    send_json(self, 400, {"error": "The request must be a JSON object."})
                    return
                try:
                    job = create_job(payload.get("action", "analyze"), payload)
                except JobRefused as refused:
                    send_json(self, refused.status, {"error": str(refused)})
                    return
                send_json(self, 202, {"job": job})
            elif parsed.path == "/api/contact-names":
                try:
                    summary = export_contact_names()
                except PermissionError as exc:
                    send_json(self, 403, {"error": str(exc), "permission": "contacts"})
                    return
                send_json(self, 200, {
                    "contactNames": summary,
                    "message": summary.get("message") or f"Loaded {summary.get('count', 0):,} contacts.",
                })
            elif parsed.path == "/api/cache/preview":
                with PREVIEW_CACHE_LOCK:
                    PREVIEW_CACHE.clear()
                clear_cache_namespace(PREVIEW_CACHE_NAMESPACE)
                send_json(self, 200, {"cleared": True})
            elif parsed.path == "/api/setup/complete":
                send_json(self, 200, {"setup": write_setup_marker(setup_marker_updates(parse_body(self)))})
            elif parsed.path == "/api/ask":
                payload = parse_body(self)
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                raw_history = payload.get("history")
                send_json(self, 200, ask_messages_payload(
                    messages_path,
                    payload.get("question", ""),
                    contact=payload.get("contact", ""),
                    limit=int(payload.get("limit", 8) or 8),
                    model=payload.get("model", DEFAULT_MODEL),
                    use_ai=str(payload.get("ai", True)).lower() not in {"0", "false", "no", "off"},
                    history=raw_history if isinstance(raw_history, list) else None,
                ))
            elif parsed.path == "/api/ask/stream":
                payload = parse_body(self)
                self.handle_ask_stream(payload)
            else:
                send_json(self, 404, {"error": "Not found"})
        except Exception as exc:
            send_json(self, 500, {"error": str(exc)})

    def handle_ask_stream(self, payload: dict):
        """Server-sent events for /api/ask: status updates while retrieving,
        answer text deltas as the model writes, then the full payload."""
        messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
        raw_history = payload.get("history")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def write_event(data: dict) -> None:
            frame = f"data: {json.dumps(data, default=json_default)}\n\n"
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

        try:
            result = ask_messages_payload(
                messages_path,
                payload.get("question", ""),
                contact=payload.get("contact", ""),
                limit=int(payload.get("limit", 8) or 8),
                model=payload.get("model", DEFAULT_MODEL),
                use_ai=str(payload.get("ai", True)).lower() not in {"0", "false", "no", "off"},
                history=raw_history if isinstance(raw_history, list) else None,
                on_event=lambda kind, text: write_event({"type": kind, "text": text}),
            )
            write_event({"type": "done", "payload": result})
        except (BrokenPipeError, ConnectionResetError):
            # client navigated away or cancelled; nothing left to tell it
            pass
        except Exception as exc:
            print(f"[ask] stream failed: {exc}", file=sys.__stderr__)
            with contextlib.suppress(Exception):
                write_event({"type": "error", "error": str(exc)})

    def handle_api_get(self, parsed):
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.path == "/api/defaults":
            # where the archive and the reports are. No database is named: an
            # update reads Messages itself (database_for).
            messages_path = DEFAULT_MESSAGES_CSV
            send_json(self, 200, {
                "defaultModel": DEFAULT_MODEL,
                "models": list(UI_MODEL_CHOICES),
                "messagesPath": str(messages_path),
                "outDir": str(OUT_DIR),
                "hasMessages": messages_path.exists(),
                "reports": list_report_files(),
                "contactNames": contacts_cache_summary(),
                "setupCompleted": bool(read_setup_marker().get("completed")),
            })
        elif parsed.path == "/api/setup/status":
            db_param = query.get("dbPath")
            send_json(self, 200, setup_status_payload(
                safe_path(db_param) if db_param else None,
                safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV),
                deep=str(query.get("deep", "0")).lower() in {"1", "true", "yes"},
            ))
        elif parsed.path == "/api/first-words":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            send_json(self, 200, {"firstWords": first_words_payload(messages_path)})
        elif parsed.path == "/api/permissions/fulldisk":
            # the onboarding wizard polls this while the user flips Full Disk
            # Access; probing the live chat.db is the only reliable signal
            live_db = Path("~/Library/Messages/chat.db").expanduser()
            status = "missing"
            if live_db.exists():
                try:
                    conn = open_read_only(live_db)
                    try:
                        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
                    finally:
                        conn.close()
                    status = "granted"
                except sqlite3.OperationalError:
                    status = "denied"
            send_json(self, 200, {"status": status, "dbPath": str(live_db)})
        elif parsed.path == "/api/semantic":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            status = sem.index_status(SEMANTIC_DIR, file_cache_signature(messages_path))
            if path_exists(messages_path):
                df, _, _ = cached_messages_bundle(messages_path)
                status["estimate"] = sem.estimate_build(df)
            send_json(self, 200, {"semantic": status})
        elif parsed.path == "/api/memories":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            send_json(self, 200, {"memories": memories_payload(messages_path)})
        elif parsed.path == "/api/overview":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            send_json(self, 200, {"overview": archive_patterns(messages_path)})
        elif parsed.path == "/api/contacts":
            # The names of groups, and the list itself before there is an
            # archive, come from Messages itself. The client names no database.
            db_path = LIVE_CHAT_DB
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            limit = int(query.get("limit", "40"))
            if path_exists(messages_path):
                df = contacts_from_messages(messages_path, limit=limit, db_path=db_path)
                send_json(self, 200, {
                    "contacts": df.to_dict(orient="records"),
                    "source": "messages",
                    "messagesPath": str(messages_path),
                    "contactNameCount": count_display_names(df),
                    "archive": archive_summary(messages_path),
                })
                return
            probe = probe_database(db_path)
            if why_unreadable(probe, archive_there=False):
                send_json(self, 400, {"error": f"There is no archive at {messages_path}, and Messages cannot be read."})
                return
            try:
                listed = list_contacts(str(db_path), limit=limit)
            except sqlite3.Error as exc:
                # a busy database can be asked again; the reason is a plain sentence either way
                stopped = reading_stopped(exc, probe, archive_there=False)
                send_json(self, 503 if isinstance(stopped, DatabaseBusy) else 500, {"error": str(stopped)})
                return
            df = decorate_contacts_frame(listed)
            send_json(self, 200, {
                "contacts": df.to_dict(orient="records"),
                "source": "database",
                "contactNameCount": count_display_names(df),
            })
        elif parsed.path == "/api/preview":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            if not path_exists(messages_path):
                send_json(self, 400, {"error": f"Messages CSV not found: {messages_path}"})
                return
            payload, cached = cached_preview_payload(
                messages_path,
                query.get("contact", ""),
                query.get("model", DEFAULT_MODEL),
                since=query.get("since"),
                until=query.get("until"),
            )
            payload = dict(payload)
            payload["cached"] = cached
            send_json(self, 200, payload)
        elif parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                send_json(self, 404, {"error": "Job not found"})
                return
            send_json(self, 200, {"job": job})
        elif parsed.path == "/api/jobs":
            send_json(self, 200, {"jobs": list_jobs()})
        elif parsed.path == "/api/reports":
            send_json(self, 200, {"reports": list_report_files()})
        elif parsed.path == "/api/day":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            contact = query.get("contact", "")
            if not contact:
                send_json(self, 400, {"error": "A conversation is needed"})
                return
            if not path_exists(messages_path):
                send_json(self, 400, {"error": f"Messages CSV not found: {messages_path}"})
                return
            try:
                place = query.get("offset", "")
                payload = day_payload(
                    messages_path,
                    contact,
                    query.get("day", ""),
                    around=query.get("around") or None,
                    offset=int(place) if place != "" else None,
                    limit=int(query.get("limit", DAY_LIMIT)),
                )
            except ValueError as error:
                send_json(self, 400, {"error": str(error)})
                return
            send_json(self, 200, payload)
        elif parsed.path == "/api/search":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            send_json(self, 200, search_messages_payload(
                messages_path,
                query=query.get("query", ""),
                contact=query.get("contact", ""),
                limit=int(query.get("limit", "60")),
            ))
        elif parsed.path == "/api/analysis":
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            report_path = safe_path(query.get("reportPath", ""))
            if not path_exists(messages_path):
                send_json(self, 400, {"error": f"Messages CSV not found: {messages_path}"})
                return
            if not report_path or not report_path.exists() or not ensure_under_root(report_path):
                send_json(self, 404, {"error": "Report not found"})
                return
            if report_path.suffix.lower() == ".html":
                markdown_path = report_path.with_suffix(".md")
                if markdown_path.exists():
                    report_path = markdown_path
            contact = query.get("contact") or report_contact(report_path)
            events_path, html_path = companion_report_paths(report_path, contact)
            payload, cached = cached_analysis_payload(
                messages_path,
                contact,
                query.get("model", DEFAULT_MODEL),
                report_path=str(report_path),
                events_path=str(events_path) if events_path else None,
                html_path=str(html_path) if html_path else None,
            )
            send_json(self, 200, {"analysis": payload, "cached": cached})
        elif parsed.path == "/api/report":
            file_path = served_report_file(query.get("path", ""))
            if not file_path:
                send_json(self, 404, {"error": "Report not found"})
                return
            data = file_path.read_bytes()
            self.send_response(200)
            for name, value in report_file_headers(file_path).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            send_json(self, 404, {"error": "Not found"})

    def serve_static(self, path: str, head_only: bool = False):
        static_root = selected_static_root()
        request_path = unquote(path)
        if request_path in {"", "/"}:
            file_path = static_root / "index.html"
        else:
            file_path = (static_root / request_path.lstrip("/")).resolve()
        if not ensure_under(file_path, static_root) or not file_path.exists() or not file_path.is_file():
            fallback = static_root / "index.html"
            path_name = Path(request_path).name
            if static_root == REACT_DIST_DIR and "." not in path_name and fallback.exists():
                file_path = fallback
            else:
                self.send_error(404)
                return
        if not ensure_under(file_path, static_root):
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        if not head_only:
            self.wfile.write(data)


def _retarget_streams_to_logfile() -> None:
    """The frozen (windowed) build gets /dev/null stdio from the bootloader,
    which would silence every print, request log, and traceback. Route all of
    it to the same log the Mac shell points users at."""
    log_dir = Path.home() / "Library" / "Logs" / "Recall"
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = open(
        log_dir / "RecallBackend.log", "a", buffering=1, encoding="utf-8", errors="replace"
    )
    sys.stdout = stream
    sys.stderr = stream
    sys.__stdout__ = stream
    sys.__stderr__ = stream
    _JOB_STDOUT._fallback = stream
    _JOB_STDERR._fallback = stream


def main():
    import signal
    import time

    from recall_paths import BUNDLED

    if BUNDLED:
        _retarget_streams_to_logfile()
    ensure_data_dirs()
    host = os.environ.get("RECALL_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("RECALL_UI_PORT", "8765"))
    server = LoopbackHTTPServer((host, port), RecallHandler)

    # clean teardown when the shell (or Sparkle relaunch) terminates us
    signal.signal(
        signal.SIGTERM,
        lambda *_: threading.Thread(target=server.shutdown, daemon=True).start(),
    )
    if BUNDLED:
        # a crashed shell must never leave an orphan listener holding the archive
        def watch_parent() -> None:
            while os.getppid() != 1:
                time.sleep(2.0)
            os._exit(0)

        threading.Thread(target=watch_parent, daemon=True).start()

    print(f"Recall UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Recall UI")


if __name__ == "__main__":
    main()
