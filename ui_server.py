#!/usr/bin/env python3
"""Local web UI server for Recall."""

import contextlib
import csv
import io
import json
import logging
import mimetypes
import os
import re
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
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from archive import Archive, ArchiveError, ArchiveNotFoundError
from chat_engine.cancel import CancelToken
from contact_names import (
    CONTACTS_CACHE_PATH,
    contacts_cache_summary,
    decorate_contacts_frame,
    export_contact_names,
)
from local_store import clear_cache_namespace, get_cache_entry, set_cache_entry, stable_cache_key
from conversation import (
    activity_series,
    compute_stats,
    day_totals,
    direction_word,
    fill_months,
    filter_conversation,
    group_chat_mask,
    hour_by_weekday,
    is_group_chat,
    month_points,
    parse_boolish,
    progression_series,
    received_mask,
    sanitize_filename,
    sent_mask,
    within_dates,
)
from group_people import group_speakers
from message_text import ATTACHMENT_PLACEHOLDER, UNSENT_LABEL, attachment_label, clip_text, display_text
from parse_imessage import list_contacts
from safe_html import story_page
from recall_paths import (
    DATA_DIR,
    SAVES_DIR,
    OUT_DIR,
    DEFAULT_MESSAGES_CSV,
    ensure_data_dirs,
)
from recall_services import chat as chat_service
from recall_services import conversations as conversation_service
from recall_services import day as day_service
from recall_services import digest_job as digest_service
from recall_services import frame as frame_service
from recall_services import inputs, models, places
from recall_services import meaning as meaning_service
from recall_services import memory as memory_service
from recall_services import names as names_service
from recall_services import reports as report_service
from recall_services import search as search_service
from recall_services.archive_state import DEFAULT_MAX_REMOVED_SHARE, ArchiveState
from recall_services.models import ModelChoiceError
from recall_services.outcomes import PullBusy, PullDone, PullRefused
from recall_services.unreadable import (  # noqa: F401 - the words of an import, under the names the server always had
    ARCHIVE_KEPT,
    DatabaseBusy,
    ImportStopped,
    reading_stopped,
    why_unreadable,
)
from setup_status import LIVE_CHAT_DB, DbProbe, derive_state, probe_database

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
REACT_DIST_DIR = ROOT / "app" / "dist"
JOBS = {}
JOBS_LOCK = threading.Lock()
# How the last `digest` job of an archive, a conversation and days ended, when a new job would
# read nothing more: by (archive path, conversation, since, until). `JOBS_LOCK` guards it.
# It is kept until an import changes that archive. It does not outlive the server.
DIGEST_ENDS: dict[tuple, str] = {}
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


class BadRequest(ValueError):
    """A request that cannot be read. The route answers 400 with the sentence."""


# no request of the app comes near this size
MAX_BODY_BYTES = 2_000_000
SERVER_FAULT = "Recall could not answer. The log of the server says why."
BAD_ARCHIVE_FILE = "The archive file that was named cannot be used."


def parse_body(handler) -> dict:
    try:
        length = int(handler.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise BadRequest("The request is too large.")
        return json.loads(handler.rfile.read(length).decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        # a length that is no number, bytes that are not UTF-8, text that is not JSON, JSON nested too deep
        if isinstance(exc, BadRequest):
            raise
        raise BadRequest("The request body must be JSON.") from exc


def server_fault(where: str) -> dict:
    """The answer to a request that failed in a way nobody planned for.

    What was raised goes to the log of the server. It can hold a path or the
    text of a query, so it does not go to the page.
    """
    with contextlib.suppress(Exception):
        print(f"[server] {where} failed", file=sys.__stderr__)
        traceback.print_exc(file=sys.__stderr__)
    return {"error": SERVER_FAULT, "code": "server_error"}


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
            keep_digest_end(JOBS[job_id])


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
    saved = saved_names()
    return [{**report, "displayName": saved.saved(report["contact"])} for report in reports]


def companion_report_paths(report_path: Path, contact: str) -> tuple[Path | None, Path | None]:
    stem = sanitize_filename(contact) if contact else report_path.stem.removeprefix("analysis_")
    out_dir = report_path.parent
    events_path = out_dir / f"events_timeline_{stem}.csv"
    html_path = out_dir / f"analysis_{stem}.html"
    return (
        events_path if events_path.exists() else None,
        html_path if html_path.exists() else None,
    )


BUILDING = "Recall is getting your archive ready. This takes a moment."
NO_ARCHIVE = "There is no archive yet. Open the setup guide to read your messages."
NOT_AN_ARCHIVE = "The archive file could not be opened. Move that file away, then set up Recall again."
ALREADY_RUNNING = "An import is already running. Wait until it ends."
NO_EXPORT_TO_LOAD = "This archive has no earlier export to load."
# A load of an export that did not end well, by archive file: the state of
# the export then, what was said, and until when that holds (None: until the
# export changes). A request does not start the load again while this holds.
BRIDGE_FAILED: dict[str, tuple[tuple | None, str, float | None]] = {}
# another writer has the archive: the load is tried again after this long
BRIDGE_RETRY_SECONDS = 15.0
EXPORT_NOT_LOADED = "The earlier export could not be loaded into the archive."


class ArchiveNotReady(Exception):
    """A request that needs the archive when there is none to read.

    `status` and `code` are the answer to the request. `job` is the job that
    is filling the archive, when one runs.
    """

    def __init__(self, sentence: str, status: int, code: str, job: dict | None = None) -> None:
        super().__init__(sentence)
        self.status = status
        self.code = code
        self.job = job

    def body(self) -> dict:
        said = {"error": str(self), "code": self.code}
        return {**said, "job": self.job} if self.job else said


def saved_names() -> names_service.Names:
    """The saved names, from the Contacts cache only. No request reads the Contacts database of the Mac."""
    return names_service.cached(CONTACTS_CACHE_PATH)


def contact_records() -> list | None:
    """The records of the Contacts cache for an import, or None to keep the people that the archive has."""
    return names_service.records(CONTACTS_CACHE_PATH)


def named_archive(payload_or_query: dict) -> Path:
    """The archive file that a request names, or the one of the app.

    Raises BadRequest for a name that is not text, that no system can hold,
    or that names no file (a folder root).
    """
    named = payload_or_query.get("messagesPath")
    if named is not None and not isinstance(named, str):
        raise BadRequest(BAD_ARCHIVE_FILE)
    try:
        path = safe_path(named, DEFAULT_MESSAGES_CSV)
        usable = bool(path.name) and "\x00" not in str(path)
    except (OSError, ValueError, RuntimeError):
        usable = False
    if not usable:
        raise BadRequest(BAD_ARCHIVE_FILE)
    return path


def export_holds_rows(export: Path | None) -> bool:
    """True when an export of an earlier build holds a message. No file holds none, and neither does a file with no row.

    A file that cannot be read is not taken for an export with no message:
    what cannot be looked at is not written over from a copy.
    """
    if export is None:
        return False
    try:
        with open(export, newline="", encoding="utf-8") as held:
            rows = csv.reader(held)
            next(rows, None)
            return next(rows, None) is not None
    except FileNotFoundError:
        return False
    except (OSError, ValueError, csv.Error):
        return True


def archive_there(messages_path: Path | None) -> bool:
    """True when the app has something to read at this place: the archive, or the export that it is loaded from."""
    return places.state_for(messages_path).exists() or export_holds_rows(places.export_for(messages_path))


def holds_messages(messages_path: Path | None) -> bool:
    """True when the archive at this place holds a message, or an export waits to be loaded into it.

    An archive that cannot be opened is not taken for an empty one: what
    cannot be looked at is not written over from a copy.
    """
    try:
        return holds_rows(places.state_for(messages_path).archive())
    except ArchiveNotFoundError:
        return export_holds_rows(places.export_for(messages_path))
    except (ArchiveError, sqlite3.Error):
        return True


def database_for(action: str, payload: dict) -> Path | None:
    """The database a job reads, if it reads one.

    An archive that holds messages is updated from the database Messages
    keeps on this Mac, and from nothing else. A path the client sends does
    not change that, and neither does a copy in the data folder: an import
    that names a copy for such an archive is stopped (ImportStopped).

    Only a first setup can name a copy: an import into an archive that holds
    no message yet. The copy is read for that import only. No other job reads
    a database.
    """
    if action == "export":
        return LIVE_CHAT_DB
    if action == "setup_import":
        named = payload.get("dbPath")
        if named and not isinstance(named, str):
            raise ValueError("The path of the copy must be text.")
        if named and holds_messages(named_archive(payload)):
            raise ImportStopped(f"An update reads Messages on this Mac. It does not read a copy. {ARCHIVE_KEPT}")
        return safe_path(named, LIVE_CHAT_DB)
    return None


def forget_previews() -> None:
    """Drop every preview that the server keeps. After an import, none of the earlier archive is shown."""
    with PREVIEW_CACHE_LOCK:
        PREVIEW_CACHE.clear()
    clear_cache_namespace(PREVIEW_CACHE_NAMESPACE)
    with ANALYSIS_CACHE_LOCK:
        ANALYSIS_CACHE.clear()
    clear_cache_namespace(ANALYSIS_CACHE_NAMESPACE)


def unfinished_import() -> dict | None:
    """The import that runs now, as a copy, or None."""
    with JOBS_LOCK:
        running = [job for job in JOBS.values() if job["action"] in IMPORT_ACTIONS and job["status"] in UNFINISHED]
        return job_copy(running[0]) if running else None


def archive_for(messages_path: Path | None, wait_for_import: bool = True) -> tuple[ArchiveState, Archive]:
    """The archive that a request reads, with its state. Raises ArchiveNotReady when there is none to read.

    When there is no archive and an export of an earlier build is there, the
    load of that export is started here, and the request is told to wait. A
    request also waits while a first import fills an archive that holds
    nothing yet. An import that counts what it wrote does not wait for itself
    (`wait_for_import=False`).
    """
    state = places.state_for(messages_path)
    try:
        archive = state.archive()
    except ArchiveNotFoundError:
        raise why_no_archive(messages_path, state) from None
    except (ArchiveError, sqlite3.Error, OSError, ValueError) as exc:
        with contextlib.suppress(Exception):
            print(f"[archive] the archive could not be opened: {type(exc).__name__}", file=sys.__stderr__)
        raise ArchiveNotReady(NOT_AN_ARCHIVE, 500, "archive_failed") from exc
    filling = unfinished_import() if wait_for_import else None
    if filling and not holds_rows(archive):
        # A first import writes into the archive file itself. Until it ends there is
        # nothing to show: the request waits for the import, as it waits for a load.
        raise ArchiveNotReady(BUILDING, 503, "archive_building", filling)
    return state, archive


def holds_rows(archive: Archive) -> bool:
    """True when the archive holds a message that a screen can show, or one in no conversation."""
    summary = archive.summary()
    return bool(summary["messages"] or summary["messages_without_conversation"])


def why_no_archive(messages_path: Path | None, state: ArchiveState) -> ArchiveNotReady:
    """What a request is told when there is no archive: wait for the load, or that there is nothing to read."""
    export = places.export_for(messages_path)
    if not export_holds_rows(export):
        running = unfinished_import()
        if running:
            return ArchiveNotReady(BUILDING, 503, "archive_building", running)
        return ArchiveNotReady(NO_ARCHIVE, 400, "no_archive")
    failed = bridge_failure(state, export)
    if failed:
        return ArchiveNotReady(failed, 500, "archive_failed")
    try:
        job = create_job("bridge", {"messagesPath": str(messages_path or DEFAULT_MESSAGES_CSV)})
    except JobRefused:
        job = unfinished_import()
    return ArchiveNotReady(BUILDING, 503, "archive_building", job)


def bridge_failure(state: ArchiveState, export: Path) -> str:
    """What the last load of this export said when it did not end well, or "" when a load can be tried."""
    failed = BRIDGE_FAILED.get(str(state.path))
    if not failed or failed[0] != file_cache_signature(export):
        return ""
    if failed[2] is not None and time.monotonic() >= failed[2]:
        return ""
    return failed[1]


def note_bridge_failure(state: ArchiveState, export: Path, sentence: str, again_after: float | None = None) -> None:
    until = None if again_after is None else time.monotonic() + again_after
    BRIDGE_FAILED[str(state.path)] = (file_cache_signature(export), sentence, until)


def load_export_at_start() -> None:
    """At the start of the server: one job loads the export of an earlier build, when the archive does not hold it yet."""
    export = places.export_for(None)
    try:
        waits = export_holds_rows(export) and places.state_for(None).export_waits(export)
    except Exception:  # noqa: BLE001 - a start with an archive that cannot be looked at is still a start
        return
    if waits:
        with contextlib.suppress(JobRefused):
            create_job("bridge", {"messagesPath": str(DEFAULT_MESSAGES_CSV)})


def load_export(job_id: str, payload: dict) -> dict:
    """The `bridge` job: load the export of an earlier build into the archive. The export stays where it is."""
    messages_path = named_archive(payload)
    state, export = places.state_for(messages_path), places.export_for(messages_path)
    if export is None:
        raise ImportStopped(NO_EXPORT_TO_LOAD)
    update_job(job_id, progress={"step": 1, "of": 2, "label": "Reading your earlier export"})
    append_log(job_id, "Loading the earlier export into the archive")
    try:
        answer = state.ensure_from_export(export, contacts=contact_records())
    except Exception:
        # whatever stopped the load, the next request does not start it again for the same export
        note_bridge_failure(state, export, EXPORT_NOT_LOADED)
        raise
    if answer.status in ("busy", "failed", "cancelled", "no_export"):
        said = answer.reason or EXPORT_NOT_LOADED
        # a busy archive is tried again after a moment; any other end holds until the export changes
        note_bridge_failure(state, export, said, BRIDGE_RETRY_SECONDS if answer.status == "busy" else None)
        raise ImportStopped(said)
    BRIDGE_FAILED.pop(str(state.path), None)
    update_job(job_id, progress={"step": 2, "of": 2, "label": "Getting your archive ready"})
    if answer.changed:
        with contextlib.suppress(Exception):
            forget_previews()
    return {**what_the_archive_holds(messages_path), "status": answer.status}


def pull_into_archive(action: str, payload: dict) -> PullDone:
    """One update: a full pull of the database into the archive. Raises ImportStopped when the archive did not change.

    `database_for` names the database. The limit on what a pull removes
    guards an archive that holds a message. Only a request that says so
    (`allowRemoval`) lifts it.
    """
    db_path = database_for(action, payload)
    place = named_archive(payload)
    state = places.state_for(place)
    # An archive that holds no message has nothing that a pull can lose: the limit guards an update only.
    guarded = payload.get("allowRemoval") is not True and holds_messages(place)
    limit = DEFAULT_MAX_REMOVED_SHARE if guarded else 1.0
    outcome = state.pull_from_messages(db_path, contact_records(), max_removed_share=limit)
    if isinstance(outcome, PullDone):
        # the archive is written: nothing after this is a reason to say that the import failed
        with contextlib.suppress(Exception):
            forget_previews()
        return outcome
    if isinstance(outcome, PullRefused):
        raise ImportStopped(outcome.reason, code="removal_limit")
    if isinstance(outcome, PullBusy):
        raise ImportStopped(ALREADY_RUNNING)
    detail = getattr(outcome, "detail", "")
    if detail:
        # what led to the sentence is for the log of the server, not for the page
        with contextlib.suppress(Exception):
            print(f"[import] {detail}", file=sys.__stderr__)
    raise ImportStopped(outcome.reason)


JOB_ACTIONS = frozenset({"analyze", "export", "setup_import", "bridge", "meaning", "digest"})
# each of these writes the archive
IMPORT_ACTIONS = frozenset({"setup_import", "export", "bridge"})
# this one reads the whole archive into the index for the search by meaning
INDEX_ACTIONS = frozenset({"meaning"})
INDEX_IS_BUILT = "The index for the search by meaning is being built. Wait until it ends."
IMPORT_FIRST = "An import is running. Wait until it ends, then build the index."
# this one reads one conversation for chat, outside a question
DIGEST_ACTIONS = frozenset({"digest"})
DIGEST_IS_READ = "Recall is reading a conversation for chat. Wait until it ends."
IMPORT_BEFORE_DIGEST = "An import is running. Wait until it ends, then ask again."
NO_CONVERSATION_TO_READ = "A conversation is needed to read it for chat."
UNFINISHED = frozenset({"queued", "running"})
# Jobs that do not run together: the job that is asked for, the jobs that
# hold it up, and what the reader is told.
ONE_AT_A_TIME = (
    (IMPORT_ACTIONS, IMPORT_ACTIONS, ALREADY_RUNNING),
    # the index is built from the whole archive: no import writes the archive meanwhile
    (IMPORT_ACTIONS | INDEX_ACTIONS, INDEX_ACTIONS, INDEX_IS_BUILT),
    (INDEX_ACTIONS, IMPORT_ACTIONS, IMPORT_FIRST),
    # one conversation is read for chat at a time, and no import writes the archive meanwhile
    (IMPORT_ACTIONS | DIGEST_ACTIONS, DIGEST_ACTIONS, DIGEST_IS_READ),
    (DIGEST_ACTIONS, IMPORT_ACTIONS, IMPORT_BEFORE_DIGEST),
)


class JobRefused(Exception):
    """A request for a job that makes no job. `status` is the answer to the request."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def usable_path(value, what: str) -> None:
    """Refuse a path that is not text, or that no system can hold. A path that is not given is the default."""
    if not value:
        return
    if not isinstance(value, str):
        raise JobRefused(f"The path of {what} must be text.", 400)
    try:
        safe_path(value)
    except (OSError, ValueError, RuntimeError) as exc:
        raise JobRefused(f"The path of {what} cannot be used.", 400) from exc


def check_request(action, payload: dict) -> None:
    """Refuse a request that cannot be a job. Raises JobRefused, and no job is made."""
    if not isinstance(action, str):
        raise JobRefused("The action must be text.", 400)
    if action not in JOB_ACTIONS:
        raise JobRefused(f"Unknown job action: {action}", 400)
    usable_path(payload.get("messagesPath"), "the archive")
    usable_path(payload.get("outDir"), "the reports folder")
    for name in ("contact", "conversation", "model", "since", "until"):
        if payload.get(name) is not None and not isinstance(payload.get(name), str):
            raise JobRefused(f"{name} must be text.", 400)
    if action == "setup_import":
        usable_path(payload.get("dbPath"), "the copy")
        if not isinstance(payload.get("includeContacts", True), bool):
            raise JobRefused("includeContacts must be true or false.", 400)
        try:
            # only a first setup can name a copy
            database_for(action, payload)
        except ImportStopped as stopped:
            raise JobRefused(str(stopped), 409) from stopped


def refuse_a_second(action: str) -> None:
    """Refuse a job while a job that holds it up is not ended. The caller holds JOBS_LOCK."""
    unfinished_actions = {job["action"] for job in JOBS.values() if job["status"] in UNFINISHED}
    for asked, held_up_by, sentence in ONE_AT_A_TIME:
        if action in asked and unfinished_actions & held_up_by:
            raise JobRefused(sentence)


def job_record(action: str, payload: dict) -> dict:
    """A job as it is listed: what it does, for which conversation if any, and how far along it is."""
    now = datetime.now().isoformat(timespec="seconds")
    record = {
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
        # a fixed word for the client beside the sentence in `error`, when a failed job has one
        "code": None,
    }
    if action not in DIGEST_ACTIONS:
        return record
    # a `digest` job also says which archive, which conversation and which days it reads
    return {
        **record,
        "contact": str(payload.get("conversation") or payload.get("contact") or ""),
        "archivePath": archive_named(payload),
        "since": payload.get("since") or None,
        "until": payload.get("until") or None,
    }


def archive_named(payload: dict) -> str:
    """The resolved path of the archive file that a request names, as text. "" for a name that cannot be used."""
    try:
        return str(places.archive_path_for(named_archive(payload)).resolve())
    except Exception:  # noqa: BLE001 - a record and a starter are made for any request
        return ""


def digest_key(job: dict) -> tuple:
    """What a `digest` job reads: the archive, the conversation and the days."""
    return (job.get("archivePath", ""), job["contact"], job.get("since"), job.get("until"))


def job_copy(job: dict) -> dict:
    """A job as it is now, for an answer. It shares no list with the job, which changes in its own thread."""
    return {**job, "logs": list(job["logs"])}


def create_job(action: str, payload: dict) -> dict:
    """Make a job and start it. Raises JobRefused when no job is made.

    Whether a second import can run is decided here, once, with the list of
    jobs held. A job is in the list as queued from this moment, so two
    imports asked for together cannot each find the other and both stop.
    """
    check_request(action, payload)
    job = job_record(action, payload)
    job_id = job["id"]
    with JOBS_LOCK:
        refuse_a_second(action)
        JOBS[job_id] = job
        # The answer is the job as it was made. The job itself changes in its
        # own thread, and it can end before the answer is written.
        made = job_copy(job)
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
    return made


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

    # the threads that answer requests: what they print is never a line of a job
    answering: set[int] = set()

    def register(self, writer: JobWriter):
        with self._lock:
            self._writers[threading.get_ident()] = writer

    def unregister(self):
        with self._lock:
            self._writers.pop(threading.get_ident(), None)

    def _target(self):
        with self._lock:
            mine = threading.get_ident()
            writer = self._writers.get(mine)
            if writer is None and len(self._writers) == 1 and mine not in self.answering:
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
    """How far setup got: the three things that can be recorded, each true or
    false. A path picked in an earlier setup is not kept: a pick is for the
    one import it was made for. A value that an earlier build wrote as text
    or as a number says nothing."""
    try:
        marker = json.loads(SETUP_MARKER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(marker, dict):
        return {}
    return {key: marker[key] for key in SETUP_MARKER_KEYS if isinstance(marker.get(key), bool)}


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


def access_to_messages(probe: DbProbe) -> str:
    """Whether macOS lets this app read Messages: granted, denied, or missing when there is no database."""
    if probe.status == "fda_blocked":
        return "denied"
    if probe.status == "missing":
        return "missing"
    return "granted"


def setup_status_payload(db_path: Path | None, messages_path: Path, deep: bool = False) -> dict:
    try:
        probe = probe_database(db_path, deep=deep)
    except Exception as exc:
        # the wizard derives every screen from this -- it must never 500
        probe = DbProbe(status="error", kind="copy", path=str(db_path or ""), detail=str(exc))
    # the app has something to read: the archive, or the export that it is loaded from
    export_exists = archive_there(messages_path)
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
        # holdsMessages is the test for "is this an update": the guide and database_for use the same one
        "export": {"exists": export_exists, "path": str(messages_path), "holdsMessages": holds_messages(messages_path)},
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
            keep_digest_end(job)


def run_job(job_id: str, action: str, payload: dict):
    """Do the work of a job in its thread. When this returns or raises, the job is ended in the list."""
    try:
        work_job(job_id, action, payload)
    finally:
        fail_unfinished(job_id)
    after_job(job_id, action, payload)


def changed_the_archive(job_id: str, action: str) -> bool:
    """True when this job is an import that ended well and changed the archive."""
    if action not in IMPORT_ACTIONS:
        return False
    with JOBS_LOCK:
        job = JOBS.get(job_id) or {}
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        # a pull that ends well changed the archive; a load says whether it did
        return job.get("status") == "completed" and (action != "bridge" or result.get("status") == "built")


def keep_digest_end(job: dict) -> None:
    """Keep how a `digest` job ended when a new job of the same reading would read nothing more.

    That is a job that failed, and a job that read no new part while parts
    failed. The starter of a question then starts no job. Any other end lets
    the reading start again. The caller holds JOBS_LOCK, and calls this with
    the change that ends the job: the next question never sees the job ended
    and no word of how.
    """
    if job["action"] not in DIGEST_ACTIONS or job["status"] in UNFINISHED:
        return
    result = job["result"] if isinstance(job.get("result"), dict) else {}
    nothing_new = result.get("periodsBuilt") == 0 and (result.get("periodsFailed") or 0) > 0
    if job["status"] == "failed" or nothing_new:
        DIGEST_ENDS[digest_key(job)] = job["status"]
    else:
        DIGEST_ENDS.pop(digest_key(job), None)


def after_job(job_id: str, action: str, payload: dict) -> None:
    """What the server does by itself when a job has ended in the list. A fault here does not change how the job ended."""
    try:
        if changed_the_archive(job_id, action):
            archive = archive_named(payload)
            with JOBS_LOCK:
                # the archive holds other messages now: a reading that failed can be tried again
                for key in [key for key in DIGEST_ENDS if key[0] == archive]:
                    del DIGEST_ENDS[key]
    except Exception as exc:  # noqa: BLE001 - the job ended: nothing here is a reason to say otherwise
        with contextlib.suppress(Exception):
            print(f"[jobs] what follows a job did not run: {type(exc).__name__}", file=sys.__stderr__)
    refresh_meaning_index(job_id, action, payload)


WRITTEN_NOT_COUNTED = "The archive is written. Recall could not read it back to count it."


JOB_FAILED = "The job failed. The log of the server says why."
REPORT_NOT_OPENED = "The report is written. Recall could not open it. Look for it in Reports."


class JobStopped(Exception):
    """A job that stops for a reason told to the reader in one plain sentence. `code` is a fixed word for the client."""

    def __init__(self, sentence: str, code: str = "") -> None:
        super().__init__(sentence)
        self.code = code


def report_places(out_dir: Path) -> report_service.ReportPlaces:
    """Where a report run writes: the reports folder that the request names, and the notes in the data folder."""
    usual = report_service.default_places()
    return report_service.ReportPlaces(out_dir=out_dir, store_dir=usual.store_dir, cache_path=usual.cache_path)


def write_report(job_id: str, payload: dict) -> dict:
    """The `analyze` job: a report on one conversation, written from the archive as it is. It reads no database."""
    messages_path = named_archive(payload)
    _state, archive = archive_for(messages_path)
    contact = payload.get("contact")
    if not contact or not isinstance(contact, str):
        raise JobStopped("Select a conversation before you start a report.", "bad_request")
    model = str(payload.get("model") or models.default_id())
    since, until = payload.get("since") or None, payload.get("until") or None
    out_dir = safe_path(payload.get("outDir"), OUT_DIR)

    def told(update: dict) -> None:
        if update.get("progress"):
            update_job(job_id, progress=update["progress"])
        if update.get("log"):
            append_log(job_id, update["log"])

    append_log(job_id, f"Writing a report with {model}")
    try:
        done = report_service.run(
            contact, model, since, until, told, None, archive=archive, names=saved_names(), places=report_places(out_dir)
        )
    except ValueError as exc:
        # a day that is no day, a first day after the last: the service says it in one sentence
        raise JobStopped(str(exc), "bad_request") from exc
    return {**done.as_dict(), **opened_report(job_id, messages_path, contact, model, since, until, done)}


def opened_report(job_id: str, messages_path: Path, contact: str, model: str, since, until, done) -> dict:
    """The written report as the Reports page opens it, and the list of reports.

    The report is written: a failure here is a warning, not a failed job.
    """
    try:
        analysis, _ = cached_analysis_payload(
            messages_path,
            contact,
            model,
            since=since,
            until=until,
            report_path=str(done.report_path),
            events_path=str(done.events_path) if done.events_path else None,
            html_path=None,
        )
        return {"analysis": analysis, "reports": list_report_files()}
    except Exception:  # noqa: BLE001 - the report is on disk, so the run did its work
        append_log(job_id, REPORT_NOT_OPENED)
        with contextlib.suppress(Exception):
            traceback.print_exc(file=sys.__stderr__)
        with contextlib.suppress(Exception):
            return {"analysis": None, "reports": list_report_files(), "warning": REPORT_NOT_OPENED}
        return {"analysis": None, "reports": [], "warning": REPORT_NOT_OPENED}


def names_from_contacts(job_id: str) -> bool:
    """Read the names of Contacts into the cache, for the pull that follows. False when Contacts could not be read."""
    try:
        export_contact_names()
    except Exception:  # noqa: BLE001 - names are a nicety, never a blocker: numbers still work
        append_log(job_id, "Contacts could not be read. Some people show as numbers.")
        with contextlib.suppress(Exception):
            traceback.print_exc(file=sys.__stderr__)
        return False
    return True


def counted(messages_path: Path, pulled: PullDone, job_id: str) -> dict:
    """What an import tells of the archive that it wrote. A count that fails after the pull is a warning."""
    try:
        return what_the_archive_holds(messages_path)
    except Exception:  # noqa: BLE001 - the archive is written, so the import did its work
        append_log(job_id, WRITTEN_NOT_COUNTED)
        with contextlib.suppress(Exception):
            traceback.print_exc(file=sys.__stderr__)
        return {"messages": pulled.messages, "conversations": pulled.conversations, "warning": WRITTEN_NOT_COUNTED}


def import_messages(job_id: str, action: str, payload: dict) -> dict:
    """The `setup_import` job, for a first setup and for an update: names first, then one full pull."""
    db_path = database_for(action, payload)
    contacts_skipped = False
    if payload.get("includeContacts", True) is not False:
        update_job(job_id, progress={"step": 1, "of": 3, "label": "Matching names from Contacts"})
        append_log(job_id, "Matching names from Contacts")
        contacts_skipped = not names_from_contacts(job_id)
    update_job(job_id, progress={"step": 2, "of": 3, "label": "Reading your Messages database"})
    append_log(job_id, "Reading Messages" if db_path == LIVE_CHAT_DB else f"Reading the copy at {db_path}")
    pulled = pull_into_archive(action, payload)
    update_job(job_id, progress={"step": 3, "of": 3, "label": "Getting your archive ready"})
    append_log(job_id, "Warming the archive")
    return {**counted(named_archive(payload), pulled, job_id), "contactsSkipped": contacts_skipped}


def read_conversation(job_id: str, payload: dict) -> dict:
    """The `digest` job: read one conversation for chat, outside a question. Its lines hold labels of parts and counts."""
    _state, archive = archive_for(named_archive(payload))
    conversation = payload.get("conversation")
    if not conversation or not isinstance(conversation, str):
        raise JobStopped(NO_CONVERSATION_TO_READ, "bad_request")
    since, until = payload.get("since") or None, payload.get("until") or None

    def told(update: dict) -> None:
        if update.get("progress"):
            update_job(job_id, progress=update["progress"])
        if update.get("log"):
            append_log(job_id, update["log"])

    try:
        return digest_service.run(conversation, since, until, told, archive=archive, names=saved_names())
    except ValueError as exc:
        # a day that is no day, a first day after the last: the service says it in one sentence
        raise JobStopped(str(exc), "bad_request") from exc


def digest_starter(payload: dict) -> memory_service.StartDigest:
    """The function that reads a long conversation outside one question: it starts a `digest` job.

    It is made for the archive that the question names. It says True when the
    job started, or when a `digest` job of the same archive, conversation and
    days already runs. It says False in every other case, and it never raises.

    It starts nothing when the last job of the same reading failed, or read
    no new part while parts failed (`DIGEST_ENDS`): each question would pay
    for the same parts again. An import that changes the archive lets the
    reading start again.
    """
    named = payload.get("messagesPath")
    archive = archive_named(payload)

    def start(conversation: str, since: str | None, until: str | None) -> bool:
        asked = {"conversation": conversation, "since": since, "until": until}
        key = (archive, conversation, since or None, until or None)
        try:
            with JOBS_LOCK:
                if key in DIGEST_ENDS:
                    return False
            create_job("digest", {**asked, "messagesPath": named} if named else asked)
            return True
        except JobRefused:
            with JOBS_LOCK:
                reading = [job for job in JOBS.values() if job["action"] in DIGEST_ACTIONS and job["status"] in UNFINISHED]
                return any(digest_key(job) == key for job in reading)
        except Exception as exc:  # noqa: BLE001 - a job that cannot be started is a no, never an error of the question
            with contextlib.suppress(Exception):
                print(f"[digest] the reading of a conversation was not started: {type(exc).__name__}", file=sys.__stderr__)
            return False

    return start


def build_meaning_index(job_id: str, payload: dict) -> dict:
    """The `meaning` job: build the index for the search by meaning, or bring it up to date. Its lines hold counts only."""
    _state, archive = archive_for(named_archive(payload))

    def told(update: dict) -> None:
        if update.get("progress"):
            update_job(job_id, progress=update["progress"])
        if update.get("log"):
            append_log(job_id, update["log"])

    try:
        return meaning_service.build(archive, saved_names(), None, told)
    except meaning_service.MeaningFailure as failure:
        raise JobStopped(str(failure), failure.code) from failure


def refresh_meaning_index(job_id: str, action: str, payload: dict) -> None:
    """After an import that changed the archive: one `meaning` job brings an index that is there up to date.

    With no index nothing is started: the first build is the choice of the
    reader. A build that would embed much is not started either
    (`meaning.small_enough`): the status then says `behind`, and the reader
    sees the cost first. The job is made as every job is made, so it runs one
    at a time with the imports. A fault here does not change how the import ended.
    """
    try:
        if not changed_the_archive(job_id, action):
            return
        place = named_archive(payload)
        if not meaning_service.built(places.archive_path_for(place)):
            return
        _state, archive = archive_for(place, wait_for_import=False)
        found = meaning_service.estimate(archive, saved_names())
        if not meaning_service.small_enough(found):
            new, held = found["newSegments"], found["segments"]
            print(f"[meaning] the index waits for the reader: {new:,} of {held:,} segments are new", file=sys.__stderr__)
            return
        named = payload.get("messagesPath")
        create_job("meaning", {"messagesPath": named} if named else {})
    except JobRefused:
        pass  # an import that started meanwhile brings its own job
    except Exception as exc:  # noqa: BLE001 - the import ended: nothing here is a reason to say otherwise
        with contextlib.suppress(Exception):
            print(f"[meaning] the index was not brought up to date: {type(exc).__name__}", file=sys.__stderr__)


def do_job(job_id: str, action: str, payload: dict) -> dict:
    """The work of one job. What it returns is the result of the job."""
    if action == "export":
        append_log(job_id, "Reading Messages")
        pulled = pull_into_archive(action, payload)
        return {
            "messages": pulled.messages,
            "conversations": pulled.conversations,
            "messagesPath": str(named_archive(payload)),
        }
    if action == "analyze":
        return write_report(job_id, payload)
    if action == "setup_import":
        return import_messages(job_id, action, payload)
    if action == "bridge":
        return load_export(job_id, payload)
    if action == "meaning":
        return build_meaning_index(job_id, payload)
    if action == "digest":
        return read_conversation(job_id, payload)
    raise ValueError(f"Unknown job action: {action}")


# a failure that is one plain sentence for the reader, with a fixed word for the client
TOLD_IN_A_SENTENCE = (ImportStopped, JobStopped, report_service.ReportFailure, ModelChoiceError, ArchiveNotReady, BadRequest)


def work_job(job_id: str, action: str, payload: dict):
    update_job(job_id, status="running")
    writer = JobWriter(job_id)
    _JOB_STDOUT.register(writer)
    _JOB_STDERR.register(writer)
    try:
        with contextlib.redirect_stdout(_JOB_STDOUT), contextlib.redirect_stderr(_JOB_STDERR):
            result = do_job(job_id, action, payload)
        writer.flush()
        update_job(job_id, status="completed", result=result)
    except TOLD_IN_A_SENTENCE as exc:
        update_job(job_id, status="failed", error=str(exc), code=getattr(exc, "code", "") or None)
        # What led to the sentence is for the log of the server, not for the
        # page. A log that cannot be written does not change how the job ended.
        with contextlib.suppress(Exception):
            writer.flush()
            if exc.__cause__ is not None:
                traceback.print_exception(exc, file=sys.__stderr__)
    except Exception:
        # Nobody planned for this failure. What was raised can hold a path or
        # the text of a query: it goes to the log of the server, not to the page.
        update_job(job_id, status="failed", error=JOB_FAILED, code="job_failed")
        with contextlib.suppress(Exception):
            writer.flush()
            traceback.print_exc(file=sys.__stderr__)
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
            saved = saved_by_handle(sorted(senders.unique()))
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
# stored one is not served as it was. Version 8: the figures are read from the
# archive, and the estimate is the plan of the report engine.
PREVIEW_PAYLOAD_VERSION = 8


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
        "attachment": attachment_label(row.get("attachment_types")) or (UNSENT_LABEL if _unsent(row) else ""),
        "isFromMe": mine,
        "sender": sender,
        "senderName": str(sender_names.get(sender) or ""),
    }


def _unsent(row) -> bool:
    """True for a message that its writer took back: a row of the archive with no words."""
    return bool(row.get("unsent", False))


def saved_by_handle(handles) -> dict[str, str]:
    """The saved name of each handle that has one, from the Contacts cache."""
    saved = saved_names()
    named = {str(handle): saved.saved(str(handle)) for handle in handles}
    return {handle: name for handle, name in named.items() if name}


def report_estimate(archive: Archive, contact: str, model: str, since, until) -> tuple[dict | None, str]:
    """What a report run that starts now would send and cost, or no estimate and one sentence that says why."""
    try:
        return report_service.estimate(contact, model, since, until, archive=archive, names=saved_names()), ""
    except (report_service.ReportFailure, ModelChoiceError, ValueError) as exc:
        return None, str(exc)


def stats_payload(stats) -> dict:
    """The figures of a conversation under the names that the app reads."""
    return {
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
    }


NO_FIGURES = "This conversation is no longer in the archive. Its figures cannot be shown."


def empty_preview() -> dict:
    """The preview of a conversation that the archive no longer holds: no figure, and no report can be run.

    A report that was written about it can still be read.
    """
    nothing = pd.DataFrame({"timestamp": pd.to_datetime([])})
    return {
        "stats": stats_payload(compute_stats(nothing)),
        "estimate": None,
        "estimateError": NO_FIGURES,
        "dynamics": {},
        "monthly": [],
        "activity": activity_series(nothing),
        "daily": day_totals(nothing),
        "recentMessages": [],
    }


def preview_payload(messages_path: Path, contact: str, model: str, since=None, until=None):
    state, archive = archive_for(messages_path)
    full_df, _, _ = cached_messages_bundle(messages_path)
    since_dt = parse_optional_date(since)
    until_dt = parse_optional_date(until)
    df = within_dates(full_df, since_dt, until_dt)
    conv = filter_conversation(df, contact)
    if conv.empty:
        raise ValueError("No messages found for this conversation in these dates.")

    days = {"since": since or None, "until": until or None}
    attachments_df = frame_service.attachments_frame(state, str(contact), **days)
    reactions_df = frame_service.reactions_frame(state, str(contact), **days)
    stats = compute_stats(conv, attachments_df, reactions_df)
    monthly = fill_months(progression_series(conv))
    # the dates are chosen from the whole conversation, so this does not narrow with them
    daily = day_totals(conv if df is full_df else filter_conversation(full_df, contact))
    estimate, no_estimate = report_estimate(archive, str(contact), model, days["since"], days["until"])
    recent = conv.tail(RECENT_MESSAGE_COUNT)
    in_group = is_group_chat(str(contact))
    recent_senders = sorted({_row_str(row, "sender") for _, row in recent.iterrows()} - {""}) if in_group else []
    sender_names = saved_by_handle(recent_senders)

    return {
        "stats": stats_payload(stats),
        "estimate": estimate,
        **({"estimateError": no_estimate} if no_estimate else {}),
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


def preview_cache_key(messages_path: Path, contact: str, model: str, since=None, until=None) -> tuple:
    return (
        PREVIEW_PAYLOAD_VERSION,
        # one content of the archive, also after a new start of the server
        places.state_for(messages_path).file_mark(),
        # the estimate leaves out what earlier report runs read
        file_cache_signature(report_service.default_places().cache_path),
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
    return story_page(text)[0]


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
# safe_html is such a change. Version 6: a report of the report engine gives
# moments that open their day, and a story is read chapter by chapter.
ANALYSIS_PAYLOAD_VERSION = 6


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


EVENT_KEYS = ("date", "title", "detail", "category", "quote")


def events_of_file(events_path: str | None) -> list[dict]:
    """The events of a report of the older kind, as its events file holds them.

    Every cell is read as the text it is: a title "NA" stays "NA", and "5"
    stays "5".
    """
    events_file_path = safe_path(events_path) if events_path else None
    if not events_file_path or not events_file_path.exists():
        return []
    rows = pd.read_csv(events_file_path, dtype=str, keep_default_na=False)
    return [{**{key: str(row.get(key, "")) for key in EVENT_KEYS}, "score": _score(row.get("score", ""))} for _, row in rows.iterrows()]


def _score(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def report_moments(contact: str, report_text: str, events_path: str | None, archive: Archive) -> list[dict]:
    """The moments of a report. A report of the report engine gives each moment what opens its day."""
    report = report_service.saved_report(contact, report_text) if report_text else None
    if report is None:
        return events_of_file(events_path)
    return report_service.moments_for(report, archive)


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
    _state, archive = archive_for(messages_path)
    summary = report_text = ""
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
        try:
            # the archive no longer holds those days; the whole conversation is the closest truth
            preview, _ = cached_preview_payload(messages_path, contact, model)
        except ValueError:
            # the archive no longer holds the conversation: its story is on disk and can still be read
            preview = empty_preview()
    stats = preview["stats"]

    events = report_moments(contact, report_text, events_path, archive)
    story, shown_plain = story_page(summary)

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
        "summaryHtml": story,
        # a chapter of the story is too long and dense to read as chapters: the page says so
        "shownPlain": shown_plain,
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


def bundle_or_none(messages_path: Path):
    """The frames of the archive, or None when there is no archive to read yet."""
    try:
        return cached_messages_bundle(messages_path)
    except ArchiveNotReady as waiting:
        if waiting.code != "no_archive":
            raise
        return None


def archive_summary(messages_path: Path, wait_for_import: bool = True) -> dict:
    """Size and span of the whole archive. The conversation list is capped, so
    totals come from here and never from adding up that list."""
    try:
        _state, archive = archive_for(messages_path, wait_for_import)
    except ArchiveNotReady as waiting:
        if waiting.code != "no_archive":
            raise
        return {"messages": 0, "conversations": 0, "firstMessage": None, "lastMessage": None}
    return conversation_service.summary(archive)


_PATTERNS_CACHE: dict = {"sig": None, "value": None}
_PATTERNS_LOCK = threading.Lock()


def what_the_archive_holds(messages_path: Path) -> dict:
    """What an import tells of the archive it wrote: the count that the Overview shows, and the years.

    One count is used everywhere: a message in no conversation is shown
    nowhere in the app, so it is not counted here either (archive_summary).
    """
    # the import that asks this is the one that filled the archive: it does not wait for itself
    summary = archive_summary(messages_path, wait_for_import=False)
    first, last = summary["firstMessage"], summary["lastMessage"]
    return {
        "messages": summary["messages"],
        "conversations": summary["conversations"],
        "firstYear": int(first[:4]) if first else None,
        "lastYear": int(last[:4]) if last else None,
    }


def archive_patterns(messages_path: Path) -> dict:
    """The shape of the whole archive, for the Overview: messages by month in
    both voices, by day, and by hour of each weekday. Worked out once for one
    content of the archive."""
    # the signature first: a frame is never kept under a signature that is newer than it
    sig = archive_signature(places.state_for(messages_path))
    bundle = bundle_or_none(messages_path)
    if bundle is None:
        nothing = pd.DataFrame({"timestamp": pd.to_datetime([])})
        return {"monthly": [], "daily": day_totals(nothing), "rhythm": hour_by_weekday(nothing)}
    # held while it is worked out: requests that arrive together wait for one answer
    with _PATTERNS_LOCK:
        if _PATTERNS_CACHE["sig"] == sig:
            return _PATTERNS_CACHE["value"]
        df = bundle[0]
        patterns = {"monthly": month_points(df), "daily": day_totals(df), "rhythm": hour_by_weekday(df)}
        _PATTERNS_CACHE.update({"sig": sig, "value": patterns})
        return patterns


def count_display_names(df: pd.DataFrame) -> int:
    if df.empty or "display_name" not in df.columns:
        return 0
    return int(df["display_name"].fillna("").astype(str).str.len().gt(0).sum())


def list_jobs() -> list[dict]:
    with JOBS_LOCK:
        jobs = [job_copy(job) for job in JOBS.values()]
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


def message_id_text(row) -> str:
    """The id of a message, written one way wherever a payload names one."""
    return _row_str(row, "message_id")


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


# The frame of the archive, the rollup of its conversations and their names,
# for the figures: reading every message again for each request costs seconds.
# Keyed by bundle_signature, so an update and a change of names both renew it.
_ASK_CACHE_LOCK = threading.Lock()
_ASK_CACHE: dict = {"sig": None, "df": None, "catalog": None, "names": None, "loading": None}

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
    sig = "|".join(str(part) for part in places.state_for(messages_path).file_mark())
    bundle = bundle_or_none(messages_path)
    if bundle is None:
        return empty
    df, catalog, names = bundle
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
    bundle = bundle_or_none(messages_path)
    if bundle is None:
        return empty
    df, _catalog, names = bundle
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
            "previewSender": saved_names().saved(sender) if in_group else "",
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
        # words as the screen shows them: the mark of an attachment is no word
        worded = past_today["text"].map(display_text).ne("")
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


def _parse_archive(state: ArchiveState) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The archive as the frame that the figures read, and the rollup of its conversations."""
    whole = frame_service.messages_frame(state)
    # A message in no conversation is shown nowhere in the app: no figure counts it.
    df = whole[whole["chat_id"].notna()]
    return df, _conversation_catalog(df)


def archive_signature(state: ArchiveState) -> tuple:
    """What one content of one archive is known by, in this run of the server."""
    return (str(state.path), state.generation, state.file_mark())


def bundle_signature(messages_path: Path) -> tuple:
    """What the cached archive is made from: the archive, and for its names, the Contacts cache."""
    return (archive_signature(places.state_for(messages_path)), file_cache_signature(CONTACTS_CACHE_PATH))


def cached_messages_bundle(messages_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(messages df, conversation catalog, display-name map), kept until the
    archive or the Contacts cache changes. When only the names changed, the
    archive is not read again: the names are worked out over the frame already
    held. Callers that arrive together wait for the first one instead of each
    reading the whole archive (app launch fires four requests at once).

    Raises ArchiveNotReady when there is no archive to read."""
    state, archive = archive_for(messages_path)
    sig = bundle_signature(messages_path)
    archive_sig = sig[0]
    while True:
        with _ASK_CACHE_LOCK:
            held = _ASK_CACHE["sig"]
            has_frame = _ASK_CACHE["df"] is not None
            if held == sig and has_frame:
                return _ASK_CACHE["df"], _ASK_CACHE["catalog"], _ASK_CACHE["names"]
            loading = _ASK_CACHE.get("loading")
            if loading is None or loading.is_set():
                loading = threading.Event()
                _ASK_CACHE["loading"] = loading
                same_archive = has_frame and held is not None and held[0] == archive_sig
                parsed = (_ASK_CACHE["df"], _ASK_CACHE["catalog"]) if same_archive else None
                break  # this thread builds; everyone else waits below
        loading.wait(timeout=180)

    try:
        df, catalog = parsed if parsed is not None else _parse_archive(state)
        # one rule for names on every screen: a saved name, the title of a group, or its saved people
        names = saved_names().conversations(archive.conversations())
        with _ASK_CACHE_LOCK:
            _ASK_CACHE.update({"sig": sig, "df": df, "catalog": catalog, "names": names})
        return df, catalog, names
    finally:
        loading.set()


# a question that the request itself makes impossible: the answer is 400, not a fault of the provider
ASK_REQUEST_FAULTS = frozenset(
    {"empty_question", "question_too_long", "unknown_model", "model_not_offered", "reasoning_level"}
)
ASK_ID_CHARS = 80
# the questions that are being answered now, by the id that the app gave each
ASKING: dict[str, CancelToken] = {}
ASKING_LOCK = threading.Lock()


def ask_arguments(payload: dict) -> dict:
    """What a chat question is asked with, as the routes receive it."""
    history = payload.get("history")
    return {
        "question": payload.get("question", ""),
        "contact": payload.get("contact", ""),
        "model": payload.get("model", ""),
        # one of the reasoning levels of the model, or nothing for the effort of the registry
        "reasoning_effort": payload.get("reasoningEffort", ""),
        "history": history if isinstance(history, list) else None,
        "use_ai": str(payload.get("ai", True)).lower() not in {"0", "false", "no", "off"},
    }


def ask_id(payload: dict) -> str:
    """The id that the app gave a question, or "" when it gave none that can be used."""
    given = payload.get("requestId")
    return given[:ASK_ID_CHARS] if isinstance(given, str) else ""


@contextlib.contextmanager
def asking(request_id: str):
    """A cancel token for one question, found by its id while the question is answered."""
    token = CancelToken()
    if request_id:
        with ASKING_LOCK:
            ASKING[request_id] = token
    try:
        yield token
    finally:
        with ASKING_LOCK:
            if ASKING.get(request_id) is token:
                del ASKING[request_id]


def cancel_asking(request_id: str) -> bool:
    """Stop the question with this id. False when no such question is being answered."""
    with ASKING_LOCK:
        token = ASKING.get(request_id) if request_id else None
    if token is None:
        return False
    token.cancel()
    return True


def answer_question(payload: dict) -> tuple[int, dict]:
    """The answer to /api/ask: the status and the body."""
    _state, archive = archive_for(named_archive(payload))
    asked = ask_arguments(payload)
    try:
        with asking(ask_id(payload)) as token:
            return 200, chat_service.ask(
                asked["question"],
                asked["contact"],
                asked["model"],
                asked["history"],
                token,
                archive=archive,
                names=saved_names(),
                use_ai=asked["use_ai"],
                reasoning_effort=asked["reasoning_effort"],
                start_digest=digest_starter(payload),
            )
    except chat_service.ChatFailure as failure:
        said = {"error": str(failure), "code": failure.code}
        if failure.retry_after is not None:
            said = {**said, "retry_after": failure.retry_after}
        return (400 if failure.code in ASK_REQUEST_FAULTS else 502), said


def contacts_answer(query: dict) -> tuple[int, dict]:
    """The answer to /api/contacts: the conversations of the archive, or of Messages before there is an archive."""
    messages_path = named_archive(query)
    try:
        _state, archive = archive_for(messages_path)
    except ArchiveNotReady as waiting:
        if waiting.code != "no_archive":
            raise
        return contacts_from_database(query)
    # the answer names the archive file itself, as /api/defaults does
    place = str(places.archive_path_for(messages_path))
    return 200, conversation_service.payload(archive, saved_names(), query.get("limit", ""), place)


def contacts_from_database(query: dict) -> tuple[int, dict]:
    """The list read from Messages itself, before there is an archive. The client names no database."""
    limit = inputs.whole("limit", query.get("limit", ""), conversation_service.DEFAULT_LIMIT)
    probe = probe_database(LIVE_CHAT_DB)
    if why_unreadable(probe, note=""):
        return 400, {"error": "There is no archive yet, and Messages cannot be read.", "code": "no_archive"}
    try:
        listed = list_contacts(str(LIVE_CHAT_DB), limit=max(0, limit or 0))
    except sqlite3.Error as exc:
        # a busy database can be asked again; the reason is a plain sentence either way
        stopped = reading_stopped(exc, probe, note="")
        return (503 if isinstance(stopped, DatabaseBusy) else 500), {"error": str(stopped)}
    df = decorate_contacts_frame(listed, name_lookup=saved_by_handle)
    return 200, {
        "contacts": df.to_dict(orient="records"),
        "source": "database",
        "contactNameCount": count_display_names(df),
    }


_QUERY_IN_A_LOG_LINE = re.compile(r"\?\S*")


class RecallHandler(BaseHTTPRequestHandler):
    server_version = "RecallUI/1.0"

    def log_message(self, format, *args):
        # The query of a request holds handles and search words. The log is a
        # file that the reader is pointed to: it gets the path and no query.
        line = _QUERY_IN_A_LOG_LINE.sub("", format % args)
        sys.__stderr__.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), line))

    @contextlib.contextmanager
    def answering(self):
        """Mark this thread as one that answers a request, so nothing it prints becomes a line of a job."""
        mine = threading.get_ident()
        _JobStreamRouter.answering.add(mine)
        try:
            yield
        finally:
            _JobStreamRouter.answering.discard(mine)

    def do_OPTIONS(self):
        # no CORS grant: the UI is same-origin, so a preflight can only come
        # from a cross-origin page probing the archive
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        with self.answering():
            self.answer_get()

    def do_POST(self):
        with self.answering():
            self.answer_post()

    def answer_get(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                if not is_local_api_request(self):
                    send_json(self, 403, {"error": "Forbidden: non-local origin"})
                    return
                self.handle_api_get(parsed)
            else:
                self.serve_static(parsed.path)
        except BadRequest as exc:
            send_json(self, 400, {"error": str(exc)})
        except ArchiveNotReady as waiting:
            send_json(self, waiting.status, waiting.body())
        except Exception:
            send_json(self, 500, server_fault(f"GET {parsed.path}"))

    def do_HEAD(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                send_json(self, 405, {"error": "Method not allowed"})
            else:
                self.serve_static(parsed.path, head_only=True)
        except Exception:
            send_json(self, 500, server_fault(f"HEAD {parsed.path}"))

    def answer_post(self):
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
                forget_previews()
                send_json(self, 200, {"cleared": True})
            elif parsed.path == "/api/setup/complete":
                send_json(self, 200, {"setup": write_setup_marker(setup_marker_updates(parse_body(self)))})
            elif parsed.path == "/api/ask":
                status, answer = answer_question(self.body_object())
                send_json(self, status, answer)
            elif parsed.path == "/api/ask/cancel":
                send_json(self, 200, {"cancelled": cancel_asking(ask_id(self.body_object()))})
            elif parsed.path == "/api/ask/stream":
                self.handle_ask_stream(self.body_object())
            else:
                send_json(self, 404, {"error": "Not found"})
        except BadRequest as exc:
            send_json(self, 400, {"error": str(exc)})
        except ArchiveNotReady as waiting:
            send_json(self, waiting.status, waiting.body())
        except Exception:
            send_json(self, 500, server_fault(f"POST {parsed.path}"))

    def body_object(self) -> dict:
        """The body of a request as a record. Any other JSON value is refused."""
        payload = parse_body(self)
        if not isinstance(payload, dict):
            raise BadRequest("The request must be a JSON object.")
        return payload

    def handle_ask_stream(self, payload: dict):
        """Server-sent events for a chat question: status lines, the words of
        the answer as the model writes them, then the whole answer or one
        plain sentence that says why there is none.

        A request that cannot be answered at all (no archive, a bad archive
        file) is refused before the stream starts, as on every other route."""
        _state, archive = archive_for(named_archive(payload))
        asked = ask_arguments(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def write_event(data: dict) -> None:
            frame = f"data: {json.dumps(data, default=json_default)}\n\n"
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

        with asking(ask_id(payload)) as token:
            try:
                events = chat_service.stream(
                    asked["question"],
                    asked["contact"],
                    asked["model"],
                    asked["history"],
                    token,
                    archive=archive,
                    names=saved_names(),
                    use_ai=asked["use_ai"],
                    reasoning_effort=asked["reasoning_effort"],
                    start_digest=digest_starter(payload),
                )
                try:
                    for event in events:
                        write_event(event)
                finally:
                    # closing the events stops the model call
                    events.close()
            except OSError:
                # the reader left or stopped the question: nothing is left to tell, and the model call ends
                token.cancel()
            except Exception:
                # the stream is open: the fault is told as its last event, not as a second answer
                token.cancel()
                with contextlib.suppress(OSError):
                    write_event({"type": "error", **server_fault("POST /api/ask/stream")})

    def handle_api_get(self, parsed):
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.path == "/api/defaults":
            # where the archive and the reports are. No database is named: an
            # update reads Messages itself (database_for).
            send_json(self, 200, {
                # the offered models: one registry for the list, the levels and every route that takes a model
                **models.defaults_payload(),
                "messagesPath": str(places.archive_path_for(None)),
                "outDir": str(OUT_DIR),
                # an archive that a first import is still filling holds nothing yet: setup is not done
                "hasMessages": holds_messages(None),
                "reports": list_report_files(),
                "contactNames": contacts_cache_summary(),
                "setupCompleted": bool(read_setup_marker().get("completed")),
            })
        elif parsed.path == "/api/setup/status":
            db_param = query.get("dbPath")
            send_json(self, 200, setup_status_payload(
                safe_path(db_param) if db_param else None,
                named_archive(query),
                deep=str(query.get("deep", "0")).lower() in {"1", "true", "yes"},
            ))
        elif parsed.path == "/api/first-words":
            send_json(self, 200, {"firstWords": first_words_payload(named_archive(query))})
        elif parsed.path == "/api/permissions/fulldisk":
            # the same check that the guide uses: it tells a database that macOS hides from one that is not there
            send_json(self, 200, {
                "status": access_to_messages(probe_database(LIVE_CHAT_DB)),
                "dbPath": str(LIVE_CHAT_DB),
            })
        elif parsed.path == "/api/memories":
            send_json(self, 200, {"memories": memories_payload(named_archive(query))})
        elif parsed.path == "/api/overview":
            send_json(self, 200, {"overview": archive_patterns(named_archive(query))})
        elif parsed.path == "/api/contacts":
            try:
                status, answer = contacts_answer(query)
            except ValueError as error:
                send_json(self, 400, {"error": str(error)})
                return
            send_json(self, status, answer)
        elif parsed.path == "/api/preview":
            try:
                payload, cached = cached_preview_payload(
                    named_archive(query),
                    query.get("contact", ""),
                    query.get("model") or models.default_id(),
                    since=query.get("since"),
                    until=query.get("until"),
                )
            except ValueError as error:
                # no message in these dates, a day that is no day: a fault of the request, in one sentence
                send_json(self, 400, {"error": str(error)})
                return
            payload = dict(payload)
            payload["cached"] = cached
            send_json(self, 200, payload)
        elif parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                # a copy: the job changes in its own thread while the answer is written
                job = job_copy(JOBS[job_id]) if job_id in JOBS else None
            if not job:
                send_json(self, 404, {"error": "Job not found"})
                return
            send_json(self, 200, {"job": job})
        elif parsed.path == "/api/jobs":
            send_json(self, 200, {"jobs": list_jobs()})
        elif parsed.path == "/api/reports":
            send_json(self, 200, {"reports": list_report_files()})
        elif parsed.path == "/api/meaning":
            # the index for the search by meaning: what it holds, and what a build would send and cost now
            _state, archive = archive_for(named_archive(query))
            send_json(self, 200, meaning_service.payload(archive, saved_names()))
        elif parsed.path == "/api/day":
            _state, archive = archive_for(named_archive(query))
            try:
                payload = day_service.payload(
                    archive,
                    saved_names(),
                    query.get("contact", ""),
                    query.get("day", ""),
                    around=query.get("around") or None,
                    offset=query.get("offset", ""),
                    limit=query.get("limit", ""),
                )
            except ValueError as error:
                send_json(self, 400, {"error": str(error)})
                return
            send_json(self, 200, payload)
        elif parsed.path == "/api/search":
            # words, filters, order, and `prefix=true` while the reader types the last word
            _state, archive = archive_for(named_archive(query))
            try:
                payload = search_service.answer(archive, saved_names(), query)
            except ValueError as error:
                send_json(self, 400, {"error": str(error)})
                return
            send_json(self, 200, payload)
        elif parsed.path == "/api/stats":
            # counts with the filters of search; imported here so that this route is one region of the file
            from recall_services import stats as stats_service

            _state, archive = archive_for(named_archive(query))
            try:
                payload = stats_service.answer(archive, saved_names(), query)
            except ValueError as error:
                send_json(self, 400, {"error": str(error)})
                return
            send_json(self, 200, payload)
        elif parsed.path == "/api/analysis":
            messages_path = named_archive(query)
            # only a report file is opened as a report: the same test as for /api/report
            report_path = served_report_file(query.get("reportPath", ""))
            if not report_path:
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
                query.get("model") or models.default_id(),
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


def log_to_the_server_log() -> None:
    """Send what the code logs to the log of the server.

    With no handler, the logging module writes to the stream that a job has
    redirected, and a traceback then becomes a line of that job on the page.
    """
    handler = logging.StreamHandler(sys.__stderr__)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    # what a question spent on a digest is a line of the log, though it is no warning
    logging.getLogger(memory_service.SPEND_LOG).setLevel(logging.INFO)


def main():
    import signal

    from recall_paths import BUNDLED

    if BUNDLED:
        _retarget_streams_to_logfile()
    log_to_the_server_log()
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
    # an export of an earlier build is loaded into the archive by one job; the app waits for it
    load_export_at_start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Recall UI")
    finally:
        # let go of the port at once: the Mac shell asks for the same port at its next launch
        server.server_close()


if __name__ == "__main__":
    main()
