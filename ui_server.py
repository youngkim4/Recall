#!/usr/bin/env python3
"""Local web UI server for Recall."""

import contextlib
import io
import json
import mimetypes
import os
import re
import sqlite3
import sys
import threading
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime
from html import escape
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

import markdown as markdown_lib
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
    compute_stats,
    filter_conversation,
    load_attachments,
    load_messages,
    load_reactions,
    parse_boolish,
    progression_series,
    sanitize_filename,
)
from openai_client import _call_openai, _call_openai_stream
from parse_imessage import extract_attachments, extract_messages, extract_reactions, list_contacts
from retrieval_planner import build_catalog, plan_retrieval
import semantic_index as sem
from recall_paths import (
    DATA_DIR,
    SAVES_DIR,
    OUT_DIR,
    DEFAULT_MESSAGES_CSV,
    SNAPSHOT_DB,
    ensure_data_dirs,
)
from setup_status import LIVE_CHAT_DB, DbProbe, derive_state, probe_database

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


def list_report_files():
    reports = []
    out_dir = OUT_DIR
    if not out_dir.exists():
        return reports
    for path in sorted(out_dir.glob("analysis_*.*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.suffix.lower() not in {".html", ".md"}:
            continue
        contact = path.stem.removeprefix("analysis_")
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


def merge_snapshot_into(db_path: Path, messages_path: Path) -> int:
    """Restore messages the live database has offloaded to iCloud ("Optimize
    Storage") by merging the frozen chat.db snapshot into the just-written
    export. Returns the count of recovered rows. Never raises — a refresh must
    not fail because of the merge.
    """
    snapshot = SNAPSHOT_DB
    try:
        if not snapshot.exists() or snapshot.resolve() == db_path.resolve():
            return 0
        from merge_export import merge as _merge

        merged, recovered = _merge(str(snapshot), str(messages_path))
        count = int(len(recovered))
        if count:
            merged.to_csv(messages_path, index=False)
            print(f"Recovered {count:,} message(s) from snapshot {snapshot.name}")
        return count
    except Exception as exc:  # noqa: BLE001 - export must survive merge failures
        print(f"Snapshot merge skipped: {exc}")
        return 0


def export_database(db_path: Path, messages_path: Path):
    try:
        df = extract_messages(str(db_path))
    except sqlite3.OperationalError as exc:
        text = str(exc).lower()
        if any(s in text for s in ("unable to open", "permission", "not permitted", "authoriz")):
            raise PermissionError(
                f"Couldn't read the Messages database at {db_path}. If this is your live database "
                "(~/Library/Messages/chat.db), grant Full Disk Access to this app in "
                "System Settings > Privacy & Security > Full Disk Access, then reopen Recall."
            ) from exc
        raise
    messages_path.parent.mkdir(parents=True, exist_ok=True)
    # write to a temp file and swap in atomically: requests served during the
    # export keep reading the complete old file instead of a half-written one
    tmp_path = messages_path.with_name(messages_path.stem + ".tmp-export" + messages_path.suffix)
    df.to_csv(tmp_path, index=False)

    # Merge the frozen snapshot to recover messages the live DB has pruned.
    recovered = merge_snapshot_into(db_path, tmp_path)
    os.replace(tmp_path, messages_path)

    attachments_df = extract_attachments(str(db_path))
    if not attachments_df.empty:
        attachments_df.to_csv(messages_path.with_name(messages_path.stem + "_attachments.csv"), index=False)

    reactions_df = extract_reactions(str(db_path))
    if not reactions_df.empty:
        reactions_df.to_csv(messages_path.with_name(messages_path.stem + "_reactions.csv"), index=False)

    return {
        "messages": len(df) + recovered,
        "recovered": recovered,
        "attachments": len(attachments_df),
        "reactions": len(reactions_df),
        "messagesPath": str(messages_path),
    }


def create_job(action: str, payload: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "action": action,
        "status": "queued",
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "updatedAt": datetime.now().isoformat(timespec="seconds"),
        "logs": [],
        "result": None,
        "error": None,
    }
    with JOBS_LOCK:
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
    thread.start()
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
        self._target().flush()


_JOB_STDOUT = _JobStreamRouter(sys.stdout)
_JOB_STDERR = _JobStreamRouter(sys.stderr)


SETUP_MARKER_PATH = SAVES_DIR / "setup.json"


def read_setup_marker() -> dict:
    try:
        marker = json.loads(SETUP_MARKER_PATH.read_text(encoding="utf-8"))
        return marker if isinstance(marker, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_setup_marker(updates: dict) -> dict:
    """Persist onboarding state server-side: WKWebView localStorage can be
    cleared while saves/ survives. Atomic so a crash never corrupts it."""
    marker = {**read_setup_marker(), **updates}
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


def run_job(job_id: str, action: str, payload: dict):
    update_job(job_id, status="running")
    writer = JobWriter(job_id)
    _JOB_STDOUT.register(writer)
    _JOB_STDERR.register(writer)
    try:
        with contextlib.redirect_stdout(_JOB_STDOUT), contextlib.redirect_stderr(_JOB_STDERR):
            if action == "export":
                db_path = safe_path(payload.get("dbPath"))
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                if not path_exists(db_path):
                    raise FileNotFoundError(f"Database not found: {db_path}")
                append_log(job_id, f"Exporting {db_path}")
                result = export_database(db_path, messages_path)
            elif action == "analyze":
                db_path = safe_path(payload.get("dbPath"))
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                if payload.get("extractFirst"):
                    if not path_exists(db_path):
                        raise FileNotFoundError(f"Database not found: {db_path}")
                    append_log(job_id, "Refreshing CSV export before analysis")
                    export_database(db_path, messages_path)
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
                import time as _time

                db_path = safe_path(payload.get("dbPath"), LIVE_CHAT_DB)
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                include_contacts = bool(payload.get("includeContacts", True))

                with JOBS_LOCK:
                    duplicate = any(
                        j["id"] != job_id
                        and j["action"] in {"setup_import", "export"}
                        and j["status"] in {"queued", "running"}
                        for j in JOBS.values()
                    )
                if duplicate:
                    raise ValueError("An import is already running.")

                update_job(job_id, progress={"step": 1, "of": 3, "label": "Reading your Messages database"})
                if not path_exists(db_path):
                    raise FileNotFoundError(f"Database not found: {db_path}")
                append_log(job_id, f"Exporting {db_path}")
                try:
                    export_result = export_database(db_path, messages_path)
                except Exception as exc:
                    if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                        # Messages mid-write; one quiet retry covers most of it
                        append_log(job_id, "Database is busy -- retrying in a moment")
                        _time.sleep(2)
                        export_result = export_database(db_path, messages_path)
                    else:
                        raise

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
                years = pd.to_datetime(df["timestamp"], errors="coerce").dt.year.dropna()
                result = {
                    "messages": int(len(df)),
                    "conversations": int(len(catalog)),
                    "firstYear": int(years.min()) if len(years) else None,
                    "lastYear": int(years.max()) if len(years) else None,
                    "contactsSkipped": contacts_skipped,
                }
                if isinstance(export_result, dict) and export_result.get("recovered"):
                    result["recovered"] = export_result["recovered"]
            elif action == "semantic":
                messages_path = safe_path(payload.get("messagesPath"), DEFAULT_MESSAGES_CSV)
                if not path_exists(messages_path):
                    raise FileNotFoundError(f"Messages CSV not found: {messages_path}")
                with JOBS_LOCK:
                    duplicate = any(
                        j["id"] != job_id
                        and j["action"] == "semantic"
                        and j["status"] in {"queued", "running"}
                        for j in JOBS.values()
                    )
                if duplicate:
                    raise ValueError("A semantic index build is already running.")
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
    out["quietDays"] = int((pd.Timestamp.now() - last_ts).days)

    if "sender" in frame.columns:
        inbound = frame[frame["sender"].fillna("").astype(str) != ""]
        if not inbound.empty and inbound["sender"].astype(str).nunique() > 1:
            shares = inbound.groupby(inbound["sender"].astype(str)).size().sort_values(ascending=False)
            total = int(shares.sum())
            speaker_names = resolve_contact_names(list(shares.index[:5]))
            out["topSpeakers"] = [
                {
                    "name": speaker_names.get(sender) or contact_label({"chatId": sender}),
                    "count": int(count),
                    "share": round(int(count) / total, 3),
                }
                for sender, count in shares.head(5).items()
            ]
    return out


def preview_payload(messages_path: Path, contact: str, model: str, since=None, until=None):
    # the warm bundle replaces what used to be TWO full CSV parses per preview
    # (one here, one inside estimate_cost) -- ~4s of dead 'Calculating...' time
    full_df, _, _ = cached_messages_bundle(messages_path)
    df = full_df
    since_dt = parse_optional_date(since)
    until_dt = parse_optional_date(until)
    if since_dt:
        df = df[df["timestamp"] >= pd.Timestamp(since_dt)]
    if until_dt:
        df = df[df["timestamp"] <= pd.Timestamp(until_dt)]
    conv = filter_conversation(df, contact)
    if conv.empty:
        raise ValueError(f"No messages found for contact '{contact}'")

    attachments_df = load_attachments(str(messages_path))
    reactions_df = load_reactions(str(messages_path))
    stats = compute_stats(conv, attachments_df, reactions_df)
    monthly = progression_series(conv).tail(72)
    estimate = estimate_cost(
        str(messages_path), contact, model=model,
        since=since_dt, until=until_dt, frame=df,
    )
    recent = conv.tail(18)

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
        "recentMessages": [
            {
                "timestamp": row["timestamp"],
                "text": str(row.get("text", ""))[:240],
                "isFromMe": row.get("is_from_me"),
            }
            for _, row in recent.iterrows()
        ],
    }


def companion_data_path(messages_path: Path, suffix: str) -> Path:
    return messages_path.with_name(messages_path.stem + suffix)


def preview_cache_key(messages_path: Path, contact: str, model: str, since=None, until=None) -> tuple:
    return (
        file_cache_signature(messages_path),
        file_cache_signature(companion_data_path(messages_path, "_attachments.csv")),
        file_cache_signature(companion_data_path(messages_path, "_reactions.csv")),
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
    if not text:
        return ""
    escaped = escape(text)
    # escaping turns blockquote markers into &gt; before markdown can see
    # them; restore line-leading ones so quoted messages render as quotes
    escaped = re.sub(r"(?m)^(\s*)&gt; ?", r"\1> ", escaped)
    return markdown_lib.markdown(
        escaped,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )


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


def file_cache_signature(path: Path | str | None) -> tuple | None:
    if not path:
        return None
    file_path = Path(path).resolve()
    try:
        stat = file_path.stat()
    except OSError:
        return (str(file_path), None, None)
    return (str(file_path), stat.st_mtime_ns, stat.st_size)


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
        file_cache_signature(messages_path),
        str(contact or ""),
        str(model or ""),
        str(since or ""),
        str(until or ""),
        file_cache_signature(report_path),
        file_cache_signature(events_path),
        file_cache_signature(html_path),
        file_cache_signature(CONTACTS_CACHE_PATH),
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
    preview, _ = cached_preview_payload(messages_path, contact, model, since=since, until=until)
    stats = preview["stats"]
    summary = ""

    if report_path:
        report_file_path = safe_path(report_path)
        if report_file_path and report_file_path.exists():
            summary = markdown_section(
                report_file_path.read_text(encoding="utf-8"),
                "Summary",
                stop_at_next_heading=False,
            )

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
    display_name = resolve_contact_names([contact]).get(contact, "")
    return {
        "contact": contact,
        "contactDisplayName": display_name,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "summaryHtml": markdown_to_html(summary),
        "stats": stats,
        "estimate": preview["estimate"],
        "monthly": preview["monthly"],
        "recentMessages": preview["recentMessages"],
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
        if str(chat_id or "").startswith("chat")
    })
    if not wanted:
        return {}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
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
    missing_group_names = display_names.eq("") & chat_ids.str.startswith("chat")
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


def contacts_from_messages(messages_path: Path, limit: int = 40, name_lookup=None, db_path: Path | None = None) -> pd.DataFrame:
    """List conversations from the same messages CSV used for preview/reporting."""
    df, _, _ = cached_messages_bundle(messages_path)
    if "chat_id" not in df.columns:
        raise ValueError("messages CSV must contain a 'chat_id' column")

    conversations = df[df["chat_id"].notna()].copy()
    if conversations.empty:
        return pd.DataFrame(columns=["chat_id", "message_count", "first_msg", "last_msg", "is_group", "display_name"])

    conversations["chat_id"] = conversations["chat_id"].astype(str)
    aggregations = {
        "message_count": ("chat_id", "size"),
        "first_msg": ("timestamp", "min"),
        "last_msg": ("timestamp", "max"),
    }
    if "chat_display_name" in conversations.columns:
        conversations["chat_display_name"] = conversations["chat_display_name"].fillna("").astype(str).str.strip()
        aggregations["display_name"] = ("chat_display_name", first_nonempty)

    grouped = (
        conversations
        .groupby("chat_id", as_index=False)
        .agg(**aggregations)
    )
    if "display_name" not in grouped.columns:
        grouped["display_name"] = ""
    grouped["is_group"] = grouped["chat_id"].str.startswith("chat").astype(int)
    grouped = grouped.sort_values(["message_count", "last_msg"], ascending=[False, False])
    grouped = fill_group_display_names(grouped.head(limit), db_path)
    return decorate_contacts_frame(grouped, name_lookup)


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


def _is_from_me_value(value):
    """Return a JSON-safe outbound flag (0/1) or None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def message_row_payload(
    row,
    name_map: dict[str, str] | None = None,
    sender_map: dict[str, str] | None = None,
) -> dict:
    chat_id = _row_str(row, "chat_id")
    text = _row_str(row, "text")
    sender = _row_str(row, "sender")
    display_name = (name_map or {}).get(chat_id, "")
    from_me = _is_from_me_value(row.get("is_from_me"))
    outbound = str(from_me).strip() in {"1", "true", "True"}
    # who actually said it (matters in group chats); blank for my own messages
    sender_name = ""
    if not outbound and sender:
        sender_name = (sender_map or {}).get(sender, "")
        if not sender_name and chat_id.startswith("chat"):
            sender_name = contact_label({"chatId": sender})
    return {
        "messageId": _row_str(row, "message_id"),
        "chatId": chat_id,
        "displayName": display_name,
        "senderName": sender_name,
        "timestamp": _row_str(row, "timestamp") or None,
        "sender": sender,
        "text": text[:600],
        "isFromMe": from_me,
    }


def search_messages_payload(messages_path: Path, query: str = "", contact: str = "", limit: int = 60) -> dict:
    if not path_exists(messages_path):
        raise FileNotFoundError(f"Messages CSV not found: {messages_path}")

    limit = max(1, min(int(limit or 60), 200))
    df, _, _ = cached_messages_bundle(messages_path)
    if contact:
        df = filter_conversation(df, contact)

    needle = str(query or "").strip()
    if needle:
        text = df["text"].fillna("").astype(str)
        df = df[text.str.contains(needle, case=False, na=False, regex=False)]
    else:
        df = df.tail(limit)

    if df.empty:
        return {"query": needle, "contact": contact, "results": [], "count": 0}

    df = df.sort_values("timestamp", ascending=False).head(limit)
    name_map = resolve_contact_names(df["chat_id"].dropna().astype(str).unique())
    return {
        "query": needle,
        "contact": contact,
        "count": int(len(df)),
        "results": [message_row_payload(row, name_map) for _, row in df.iterrows()],
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


def contact_label(citation: dict) -> str:
    """Human label for a citation's contact. Never expose a raw handle/number."""
    name = str(citation.get("displayName") or "").strip()
    if name:
        return name
    handle = str(citation.get("chatId") or "").strip()
    if not handle:
        return "an unsaved contact"
    if handle.startswith("chat"):
        return "a group chat"
    digits = re.sub(r"\D", "", handle)
    if digits:
        return f"an unsaved contact (ending {digits[-4:]})"
    return "an unsaved contact"


def citation_context(citations: list[dict]) -> str:
    lines = []
    for idx, citation in enumerate(citations, start=1):
        outbound = str(citation.get("isFromMe")).strip().lower() in {"1", "true"}
        convo = contact_label(citation)
        speaker = "You" if outbound else (citation.get("senderName") or convo)
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
        name = contact_label(citation)
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
        is_group = cid.startswith("chat")
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
            outbound = str(_is_from_me_value(row.get("is_from_me"))).strip() in {"1", "true", "True"}
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


def _conversation_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Per-conversation rollup (chat_id, message_count, last_msg, display_name)."""
    convo = df[df["chat_id"].notna()].copy() if "chat_id" in df.columns else pd.DataFrame()
    if convo.empty:
        return pd.DataFrame(columns=["chat_id", "message_count", "last_msg", "display_name"])
    convo["chat_id"] = convo["chat_id"].astype(str)
    aggregations = {
        "message_count": ("chat_id", "size"),
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


_FW_URL_ONLY_RE = re.compile(r"^https?://\S+$", re.I)
_FW_LETTER_RE = re.compile(r"[^\W\d_]")
_FW_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿❤♥]")


def _first_real_texts(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows whose text reads like words someone typed: not attachment-only,
    not a bare link, carries at least one letter or an emoji."""
    texts = rows["text"].fillna("").astype(str).str.strip()
    ok = (
        texts.ne("")
        & ~texts.str.contains("￼", regex=False)
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

    direct = catalog[~catalog["chat_id"].astype(str).str.startswith("chat")].copy()
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
        first = real.iloc[0]
        first_dir = str(first.get("direction") or "")
        reply = None
        later = real.iloc[1:]
        if not later.empty and "direction" in later.columns:
            opposite = later[later["direction"].fillna("").astype(str) != first_dir].head(1)
            if not opposite.empty:
                row = opposite.iloc[0]
                reply = {
                    "text": str(row["text"]).strip()[:280],
                    "direction": str(row.get("direction") or ""),
                    "timestamp": str(row["timestamp"]),
                }
        cid = str(first["chat_id"])
        display = names.get(cid) or contact_label({"chatId": cid})
        first_ts = pd.Timestamp(first["timestamp"])
        entries.append({
            "person": display,
            "chatId": cid,
            "timestamp": str(first["timestamp"]),
            "yearsAgo": max(0, int((pd.Timestamp.now() - first_ts).days // 365)),
            "direction": first_dir,
            "text": str(first["text"]).strip()[:280],
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


def memories_payload(messages_path: Path) -> dict:
    """Proactive memories computed from the cached dataframe -- no AI calls.
    On-this-day moments from past years, upcoming first-message anniversaries,
    and reconnect nudges for high-volume conversations gone quiet."""
    empty = {"onThisDay": [], "anniversaries": [], "reconnect": []}
    if not path_exists(messages_path):
        return empty
    df, _catalog, names = cached_messages_bundle(messages_path)
    if df.empty or "timestamp" not in df.columns:
        return empty

    now = pd.Timestamp.now()
    cids = df["chat_id"].astype(str)

    def label(cid: str) -> str:
        name = names.get(cid)
        if name:
            return name
        digits = re.sub(r"\D", "", cid)
        if cid.startswith("chat"):
            return f"Group ·{digits[-4:]}" if digits else "Group chat"
        return f"Unsaved ·{digits[-4:]}" if digits else "Unsaved contact"

    def memory_preview(texts: pd.Series) -> str:
        """A quote that reads like a memory: conversational, never a link or a paste."""
        clean = texts[
            ~texts.str.contains("http", case=False, na=False)
            & texts.str.contains(r"[a-zA-Z]{3}", na=False)
        ]
        if clean.empty:
            return ""
        lengths = clean.str.len()
        sweet = clean[(lengths >= 25) & (lengths <= 130)]
        if not sweet.empty:
            return str(sweet.loc[sweet.str.len().idxmax()])
        return str(clean.loc[lengths.idxmax()])[:120]

    # --- on this day, in past years ---
    on_this_day: list[dict] = []
    ts = df["timestamp"]
    day_mask = (
        (ts.dt.month == now.month)
        & (ts.dt.day == now.day)
        & (ts.dt.year < now.year)
        & df["text"].fillna("").astype(str).str.strip().ne("")
    )
    past_today = df[day_mask]
    if not past_today.empty:
        grouped = (
            past_today.groupby(
                [past_today["timestamp"].dt.year.rename("year"), cids[day_mask].rename("cid")]
            )
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
            .head(6)
        )
        for _, row in grouped.iterrows():
            year, cid, count = int(row["year"]), str(row["cid"]), int(row["count"])
            day_rows = past_today[
                (past_today["timestamp"].dt.year == year) & (cids[day_mask] == cid)
            ]
            texts = day_rows["text"].astype(str)
            preview = memory_preview(texts) if not texts.empty else ""
            on_this_day.append(
                {
                    "chatId": cid,
                    "name": label(cid),
                    "year": year,
                    "yearsAgo": int(now.year - year),
                    "count": count,
                    "preview": preview,
                }
            )

    firsts = df.groupby(cids)["timestamp"].min()
    lasts = df.groupby(cids)["timestamp"].max()
    counts = df.groupby(cids).size()

    # --- first-message anniversaries in the next week (named contacts only) ---
    anniversaries: list[dict] = []
    for cid, first in firsts.items():
        if int(counts.get(cid, 0)) < 300 or not names.get(str(cid)):
            continue
        years = int(now.year - first.year)
        if years < 1:
            continue
        try:
            next_anniversary = first.replace(year=now.year)
        except ValueError:  # Feb 29 origin
            continue
        in_days = int((next_anniversary.normalize() - now.normalize()).days)
        if 0 <= in_days <= 7:
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
    for cid, last in lasts.items():
        cid = str(cid)
        if cid.startswith("chat") or int(counts.get(cid, 0)) < 1000:
            continue
        quiet_days = int((now - last).days)
        if quiet_days < 90:
            continue
        reconnect.append(
            {
                "chatId": cid,
                "name": label(cid),
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


def cached_messages_bundle(messages_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(messages df, conversation catalog, display-name map) cached on file signature.
    Concurrent cold callers wait for the first loader instead of each parsing
    the 75MB CSV in parallel (app launch fires four endpoints at once)."""
    sig = file_cache_signature(messages_path)
    while True:
        with _ASK_CACHE_LOCK:
            if sig and _ASK_CACHE["sig"] == sig and _ASK_CACHE["df"] is not None:
                return _ASK_CACHE["df"], _ASK_CACHE["catalog"], _ASK_CACHE["names"]
            loading = _ASK_CACHE.get("loading")
            if loading is None or loading.is_set():
                loading = threading.Event()
                _ASK_CACHE["loading"] = loading
                break  # this thread builds; everyone else waits below
        loading.wait(timeout=180)

    try:
        df = load_messages(str(messages_path))
        if "text" not in df.columns:
            df["text"] = ""
        catalog = _conversation_catalog(df)
        names: dict[str, str] = {}
        if not catalog.empty:
            names = dict(resolve_contact_names(catalog["chat_id"].tolist()))
            # groups have no Contacts entry; their iMessage display name is the label
            for _, row in catalog.iterrows():
                cid = str(row["chat_id"])
                display = str(row.get("display_name") or "").strip()
                if display and not names.get(cid):
                    names[cid] = display
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
        scope |= set(cids[sent & cids.str.startswith("chat")].unique())
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

    handles = [cid for cid in plan_ids if cid and not str(cid).startswith("chat")]

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
        if date_from:
            df = df[df["timestamp"] >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df["timestamp"] < pd.Timestamp(date_to) + pd.Timedelta(days=1)]
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
                action = payload.get("action", "analyze")
                send_json(self, 202, {"job": create_job(action, payload)})
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
                payload = parse_body(self)
                updates: dict = {}
                for key in ("completed", "skipped", "firstWordsShown"):
                    if key in payload:
                        updates[key] = bool(payload[key])
                if payload.get("pickedDbPath"):
                    updates["pickedDbPath"] = str(safe_path(str(payload["pickedDbPath"])))
                send_json(self, 200, {"setup": write_setup_marker(updates)})
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
            db_path = SNAPSHOT_DB
            messages_path = DEFAULT_MESSAGES_CSV
            send_json(self, 200, {
                "defaultModel": DEFAULT_MODEL,
                "models": list(UI_MODEL_CHOICES),
                "dbPath": str(db_path) if db_path.exists() else str(Path("~/Library/Messages/chat.db").expanduser()),
                "messagesPath": str(messages_path),
                "outDir": str(OUT_DIR),
                "hasDb": db_path.exists(),
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
                    conn = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)
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
        elif parsed.path == "/api/contacts":
            db_path = safe_path(query.get("dbPath"))
            messages_path = safe_path(query.get("messagesPath"), DEFAULT_MESSAGES_CSV)
            limit = int(query.get("limit", "40"))
            if path_exists(messages_path):
                df = contacts_from_messages(messages_path, limit=limit, db_path=db_path)
                send_json(self, 200, {
                    "contacts": df.to_dict(orient="records"),
                    "source": "messages",
                    "messagesPath": str(messages_path),
                    "contactNameCount": count_display_names(df),
                })
                return
            if not path_exists(db_path):
                send_json(self, 400, {"error": f"Message export or database not found: {messages_path} / {db_path}"})
                return
            df = decorate_contacts_frame(list_contacts(str(db_path), limit=limit))
            send_json(self, 200, {
                "contacts": df.to_dict(orient="records"),
                "source": "database",
                "dbPath": str(db_path),
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
            report_path = safe_path(unquote(query.get("reportPath", "")))
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
            contact = query.get("contact") or report_path.stem.removeprefix("analysis_")
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
            raw_path = unquote(query.get("path", ""))
            file_path = safe_path(raw_path)
            if not file_path or not file_path.exists() or not ensure_under_root(file_path):
                send_json(self, 404, {"error": "Report not found"})
                return
            content_type = mimetypes.guess_type(str(file_path))[0] or "text/plain"
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
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
