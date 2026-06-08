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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import markdown as markdown_lib
import pandas as pd
from openai import OpenAI

from ai_config import DEFAULT_MODEL, UI_MODEL_CHOICES
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
    progression_series,
    sanitize_filename,
)
from openai_client import _call_openai
from parse_imessage import extract_attachments, extract_messages, extract_reactions, list_contacts

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
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            append_log(self.job_id, line)
        return len(text)

    def flush(self):
        if self._buffer:
            append_log(self.job_id, self._buffer)
            self._buffer = ""


def json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def send_json(handler, status: int, payload: dict):
    data = json.dumps(payload, default=json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


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
    return ensure_under(path, ROOT)


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
    out_dir = ROOT / "out"
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
    snapshot = ROOT / "chat.db"
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
    df.to_csv(messages_path, index=False)

    # Merge the frozen snapshot to recover messages the live DB has pruned.
    recovered = merge_snapshot_into(db_path, messages_path)

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

    thread = threading.Thread(target=run_job, args=(job_id, action, payload), daemon=True)
    thread.start()
    return job


def run_job(job_id: str, action: str, payload: dict):
    update_job(job_id, status="running")
    writer = JobWriter(job_id)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            if action == "export":
                db_path = safe_path(payload.get("dbPath"))
                messages_path = safe_path(payload.get("messagesPath"), ROOT / "messages.csv")
                if not path_exists(db_path):
                    raise FileNotFoundError(f"Database not found: {db_path}")
                append_log(job_id, f"Exporting {db_path}")
                result = export_database(db_path, messages_path)
            elif action == "analyze":
                db_path = safe_path(payload.get("dbPath"))
                messages_path = safe_path(payload.get("messagesPath"), ROOT / "messages.csv")
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
                out_dir = str(safe_path(payload.get("outDir"), ROOT / "out"))
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
            else:
                raise ValueError(f"Unknown job action: {action}")
        writer.flush()
        update_job(job_id, status="completed", result=result)
    except Exception as exc:
        writer.flush()
        append_log(job_id, traceback.format_exc())
        update_job(job_id, status="failed", error=str(exc))


def parse_optional_date(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def preview_payload(messages_path: Path, contact: str, model: str, since=None, until=None):
    df = load_messages(str(messages_path), parse_optional_date(since), parse_optional_date(until))
    conv = filter_conversation(df, contact)
    if conv.empty:
        raise ValueError(f"No messages found for contact '{contact}'")

    attachments_df = load_attachments(str(messages_path))
    reactions_df = load_reactions(str(messages_path))
    stats = compute_stats(conv, attachments_df, reactions_df)
    monthly = progression_series(conv).tail(72)
    estimate = estimate_cost(str(messages_path), contact, model=model, since=parse_optional_date(since), until=parse_optional_date(until))
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
    return markdown_lib.markdown(
        escape(text),
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
    """Read custom group-chat display names from chat.db when available."""
    if not path_exists(db_path):
        return {}

    wanted = sorted({
        str(chat_id)
        for chat_id in chat_ids
        if str(chat_id or "").startswith("chat")
    })
    if not wanted:
        return {}

    conn = sqlite3.connect(str(db_path))
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
    df = load_messages(str(messages_path))
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


def message_row_payload(row, name_map: dict[str, str] | None = None) -> dict:
    chat_id = _row_str(row, "chat_id")
    text = _row_str(row, "text")
    display_name = (name_map or {}).get(chat_id, "")
    return {
        "messageId": _row_str(row, "message_id"),
        "chatId": chat_id,
        "displayName": display_name,
        "timestamp": _row_str(row, "timestamp") or None,
        "sender": _row_str(row, "sender"),
        "text": text[:600],
        "isFromMe": _is_from_me_value(row.get("is_from_me")),
    }


def search_messages_payload(messages_path: Path, query: str = "", contact: str = "", limit: int = 60) -> dict:
    if not path_exists(messages_path):
        raise FileNotFoundError(f"Messages CSV not found: {messages_path}")

    limit = max(1, min(int(limit or 60), 200))
    df = load_messages(str(messages_path))
    if contact:
        df = filter_conversation(df, contact)
    if "text" not in df.columns:
        df["text"] = ""

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
    "about",
    "after",
    "again",
    "before",
    "could",
    "does",
    "from",
    "have",
    "into",
    "like",
    "messages",
    "that",
    "their",
    "there",
    "they",
    "this",
    "thread",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def query_terms(question: str) -> list[str]:
    terms = []
    for term in re.findall(r"[a-z0-9']+", str(question or "").lower()):
        if len(term) < 3 or term in ASK_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms[:8]


def citation_context(citations: list[dict]) -> str:
    lines = []
    for idx, citation in enumerate(citations, start=1):
        name = citation.get("displayName") or citation.get("chatId") or "conversation"
        timestamp = citation.get("timestamp") or "unknown time"
        speaker = "me" if str(citation.get("isFromMe")).lower() in {"1", "true"} else "them"
        text = str(citation.get("text") or "").replace("\n", " ").strip()
        lines.append(f"[{idx}] {name} / {timestamp} / {speaker}: {text}")
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
        name = citation.get("displayName") or citation.get("chatId") or "conversation"
        text = str(citation.get("text") or "").replace("\n", " ").strip()
        if text:
            highlights.append(f"{name}: {text[:180]}")

    if not highlights:
        return lead
    return lead + "\n\n" + "\n".join(f"- {highlight}" for highlight in highlights)


def ai_ask_answer(question: str, citations: list[dict], scope_label: str, model: str) -> str:
    if not os.environ.get("OPENAI_API_KEY") or not citations:
        return ""

    system = (
        "You are Recall, a private iMessage archive assistant. "
        "Answer like a careful chatbot: direct, natural, and useful. "
        "Use only the provided message excerpts. If the excerpts are not enough, say what the evidence can and cannot support. "
        "Cite evidence inline with bracket numbers like [1]. Keep the answer concise unless the user asks for detail."
    )
    user = (
        f"Question: {question}\n"
        f"Scope: {scope_label}\n\n"
        "Message excerpts:\n"
        f"{citation_context(citations)}"
    )
    return _call_openai(
        OpenAI(),
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model or DEFAULT_MODEL,
        max_retries=1,
        reasoning_effort="low",
        verbosity="low",
    )


def ask_messages_payload(
    messages_path: Path,
    question: str,
    contact: str = "",
    limit: int = 8,
    model: str = DEFAULT_MODEL,
    use_ai: bool = False,
) -> dict:
    if not path_exists(messages_path):
        raise FileNotFoundError(f"Messages CSV not found: {messages_path}")

    terms = query_terms(question)
    df = load_messages(str(messages_path))
    if contact:
        df = filter_conversation(df, contact)
    if "text" not in df.columns:
        df["text"] = ""

    text = df["text"].fillna("").astype(str)
    if terms:
        scores = pd.Series(0, index=df.index)
        for term in terms:
            scores = scores + text.str.contains(term, case=False, na=False, regex=False).astype(int)
        matches = df[scores > 0].copy()
        if not matches.empty:
            matches["score"] = scores[scores > 0]
            matches = matches.sort_values(["score", "timestamp"], ascending=[False, False])
        else:
            matches = df.tail(limit).copy().sort_values("timestamp", ascending=False)
    else:
        matches = df.tail(limit).copy().sort_values("timestamp", ascending=False)

    matches = matches.head(max(1, min(int(limit or 8), 20)))
    name_map = resolve_contact_names(matches["chat_id"].dropna().astype(str).unique()) if not matches.empty else {}
    citations = [message_row_payload(row, name_map) for _, row in matches.iterrows()]
    scope_label = name_map.get(contact, contact) if contact else "All conversations"
    answer_mode = "local"
    answer = ""
    if use_ai:
        try:
            answer = ai_ask_answer(str(question or "").strip(), citations, scope_label, model)
            if answer:
                answer_mode = "ai"
        except Exception:
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
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
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
            elif parsed.path == "/api/ask":
                payload = parse_body(self)
                messages_path = safe_path(payload.get("messagesPath"), ROOT / "messages.csv")
                send_json(self, 200, ask_messages_payload(
                    messages_path,
                    payload.get("question", ""),
                    contact=payload.get("contact", ""),
                    limit=int(payload.get("limit", 8) or 8),
                    model=payload.get("model", DEFAULT_MODEL),
                    use_ai=str(payload.get("ai", True)).lower() not in {"0", "false", "no", "off"},
                ))
            else:
                send_json(self, 404, {"error": "Not found"})
        except Exception as exc:
            send_json(self, 500, {"error": str(exc)})

    def handle_api_get(self, parsed):
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.path == "/api/defaults":
            db_path = ROOT / "chat.db"
            messages_path = ROOT / "messages.csv"
            send_json(self, 200, {
                "defaultModel": DEFAULT_MODEL,
                "models": list(UI_MODEL_CHOICES),
                "dbPath": str(db_path) if db_path.exists() else str(Path("~/Library/Messages/chat.db").expanduser()),
                "messagesPath": str(messages_path),
                "outDir": str(ROOT / "out"),
                "hasDb": db_path.exists(),
                "hasMessages": messages_path.exists(),
                "reports": list_report_files(),
                "contactNames": contacts_cache_summary(),
            })
        elif parsed.path == "/api/contacts":
            db_path = safe_path(query.get("dbPath"))
            messages_path = safe_path(query.get("messagesPath"), ROOT / "messages.csv")
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
            messages_path = safe_path(query.get("messagesPath"), ROOT / "messages.csv")
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
            messages_path = safe_path(query.get("messagesPath"), ROOT / "messages.csv")
            send_json(self, 200, search_messages_payload(
                messages_path,
                query=query.get("query", ""),
                contact=query.get("contact", ""),
                limit=int(query.get("limit", "60")),
            ))
        elif parsed.path == "/api/analysis":
            messages_path = safe_path(query.get("messagesPath"), ROOT / "messages.csv")
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


def main():
    host = os.environ.get("RECALL_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("RECALL_UI_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), RecallHandler)
    print(f"Recall UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Recall UI")


if __name__ == "__main__":
    main()
