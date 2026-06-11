"""Best-effort local contact name resolution for iMessage handles."""

from __future__ import annotations

import os
import re
import sqlite3
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from recall_paths import SAVES_DIR


ROOT = Path(__file__).resolve().parent
ADDRESS_BOOK_DIR = Path("~/Library/Application Support/AddressBook").expanduser()
CONTACTS_CACHE_PATH = SAVES_DIR / "contact_names.json"
CONTACTS_EXPORTER_APP = SAVES_DIR / "Recall Contacts Exporter.app"
CONTACTS_EXPORTER_PATH = CONTACTS_EXPORTER_APP / "Contents" / "MacOS" / "RecallContactsExporter"
CONTACTS_EXPORTER_SCRIPT = ROOT / "scripts" / "export_contacts.swift"
CONTACTS_EXPORTER_PLIST = ROOT / "scripts" / "export_contacts_Info.plist"


def handle_keys(value: object) -> set[str]:
    """Return normalized lookup keys for a phone number or email-like handle."""
    text = str(value or "").strip()
    if not text:
        return set()
    if "@" in text:
        return {text.lower()}

    digits = re.sub(r"\D", "", text)
    if not digits:
        return {text.lower()}

    keys = {digits}
    if len(digits) > 10:
        keys.add(digits[-10:])
    return keys


def find_contact_databases(base_dir: Path | None = None) -> list[Path]:
    """Find readable macOS Contacts database files."""
    root = base_dir or ADDRESS_BOOK_DIR
    if not root.exists():
        return []

    candidates = list(root.glob("AddressBook-v*.abcddb"))
    candidates.extend(root.glob("Sources/*/AddressBook-v*.abcddb"))
    candidates.extend(root.glob("Sources/*/*.abcddb"))
    unique = {path.resolve(): path for path in candidates if path.is_file()}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_contact_names(
    handles: Iterable[object],
    base_dir: Path | None = None,
    cache_path: Path | None = CONTACTS_CACHE_PATH,
) -> dict[str, str]:
    """Resolve iMessage handles to local Contacts names when possible."""
    originals = [str(handle) for handle in handles if str(handle or "").strip()]
    if not originals:
        return {}

    key_to_handles: dict[str, list[str]] = {}
    for handle in dict.fromkeys(originals):
        for key in handle_keys(handle):
            key_to_handles.setdefault(key, []).append(handle)

    resolved = load_cached_contact_names(originals, cache_path)
    remaining_handles = [handle for handle in originals if handle not in resolved]
    if not remaining_handles:
        return resolved

    remaining_keys = set()
    for handle in remaining_handles:
        remaining_keys.update(handle_keys(handle))

    wanted_keys = set(key_to_handles)
    for db_path in find_contact_databases(base_dir):
        try:
            key_to_name = read_contact_database(db_path, wanted_keys & remaining_keys)
        except (OSError, PermissionError, sqlite3.Error):
            continue
        for key, name in key_to_name.items():
            for handle in key_to_handles.get(key, []):
                resolved.setdefault(handle, name)
        if len(resolved) == len(set(originals)):
            break
    return resolved


def decorate_contacts_frame(df, name_lookup: Callable[[Iterable[str]], dict[str, str]] | None = None):
    """Add or fill a display_name column to a conversations DataFrame."""
    if df.empty or "chat_id" not in df.columns:
        result = df.copy()
        if "display_name" not in result.columns:
            result["display_name"] = ""
        return result

    result = df.copy()
    if "display_name" not in result.columns:
        result["display_name"] = ""

    handles = result["chat_id"].dropna().astype(str).unique().tolist()
    names = name_lookup(handles) if name_lookup else resolve_contact_names(handles)
    existing = result["display_name"].fillna("").astype(str).str.strip()
    resolved = result["chat_id"].astype(str).map(names).fillna("").astype(str).str.strip()
    result["display_name"] = existing.where(existing.str.len().gt(0), resolved)
    return result


def load_cached_contact_names(handles: Iterable[object], cache_path: Path | None = CONTACTS_CACHE_PATH) -> dict[str, str]:
    """Resolve handles from a locally exported Contacts cache."""
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    records = payload.get("contacts", []) if isinstance(payload, dict) else []
    key_to_name: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        name = clean_name_part(record.get("name"))
        if not name:
            continue
        values = list(record.get("phones") or []) + list(record.get("emails") or [])
        for value in values:
            for key in handle_keys(value):
                key_to_name.setdefault(key, name)

    resolved: dict[str, str] = {}
    for handle in handles:
        text = str(handle)
        for key in handle_keys(text):
            if key in key_to_name:
                resolved[text] = key_to_name[key]
                break
    return resolved


def contacts_cache_summary(cache_path: Path | None = CONTACTS_CACHE_PATH) -> dict:
    if not cache_path:
        return {"exists": False, "count": 0, "exportedAt": None}
    path = Path(cache_path)
    if not path.exists():
        return {"exists": False, "count": 0, "exportedAt": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": True, "count": 0, "exportedAt": None, "invalid": True}
    contacts = payload.get("contacts", []) if isinstance(payload, dict) else []
    return {
        "exists": True,
        "count": len(contacts),
        "exportedAt": payload.get("exportedAt") if isinstance(payload, dict) else None,
    }


def _bundled_contacts_exporter() -> Path | None:
    """Pre-built exporter shipped inside the packaged app, so customer machines
    never need swiftc/Xcode tools. Set by the Mac shell."""
    env = os.environ.get("RECALL_CONTACTS_EXPORTER_APP", "")
    if not env:
        return None
    app = Path(env)
    binary = app / "Contents" / "MacOS" / "RecallContactsExporter"
    if binary.is_file() and os.access(binary, os.X_OK):
        return app
    return None


def export_contact_names(cache_path: Path | None = CONTACTS_CACHE_PATH, timeout: int = 120) -> dict:
    """Run the local macOS Contacts exporter and return cache metadata."""
    path = Path(cache_path or CONTACTS_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    exporter_app = _bundled_contacts_exporter()
    if exporter_app is None:
        compile_contacts_exporter()
        exporter_app = CONTACTS_EXPORTER_APP

    result = subprocess.run(
        ["open", "-W", "-n", str(exporter_app), "--args", str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Contacts export failed.").strip()
        if "denied" in message.lower() or "not granted" in message.lower():
            raise PermissionError(
                "Contacts access was denied. Enable Contacts access for this app in macOS Settings, then try again."
            )
        raise RuntimeError(message)

    summary = contacts_cache_summary(path)
    if not summary.get("exists"):
        raise PermissionError(
            "Contacts access was denied or not completed. Enable Recall Contacts Exporter in macOS Settings, then try again."
        )
    summary["message"] = (result.stdout or "").strip()
    return summary


def compile_contacts_exporter() -> None:
    if not CONTACTS_EXPORTER_SCRIPT.exists():
        raise RuntimeError(f"Contacts exporter missing: {CONTACTS_EXPORTER_SCRIPT}")
    swiftc = shutil.which("swiftc")
    command = [swiftc] if swiftc else ["xcrun", "swiftc"]
    macos_dir = CONTACTS_EXPORTER_APP / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONTACTS_EXPORTER_PLIST, CONTACTS_EXPORTER_APP / "Contents" / "Info.plist")
    command.extend([
        str(CONTACTS_EXPORTER_SCRIPT),
        "-o",
        str(CONTACTS_EXPORTER_PATH),
    ])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Could not compile Contacts exporter.").strip())
    CONTACTS_EXPORTER_PATH.chmod(0o755)
    codesign = shutil.which("codesign")
    if codesign:
        subprocess.run(
            [codesign, "--force", "--deep", "--sign", "-", str(CONTACTS_EXPORTER_APP)],
            capture_output=True,
            text=True,
            check=False,
        )


def read_contact_database(db_path: Path, wanted_keys: set[str]) -> dict[str, str]:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = table_names(conn)
        people = load_people(conn, tables)
        if not people:
            return {}

        matches: dict[str, str] = {}
        for table in tables:
            lowered = table.lower()
            if "phone" in lowered:
                matches.update(read_values(conn, table, PHONE_VALUE_COLUMNS, people, wanted_keys))
            elif "email" in lowered:
                matches.update(read_values(conn, table, EMAIL_VALUE_COLUMNS, people, wanted_keys))
        return matches
    finally:
        conn.close()


PHONE_VALUE_COLUMNS = (
    "ZNORMALIZEDNUMBER",
    "ZFULLNUMBER",
    "ZNUMBER",
    "ZPHONE",
    "ZVALUE",
    "ZADDRESS",
)
EMAIL_VALUE_COLUMNS = (
    "ZADDRESS",
    "ZEMAIL",
    "ZVALUE",
)
OWNER_COLUMNS = (
    "ZOWNER",
    "ZPERSON",
    "ZCONTACT",
    "ZRECORD",
    "ZPARENT",
)
PERSON_TABLE_HINTS = (
    "zabcdrecord",
    "zperson",
    "zcontact",
)
NAME_COLUMNS = (
    "ZFIRSTNAME",
    "ZMIDDLENAME",
    "ZLASTNAME",
    "ZDISPLAYNAME",
    "ZCOMPOSITE_NAME",
    "ZFULLNAME",
    "ZNAME",
    "ZNICKNAME",
    "ZORGANIZATION",
    "ZCOMPANY",
)


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [str(row["name"]) for row in rows if row["name"]]


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return [str(row["name"]) for row in rows]


def load_people(conn: sqlite3.Connection, tables: list[str]) -> dict[str, str]:
    people: dict[str, str] = {}
    for table in tables:
        lowered = table.lower()
        if not any(hint in lowered for hint in PERSON_TABLE_HINTS):
            continue
        columns = table_columns(conn, table)
        if not any(column in columns for column in NAME_COLUMNS):
            continue

        pk_expr = quote_identifier("Z_PK") if "Z_PK" in columns else "rowid"
        rows = conn.execute(f"SELECT {pk_expr} AS __pk__, * FROM {quote_identifier(table)}").fetchall()
        for row in rows:
            name = display_name_from_row(row)
            if name:
                people[str(row["__pk__"])] = name
    return people


def read_values(
    conn: sqlite3.Connection,
    table: str,
    value_candidates: tuple[str, ...],
    people: dict[str, str],
    wanted_keys: set[str],
) -> dict[str, str]:
    columns = table_columns(conn, table)
    owner_columns = [column for column in OWNER_COLUMNS if column in columns]
    value_columns = [column for column in value_candidates if column in columns]
    if not owner_columns or not value_columns:
        return {}

    matches: dict[str, str] = {}
    for owner_column in owner_columns:
        for value_column in value_columns:
            sql = (
                f"SELECT {quote_identifier(owner_column)} AS __owner__, "
                f"{quote_identifier(value_column)} AS __value__ "
                f"FROM {quote_identifier(table)} "
                f"WHERE {quote_identifier(value_column)} IS NOT NULL"
            )
            for row in conn.execute(sql):
                name = people.get(str(row["__owner__"]))
                if not name:
                    continue
                for key in handle_keys(row["__value__"]):
                    if key in wanted_keys:
                        matches.setdefault(key, name)
    return matches


def display_name_from_row(row: sqlite3.Row) -> str:
    values = {key: clean_name_part(row[key]) for key in row.keys()}
    display = first_present(values, "ZDISPLAYNAME", "ZCOMPOSITE_NAME", "ZFULLNAME", "ZNAME")
    parts = [
        first_present(values, "ZFIRSTNAME"),
        first_present(values, "ZMIDDLENAME"),
        first_present(values, "ZLASTNAME"),
    ]
    personal_name = " ".join(part for part in parts if part).strip()
    return personal_name or display or first_present(values, "ZNICKNAME", "ZORGANIZATION", "ZCOMPANY")


def first_present(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return ""


def clean_name_part(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
