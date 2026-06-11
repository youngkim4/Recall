"""Fresh-install probing: what state is the user's Messages source in?

The onboarding wizard derives every screen from these answers, so the rules
are strict: never crash (the endpoint always answers 200), never write a
byte to the database (read-only sqlite URIs -- a bare connect() would CREATE
a missing file), and never trust a plain FileNotFoundError on the live path:
without Full Disk Access, macOS TCC reports protected files as missing
rather than permission-denied.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LIVE_MESSAGES_DIR = Path("~/Library/Messages").expanduser()
LIVE_CHAT_DB = LIVE_MESSAGES_DIR / "chat.db"

_APPLE_EPOCH_UNIX = 978307200  # 2001-01-01 in unix seconds
_SQLITE_MAGIC = b"SQLite format 3"


@dataclass(frozen=True)
class DbProbe:
    status: str  # readable | fda_blocked | missing | empty | invalid | locked | error
    kind: str  # live | copy
    path: str
    detail: str = ""
    approx_messages: int | None = None
    conversations: int | None = None
    first_year: int | None = None
    last_year: int | None = None


def probe_database(path: Path | str | None = None, deep: bool = False) -> DbProbe:
    target = Path(path).expanduser() if path else LIVE_CHAT_DB
    try:
        target = target.resolve()
    except OSError:
        pass
    kind = "live" if str(target).startswith(str(LIVE_MESSAGES_DIR)) else "copy"

    def result(status: str, detail: str = "", **extra) -> DbProbe:
        return DbProbe(status=status, kind=kind, path=str(target), detail=detail, **extra)

    # byte probe first: existence + permission + SQLite magic
    try:
        with open(target, "rb") as fh:
            head = fh.read(16)
    except PermissionError:
        return result("fda_blocked")
    except FileNotFoundError:
        if kind == "live":
            # TCC can HIDE the file instead of denying it: only conclude
            # "missing" when the parent directory lists without chat.db
            try:
                names = os.listdir(target.parent)
            except PermissionError:
                return result("fda_blocked")
            except OSError:
                return result("missing")
            return result("fda_blocked" if target.name in names else "missing")
        return result("missing")
    except IsADirectoryError:
        return result("invalid", detail="that is a folder, not a database file")
    except OSError as exc:
        if getattr(exc, "errno", None) in (1, 13):  # EPERM / EACCES
            return result("fda_blocked")
        return result("error", detail=str(exc))

    if not head.startswith(_SQLITE_MAGIC):
        return result("invalid", detail="not a SQLite database")

    try:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.OperationalError as exc:
        return result(_locked_or_invalid(exc), detail=str(exc))

    try:
        try:
            tables = {
                name
                for (name,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        except sqlite3.DatabaseError as exc:
            return result(_locked_or_invalid(exc), detail=str(exc))
        if "message" not in tables:
            return result("invalid", detail="no message table -- not a Messages database")
        try:
            row = conn.execute("SELECT MAX(ROWID) FROM message").fetchone()
        except sqlite3.DatabaseError as exc:
            return result(_locked_or_invalid(exc), detail=str(exc))
        max_rowid = row[0] if row else None
        if not max_rowid:
            return result("empty")
        extra: dict = {}
        if deep:
            extra = _deep_stats(conn)
        return result("readable", approx_messages=int(max_rowid), **extra)
    finally:
        conn.close()


def _locked_or_invalid(exc: Exception) -> str:
    message = str(exc).lower()
    return "locked" if ("locked" in message or "busy" in message) else "invalid"


def _deep_stats(conn: sqlite3.Connection) -> dict:
    out: dict = {}
    try:
        row = conn.execute("SELECT COUNT(*) FROM chat").fetchone()
        if row and row[0]:
            out["conversations"] = int(row[0])
    except sqlite3.DatabaseError:
        pass
    try:
        row = conn.execute("SELECT MIN(date), MAX(date) FROM message WHERE date > 0").fetchone()
        if row and row[0]:
            out["first_year"] = _apple_date_year(row[0])
            out["last_year"] = _apple_date_year(row[1])
    except sqlite3.DatabaseError:
        pass
    return out


def _apple_date_year(value) -> int | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds > 1e12:  # nanoseconds since 2001
        seconds /= 1e9
    try:
        return datetime.fromtimestamp(seconds + _APPLE_EPOCH_UNIX, tz=timezone.utc).year
    except (OverflowError, OSError, ValueError):
        return None


def derive_state(db: DbProbe, export_exists: bool) -> str:
    """Wizard phase from probe + whether an archive already exists. An
    existing export always wins: the app is fully usable from the CSV, so
    the wizard must never trap a user whose live db is merely blocked."""
    if export_exists:
        return "ready"
    if db.status == "readable":
        return "needs_export"
    if db.status == "fda_blocked":
        return "needs_permission"
    if db.status in ("missing", "empty"):
        return "no_messages"
    if db.status == "locked":
        return "db_locked"
    return "db_invalid"
