"""Small local SQLite store for app-owned cache and state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE_PATH = ROOT / "saves" / "recall_store.sqlite3"
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def stable_cache_key(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def connect_store(path: Path | None = None) -> sqlite3.Connection:
    store_path = Path(path or DEFAULT_STORE_PATH)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(store_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache_entries (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT NOT NULL,
            PRIMARY KEY (namespace, key)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cache_entries_last_accessed
        ON cache_entries(namespace, last_accessed_at)
    """)
    conn.execute(
        """
        INSERT INTO app_meta(key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def get_cache_entry(namespace: str, key: str, path: Path | None = None) -> dict | None:
    with connect_store(path) as conn:
        row = conn.execute(
            """
            SELECT value_json, metadata_json
            FROM cache_entries
            WHERE namespace = ? AND key = ?
            """,
            (namespace, key),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            """
            UPDATE cache_entries
            SET last_accessed_at = ?
            WHERE namespace = ? AND key = ?
            """,
            (utc_now(), namespace, key),
        )
        conn.commit()

    try:
        return {
            "value": json.loads(row["value_json"]),
            "metadata": json.loads(row["metadata_json"]),
        }
    except json.JSONDecodeError:
        delete_cache_entry(namespace, key, path)
        return None


def set_cache_entry(
    namespace: str,
    key: str,
    value: Any,
    metadata: dict | None = None,
    path: Path | None = None,
    limit: int | None = None,
) -> None:
    now = utc_now()
    with connect_store(path) as conn:
        conn.execute(
            """
            INSERT INTO cache_entries(
                namespace,
                key,
                value_json,
                metadata_json,
                created_at,
                updated_at,
                last_accessed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value_json = excluded.value_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at,
                last_accessed_at = excluded.last_accessed_at
            """,
            (
                namespace,
                key,
                stable_json(value),
                stable_json(metadata or {}),
                now,
                now,
                now,
            ),
        )
        if limit:
            prune_cache_namespace(conn, namespace, limit)
        conn.commit()


def delete_cache_entry(namespace: str, key: str, path: Path | None = None) -> None:
    with connect_store(path) as conn:
        conn.execute(
            "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        conn.commit()


def clear_cache_namespace(namespace: str, path: Path | None = None) -> None:
    with connect_store(path) as conn:
        conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
        conn.commit()


def prune_cache_namespace(conn: sqlite3.Connection, namespace: str, limit: int) -> None:
    conn.execute(
        """
        DELETE FROM cache_entries
        WHERE namespace = ?
          AND key NOT IN (
            SELECT key
            FROM cache_entries
            WHERE namespace = ?
            ORDER BY last_accessed_at DESC
            LIMIT ?
          )
        """,
        (namespace, namespace, limit),
    )
