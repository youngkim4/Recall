"""Period cache: the result of each map call, in one small SQLite file.

The key is a hash of the prompt version, the model id, the speaker names and
the period transcript. It holds nothing else: not the notes carried over from
earlier periods, and not the conversation id. A period is read from the cache
whenever the model would be shown the same text again.

The file holds summaries and quotes. Keep it next to the archive.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from report_engine.errors import StoreError
from report_engine.ports import LLMUsage

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
BUSY_TIMEOUT_S = 10.0
PRIVATE_FILE_MODE = 0o600
PRIVATE_FOLDER_MODE = 0o700
QUERY_CHUNK = 500

_CREATE = """
CREATE TABLE IF NOT EXISTS period_results (
    key TEXT PRIMARY KEY,
    conversation TEXT NOT NULL,
    label TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL
)
"""
_COLUMNS = "key, payload, input_tokens, cached_input_tokens, output_tokens, cost_usd, created_at"


def period_key(prompt_version: str, model_id: str, labels: Sequence[str], transcript: str) -> str:
    """Hash of the four things that decide what a map call returns."""
    material = json.dumps(
        [prompt_version, model_id, list(labels), transcript],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """One map result as the model returned it, with what the call cost."""

    key: str
    payload: Mapping[str, Any]
    usage: LLMUsage
    cost_usd: float
    created_at: str = ""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _chunks(keys: Sequence[str]) -> Iterator[Sequence[str]]:
    for start in range(0, len(keys), QUERY_CHUNK):
        yield keys[start : start + QUERY_CHUNK]


class PeriodCache:
    """Opens the file per operation, so threads and processes can share it."""

    def __init__(
        self, path: str | os.PathLike[str], now: Callable[[], datetime] = _utc_now
    ) -> None:
        self._path = Path(path)
        self._now = now
        try:
            self._path.parent.mkdir(mode=PRIVATE_FOLDER_MODE, parents=True, exist_ok=True)
        except OSError as exc:
            raise StoreError(f"can not create the cache folder: {exc}") from exc
        with self._open() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > CACHE_SCHEMA_VERSION:
                raise StoreError(
                    f"period cache has schema version {version}; this build reads up to {CACHE_SCHEMA_VERSION}"
                )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE)
            conn.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
        self._keep_private()

    def _keep_private(self) -> None:
        """The file holds summaries and quotes: it is for its owner only."""
        try:
            os.chmod(self._path, PRIVATE_FILE_MODE)
        except OSError as exc:
            logger.warning("the period cache file could not be made private: %s", exc)

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S)
        except sqlite3.Error as exc:
            raise StoreError(f"can not open the period cache: {exc}") from exc
        try:
            with conn:
                yield conn
        except sqlite3.Error as exc:
            raise StoreError(f"period cache failed: {exc}") from exc
        finally:
            conn.close()

    def _stamp(self) -> str:
        return self._now().astimezone(UTC).isoformat(timespec="seconds")

    def get_many(self, keys: Sequence[str], touch: bool = False) -> dict[str, CacheEntry]:
        """Entries for the keys that have one. `touch` marks them as used now.
        Marking and the removal of broken entries are housekeeping: when they
        fail, the entries that were read are still returned."""
        rows: list[tuple[Any, ...]] = []
        with self._open() as conn:
            for chunk in _chunks(list(keys)):
                marks = ",".join("?" * len(chunk))
                rows.extend(
                    conn.execute(
                        f"SELECT {_COLUMNS} FROM period_results WHERE key IN ({marks})", chunk
                    )
                )
        entries: dict[str, CacheEntry] = {}
        broken: list[str] = []
        for row in rows:
            entry = _entry_from_row(row)
            if entry is None:
                broken.append(row[0])
            else:
                entries[entry.key] = entry
        try:
            if broken:
                self._delete(broken)
            if touch and entries:
                self._touch(list(entries))
        except StoreError as exc:
            logger.warning("period cache housekeeping failed: %s", exc)
        return entries

    def get(self, key: str) -> CacheEntry | None:
        return self.get_many([key]).get(key)

    def put(
        self,
        entry: CacheEntry,
        *,
        conversation: str,
        label: str,
        model: str,
        prompt_version: str,
    ) -> None:
        """Store one result. An entry with the same key is replaced."""
        try:
            payload = json.dumps(entry.payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise StoreError(f"period result is not JSON: {exc}") from exc
        stamp = self._stamp()
        with self._open() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO period_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.key,
                    conversation,
                    label,
                    model,
                    prompt_version,
                    payload,
                    entry.usage.input_tokens,
                    entry.usage.cached_input_tokens,
                    entry.usage.output_tokens,
                    entry.cost_usd,
                    stamp,
                    stamp,
                ),
            )

    def prune(self, unused_for_days: int) -> int:
        """Delete entries no run has read for this many days. Returns how many."""
        cutoff = (self._now().astimezone(UTC) - timedelta(days=unused_for_days)).isoformat(
            timespec="seconds"
        )
        with self._open() as conn:
            return conn.execute(
                "DELETE FROM period_results WHERE last_used_at < ?", (cutoff,)
            ).rowcount

    def clear(self) -> int:
        with self._open() as conn:
            return conn.execute("DELETE FROM period_results").rowcount

    def count(self) -> int:
        with self._open() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM period_results").fetchone()[0])

    def _touch(self, keys: Sequence[str]) -> None:
        stamp = self._stamp()
        with self._open() as conn:
            for chunk in _chunks(keys):
                marks = ",".join("?" * len(chunk))
                conn.execute(
                    f"UPDATE period_results SET last_used_at = ? WHERE key IN ({marks})",
                    (stamp, *chunk),
                )

    def _delete(self, keys: Sequence[str]) -> None:
        with self._open() as conn:
            for chunk in _chunks(keys):
                marks = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM period_results WHERE key IN ({marks})", chunk)


def _entry_from_row(row: tuple[Any, ...]) -> CacheEntry | None:
    key, payload_text, input_tokens, cached_tokens, output_tokens, cost_usd, created_at = row
    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError):
        payload = None
    if not isinstance(payload, dict):
        logger.warning("period cache entry %s is unreadable; it is dropped", str(key)[:8])
        return None
    return CacheEntry(
        key=key,
        payload=payload,
        usage=LLMUsage(
            input_tokens=int(input_tokens),
            cached_input_tokens=int(cached_tokens),
            output_tokens=int(output_tokens),
        ),
        cost_usd=float(cost_usd),
        created_at=str(created_at),
    )
