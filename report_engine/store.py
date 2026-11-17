"""Reports on disk.

    <root>/<conversation folder>/<report id>.json

One folder per conversation, one file per run. A file is written whole or not
at all, and a run never replaces the file of an older run. The folder name is
a readable prefix plus a hash of the exact conversation id, so two ids never
share a folder.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from report_engine.errors import ReportNotFound, StoreError
from report_engine.markdown import render_markdown
from report_engine.models import SCHEMA_VERSION, Report, ReportStatus, Scope

__all__ = ["ReportEntry", "ReportStore", "conversation_folder", "render_markdown", "write_text_atomic"]

logger = logging.getLogger(__name__)

REPORT_SUFFIX = ".json"
FOLDER_PREFIX_CHARS = 40
FOLDER_HASH_CHARS = 10
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


@dataclass(frozen=True)
class ReportEntry:
    """One saved run, as a list shows it."""

    id: str
    conversation: str
    created_at: str
    model: str
    status: ReportStatus
    scope: Scope
    cost_usd: float
    periods: int
    moments: int
    path: Path


def conversation_folder(conversation: str) -> str:
    kept = "".join(char for char in conversation if char.isalnum() or char in "+-_")
    prefix = kept[:FOLDER_PREFIX_CHARS] or "conversation"
    digest = hashlib.sha256(conversation.encode("utf-8")).hexdigest()[:FOLDER_HASH_CHARS]
    return f"{prefix}-{digest}"


def _fsync_folder(folder: Path) -> None:
    try:
        handle = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass  # not every file system can sync a folder; the file itself is synced
    finally:
        os.close(handle)


def _write_temp(folder: Path, name: str, text: str) -> Path:
    temp = folder / f".{name}.{secrets.token_hex(4)}.tmp"
    with open(temp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return temp


def write_text_atomic(path: str | os.PathLike[str], text: str) -> Path:
    """Write a text file in one step. An existing file is replaced."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = _write_temp(target.parent, target.name, text)
    except OSError as exc:
        raise StoreError(f"can not write {target.name}: {exc}") from exc
    try:
        os.replace(temp, target)
    except OSError as exc:
        temp.unlink(missing_ok=True)
        raise StoreError(f"can not write {target.name}: {exc}") from exc
    _fsync_folder(target.parent)
    return target


def _publish_new(temp: Path, target: Path) -> None:
    """Give a finished temp file its final name. Never replaces a file."""
    try:
        os.link(temp, target)
        return
    except FileExistsError as exc:
        raise StoreError(f"report {target.stem} already exists; a run never overwrites one") from exc
    except OSError:
        pass  # no hard links on this file system
    if target.exists():
        raise StoreError(f"report {target.stem} already exists; a run never overwrites one")
    os.replace(temp, target)


def _read(path: Path) -> Report:
    try:
        report = Report.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise StoreError(f"report file {path.name} can not be read: {type(exc).__name__}") from exc
    if report.schema_version > SCHEMA_VERSION:
        raise StoreError(
            f"report file {path.name} was written by a newer version "
            f"(schema {report.schema_version}; this build reads up to {SCHEMA_VERSION})"
        )
    return report


def _entry(report: Report, path: Path) -> ReportEntry:
    return ReportEntry(
        id=report.id,
        conversation=report.conversation,
        created_at=report.created_at,
        model=report.model,
        status=report.status,
        scope=report.scope,
        cost_usd=report.usage.cost_usd,
        periods=len(report.periods),
        moments=len(report.moments),
        path=path,
    )


class ReportStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def _folder(self, conversation: str) -> Path:
        return self._root / conversation_folder(conversation)

    def _path(self, conversation: str, report_id: str) -> Path:
        if not isinstance(report_id, str) or not _SAFE_ID.fullmatch(report_id):
            raise StoreError(f"not a usable report id: {report_id!r}")
        return self._folder(conversation) / f"{report_id}{REPORT_SUFFIX}"

    def save(self, report: Report) -> Path:
        """Write one run. Raises StoreError when a report with this id exists."""
        target = self._path(report.conversation, report.id)
        if target.exists():
            raise StoreError(f"report {report.id} already exists; a run never overwrites one")
        text = report.model_dump_json(indent=2) + "\n"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = _write_temp(target.parent, target.name, text)
        except OSError as exc:
            raise StoreError(f"can not write report {report.id}: {exc}") from exc
        try:
            _publish_new(temp, target)
        except OSError as exc:
            raise StoreError(f"can not write report {report.id}: {exc}") from exc
        finally:
            temp.unlink(missing_ok=True)
        _fsync_folder(target.parent)
        return target

    def list_reports(self, conversation: str | None = None) -> list[ReportEntry]:
        """Saved runs, the newest first. A file that can not be read is skipped and logged."""
        if conversation is not None:
            folders = [self._folder(conversation)]
        elif self._root.is_dir():
            folders = sorted(path for path in self._root.iterdir() if path.is_dir())
        else:
            folders = []
        entries: list[ReportEntry] = []
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in folder.glob(f"[!.]*{REPORT_SUFFIX}"):
                try:
                    entries.append(_entry(_read(path), path))
                except StoreError as exc:
                    logger.warning("skipped a report file: %s", exc)
        return sorted(entries, key=lambda entry: (entry.created_at, entry.id), reverse=True)

    def load_report(self, conversation: str, report_id: str) -> Report:
        path = self._path(conversation, report_id)
        if not path.is_file():
            raise ReportNotFound(f"no report {report_id} for this conversation")
        return _read(path)

    def latest_report(self, conversation: str) -> Report | None:
        entries = self.list_reports(conversation)
        return _read(entries[0].path) if entries else None
