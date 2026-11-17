"""Import a Messages database (chat.db) into the archive.

One run is one transaction. It either commits everything or leaves the
archive data as it was. Messages are upserted by guid and never deleted, so
the archive keeps what the source lost.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from archive import connection, directory, ledger, runs
from archive.classify import classify
from archive.errors import ArchiveError, ImportBusyError, ImportFailed
from archive.people import ContactGroup, contact_groups
from archive.records import (
    AttachmentRow,
    MessageRecord,
    MessageRow,
    ReactionRecord,
    ReactionRow,
    Skipped,
    SourceAttachment,
    SourceMessage,
)
from archive.source import SourceReader, open_snapshot

WORK_DIR_NAME = ".recall-import-work"
BATCH_ROWS = 1000
# An incremental run reads these many rows below its mark again. Recent rows
# can still change: an attachment arrives, a chat link is written late.
OVERLAP_ROWS = 2000
CHECKPOINT_ROWS = 8
LOCK_WAIT_SECONDS = 2.0

Progress = Callable[[dict[str, Any]], None]
COUNT_NAMES = (
    "rows_seen",
    "messages_seen",
    "reactions_seen",
    "system_seen",
    "messages_added",
    "messages_updated",
    "reactions_added",
    "system_added",
    "attachments_added",
    "conversations_added",
    "handles_added",
    "blob_texts",
    "decode_failures",
)


class _Cancelled(Exception):
    """The caller asked to stop."""


@dataclass(frozen=True)
class ImportResult:
    """What one import run did. `status` is `ok` or `cancelled`."""

    run_id: int
    status: str
    mode: str  # full | incremental: what the run read, not what was asked
    source_path: str
    archive_path: str
    started_at: int
    finished_at: int
    rows_seen: int = 0
    messages_seen: int = 0
    reactions_seen: int = 0
    system_seen: int = 0
    messages_added: int = 0
    messages_updated: int = 0
    reactions_added: int = 0
    system_added: int = 0
    attachments_added: int = 0
    conversations_added: int = 0
    handles_added: int = 0
    blob_texts: int = 0
    decode_failures: int = 0
    skipped: Mapping[str, int] = field(default_factory=dict)  # reason -> rows not stored
    counts: Mapping[str, int] = field(default_factory=dict)  # every counter of the run

    def as_dict(self) -> dict[str, Any]:
        """A plain dict that json.dumps accepts."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return {**values, "skipped": dict(self.skipped), "counts": dict(self.counts)}


@dataclass
class _Run:
    """Working state of one run."""

    source: Path
    archive: Path
    started_at: int
    mode: str
    report: Progress | None
    should_stop: Callable[[], bool]
    run_id: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    touched: set[int] = field(default_factory=set)

    def tell(self, phase: str, done: int, total: int) -> None:
        if self.report is not None and total > 0:
            self.report({"phase": phase, "done": min(done, total), "total": total})

    def check_stop(self) -> None:
        if self.should_stop():
            raise _Cancelled


@dataclass(frozen=True)
class _Lookups:
    """Source ROWIDs to archive ids."""

    handles: Mapping[int, int]
    handle_text: Mapping[int, str]
    conversations: Mapping[int, int]


def _stop_check(cancel: object) -> Callable[[], bool]:
    if cancel is None:
        return lambda: False
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return lambda: bool(is_set())
    if callable(cancel):
        return lambda: bool(cancel())
    raise TypeError("cancel must be a callable or an object with is_set()")


def _extra_json(record: MessageRecord, lookups: _Lookups) -> str | None:
    extra = dict(record.extra)
    other = lookups.handle_text.get(record.other_handle_rowid) if record.other_handle_rowid else None
    if other:
        extra["other_handle"] = other
    return json.dumps(extra, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if extra else None


def _message_row(record: MessageRecord, conversation_id: int, lookups: _Lookups) -> MessageRow:
    sender = lookups.handles.get(record.handle_rowid) if record.handle_rowid is not None else None
    return MessageRow(
        guid=record.guid,
        conversation_id=conversation_id,
        sender_handle_id=sender,
        is_from_me=int(record.is_from_me),
        sent_ns=record.sent_ns,
        text=record.text,
        kind=record.kind,
        service=record.service,
        reply_to_guid=record.reply_to_guid,
        edited_at=record.edited_at,
        unsent=int(record.unsent),
        item_type=record.item_type,
        text_source=record.text_source,
        extra=_extra_json(record, lookups),
        source_rowid=record.source_rowid,
    )


def _reaction_row(record: ReactionRecord, lookups: _Lookups) -> ReactionRow:
    return ReactionRow(
        guid=record.guid,
        target_guid=record.target_guid,
        target_part=record.target_part,
        conversation_id=lookups.conversations.get(record.chat_rowid) if record.chat_rowid is not None else None,
        sender_handle_id=lookups.handles.get(record.handle_rowid) if record.handle_rowid is not None else None,
        is_from_me=int(record.is_from_me),
        kind=record.kind,
        emoji=record.emoji,
        removed=int(record.removed),
        sent_ns=record.sent_ns,
        source_rowid=record.source_rowid,
    )


def _attachment_rows(message_guid: str, attachments: Sequence[SourceAttachment]) -> list[AttachmentRow]:
    rows = []
    for position, item in enumerate(attachments):
        name = (item.transfer_name or "").strip() or os.path.basename((item.filename or "").strip())
        rows.append(
            AttachmentRow(
                guid=(item.guid or "").strip() or f"{message_guid}/{position}",
                message_guid=message_guid,
                position=position,
                mime_type=(item.mime_type or "").strip() or None,
                file_name=name or None,
                path=(item.filename or "").strip() or None,
                total_bytes=item.total_bytes if item.total_bytes is not None and item.total_bytes >= 0 else None,
            )
        )
    return rows


def _store_batch(
    conn: sqlite3.Connection, run: _Run, reader: SourceReader, lookups: _Lookups, batch: Sequence[SourceMessage]
) -> None:
    """Classify one batch of source rows and write it."""
    attachments = reader.attachments_for([row.rowid for row in batch])
    messages: list[MessageRow] = []
    reactions: list[ReactionRow] = []
    files: list[AttachmentRow] = []
    counts = run.counts
    for row in batch:
        counts["rows_seen"] += 1
        record = classify(row, has_attachments=row.rowid in attachments)
        if isinstance(record, Skipped):
            counts[f"skipped_{record.reason}"] += 1
            counts["decode_failures"] += int(record.decode_failed)
            continue
        if isinstance(record, ReactionRecord):
            counts["reactions_seen"] += 1
            reactions.append(_reaction_row(record, lookups))
            continue
        conversation_id = lookups.conversations.get(record.chat_rowid) if record.chat_rowid is not None else None
        if conversation_id is None:
            counts["skipped_no_conversation"] += 1
            continue
        counts["system_seen" if record.kind == "system" else "messages_seen"] += 1
        counts["blob_texts"] += int(record.text_source == "blob")
        counts["decode_failures"] += int(record.decode_failed)
        messages.append(_message_row(record, conversation_id, lookups))
        files.extend(_attachment_rows(record.guid, attachments.get(row.rowid, ())))

    written = ledger.write_messages(conn, run.run_id, messages)
    counts["messages_added"] += written.messages_added
    counts["system_added"] += written.system_added
    counts["messages_updated"] += written.messages_updated
    counts["reactions_added"] += ledger.write_reactions(conn, run.run_id, reactions)
    counts["attachments_added"] += ledger.write_attachments(conn, files)
    run.touched.update(written.conversations)


def _sync_directory(conn: sqlite3.Connection, run: _Run, reader: SourceReader) -> _Lookups:
    source_handles = reader.handles()
    handle_ids, handles_added = directory.sync_handles(conn, source_handles)
    conversation_ids, conversations_added = directory.sync_conversations(conn, reader.chats())
    directory.sync_members(
        conn,
        (
            (conversation_ids[chat], handle_ids[handle])
            for chat, handle in reader.members()
            if chat in conversation_ids and handle in handle_ids
        ),
    )
    run.counts["handles_added"] += handles_added
    run.counts["conversations_added"] += conversations_added
    return _Lookups(handle_ids, {rowid: text.strip() for rowid, text in source_handles}, conversation_ids)


def _read_source(conn: sqlite3.Connection, run: _Run, reader: SourceReader, full: bool) -> None:
    """Read the rows this run needs and write them."""
    mark = 0 if full else runs.verified_mark(conn, reader.guid_at)
    run.mode = "incremental" if mark else "full"
    lookups = _sync_directory(conn, run, reader)
    start = max(0, mark - OVERLAP_ROWS)
    total = reader.count_after(start)
    done = 0
    for batch in reader.messages_after(start, BATCH_ROWS):
        _store_batch(conn, run, reader, lookups, batch)
        done += len(batch)
        run.tell("messages", done, total)
        run.check_stop()
    changed = 0
    for batch in reader.changed_messages(start, BATCH_ROWS):  # edits and unsends of older messages
        _store_batch(conn, run, reader, lookups, batch)
        changed += len(batch)
        run.check_stop()
    run.tell("changes", changed, changed)


def _finish(conn: sqlite3.Connection, run: _Run, groups: Sequence[ContactGroup] | None) -> None:
    directory.apply_people(conn, groups)
    directory.refresh_totals(conn, run.touched)
    directory.link_conversations(conn)
    directory.drop_unused_people(conn)


def _run_import(conn: sqlite3.Connection, run: _Run, groups: Sequence[ContactGroup] | None, full: bool) -> None:
    """Everything inside the transaction."""
    run.check_stop()
    run.run_id = runs.start_run(conn, str(run.source), run.mode, run.started_at)

    def copy_step(done: int, total: int) -> None:
        run.tell("copy", done, total)
        run.check_stop()

    with open_snapshot(run.source, run.archive.parent / WORK_DIR_NAME, copy_step) as reader:
        _read_source(conn, run, reader, full)
        run.tell("finish", 0, 1)
        _finish(conn, run, groups)
        checkpoints = reader.top_rows(CHECKPOINT_ROWS)
    runs.finish_run(
        conn, run.run_id, mode=run.mode, counts=dict(run.counts), checkpoints=checkpoints, finished_at=int(time.time())
    )
    run.tell("finish", 1, 1)


def _take_write_lock(conn: sqlite3.Connection) -> None:
    """Start the import transaction. Only one import can write to an archive at a time."""
    conn.execute(f"PRAGMA busy_timeout={int(LOCK_WAIT_SECONDS * 1000)}")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        raise ImportBusyError("another import is writing to this archive") from exc
    finally:
        conn.execute(f"PRAGMA busy_timeout={int(connection.BUSY_SECONDS * 1000)}")


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def _record_end(conn: sqlite3.Connection, run: _Run, status: str, error: str | None) -> int:
    return runs.record_ended_run(
        conn,
        source_path=str(run.source),
        mode=run.mode,
        status=status,
        error=error,
        started_at=run.started_at,
        finished_at=int(time.time()),
    )


def _record_failure(conn: sqlite3.Connection, run: _Run, cause: BaseException) -> int | None:
    """Log a failed run. A failure to log must not hide the first error."""
    try:
        return _record_end(conn, run, "failed", f"{type(cause).__name__}: {cause}"[:2000])
    except sqlite3.Error:
        return None


def _after_commit(conn: sqlite3.Connection) -> None:
    """Housekeeping. The run is already safe: an error here is not a failed import."""
    try:
        conn.execute("PRAGMA optimize")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        return


def _result(run: _Run, status: str, run_id: int, counts: Mapping[str, int]) -> ImportResult:
    skipped = {
        name[len("skipped_") :]: value for name, value in counts.items() if name.startswith("skipped_") and value
    }
    return ImportResult(
        run_id=run_id,
        status=status,
        mode=run.mode,
        source_path=str(run.source),
        archive_path=str(run.archive),
        started_at=run.started_at,
        finished_at=int(time.time()),
        skipped=MappingProxyType(skipped),
        counts=MappingProxyType(dict(counts)),
        **{name: int(counts.get(name, 0)) for name in COUNT_NAMES},
    )


def import_chat_db(
    source_path: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
    *,
    contacts: Sequence[Mapping[str, object]] | None = None,
    full: bool = False,
    progress: Progress | None = None,
    cancel: object = None,
) -> ImportResult:
    """Import `source_path` (a chat.db) into the archive at `archive_path`.

    - `contacts`: records `{name, phones, emails}` from the Contacts cache.
      None keeps the people the archive has.
    - `full`: read every row. The default reads only rows that are new since
      the last run of the same database, plus edited and unsent rows.
    - `progress`: called with `{"phase", "done", "total"}`. Phases are `copy`,
      `messages`, `changes` and `finish`.
    - `cancel`: a callable that returns bool, or an object with `is_set()`.

    Returns an ImportResult with status `ok` or `cancelled`. A failure raises
    an ArchiveError after the failed run is recorded.
    """
    groups, contacts_skipped = contact_groups(contacts) if contacts is not None else (None, 0)
    run = _Run(
        source=Path(source_path).expanduser(),
        archive=Path(archive_path).expanduser(),
        started_at=int(time.time()),
        mode="full" if full else "incremental",
        report=progress,
        should_stop=_stop_check(cancel),
    )
    run.counts["contacts_skipped"] = contacts_skipped
    conn = connection.open_writer(run.archive)
    try:
        _take_write_lock(conn)
        try:
            _run_import(conn, run, groups, full)
            conn.execute("COMMIT")
        except _Cancelled:
            _rollback(conn)
            return _result(run, "cancelled", _record_end(conn, run, "cancelled", None), {})
        except BaseException as exc:
            _rollback(conn)
            failed_run = _record_failure(conn, run, exc)
            if isinstance(exc, ArchiveError) or not isinstance(exc, Exception):
                raise
            raise ImportFailed(f"import failed: {exc}", run_id=failed_run) from exc
        _after_commit(conn)
        return _result(run, "ok", run.run_id, run.counts)
    finally:
        conn.close()
