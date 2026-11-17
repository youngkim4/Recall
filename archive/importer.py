"""Import a Messages database (chat.db) into the archive.

The archive mirrors the source. A full pull reads every row: after it the
archive holds exactly the messages of the source. A row that the source
still holds keeps its id (upsert by guid). A row whose guid is gone from the
source is removed. An incremental run reads only new and changed rows and
removes nothing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from archive import directory, ledger, mirror, run_control, runs
from archive.classify import classify
from archive.people import ContactGroup, contact_groups
from archive.records import (
    AttachmentRow,
    MessageRecord,
    MessageRow,
    ReactionRecord,
    ReactionRow,
    Record,
    Skipped,
    SourceAttachment,
    SourceChat,
    SourceMessage,
)
from archive.run_control import ImportResult, Progress, Run
from archive.schema import INDEXED_TEXT_CHARS
from archive.source import SourceReader, open_snapshot

__all__ = ["ImportResult", "import_chat_db", "work_dir"]

BATCH_ROWS = 1000
# An incremental run reads again the rows whose ROWID is at most this far
# below its mark. Recent rows can still change: an attachment arrives, a
# chat link is written late.
OVERLAP_ROWS = 2000
CHECKPOINT_ROWS = 8


@dataclass(frozen=True)
class _Lookups:
    """Source ROWIDs to archive ids."""

    handles: Mapping[int, int]
    handle_text: Mapping[int, str]
    conversations: Mapping[int, int]


def work_dir(archive_path: Path) -> Path:
    """The private folder that holds the copy of the source during a run. Each archive has its own."""
    return archive_path.parent / f".{archive_path.name}.import-work"


def _extra_json(record: MessageRecord, lookups: _Lookups) -> str | None:
    extra = dict(record.extra)
    other = lookups.handle_text.get(record.other_handle_rowid) if record.other_handle_rowid else None
    if other:
        extra["other_handle"] = other
    return json.dumps(extra, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if extra else None


def _message_row(record: MessageRecord, conversation_id: int | None, lookups: _Lookups) -> MessageRow:
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


def _classified(row: SourceMessage, has_attachments: bool) -> Record:
    """Classify one row. A row with a value that cannot be used is skipped: it must not stop every import."""
    try:
        return classify(row, has_attachments=has_attachments)
    except (ValueError, OverflowError, TypeError):
        return Skipped("bad_row")


def _store_batch(
    conn: sqlite3.Connection, run: Run, reader: SourceReader, lookups: _Lookups, batch: Sequence[SourceMessage]
) -> None:
    """Classify one batch of source rows and write it."""
    attachments = reader.attachments_for([row.rowid for row in batch])
    messages: list[MessageRow] = []
    reactions: list[ReactionRow] = []
    files: list[AttachmentRow] = []
    counts = run.counts
    for row in batch:
        counts["rows_seen"] += 1
        record = _classified(row, row.rowid in attachments)
        if isinstance(record, Skipped):
            counts[f"skipped_{record.reason}"] += 1
            counts["decode_failures"] += int(record.decode_failed)
            continue
        if isinstance(record, ReactionRecord):
            counts["reactions_seen"] += 1
            reactions.append(_reaction_row(record, lookups))
            continue
        conversation_id = lookups.conversations.get(record.chat_rowid) if record.chat_rowid is not None else None
        counts["without_conversation"] += int(conversation_id is None)  # stored, in no conversation
        counts["system_seen" if record.kind == "system" else "messages_seen"] += 1
        counts["blob_texts"] += int(record.text_source == "blob")
        counts["decode_failures"] += int(record.decode_failed)
        counts["oversize_texts"] += int(len(record.text) > INDEXED_TEXT_CHARS)
        messages.append(_message_row(record, conversation_id, lookups))
        files.extend(_attachment_rows(record.guid, attachments.get(row.rowid, ())))

    moved = ledger.drop_other_kind(conn, [row.guid for row in messages], [row.guid for row in reactions])
    counts["messages_removed"] += moved.messages_removed
    counts["reactions_removed"] += moved.reactions_removed
    written = ledger.write_messages(conn, run.run_id, messages)
    counts["messages_added"] += written.messages_added
    counts["system_added"] += written.system_added
    counts["messages_updated"] += written.messages_updated
    counts["system_updated"] += written.system_updated
    run.touched.update(moved.conversations | written.conversations)
    tapbacks = ledger.write_reactions(conn, run.run_id, reactions)
    counts["reactions_added"] += tapbacks.added
    counts["reactions_updated"] += tapbacks.updated
    run.waiting.extend(tapbacks.waiting)
    records = ledger.write_attachments(conn, [row.guid for row in messages], files)
    counts["attachments_added"] += records.added
    counts["attachments_removed"] += records.removed


def _settle_waiting(conn: sqlite3.Connection, run: Run) -> None:
    """Store the tapbacks whose target came in a later batch. The others have no target in the archive."""
    for start in range(0, len(run.waiting), BATCH_ROWS):
        settled = ledger.write_reactions(conn, run.run_id, run.waiting[start : start + BATCH_ROWS])
        run.counts["reactions_added"] += settled.added
        run.counts["reactions_updated"] += settled.updated
        run.counts["skipped_no_target"] += len(settled.waiting)
    run.waiting.clear()


def _sync_directory(
    conn: sqlite3.Connection,
    run: Run,
    reader: SourceReader,
    source_handles: Sequence[tuple[int, str]],
    chats: Sequence[SourceChat],
    *,
    replace: bool,
) -> _Lookups:
    handle_ids, handles_added = directory.sync_handles(conn, source_handles)
    conversation_ids, conversations_added = directory.sync_conversations(conn, chats)
    changed = directory.sync_members(
        conn,
        (
            (conversation_ids[chat], handle_ids[handle])
            for chat, handle in reader.members()
            if chat in conversation_ids and handle in handle_ids
        ),
        replace=replace,
    )
    run.touched.update(changed)  # a member that joined: count what that member wrote
    run.counts["handles_added"] += handles_added
    run.counts["conversations_added"] += conversations_added
    return _Lookups(handle_ids, {rowid: text.strip() for rowid, text in source_handles}, conversation_ids)


def _read_rows(conn: sqlite3.Connection, run: Run, reader: SourceReader, lookups: _Lookups, start: int) -> None:
    """Read every row above ROWID `start`, then the older rows with an edit or unsend mark."""
    total = reader.count_after(start)
    done = 0
    run.tell("messages", done, total)
    for batch in reader.messages_after(start, BATCH_ROWS):
        _store_batch(conn, run, reader, lookups, batch)
        done += len(batch)
        run.tell("messages", done, total)
        run.check_stop()
    changed = 0
    for batch in reader.changed_messages(start, BATCH_ROWS):
        _store_batch(conn, run, reader, lookups, batch)
        changed += len(batch)
        run.check_stop()
    run.tell("changes", changed, changed)
    _settle_waiting(conn, run)


def _remove_gone(conn: sqlite3.Connection, run: Run) -> None:
    """The mirror step of a full pull: remove the rows whose guid the source no longer has."""
    removed = mirror.remove_gone(conn)
    run.counts["messages_removed"] += removed.messages
    run.counts["system_removed"] += removed.system
    run.counts["reactions_removed"] += removed.reactions
    run.counts["attachments_removed"] += removed.attachments


def _prune_directory(
    conn: sqlite3.Connection, run: Run, chats: Sequence[SourceChat], source_handles: Sequence[tuple[int, str]]
) -> None:
    """The last step of a full pull: tapbacks, conversations and handles that nothing holds any more."""
    run.counts["reactions_removed"] += mirror.drop_orphan_tapbacks(conn)  # the target became a tapback itself
    directory.drop_gone(
        conn,
        {(chat.identifier or "").strip() for chat in chats},
        {text.strip() for _, text in source_handles},
    )


def _pull(conn: sqlite3.Connection, run: Run, reader: SourceReader, full: bool, max_removed_share: float) -> None:
    """Read the rows this run needs and write them. A full pull first removes what is gone from the source."""
    replaces_bridge = mirror.has_bridge_rows(conn)
    mark = 0 if full or replaces_bridge else runs.verified_mark(conn, reader.guid_at)
    is_full = mark == 0
    run.mode = "full" if is_full else "incremental"
    if is_full:
        mirror.note_source_guids(conn, reader.guids(BATCH_ROWS))  # small chunks: memory stays flat
        mirror.check_removal_limit(conn, max_removed_share)  # before the first change
        run.counts["legacy_rows_replaced"] = mirror.clear_bridge_rows(conn)
        _remove_gone(conn, run)
    source_handles = reader.handles()
    chats = reader.chats()
    lookups = _sync_directory(conn, run, reader, source_handles, chats, replace=is_full)
    _read_rows(conn, run, reader, lookups, max(0, mark - OVERLAP_ROWS))
    if is_full:
        _prune_directory(conn, run, chats, source_handles)
        run.touched.update(directory.conversation_ids(conn))  # a full pull counts every total again
    elif run.counts["messages_removed"]:  # a message became a tapback: the tapbacks on it have no target now
        run.counts["reactions_removed"] += mirror.drop_orphan_tapbacks(conn)


def _run_import(
    conn: sqlite3.Connection, run: Run, groups: Sequence[ContactGroup] | None, full: bool, max_removed_share: float
) -> None:
    """Everything inside the transaction."""
    run.check_stop()
    run.run_id = runs.start_run(conn, str(run.source), run.mode, run.started_at)

    def copy_step(done: int, total: int) -> None:
        run.tell("copy", done, total)
        run.check_stop()

    with open_snapshot(run.source, work_dir(run.archive), copy_step) as reader:
        _pull(conn, run, reader, full, max_removed_share)
        run.tell("finish", 0, 1)
        directory.settle(conn, groups, run.touched)
        checkpoints = reader.top_rows(CHECKPOINT_ROWS)
    runs.finish_run(
        conn, run.run_id, mode=run.mode, counts=dict(run.counts), checkpoints=checkpoints, finished_at=int(time.time())
    )
    run.tell("finish", 1, 1)


def import_chat_db(
    source_path: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
    *,
    contacts: Sequence[Mapping[str, object]] | None = None,
    full: bool = True,
    max_removed_share: float = mirror.DEFAULT_MAX_REMOVED_SHARE,
    progress: Progress | None = None,
    cancel: object = None,
) -> ImportResult:
    """Pull `source_path` (a chat.db) into the archive at `archive_path`.

    - `contacts`: records `{name, phones, emails}` from the Contacts cache.
      None keeps the people the archive has.
    - `full`: read every row and make the archive equal to the source: a
      message that the source no longer has is removed. False reads only the
      rows that are new since the last run of the same database, plus edited
      and unsent rows, and removes nothing.
    - `max_removed_share`: the safety valve of a full pull. When the source
      lacks more than this share of the stored messages, the pull raises
      RemovalLimitError before it changes anything. 1 switches the valve off.
    - `progress`: called with `{"phase", "done", "total"}`. Phases are `copy`,
      `messages`, `changes` and `finish`.
    - `cancel`: a callable that returns bool, or an object with `is_set()`.

    Returns an ImportResult with status `ok` or `cancelled`. A failure raises
    an ArchiveError after the failed run is recorded.
    """
    share = mirror.check_share(max_removed_share)
    groups, contacts_skipped = contact_groups(contacts) if contacts is not None else (None, 0)
    run = run_control.new_run(source_path, archive_path, "full" if full else "incremental", progress, cancel)
    run.counts["contacts_skipped"] = contacts_skipped
    return run_control.execute(run, lambda conn, state: _run_import(conn, state, groups, bool(full), share))
