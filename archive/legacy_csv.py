"""Bridge: load one legacy CSV export into an empty archive.

The export is the file that the old pipeline wrote (parse_imessage.py). The
bridge exists so that the rest of the app can be built and measured before
the importer reads the Messages database. A later full pull from the
database replaces everything the bridge loaded.

What the export cannot say, and what the bridge does about it:
- It has no guid. A row gets the id `legacy:<ROWID>`.
- It has no member list. The members of a conversation are the senders seen
  in it, plus the handle that names a one-to-one conversation.
- It does not mark tapbacks. A row whose whole text is a tapback phrase
  (`Loved "..."` and the like) is stored as kind `system`: it is hidden by
  default and is not counted as a message.
- Its time stamps are local with no zone. They are read in the zone of this
  machine. A time that the clocks repeat in autumn is read as its first pass.
- Its text is what the old exporter wrote. Text that the exporter cut or
  cleaned is not restored here.
"""

from __future__ import annotations

import csv
import io
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from archive import directory, ledger, mirror, run_control, runs
from archive.classify import OBJECT_REPLACEMENT, stored_text
from archive.errors import ArchiveNotEmptyError, SourceError, SourceFormatError
from archive.filters import MAX_INT
from archive.handles import is_group_identifier
from archive.people import ContactGroup, contact_groups
from archive.records import AttachmentRow, MessageRow
from archive.run_control import ImportResult, Progress, Run
from archive.schema import INDEXED_TEXT_CHARS
from archive.timeconv import APPLE_EPOCH_UNIX, NS_PER_SECOND, local_to_unix

__all__ = ["import_legacy_csv"]

MODE = "legacy_csv"
BATCH_ROWS = 1000
# every column that parse_imessage.py writes
COLUMNS = (
    "message_id",
    "timestamp",
    "sender",
    "text",
    "chat_id",
    "chat_display_name",
    "is_from_me",
    "service",
    "attachment_types",
    "attachment_files",
)
ATTACHMENT_LABEL = "[Attachment]"  # the old exporter wrote this for a message that is only attachments
MAX_FIELD_CHARS = 2**31 - 1  # some texts of the export are larger than the csv module's default limit
_TRUE, _FALSE = frozenset(("1", "1.0", "true")), frozenset(("0", "0.0", "false"))
_QUOTED = '["' + chr(0x201C) + '].*["' + chr(0x201D) + "]"  # text in straight or curly double quotes
_TARGET = rf"(?:{_QUOTED}|an image|a movie|an audio message|an attachment)"
# English tapback summaries, as Messages writes them into the text of a tapback row
_TAPBACK_PHRASE = re.compile(
    rf"(?:Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) {_TARGET}"
    rf"|Removed an? (?:heart|like|dislike|laugh|exclamation|question mark) from {_TARGET}"
    rf"|Reacted \S{{1,16}} to {_QUOTED}",
    re.DOTALL,
)
_field_limit_lock = threading.Lock()
_field_limit_state = {"loads": 0, "kept": 0}  # loads that run now, and the limit to put back after the last one


@dataclass
class _Directory:
    """Ids of the conversations and handles made so far, and what the rows said about them."""

    conversations: dict[str, int] = field(default_factory=dict)
    titles: dict[str, str] = field(default_factory=dict)
    services: dict[str, str] = field(default_factory=dict)
    handles: dict[str, int] = field(default_factory=dict)
    members: set[tuple[int, int]] = field(default_factory=set)


def is_tapback_phrase(text: str) -> bool:
    """True when the whole text is the summary that Messages writes for a tapback."""
    return _TAPBACK_PHRASE.fullmatch(text) is not None


@contextmanager
def _large_fields() -> Iterator[None]:
    """Let the csv module read fields of any size while a load runs.

    The limit belongs to the whole process. Loads that overlap share it: the
    first one raises it and the last one puts the old value back.
    """
    with _field_limit_lock:
        if _field_limit_state["loads"] == 0:
            _field_limit_state["kept"] = csv.field_size_limit(MAX_FIELD_CHARS)
        _field_limit_state["loads"] += 1
    try:
        yield
    finally:
        with _field_limit_lock:
            _field_limit_state["loads"] -= 1
            if _field_limit_state["loads"] == 0:
                csv.field_size_limit(_field_limit_state["kept"])


def _require_empty(conn: sqlite3.Connection) -> None:
    for table in ("messages", "reactions", "conversations", "handles"):
        if conn.execute(f"SELECT EXISTS (SELECT 1 FROM {table})").fetchone()[0]:
            raise ArchiveNotEmptyError("the legacy export bridge loads only into an empty archive")


def _header(reader: Iterator[list[str]]) -> list[str]:
    header = next(reader, None) or []
    missing = [name for name in COLUMNS if name not in header]
    if missing:
        raise SourceFormatError(f"the export has no column {', '.join(missing)}")
    if len(set(header)) != len(header):
        raise SourceFormatError("the export names a column more than one time")
    return header


def _rows(path: Path, run: Run) -> Iterator[list[dict[str, str] | None]]:
    """The rows of the export in batches. A row with too few or too many fields is None.

    Progress is counted in bytes of the file. The reader is strict: a quote
    that is never closed is an error, not one field that holds the rest of the file.
    """
    total = path.stat().st_size
    with path.open("rb") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), strict=True)
        header = _header(reader)
        batch: list[dict[str, str] | None] = []
        for values in reader:
            if not values:
                continue  # an empty line
            batch.append(dict(zip(header, values, strict=True)) if len(values) == len(header) else None)
            if len(batch) == BATCH_ROWS:
                yield batch
                batch = []
                run.tell("messages", raw.tell(), total)
        if batch:
            yield batch
    run.tell("messages", total, total)


def _content(text: str, has_files: bool) -> tuple[str, str] | None:
    """(kind, text) of a row, or None for a row with nothing to show."""
    if text == ATTACHMENT_LABEL or (has_files and text.startswith(ATTACHMENT_LABEL + " ")):
        return "attachment", ""
    kept = stored_text(text)
    if is_tapback_phrase(kept):
        return "system", kept
    if kept.strip():
        return "text", kept
    if has_files or OBJECT_REPLACEMENT in text:
        return "attachment", kept
    return ("text", kept) if kept else None


def _files(guid: str, types: str, paths: str) -> list[AttachmentRow]:
    """Attachment records from the two `|` lists of the export.

    The old exporter left a missing value out of its list. Lists of different
    length cannot be paired, so no record then gets a type.
    """
    kinds = types.split("|") if types else []
    names = paths.split("|") if paths else []
    paired = len(kinds) == len(names) or not names
    rows = []
    for position in range(max(len(kinds), len(names))):
        path = names[position].strip() if position < len(names) else ""
        mime_type = kinds[position].strip() if paired and position < len(kinds) else ""
        rows.append(
            AttachmentRow(
                guid=f"{guid}/{position}",
                message_guid=guid,
                position=position,
                mime_type=mime_type or None,
                file_name=os.path.basename(path) or None,
                path=path or None,
                total_bytes=None,
            )
        )
    return rows


def _handle_id(conn: sqlite3.Connection, run: Run, known: _Directory, text: str) -> int:
    if text not in known.handles:
        known.handles[text] = directory.add_handle(conn, text)
        run.counts["handles_added"] += 1
    return known.handles[text]


def _conversation_id(conn: sqlite3.Connection, run: Run, known: _Directory, row: Mapping[str, str]) -> int:
    """The conversation of a row. The first row of a chat makes it; later rows can name its title and service."""
    identifier = row["chat_id"].strip()
    title, service = row["chat_display_name"].strip(), row["service"].strip()
    if identifier not in known.conversations:
        known.conversations[identifier] = directory.add_conversation(conn, identifier)
        run.counts["conversations_added"] += 1
        if not is_group_identifier(identifier):
            known.members.add((known.conversations[identifier], _handle_id(conn, run, known, identifier)))
    if title:
        known.titles[identifier] = title
    services = [name for name in (known.services.get(identifier, ""), service) if name]
    if services:
        known.services[identifier] = min(services, key=directory.service_rank)
    return known.conversations[identifier]


def _message(conn: sqlite3.Connection, run: Run, known: _Directory, row: Mapping[str, str] | None) -> MessageRow | str:
    """The archive row for one row of the export, or the reason it is skipped."""
    if row is None:
        return "bad_row"
    number, direction = row["message_id"].strip(), row["is_from_me"].strip().lower()
    if not number.isascii() or not number.isdigit() or int(number) > MAX_INT or direction not in _TRUE | _FALSE:
        return "bad_row"
    sent = local_to_unix(row["timestamp"])
    if sent is None or sent <= APPLE_EPOCH_UNIX:  # the exporter wrote the start of Apple time for a row with no date
        return "no_date"
    has_files = bool(row["attachment_types"] or row["attachment_files"])
    content = _content(row["text"], has_files)
    if content is None:
        return "empty"
    kind, text = content
    # a row with no chat id is a message that belongs to no conversation
    conversation_id = _conversation_id(conn, run, known, row) if row["chat_id"].strip() else None
    sender = row["sender"].strip()
    is_from_me = direction in _TRUE
    sender_id = None
    if sender and not is_from_me:  # on a sent row the export names the other person, not the sender
        sender_id = _handle_id(conn, run, known, sender)
        if conversation_id is not None:
            known.members.add((conversation_id, sender_id))
    return MessageRow(
        guid=mirror.legacy_guid(int(number)),
        conversation_id=conversation_id,
        sender_handle_id=sender_id,
        is_from_me=int(is_from_me),
        sent_ns=sent * NS_PER_SECOND,
        text=text,
        kind=kind,
        service=row["service"].strip(),
        reply_to_guid=None,
        edited_at=None,
        unsent=0,
        item_type=0,
        text_source="export" if text else "none",
        extra=None,
        source_rowid=int(number),
    )


def _store_batch(
    conn: sqlite3.Connection, run: Run, known: _Directory, batch: Sequence[Mapping[str, str] | None]
) -> None:
    kept: dict[str, tuple[MessageRow, list[AttachmentRow]]] = {}
    counts = run.counts
    for row in batch:
        counts["rows_seen"] += 1
        found = _message(conn, run, known, row)
        if isinstance(found, str) or row is None:
            counts[f"skipped_{found}"] += 1
            continue
        counts["system_seen" if found.kind == "system" else "messages_seen"] += 1
        counts["reaction_phrases"] += int(found.kind == "system")
        counts["without_conversation"] += int(found.conversation_id is None)
        counts["oversize_texts"] += int(len(found.text) > INDEXED_TEXT_CHARS)
        counts["messages_updated"] += int(found.guid in kept)  # one ROWID twice in the export: the last row wins
        kept[found.guid] = (found, _files(found.guid, row["attachment_types"], row["attachment_files"]))
    messages = [message for message, _ in kept.values()]
    written = ledger.write_messages(conn, run.run_id, messages)
    counts["messages_added"] += written.messages_added
    counts["system_added"] += written.system_added
    counts["messages_updated"] += written.messages_updated
    counts["system_updated"] += written.system_updated
    run.touched.update(written.conversations)
    files = [record for _, records in kept.values() for record in records]
    counts["attachments_added"] += ledger.write_attachments(conn, [row.guid for row in messages], files).added


def _load(conn: sqlite3.Connection, run: Run, groups: Sequence[ContactGroup] | None) -> None:
    """Everything inside the transaction."""
    run.check_stop()
    _require_empty(conn)
    run.run_id = runs.start_run(conn, str(run.source), MODE, run.started_at)
    known = _Directory()
    for batch in _rows(run.source, run):
        _store_batch(conn, run, known, batch)
        run.check_stop()
    run.tell("finish", 0, 1)
    directory.set_details(
        conn,
        [
            (known.titles.get(identifier, ""), known.services.get(identifier, ""), conversation_id)
            for identifier, conversation_id in known.conversations.items()
        ],
    )
    directory.sync_members(conn, known.members, replace=False)
    directory.settle(conn, groups, run.touched)
    runs.finish_run(conn, run.run_id, mode=MODE, counts=dict(run.counts), checkpoints=(), finished_at=int(time.time()))
    run.tell("finish", 1, 1)


def _guarded_load(conn: sqlite3.Connection, run: Run, groups: Sequence[ContactGroup] | None) -> None:
    """Run the load. An error of the file becomes an error of this package, with no text of the file in it."""
    try:
        with _large_fields():
            _load(conn, run, groups)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise SourceFormatError(f"the export is not a CSV file this bridge can read: {type(exc).__name__}") from None
    except OSError as exc:
        raise SourceError(f"cannot read the export: {type(exc).__name__}") from None


def import_legacy_csv(
    csv_path: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
    *,
    contacts: Sequence[Mapping[str, object]] | None = None,
    progress: Progress | None = None,
    cancel: object = None,
) -> ImportResult:
    """Load one legacy export (`messages.csv`) into the EMPTY archive at `archive_path`.

    - An archive that holds anything is refused with ArchiveNotEmptyError.
    - `contacts`, `progress` and `cancel` are the same as for import_chat_db.
      The progress of phase `messages` is counted in bytes of the file.

    Returns an ImportResult with mode `legacy_csv`. A later import_chat_db
    replaces every row that the bridge loaded.
    """
    groups, contacts_skipped = contact_groups(contacts) if contacts is not None else (None, 0)
    run = run_control.new_run(csv_path, archive_path, MODE, progress, cancel)
    if not run.source.is_file():
        raise SourceError(f"no export file at {run.source}")
    run.counts["contacts_skipped"] = contacts_skipped
    return run_control.execute(run, lambda conn, state: _guarded_load(conn, state, groups))
