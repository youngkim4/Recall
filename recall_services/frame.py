"""The archive as the frames that the figures on pandas read.

`compute_stats`, `activity_series`, `compute_dynamics`, the day view and the
moments of the server read one pandas frame with the columns of the export
(`messages.csv`). `messages_frame()` gives that frame from the archive,
through `archive.compat.messages_frame`, so that code can read the archive
without a change.

What `message_id` holds:
- After a pull from Messages it is Apple's guid of the message.
- In an archive that the legacy bridge loaded it is `legacy:<ROWID>`.
- The export held the ROWID as a number. Saved chats and saved messages of
  the browser hold that number. `recall_services.messages.find_saved` reads
  such an id with `Archive.message_by_source_rowid`. That answer is right only
  while the Messages database is the one that the number came from.

Differences from the export, beyond those of `archive/README.md`:
- A message that is only attachments has the placeholder U+FFFC as its text,
  as the exporter of today writes it. `archive.compat` writes the label of an
  older exporter there (`[Attachment] image/jpeg | path`), and the server
  would show that label as words.

The frame holds every message in memory, as the server holds the export
today. It is made once for one state of the archive.

`attachments_frame` and `reactions_frame` stand in for the two files that the
export wrote beside the messages. `compute_stats` reads them for the media
figures of one conversation. A tapback has the time of the message that was
tapped: the archive does not give the time of a tapback. An archive that the
bridge loaded holds no tapbacks.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

import pandas as pd

from archive import Archive, compat
from message_text import ATTACHMENT_PLACEHOLDER
from recall_services import archive_state
from recall_services.archive_state import ArchiveState

ATTACHMENT_COLUMNS = ("message_id", "chat_id", "mime_type", "is_from_me", "timestamp", "category")
REACTION_COLUMNS = ("message_id", "chat_id", "reaction_type", "is_add", "is_from_me", "timestamp")
# the kinds that the exporter of today reads and that `compute_reaction_stats` counts
REACTION_KINDS = frozenset({"love", "like", "dislike", "laugh", "emphasis", "question"})
_ONLY_FILES_KIND = "attachment"

_lock = threading.Lock()
_held: weakref.WeakKeyDictionary[ArchiveState, tuple[tuple[Any, ...], pd.DataFrame]] = weakref.WeakKeyDictionary()


def _old_label(types: object, files: object) -> str:
    """The text that archive.compat writes for a message that is only attachments."""
    if isinstance(types, str) and types and isinstance(files, str) and files:
        return f"{compat.ATTACHMENT_LABEL} {types} | {files}"
    if isinstance(types, str) and types:
        return f"{compat.ATTACHMENT_LABEL} {types}"
    return compat.ATTACHMENT_LABEL


def _typed_labels(raw: pd.DataFrame, archive: Archive) -> pd.Series:
    """True for a row whose text is the bare label and that a person typed: its kind is not `attachment`.

    The frame of archive.compat has no kind. The bare label is rare, so each
    such row is looked up.
    """
    bare = raw["text"] == compat.ATTACHMENT_LABEL
    typed = {
        message_id
        for message_id in raw.loc[bare, "message_id"]
        if (archive.message(message_id) or {}).get("kind") != _ONLY_FILES_KIND
    }
    return raw["message_id"].isin(typed)


def _as_the_export_of_today(raw: pd.DataFrame, archive: Archive) -> pd.DataFrame:
    labels = [_old_label(types, files) for types, files in zip(raw["attachment_types"], raw["attachment_files"])]
    only_files = (raw["text"] == pd.Series(labels, index=raw.index, dtype=object)) & ~_typed_labels(raw, archive)
    return raw.assign(text=raw["text"].where(~only_files, ATTACHMENT_PLACEHOLDER))


def _key(state: ArchiveState) -> tuple[Any, ...]:
    """What one state of the archive is known by: its generation and the mark of its files."""
    return (state.generation, state.file_mark())


def messages_frame(state: ArchiveState | None = None) -> pd.DataFrame:
    """Every message of the archive, oldest first, with the columns of the export.

    With no state given, the shared state of the server. The frame is made
    once for one state of the archive. Callers that arrive together wait for
    one frame. Do not change the frame: every caller gets the same one.
    """
    chosen = state if state is not None else archive_state.shared()
    with _lock:
        key = _key(chosen)
        held = _held.get(chosen)
        if held is not None and held[0] == key:
            return held[1]
        archive = chosen.archive()
        made = _as_the_export_of_today(compat.messages_frame(archive), archive)
        _held[chosen] = (key, made)
        return made


def category_of(mime_type: str) -> str:
    """The kind of an attachment, by the rule of `extract_attachments` in parse_imessage.py."""
    mime = str(mime_type or "").lower()
    if "gif" in mime:
        return "gif"
    for start, category in (("image", "photo"), ("video", "video"), ("audio", "audio")):
        if mime.startswith(start):
            return category
    return "document" if "pdf" in mime or "document" in mime else "other"


def _stamped(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    made = pd.DataFrame(rows, columns=list(columns))
    return made.assign(timestamp=pd.to_datetime(made["timestamp"], format=compat.LOCAL_TIMESTAMP_FORMAT))


def attachments_frame(
    state: ArchiveState, conversation_id: str, *, since: str | None = None, until: str | None = None
) -> pd.DataFrame:
    """One row for each attachment of one conversation, with the columns that `compute_stats` reads."""
    rows: list[dict[str, Any]] = []
    messages = state.archive().iter_messages(conversation_id, since=since, until=until) if conversation_id else ()
    for message in messages:
        for mime_type in message["attachments"]:
            rows.append(
                {
                    "message_id": message["id"],
                    "chat_id": conversation_id,
                    "mime_type": mime_type,
                    "is_from_me": int(message["is_from_me"]),
                    "timestamp": message["timestamp"],
                    "category": category_of(mime_type),
                }
            )
    return _stamped(rows, ATTACHMENT_COLUMNS)


def reactions_frame(
    state: ArchiveState, conversation_id: str, *, since: str | None = None, until: str | None = None
) -> pd.DataFrame:
    """One row for each tapback that stands on a message of one conversation.

    A person has at most one tapback on a message, the last one, and a
    tapback that was taken back is no row. The export file of today held one
    row for each tapback that was ever added. A tapback with any emoji or
    with a sticker is no row: the exporter of today does not read those, and
    the figures have no place for them. The days of `since` and `until` are
    the days of the messages.
    """
    archive = state.archive()
    messages = archive.iter_messages(conversation_id, since=since, until=until) if conversation_id else ()
    sent_at = {message["id"]: message["timestamp"] for message in messages}
    rows = [
        {
            "message_id": message_id,
            "chat_id": conversation_id,
            "reaction_type": tapback["kind"],
            "is_add": True,
            "is_from_me": int(tapback["is_from_me"]),
            "timestamp": sent_at[message_id],
        }
        for message_id, tapbacks in archive.reactions(list(sent_at)).items()
        for tapback in tapbacks
        if tapback["kind"] in REACTION_KINDS
    ]
    return _stamped(rows, REACTION_COLUMNS)
