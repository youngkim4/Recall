"""Plain records that move between the source reader, the classifier and the writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class SourceMessage(NamedTuple):
    """One row of chat.db's message table. A column the source does not have is None."""

    rowid: int
    guid: str | None
    text: str | None
    body: bytes | None
    handle_rowid: int | None
    is_from_me: int | None
    date: int | None
    date_delivered: int | None
    date_read: int | None
    service: str | None
    item_type: int | None
    associated_guid: str | None
    associated_type: int | None
    associated_emoji: str | None
    thread_guid: str | None
    date_edited: int | None
    date_retracted: int | None
    group_title: str | None
    group_action_type: int | None
    other_handle: int | None
    balloon_bundle_id: str | None
    expressive_style: str | None
    subject: str | None
    is_audio: int | None
    chat_rowid: int | None


class SourceAttachment(NamedTuple):
    """Metadata of one attachment. The file itself is never read."""

    message_rowid: int
    rowid: int
    guid: str | None
    mime_type: str | None
    transfer_name: str | None
    filename: str | None
    total_bytes: int | None


class SourceChat(NamedTuple):
    """One row of chat.db's chat table."""

    rowid: int
    identifier: str | None
    title: str | None
    service: str | None


class MessageRow(NamedTuple):
    """A row of the archive's messages table, without its id and import ids."""

    guid: str
    conversation_id: int
    sender_handle_id: int | None
    is_from_me: int
    sent_ns: int
    text: str
    kind: str
    service: str
    reply_to_guid: str | None
    edited_at: int | None
    unsent: int
    item_type: int
    text_source: str
    extra: str | None
    source_rowid: int


class ReactionRow(NamedTuple):
    """A row of the archive's reactions table, without its id and import ids."""

    guid: str
    target_guid: str
    target_part: int | None
    conversation_id: int | None
    sender_handle_id: int | None
    is_from_me: int
    kind: str
    emoji: str | None
    removed: int
    sent_ns: int
    source_rowid: int


class AttachmentRow(NamedTuple):
    """A row of the archive's attachments table, with the guid of its message."""

    guid: str
    message_guid: str
    position: int
    mime_type: str | None
    file_name: str | None
    path: str | None
    total_bytes: int | None


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """A message or a group event, ready for the archive. Ids are still source ROWIDs."""

    guid: str
    source_rowid: int
    chat_rowid: int | None
    handle_rowid: int | None
    is_from_me: bool
    sent_ns: int
    text: str
    kind: str
    service: str
    reply_to_guid: str | None
    edited_at: int | None
    unsent: bool
    item_type: int
    text_source: str
    extra: tuple[tuple[str, object], ...] = ()
    other_handle_rowid: int | None = None
    decode_failed: bool = False


@dataclass(frozen=True, slots=True)
class ReactionRecord:
    """A tapback. It points at its target by guid."""

    guid: str
    source_rowid: int
    chat_rowid: int | None
    handle_rowid: int | None
    is_from_me: bool
    sent_ns: int
    target_guid: str
    target_part: int | None
    kind: str
    emoji: str | None
    removed: bool


@dataclass(frozen=True, slots=True)
class Skipped:
    """A source row that is not stored. `reason` is one of SKIP_REASONS."""

    reason: str
    decode_failed: bool = False


SKIP_REASONS: tuple[str, ...] = ("no_guid", "no_date", "no_target", "empty", "no_conversation")

Record = MessageRecord | ReactionRecord | Skipped
