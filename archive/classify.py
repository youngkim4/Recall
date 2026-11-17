"""Decide what one row of chat.db's message table is.

- A tapback (associated_message_type 2000-2007 or 3000-3007) is a reaction.
  Another code from 2000 to 3999 is a tapback kind this version does not
  know: the row is skipped, never stored as a text message.
- A row with item_type other than 0 is a group event: kind `system`.
- A row with text is kind `text`.
- A row with no visible text and an attachment is kind `attachment`.
- An unsent message is kept with empty text.
- Any other row has nothing to show and is skipped.

Text is stored as Messages has it. Nothing is trimmed and no invisible
character is dropped. Only two characters are never stored: the attachment
placeholder U+FFFC, because each attachment is a row of its own, and NUL.
"""

from __future__ import annotations

from archive.errors import TypedStreamError
from archive.records import (
    MessageRecord,
    ReactionRecord,
    Record,
    Skipped,
    SourceMessage,
)
from archive.tapbacks import in_tapback_range, split_target, tapback
from archive.timeconv import apple_to_unix_ns, apple_to_unix_seconds
from archive.typedstream import decode_attributed_body

OBJECT_REPLACEMENT = chr(0xFFFC)  # U+FFFC stands for an attachment inside the text
_NEVER_STORED = str.maketrans("", "", OBJECT_REPLACEMENT + "\x00")
# the send time comes from the first of these columns that holds a date
_DATE_COLUMNS = ("date", "date_delivered", "date_read")


def stored_text(text: str | None) -> str:
    """The text the archive stores: as given, without attachment placeholders and NUL."""
    return text.translate(_NEVER_STORED) if text else ""


def message_text(text: str | None, body: bytes | None) -> tuple[str, str, bool]:
    """(text as the source has it, where it came from, blob could not be read).

    The text column wins when it has text. New macOS versions leave it empty
    and keep the text only in attributedBody.
    """
    if text:
        return text, "column", False
    if not body:
        return "", "none", False
    try:
        from_blob = decode_attributed_body(body)
    except TypedStreamError:
        return "", "none", True
    return from_blob, ("blob" if from_blob else "none"), False


def _sent_ns(row: SourceMessage) -> tuple[int, str] | None:
    """(send time, the column it came from)."""
    for column in _DATE_COLUMNS:
        unix_ns = apple_to_unix_ns(getattr(row, column))
        if unix_ns is not None:
            return unix_ns, column
    return None


def _extra(row: SourceMessage, *, is_system: bool, date_column: str) -> tuple[tuple[str, object], ...]:
    """Rare source fields that no column of the archive holds."""
    fields: list[tuple[str, object]] = []
    if row.subject:
        fields.append(("subject", row.subject))
    if row.balloon_bundle_id:
        fields.append(("balloon_bundle_id", row.balloon_bundle_id))
    if row.expressive_style:
        fields.append(("expressive_send_style_id", row.expressive_style))
    if row.is_audio:
        fields.append(("is_audio_message", True))
    if is_system and row.group_action_type is not None:
        fields.append(("group_action_type", int(row.group_action_type)))
    if row.associated_type:
        fields.append(("associated_message_type", int(row.associated_type)))
        if row.associated_guid:
            fields.append(("associated_message_guid", row.associated_guid))
    if date_column != _DATE_COLUMNS[0]:
        fields.append(("date_source", date_column))  # the row has no send date: this time is later
    return tuple(fields)


def _reaction(row: SourceMessage, guid: str, sent_ns: int, kind: str, removed: bool) -> Record:
    target = split_target(row.associated_guid)
    if target is None:
        return Skipped("no_target")
    is_from_me = bool(row.is_from_me)
    return ReactionRecord(
        guid=guid,
        source_rowid=row.rowid,
        chat_rowid=row.chat_rowid,
        handle_rowid=None if is_from_me else (row.handle_rowid or None),
        is_from_me=is_from_me,
        sent_ns=sent_ns,
        target_guid=target[0],
        target_part=target[1],
        kind=kind,
        emoji=(row.associated_emoji or None) if kind == "emoji" else None,
        removed=removed,
    )


def _content(row: SourceMessage, has_attachments: bool, is_system: bool) -> tuple[str, str, str, bool] | Skipped:
    """(kind, text, text source, blob unreadable), or Skipped for a row with nothing to show."""
    if is_system:
        return "system", stored_text(row.group_title), "none", False
    raw, text_source, decode_failed = message_text(row.text, row.body)
    text = stored_text(raw)
    if text.strip():
        return "text", text, text_source, decode_failed
    if has_attachments or OBJECT_REPLACEMENT in raw:
        return "attachment", text, (text_source if text else "none"), decode_failed
    if text:
        return "text", text, text_source, decode_failed  # white space only: it is still what was sent
    if apple_to_unix_ns(row.date_retracted) is not None:
        return "text", "", "none", decode_failed
    return Skipped("empty", decode_failed=decode_failed)


def classify(row: SourceMessage, *, has_attachments: bool) -> Record:
    """Turn a source row into a message, a reaction or a skip."""
    guid = (row.guid or "").strip()
    if not guid:
        return Skipped("no_guid")
    sent = _sent_ns(row)
    if sent is None:
        return Skipped("no_date")
    sent_ns, date_column = sent
    found = tapback(row.associated_type)
    if found is not None:
        return _reaction(row, guid, sent_ns, found[0], found[1])
    if in_tapback_range(row.associated_type):
        return Skipped("unknown_tapback")

    is_system = bool(row.item_type)
    content = _content(row, has_attachments, is_system)
    if isinstance(content, Skipped):
        return content
    kind, text, text_source, decode_failed = content
    is_from_me = bool(row.is_from_me)
    return MessageRecord(
        guid=guid,
        source_rowid=row.rowid,
        chat_rowid=row.chat_rowid,
        handle_rowid=None if is_from_me else (row.handle_rowid or None),
        is_from_me=is_from_me,
        sent_ns=sent_ns,
        text=text,
        kind=kind,
        service=row.service or "",
        reply_to_guid=(row.thread_guid or "").strip() or None,
        edited_at=apple_to_unix_seconds(row.date_edited),
        unsent=apple_to_unix_ns(row.date_retracted) is not None,
        item_type=int(row.item_type or 0),
        text_source=text_source,
        extra=_extra(row, is_system=is_system, date_column=date_column),
        other_handle_rowid=(row.other_handle or None) if is_system else None,
        decode_failed=decode_failed,
    )
