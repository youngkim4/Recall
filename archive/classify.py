"""Decide what one row of chat.db's message table is.

- A tapback (associated_message_type 2000-2007 or 3000-3007) is a reaction.
- A row with item_type other than 0 is a group event: kind `system`.
- A row with text is kind `text`.
- A row with no text and at least one attachment is kind `attachment`.
- An unsent message is kept with empty text.
- Any other row has nothing to show and is skipped.
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
from archive.tapbacks import split_target, tapback
from archive.timeconv import apple_to_unix_ns, apple_to_unix_seconds
from archive.typedstream import decode_attributed_body

OBJECT_REPLACEMENT = chr(0xFFFC)  # U+FFFC stands for an attachment inside the text


def clean_text(text: str | None) -> str:
    """Text as the user wrote it: no attachment placeholders, no outer white space."""
    if not text:
        return ""
    return text.replace(OBJECT_REPLACEMENT, "").replace("\x00", "").strip()


def message_text(text: str | None, body: bytes | None) -> tuple[str, str, bool]:
    """(text, where it came from, blob could not be read).

    The text column wins when it has text. New macOS versions leave it empty
    and keep the text only in attributedBody.
    """
    from_column = clean_text(text)
    if from_column:
        return from_column, "column", False
    if not body:
        return "", "none", False
    try:
        from_blob = clean_text(decode_attributed_body(body))
    except TypedStreamError:
        return "", "none", True
    return from_blob, ("blob" if from_blob else "none"), False


def _sent_ns(row: SourceMessage) -> int | None:
    for value in (row.date, row.date_delivered, row.date_read):
        unix_ns = apple_to_unix_ns(value)
        if unix_ns is not None:
            return unix_ns
    return None


def _extra(row: SourceMessage, *, is_system: bool) -> tuple[tuple[str, object], ...]:
    """Rare source fields, kept because the source can lose the row later."""
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
        return "system", clean_text(row.group_title), "none", False
    text, text_source, decode_failed = message_text(row.text, row.body)
    if text:
        return "text", text, text_source, decode_failed
    if has_attachments:
        return "attachment", "", "none", decode_failed
    if apple_to_unix_ns(row.date_retracted) is not None:
        return "text", "", "none", decode_failed
    return Skipped("empty", decode_failed=decode_failed)


def classify(row: SourceMessage, *, has_attachments: bool) -> Record:
    """Turn a source row into a message, a reaction or a skip."""
    guid = (row.guid or "").strip()
    if not guid:
        return Skipped("no_guid")
    sent_ns = _sent_ns(row)
    if sent_ns is None:
        return Skipped("no_date")
    found = tapback(row.associated_type)
    if found is not None:
        return _reaction(row, guid, sent_ns, found[0], found[1])

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
        extra=_extra(row, is_system=is_system),
        other_handle_rowid=(row.other_handle or None) if is_system else None,
        decode_failed=decode_failed,
    )
