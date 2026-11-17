"""A message of the archive in the shape that every payload gives it.

Search results, the messages of a day and the citations of a chat answer all
have one shape (`SearchResult` in app/src/types.ts, `message_row_payload` in
ui_server.py at 89ab504):

    messageId, chatId, displayName, senderName, timestamp, sender, text, attachment, isFromMe

- `messageId` is the id of the archive: Apple's guid, or `legacy:<ROWID>` in
  an archive that the legacy bridge loaded.
- `displayName` is the name of the conversation, or "".
- `senderName` is the saved name of the writer of a received message in a
  group, or "". With one person the page already says who wrote it.
- `text` is trimmed and cut for the screen. The archive keeps it whole.

The browser saved ids of the old export: a ROWID as text. `find_saved` reads
such an id.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from archive import Archive
from message_text import UNSENT_LABEL, attachment_label, clip_text, display_text
from recall_services.names import Names

# how much of a text a list shows; the archive keeps it whole
MESSAGE_TEXT_LIMIT = 600
LEGACY_PREFIX = "legacy:"
_SYSTEM_KIND = "system"
_STAMP_CHARS = 19
# SQLite holds a ROWID in 63 bits: a longer run of digits is no ROWID
_MAX_ROWID_DIGITS = 18


def payload(message: Mapping[str, Any], display_name: str, sender_name: str) -> dict[str, Any]:
    """One message with the nine keys of a search result."""
    return {
        "messageId": str(message["id"]),
        "chatId": str(message.get("conversation") or ""),
        "displayName": display_name,
        "senderName": sender_name,
        "timestamp": message.get("timestamp") or None,
        "sender": str(message.get("sender") or ""),
        "text": clip_text(display_text(message.get("text")), MESSAGE_TEXT_LIMIT),
        "attachment": _carried(message),
        "isFromMe": bool(message["is_from_me"]),
    }


def _carried(message: Mapping[str, Any]) -> str:
    """What a message carried beside its words: "Photo", "3 photos", or the label of a message that was unsent."""
    label = attachment_label("|".join(message.get("attachments") or []))
    return label or (UNSENT_LABEL if message.get("unsent") else "")


def _conversations(archive: Archive, ids: Iterable[str]) -> dict[str, Mapping[str, Any]]:
    found = {conversation_id: archive.conversation(conversation_id) for conversation_id in ids if conversation_id}
    return {conversation_id: record for conversation_id, record in found.items() if record is not None}


def payloads(archive: Archive, names: Names, found: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Messages in their order, each with the name of its conversation and of its writer."""
    listed = list(found)
    known = _conversations(archive, dict.fromkeys(str(message.get("conversation") or "") for message in listed))
    display = names.conversations(known.values())
    groups = {conversation_id for conversation_id, record in known.items() if record.get("is_group")}

    def sender_name(message: Mapping[str, Any]) -> str:
        in_group = str(message.get("conversation") or "") in groups
        sender = str(message.get("sender") or "")
        return names.sender(sender) if in_group and not message["is_from_me"] and sender else ""

    return [
        payload(message, display.get(str(message.get("conversation") or ""), ""), sender_name(message))
        for message in listed
    ]


def _row_number(text: str) -> int | None:
    digits = text.removeprefix(LEGACY_PREFIX)
    if not digits.isascii() or not digits.isdigit() or len(digits) > _MAX_ROWID_DIGITS:
        return None
    return int(digits)


def _stamp(timestamp: object) -> str:
    return str(timestamp)[:_STAMP_CHARS].replace("T", " ")


def find_saved(
    archive: Archive, saved_id: str, *, chat_id: str | None = None, timestamp: str | None = None
) -> dict[str, Any] | None:
    """The message that an id saved by the browser names today, or None.

    Saved chats and saved messages of the app at 89ab504 hold the ROWID of the
    export as text. The archive finds such a number with
    `message_by_source_rowid`. That answer is right only while the Messages
    database is the one that the number came from: a rebuilt database numbers
    its rows again. Give the saved `chat_id` and `timestamp` too, and a
    message that does not agree with them is not returned.

    A row that is no message (a tapback phrase in a bridge archive, a group
    event) is not returned.
    """
    text = saved_id.strip() if isinstance(saved_id, str) else ""
    if not text:
        return None
    found = archive.message(text)
    if found is None:
        rowid = _row_number(text)
        found = None if rowid is None else archive.message_by_source_rowid(rowid)
    if found is None or found.get("kind") == _SYSTEM_KIND:
        return None
    if chat_id is not None and found["conversation"] != chat_id:
        return None
    if timestamp is not None and found["timestamp"] != _stamp(timestamp):
        return None
    return dict(found)
