"""Export frames made from invented messages, for the `FrameArchive` tests."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from chat_engine.ports import Message

COLUMNS = [
    "message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me", "service",
    "attachment_types", "attachment_files",
]


def export_row(message: Message, display_name: str = "") -> dict[str, Any]:
    """One message as the legacy exporter writes it."""
    types = "|".join(message["attachments"])
    text = message["text"]
    if message["kind"] == "attachment":
        text = f"[Attachment] {types} | ~/Library/file-{message['id']}" if types else "￼"
    return {
        "message_id": message["id"],
        "timestamp": message["timestamp"],
        "sender": "" if message["is_from_me"] else message["sender"],
        "text": text,
        "chat_id": message["conversation"],
        "chat_display_name": display_name,
        "is_from_me": "1" if message["is_from_me"] else "0",
        "service": "iMessage",
        "attachment_types": types,
        "attachment_files": f"~/Library/file-{message['id']}" if types else "",
    }


def frame_of(
    messages: Iterable[Message],
    conversations: Sequence[Mapping[str, Any]] = (),
    extra: Sequence[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """The export frame of the messages. `extra` rows are added as they are."""
    names = {str(row["id"]): str(row.get("name") or "") for row in conversations if row.get("is_group")}
    rows = [export_row(message, names.get(message["conversation"], "")) for message in messages]
    rows.extend({**dict.fromkeys(COLUMNS, ""), **row} for row in extra)
    return pd.DataFrame(rows, columns=COLUMNS)
