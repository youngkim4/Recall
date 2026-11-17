"""Bridge to the old CSV world: the archive as one pandas DataFrame.

The server and the report code of today read `messages.csv` into a
DataFrame. messages_frame() gives the same columns from the archive, so that
code can run on the archive before each feature is ported to the query API.

This is the only module of the package that imports pandas, and it loads
every message into memory. New code must use archive.queries instead.

Differences from the CSV export, all on purpose:
- `message_id` is the message guid (text), not a ROWID.
- Tapbacks and group events are not rows.
- `sender` is the writer's handle. It is missing for own messages. The export
  had the other person's handle on own one-to-one messages.
"""

from __future__ import annotations

import pandas as pd

from archive.queries import Archive
from archive.rows import UNKNOWN_MIME_TYPE
from archive.timeconv import LOCAL_TIMESTAMP_FORMAT, sql_local_timestamp

EXPORT_COLUMNS: tuple[str, ...] = (
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
ATTACHMENT_LABEL = "[Attachment]"

_FRAME_SQL = f"""
SELECT m.guid AS message_id,
       {sql_local_timestamp("m.sent_at")} AS timestamp,
       h.handle AS sender,
       m.text AS text,
       c.chat_identifier AS chat_id,
       NULLIF(c.title, '') AS chat_display_name,
       m.is_from_me AS is_from_me,
       NULLIF(m.service, '') AS service,
       (SELECT group_concat(COALESCE(NULLIF(a.mime_type, ''), '{UNKNOWN_MIME_TYPE}'), '|' ORDER BY a.position, a.id)
          FROM attachments a WHERE a.message_id = m.id) AS attachment_types,
       (SELECT group_concat(COALESCE(a.path, ''), '|' ORDER BY a.position, a.id)
          FROM attachments a WHERE a.message_id = m.id) AS attachment_files,
       m.kind AS kind
  FROM messages m
  LEFT JOIN conversations c ON c.id = m.conversation_id
  LEFT JOIN handles h ON h.id = m.sender_handle_id
 WHERE m.kind != 'system'
 ORDER BY m.sent_ns, m.source_rowid, m.id
"""


def _attachment_text(types: object, files: object) -> str:
    """The text the old export wrote for a message that is only attachments."""
    if isinstance(types, str) and types and isinstance(files, str) and files:
        return f"{ATTACHMENT_LABEL} {types} | {files}"
    if isinstance(types, str) and types:
        return f"{ATTACHMENT_LABEL} {types}"
    return ATTACHMENT_LABEL


def messages_frame(archive: Archive) -> pd.DataFrame:
    """Every message of the archive, oldest first, with the columns of the CSV export."""
    with archive._using() as conn:
        raw = pd.read_sql_query(_FRAME_SQL, conn)
    only_files = raw["kind"] == "attachment"
    text = raw["text"].where(
        ~only_files,
        [
            _attachment_text(types, files)
            for types, files in zip(raw["attachment_types"], raw["attachment_files"], strict=True)
        ],
    )
    frame = raw.assign(
        timestamp=pd.to_datetime(raw["timestamp"], format=LOCAL_TIMESTAMP_FORMAT),
        sender=raw["sender"].astype("string"),
        chat_id=raw["chat_id"].astype("string"),
        is_from_me=raw["is_from_me"].astype("int64"),
        text=text,
    )
    has_text = frame["text"].fillna("").astype(str) != ""  # the export had no row for a message with nothing to show
    return frame.loc[has_text, list(EXPORT_COLUMNS)].reset_index(drop=True)
