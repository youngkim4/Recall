#!/usr/bin/env python3
"""
Module: parse_imessage

Purpose:
- Extract messages from the iMessage SQLite database (chat.db)
- Normalize timestamps to local time
- Include message direction (is_from_me) and service (iMessage/SMS)
- Extract attachments and reactions metadata
- Provide a simple CLI to export results to a CSV file

Usage:
  python parse_imessage.py --db ~/Library/Messages/chat.db --out messages.csv
  python parse_imessage.py --db ~/Library/Messages/chat.db --list-contacts

Notes:
- Timestamp conversion handles seconds, microseconds, and nanoseconds since 2001-01-01.
- Ensure your shell/IDE has Full Disk Access to read chat.db.
"""
import argparse
import logging
import os
import pathlib
import re
import sqlite3
import sys
from typing import Optional, Union
from urllib.parse import quote

import pandas as pd

logger = logging.getLogger(__name__)

# Apple's iMessage DB often lives at: ~/Library/Messages/chat.db
# Ensure Terminal (or your IDE) has Full Disk Access to read it.


def imessage_timestamp_sql(column: str) -> str:
    """Build SQL that handles Apple's seconds, microseconds, or nanoseconds date values."""
    seconds_expr = (
        f"CASE "
        f"WHEN {column} IS NULL THEN NULL "
        f"WHEN ABS({column}) > 100000000000000 THEN {column} / 1000000000.0 "
        f"WHEN ABS({column}) > 100000000000 THEN {column} / 1000000.0 "
        f"ELSE {column} "
        f"END"
    )
    return f"datetime(({seconds_expr}) + strftime('%s','2001-01-01'), 'unixepoch', 'localtime')"


def open_read_only(db_path: Union[str, os.PathLike], timeout: float = 5.0) -> sqlite3.Connection:
    """Open a database to read it, and for nothing else.

    A path that holds no file makes no file: sqlite3.connect(path) would make
    an empty one. The path goes into the address quoted, so a "?", a "#" or a
    "%" in a name is part of the name. The database is not opened as one that
    cannot change: what Messages has written to the -wal file is read too.
    """
    whole = os.path.abspath(os.fspath(db_path))
    return sqlite3.connect(f"file://{quote(whole)}?mode=ro", uri=True, timeout=timeout)


def read_frame(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """The rows of a query as a frame. With no rows, the frame still has the columns.

    An error is the one sqlite3 raises. pandas.read_sql_query puts the whole
    text of the query into the error, and that text would reach the page.
    """
    cursor = conn.execute(sql, params)
    names = [column[0] for column in cursor.description]
    return pd.DataFrame.from_records(cursor.fetchall(), columns=names, coerce_float=True)


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return column names for a SQLite table."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def chat_display_name_sql(conn: sqlite3.Connection) -> str:
    """Return a safe SQL expression for chat display names across schema versions."""
    if "display_name" in table_columns(conn, "chat"):
        return "NULLIF(TRIM(chat.display_name), '')"
    return "NULL"


# attributedBody is a typedstream (NSArchiver) of an NSAttributedString:
#   ...NSString...+[length][UTF-8 text]...
_NSSTRING_MARKER = b"NSString"
_STRING_TYPE_CODE = b"+"
# from the class name to the type code: class version, superclass, then "+"
_TYPE_CODE_WINDOW = 20
# a length of 128 or more: this tag byte, then this many bytes, low byte first
_WIDE_LENGTH_BYTES = {0x81: 2, 0x82: 4}
# C0 controls other than tab and newline: not part of a message, and NUL breaks the CSV
_DROPPED_CONTROLS = re.compile("[\x00-\x08\x0b-\x1f]")
# a blob with no NSString marker: what it must be to count as a message
_FALLBACK_MAX_BYTES = 4096
_FALLBACK_MIN_CHARS = 2


def _typedstream_length(data: bytes, pos: int) -> Optional[tuple[int, int]]:
    """Read the length at pos: (length, offset of the byte after it), or None.

    A byte below 0x80 is the length. 0x81 is followed by a 2-byte length and
    0x82 by a 4-byte length, low byte first. This is not BER: 0x81 does not
    mean "one length byte follows".
    """
    if pos >= len(data):
        return None
    head = data[pos]
    if head < 0x80:
        return head, pos + 1
    width = _WIDE_LENGTH_BYTES.get(head)
    if width is None:
        return None
    end = pos + 1 + width
    if end > len(data):
        return None
    return int.from_bytes(data[pos + 1:end], "little"), end


def _nsstring_bytes(data: bytes, marker: int) -> Optional[bytes]:
    """UTF-8 bytes of the NSString whose class name is at marker, or None."""
    search_start = marker + len(_NSSTRING_MARKER)
    type_code = data.find(_STRING_TYPE_CODE, search_start, search_start + _TYPE_CODE_WINDOW)
    if type_code == -1:
        return None
    length = _typedstream_length(data, type_code + 1)
    if length is None:
        return None
    text_len, text_start = length
    text_end = text_start + text_len
    if text_end > len(data):
        return None
    return data[text_start:text_end]


def _message_text(raw: bytes) -> Optional[str]:
    """Decode the bytes of a message. None when they are not UTF-8 or hold no text."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return _DROPPED_CONTROLS.sub("", text) or None


def _fallback_text(data: bytes) -> Optional[str]:
    """Text of a blob with no NSString marker, only when the whole blob is plain text.

    Plain text is UTF-8 with no C0 control other than tab and newline, at most
    _FALLBACK_MAX_BYTES, with _FALLBACK_MIN_CHARS or more characters and a
    letter or a digit. It comes back whole, with its ends as they are.
    A run of text is never taken out of a blob, and a longer blob is not cut
    to fit: a part of a text reads as a whole message. A typedstream starts
    with a control byte, so it never passes.
    """
    if len(data) > _FALLBACK_MAX_BYTES:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if _DROPPED_CONTROLS.search(text):
        return None
    visible = text.strip()
    if len(visible) < _FALLBACK_MIN_CHARS or not any(char.isalnum() for char in visible):
        return None
    return text


def extract_text_from_attributed_body(blob: Optional[bytes]) -> Optional[str]:
    """
    Text of a message from its attributedBody blob, as Messages stored it.

    Newer macOS/iMessage versions store message text in attributedBody instead of text column.
    The blob is a typedstream of an NSAttributedString:
    ...NSString...+[length][UTF-8 text]...

    The result is the same string as the text column gives: the ends are not
    trimmed, and invisible characters stay (emoji joiners, variation selectors,
    direction marks). Only C0 controls other than tab and newline are dropped.

    Returns None when the blob holds no text or cannot be read. Never raises.
    A blob with the NSString marker gives its text or None, never a part of it.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    try:
        data = bytes(blob)
        marker = data.find(_NSSTRING_MARKER)
        if marker == -1:
            return _fallback_text(data)
        raw = _nsstring_bytes(data, marker)
        return None if raw is None else _message_text(raw)
    except Exception:
        logger.debug("Failed to extract text from attributedBody blob", exc_info=True)
        return None


def _attachment_note(types, files) -> Optional[str]:
    """What stands for the text of a message that is only an attachment, or None when it has none."""
    if pd.isna(types) and pd.isna(files):
        return None
    types, files = types or "", files or ""
    if types and files:
        return f"[Attachment] {types} | {files}"
    if types:
        return f"[Attachment] {types}"
    return "[Attachment]"


def _text_of_message(text, body, types, files) -> Optional[str]:
    """The text of one message: the text column, then attributedBody, then a note of its attachments."""
    if pd.notna(text):
        return text
    if pd.notna(body):
        extracted = extract_text_from_attributed_body(body)
        if extracted:
            return extracted
    return _attachment_note(types, files)


def extract_messages(db_path: str) -> pd.DataFrame:
    """
    Extract messages from the iMessage SQLite database.

    Apple's timestamps are stored as seconds, microseconds, or nanoseconds since
    2001-01-01 depending on OS version. This query converts them to local time.

    Returns a DataFrame with columns:
      - message_id
      - timestamp (local time)
      - sender (other party identifier)
      - text
      - chat_id (conversation identifier; for 1:1 this is typically the other party)
      - chat_display_name (custom group chat title, when available)
      - is_from_me (1 if sent by you, 0 if received)
      - service (iMessage/SMS)
      - attachment_types, attachment_files (what hangs on the message, joined with "|")

    A database with no messages gives a frame with these columns and no rows.
    """
    conn = open_read_only(db_path)
    try:
        display_name_expr = chat_display_name_sql(conn)
        # SQL projection maps core fields we need for analysis. The conversion below
        # turns Apple's 2001-01-01 epoch into a standard local timestamp.
        # Include attributedBody for messages where text is NULL (newer iMessage format)
        query = f"""
        SELECT
            message.ROWID AS message_id,
            {imessage_timestamp_sql("message.date")} AS timestamp,
            handle.id AS sender,
            message.text,
            message.attributedBody,
            chat.chat_identifier AS chat_id,
            {display_name_expr} AS chat_display_name,
            message.is_from_me AS is_from_me,
            message.service AS service,
            group_concat(attachment.mime_type, '|') AS attachment_types,
            group_concat(attachment.filename, '|') AS attachment_files
        FROM
            message
        LEFT JOIN
            handle ON message.handle_id = handle.ROWID
        LEFT JOIN
            chat_message_join ON message.ROWID = chat_message_join.message_id
        LEFT JOIN
            chat ON chat.ROWID = chat_message_join.chat_id
        LEFT JOIN
            message_attachment_join ON message.ROWID = message_attachment_join.message_id
        LEFT JOIN
            attachment ON attachment.ROWID = message_attachment_join.attachment_id
        WHERE
            message.text IS NOT NULL
            OR message.attributedBody IS NOT NULL
            OR attachment.ROWID IS NOT NULL
        GROUP BY
            message.ROWID
        ORDER BY
            timestamp ASC;
        """
        df = read_frame(conn, query)
    finally:
        conn.close()

    texts = [
        _text_of_message(text, body, types, files)
        for text, body, types, files in zip(df["text"], df["attributedBody"], df["attachment_types"], df["attachment_files"])
    ]
    # the text of a message is kept as it is; a message with none is no row
    with_text = df.drop(columns=["attributedBody"]).assign(text=pd.Series(texts, index=df.index, dtype="object"))
    return with_text.dropna(subset=["text"])


def list_contacts(db_path: str, limit: int = 20) -> pd.DataFrame:
    """
    List top contacts by message count with stats.
    
    Returns DataFrame with: chat_id, display_name, message_count, first_msg, last_msg, is_group
    """
    conn = open_read_only(db_path)
    try:
        display_name_expr = chat_display_name_sql(conn)
        query = f"""
        SELECT
            chat.chat_identifier AS chat_id,
            MAX({display_name_expr}) AS display_name,
            COUNT(message.ROWID) AS message_count,
            MIN({imessage_timestamp_sql("message.date")}) AS first_msg,
            MAX({imessage_timestamp_sql("message.date")}) AS last_msg,
            -- a group is "chat" and digits, nothing more (is_group_chat in conversation)
            CASE WHEN chat.chat_identifier GLOB 'chat[0-9]*'
                      AND substr(chat.chat_identifier, 5) NOT GLOB '*[^0-9]*'
                 THEN 1 ELSE 0 END AS is_group
        FROM message
        LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
        WHERE chat.chat_identifier IS NOT NULL
        GROUP BY chat.chat_identifier
        ORDER BY message_count DESC
        LIMIT ?;
        """
        df = read_frame(conn, query, (limit,))
    finally:
        conn.close()
    return df


def _attachment_category(mime) -> str:
    """The kind of an attachment, from its type."""
    if pd.isna(mime):
        return "unknown"
    mime = str(mime).lower()
    if "gif" in mime:
        return "gif"
    if mime.startswith("image"):
        return "photo"
    if mime.startswith("video"):
        return "video"
    if mime.startswith("audio"):
        return "audio"
    if "pdf" in mime or "document" in mime:
        return "document"
    return "other"


def extract_attachments(db_path: str) -> pd.DataFrame:
    """
    Extract attachment metadata from the iMessage database.

    Returns DataFrame with: message_id, chat_id, filename, mime_type,
    total_bytes, is_from_me, timestamp, category. A database with no
    attachments gives a frame with these columns and no rows.
    """
    conn = open_read_only(db_path)
    try:
        query = f"""
        SELECT
            message.ROWID AS message_id,
            chat.chat_identifier AS chat_id,
            attachment.filename,
            attachment.mime_type,
            attachment.total_bytes,
            message.is_from_me,
            {imessage_timestamp_sql("message.date")} AS timestamp
        FROM attachment
        JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
        JOIN message ON message.ROWID = message_attachment_join.message_id
        LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
        ORDER BY timestamp ASC;
        """
        df = read_frame(conn, query)
    finally:
        conn.close()

    categories = [_attachment_category(mime) for mime in df["mime_type"]]
    return df.assign(category=pd.Series(categories, index=df.index, dtype="object"))


_REACTION_NAMES = {0: "love", 1: "like", 2: "dislike", 3: "laugh", 4: "emphasis", 5: "question"}


def _decode_reaction(code) -> tuple[str, bool]:
    """A tapback from its type code: its name, and True when it was added.

    2000-2005 add a reaction and 3000-3005 take one away.
    """
    if pd.isna(code):
        return ("unknown", False)
    code = int(code)
    if 2000 <= code <= 2005:
        return (_REACTION_NAMES.get(code - 2000, "unknown"), True)
    if 3000 <= code <= 3005:
        return (_REACTION_NAMES.get(code - 3000, "unknown"), False)
    return ("unknown", False)


def extract_reactions(db_path: str) -> pd.DataFrame:
    """
    Extract tapback reactions from the iMessage database.

    Reactions are stored as associated messages with specific type codes:
    2000-2005 = add reaction, 3000-3005 = remove reaction
    Types: 0=love, 1=like, 2=dislike, 3=laugh, 4=emphasis, 5=question

    Returns DataFrame with: reaction_id, associated_message_guid,
    associated_message_type, is_from_me, chat_id, timestamp, reaction_type,
    is_add. A database with no reactions gives a frame with these columns and
    no rows.
    """
    conn = open_read_only(db_path)
    try:
        query = f"""
        SELECT
            m.ROWID AS reaction_id,
            m.associated_message_guid,
            m.associated_message_type,
            m.is_from_me,
            chat.chat_identifier AS chat_id,
            {imessage_timestamp_sql("m.date")} AS timestamp
        FROM message m
        LEFT JOIN chat_message_join ON m.ROWID = chat_message_join.message_id
        LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
        WHERE (m.associated_message_type BETWEEN 2000 AND 2005)
           OR (m.associated_message_type BETWEEN 3000 AND 3005)
        ORDER BY timestamp ASC;
        """
        df = read_frame(conn, query)
    finally:
        conn.close()

    decoded = [_decode_reaction(code) for code in df["associated_message_type"]]
    return df.assign(
        reaction_type=pd.Series([name for name, _ in decoded], index=df.index, dtype="object"),
        is_add=pd.Series([added for _, added in decoded], index=df.index, dtype="bool"),
    )


def search_contacts(db_path: str, query: str, limit: int = 10) -> pd.DataFrame:
    """
    Search contacts by partial chat_id or group display-name match.
    """
    conn = open_read_only(db_path)
    try:
        display_name_expr = chat_display_name_sql(conn)
        sql = f"""
        SELECT
            chat.chat_identifier AS chat_id,
            MAX({display_name_expr}) AS display_name,
            COUNT(message.ROWID) AS message_count,
            MAX({imessage_timestamp_sql("message.date")}) AS last_msg
        FROM message
        LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
        WHERE chat.chat_identifier LIKE ?
           OR {display_name_expr} LIKE ?
        GROUP BY chat.chat_identifier
        ORDER BY message_count DESC
        LIMIT ?;
        """
        like_query = f"%{query}%"
        df = read_frame(conn, sql, (like_query, like_query, limit))
    finally:
        conn.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract iMessage messages to CSV")
    parser.add_argument("--db", dest="db_path", default="chat.db", help="Path to chat.db (default: ./chat.db)")
    parser.add_argument("--out", dest="out_csv", default="messages.csv", help="Output CSV path (default: messages.csv)")
    parser.add_argument("--list-contacts", action="store_true", help="List top contacts by message count")
    parser.add_argument("--limit", type=int, default=20, help="Number of contacts to show (default: 20)")
    parser.add_argument("--attachments", action="store_true", help="Export attachments metadata")
    parser.add_argument("--reactions", action="store_true", help="Export reactions/tapbacks")
    args = parser.parse_args()

    try:
        if args.list_contacts:
            df = list_contacts(args.db_path, args.limit)
            if df.empty:
                print("No contacts found.")
                return
            print("\n📱 Top Contacts by Message Count:\n")
            print("-" * 70)
            for _, row in df.iterrows():
                group_tag = " [GROUP]" if row["is_group"] else ""
                display_name = str(row.get("display_name") or "").strip()
                label = f"{display_name} ({row['chat_id']})" if display_name else row["chat_id"]
                print(f"  {label}{group_tag}")
                print(f"    Messages: {row['message_count']:,}  |  First: {row['first_msg']}  |  Last: {row['last_msg']}")
            print("-" * 70)
            return
        
        if args.attachments:
            df = extract_attachments(args.db_path)
            p = pathlib.Path(args.out_csv)
            out_path = str(p.with_name(p.stem + "_attachments.csv"))
            df.to_csv(out_path, index=False)
            print(f"✅ Exported {len(df):,} attachments to {out_path}")
            return

        if args.reactions:
            df = extract_reactions(args.db_path)
            p = pathlib.Path(args.out_csv)
            out_path = str(p.with_name(p.stem + "_reactions.csv"))
            df.to_csv(out_path, index=False)
            print(f"✅ Exported {len(df):,} reactions to {out_path}")
            return
        
        df = extract_messages(args.db_path)
        if df.empty:
            print("No messages found in the database.")
            return
        print("Preview of extracted messages:")
        print(df.head())
        df.to_csv(args.out_csv, index=False)
        print(f"✅ Exported messages to {args.out_csv}")
    except sqlite3.Error as e:
        logger.error("SQLite error: %s", e)
        print(f"SQLite error: {e}")
        sys.exit(1)
    except Exception as ex:
        logger.exception("Unexpected error")
        print(f"An unexpected error occurred: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
