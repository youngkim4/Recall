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
- Timestamp conversion assumes nanoseconds since 2001-01-01; adjust divisor if your DB differs.
- Ensure your shell/IDE has Full Disk Access to read chat.db.
"""
import argparse
import logging
import sqlite3
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Apple's iMessage DB often lives at: ~/Library/Messages/chat.db
# Ensure Terminal (or your IDE) has Full Disk Access to read it.

def extract_text_from_attributed_body(blob: bytes) -> Optional[str]:
    """
    Extract plain text from attributedBody blob (NSAttributedString).
    
    Newer macOS/iMessage versions store message text in attributedBody instead of text column.
    The blob is a streamtyped NSAttributedString with format:
    ...NSString...+[length_byte][text_content][marker]...
    """
    if not blob:
        return None
    try:
        # The format is: ...NSString\x01+\x[len][text]...
        # Find the NSString marker and extract text after the + length indicator
        
        # Method 1: Look for the +[length][text] pattern after NSString
        ns_string_marker = b'NSString'
        idx = blob.find(ns_string_marker)
        if idx != -1:
            # Skip past NSString and look for the + marker
            search_start = idx + len(ns_string_marker)
            plus_idx = blob.find(b'+', search_start, search_start + 20)
            if plus_idx != -1 and plus_idx + 2 < len(blob):
                # Decode length using BER-style encoding:
                # - 0x00-0x7F: direct length (0-127)
                # - 0x80+: long form — low 7 bits = number of subsequent length bytes
                first_byte = blob[plus_idx + 1]
                if first_byte < 0x80:
                    text_len = first_byte
                    text_start = plus_idx + 2
                else:
                    num_len_bytes = first_byte & 0x7F
                    if 1 <= num_len_bytes <= 4 and plus_idx + 2 + num_len_bytes <= len(blob):
                        text_len = int.from_bytes(
                            blob[plus_idx + 2:plus_idx + 2 + num_len_bytes], 'big'
                        )
                        text_start = plus_idx + 2 + num_len_bytes
                    else:
                        # Fallback: treat as direct single-byte length
                        text_len = first_byte
                        text_start = plus_idx + 2

                text_end = text_start + text_len
                if 0 < text_len and text_end <= len(blob):
                    try:
                        text = blob[text_start:text_end].decode('utf-8')
                        text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
                        if text.strip():
                            return text.strip()
                    except UnicodeDecodeError:
                        pass
        
        # Method 2: Fallback - try to find readable text between markers
        # Look for text that's surrounded by non-printable bytes
        import re
        decoded = blob.decode('utf-8', errors='ignore')
        
        # Remove known metadata patterns
        cleaned = re.sub(r'(streamtyped|NSAttributedString|NSMutableString|NSString|NSObject|NSDictionary|NSNumber|NSArray)', '', decoded)
        
        # Find sequences of printable characters (min 2 chars, not just 'iI' or similar markers)
        matches = re.findall(r'[a-zA-Z0-9\s\.\,\!\?\'\"\-\:\;\@\#\$\%\&\*\(\)\[\]\{\}\/\\]+', cleaned)
        matches = [m.strip() for m in matches if len(m.strip()) >= 2 and m.strip() not in ['iI', 'II', 'ii', 'NS']]
        
        if matches:
            # Return the longest meaningful match
            return max(matches, key=len)
        
        return None
    except Exception:
        logger.debug("Failed to extract text from attributedBody blob", exc_info=True)
        return None


def extract_messages(db_path: str) -> pd.DataFrame:
    """
    Extract messages from the iMessage SQLite database.

    Apple's timestamps are stored as nanoseconds since 2001-01-01 on many systems.
    This query converts them to local time. If your DB stores seconds instead of
    nanoseconds, you may need to adjust the divisor accordingly.

    Returns a DataFrame with columns:
      - message_id
      - timestamp (local time)
      - sender (other party identifier)
      - text
      - chat_id (conversation identifier; for 1:1 this is typically the other party)
      - is_from_me (1 if sent by you, 0 if received)
      - service (iMessage/SMS)
    """
    conn = sqlite3.connect(db_path)

    # SQL projection maps core fields we need for analysis. The conversion below
    # turns Apple's 2001-01-01 epoch into a standard local timestamp.
    # Include attributedBody for messages where text is NULL (newer iMessage format)
    query = """
    SELECT
        message.ROWID AS message_id,
        datetime(message.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime') AS timestamp,
        handle.id AS sender,
        message.text,
        message.attributedBody,
        chat.chat_identifier AS chat_id,
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

    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Extract text from attributedBody where text is NULL
    def get_text(row):
        if pd.notna(row['text']):
            return row['text']
        if pd.notna(row['attributedBody']):
            extracted = extract_text_from_attributed_body(row['attributedBody'])
            if extracted:
                return extracted
        # Preserve non-text messages with attachment metadata
        attachment_types = row.get('attachment_types')
        attachment_files = row.get('attachment_files')
        if pd.notna(attachment_types) or pd.notna(attachment_files):
            types = attachment_types or ""
            files = attachment_files or ""
            if types and files:
                return f"[Attachment] {types} | {files}"
            if types:
                return f"[Attachment] {types}"
            return "[Attachment]"
        return None
    
    df['text'] = df.apply(get_text, axis=1)
    
    # Drop attributedBody column and rows with no text
    df = df.drop(columns=['attributedBody'])
    df = df.dropna(subset=['text'])
    
    return df


def list_contacts(db_path: str, limit: int = 20) -> pd.DataFrame:
    """
    List top contacts by message count with stats.
    
    Returns DataFrame with: chat_id, message_count, first_msg, last_msg, is_group
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        chat.chat_identifier AS chat_id,
        COUNT(message.ROWID) AS message_count,
        MIN(datetime(message.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime')) AS first_msg,
        MAX(datetime(message.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime')) AS last_msg,
        CASE WHEN chat.chat_identifier LIKE 'chat%' THEN 1 ELSE 0 END AS is_group
    FROM message
    LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
    WHERE chat.chat_identifier IS NOT NULL
    GROUP BY chat.chat_identifier
    ORDER BY message_count DESC
    LIMIT ?;
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def extract_attachments(db_path: str) -> pd.DataFrame:
    """
    Extract attachment metadata from the iMessage database.
    
    Returns DataFrame with: message_id, chat_id, filename, mime_type, 
    total_bytes, is_from_me, timestamp
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        message.ROWID AS message_id,
        chat.chat_identifier AS chat_id,
        attachment.filename,
        attachment.mime_type,
        attachment.total_bytes,
        message.is_from_me,
        datetime(message.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime') AS timestamp
    FROM attachment
    JOIN message_attachment_join ON attachment.ROWID = message_attachment_join.attachment_id
    JOIN message ON message.ROWID = message_attachment_join.message_id
    LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
    ORDER BY timestamp ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Categorize attachments
    def categorize(mime: Optional[str]) -> str:
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
    
    df["category"] = df["mime_type"].apply(categorize)
    return df


def extract_reactions(db_path: str) -> pd.DataFrame:
    """
    Extract tapback reactions from the iMessage database.
    
    Reactions are stored as associated messages with specific type codes:
    2000-2005 = add reaction, 3000-3005 = remove reaction
    Types: 0=love, 1=like, 2=dislike, 3=laugh, 4=emphasis, 5=question
    """
    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        m.ROWID AS reaction_id,
        m.associated_message_guid,
        m.associated_message_type,
        m.is_from_me,
        chat.chat_identifier AS chat_id,
        datetime(m.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime') AS timestamp
    FROM message m
    LEFT JOIN chat_message_join ON m.ROWID = chat_message_join.message_id
    LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
    WHERE m.associated_message_type BETWEEN 2000 AND 3005
    ORDER BY timestamp ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Decode reaction types
    reaction_names = {0: "love", 1: "like", 2: "dislike", 3: "laugh", 4: "emphasis", 5: "question"}
    
    def decode_reaction(atype: int) -> tuple:
        if pd.isna(atype):
            return ("unknown", False)
        atype = int(atype)
        if 2000 <= atype <= 2005:
            return (reaction_names.get(atype - 2000, "unknown"), True)  # added
        if 3000 <= atype <= 3005:
            return (reaction_names.get(atype - 3000, "unknown"), False)  # removed
        return ("unknown", False)
    
    df[["reaction_type", "is_add"]] = df["associated_message_type"].apply(
        lambda x: pd.Series(decode_reaction(x))
    )
    return df


def search_contacts(db_path: str, query: str, limit: int = 10) -> pd.DataFrame:
    """
    Search contacts by partial chat_id match.
    """
    conn = sqlite3.connect(db_path)
    sql = """
    SELECT
        chat.chat_identifier AS chat_id,
        COUNT(message.ROWID) AS message_count,
        MAX(datetime(message.date / 1000000000 + strftime('%s','2001-01-01'), 'unixepoch', 'localtime')) AS last_msg
    FROM message
    LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    LEFT JOIN chat ON chat.ROWID = chat_message_join.chat_id
    WHERE chat.chat_identifier LIKE ?
    GROUP BY chat.chat_identifier
    ORDER BY message_count DESC
    LIMIT ?;
    """
    df = pd.read_sql_query(sql, conn, params=(f"%{query}%", limit))
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
                print(f"  {row['chat_id']}{group_tag}")
                print(f"    Messages: {row['message_count']:,}  |  First: {row['first_msg']}  |  Last: {row['last_msg']}")
            print("-" * 70)
            return
        
        if args.attachments:
            df = extract_attachments(args.db_path)
            out_path = args.out_csv.replace(".csv", "_attachments.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Exported {len(df):,} attachments to {out_path}")
            return
        
        if args.reactions:
            df = extract_reactions(args.db_path)
            out_path = args.out_csv.replace(".csv", "_reactions.csv")
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
        print(f"SQLite error: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()
