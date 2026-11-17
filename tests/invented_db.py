"""An invented database, shaped like the one Messages keeps. No row comes from a real one."""
import sqlite3
from pathlib import Path

PERSON = "+15550100"
GROUP = "chat123"
# 2024-01-15 10:00 UTC, in the nanoseconds since 2001 that Messages keeps
START = 726915600 * 1_000_000_000
MINUTE = 60 * 1_000_000_000

SCHEMA = """
    CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
    CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
    CREATE TABLE message (
        ROWID INTEGER PRIMARY KEY, date INTEGER, handle_id INTEGER, text TEXT, attributedBody BLOB,
        is_from_me INTEGER, service TEXT, associated_message_guid TEXT, associated_message_type INTEGER DEFAULT 0
    );
    CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
    CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT, mime_type TEXT, total_bytes INTEGER);
    CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
"""

FIXTURES = Path(__file__).parent / "fixtures" / "typedstream"


def apple_archive(name: str) -> bytes:
    """An attributedBody that Apple's encoder made (tests/fixtures/typedstream)."""
    return bytes.fromhex((FIXTURES / f"{name}.hex").read_text(encoding="ascii"))


def fill(conn: sqlite3.Connection, texts, group_name="", photo=True, reaction=True, bodies=()) -> None:
    """One person and one group. `texts` are messages of the person, oldest first.

    `bodies` are messages whose text is only in attributedBody. A photo hangs
    on the first message and a reaction comes last, unless told not to.
    """
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO handle VALUES (1, ?)", (PERSON,))
    conn.execute("INSERT INTO chat VALUES (1, ?, '')", (PERSON,))
    conn.execute("INSERT INTO chat VALUES (2, ?, ?)", (GROUP, group_name))
    rows = [(text, None) for text in texts] + [(None, body) for body in bodies]
    for place, (text, body) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO message VALUES (?, ?, 1, ?, ?, ?, 'iMessage', NULL, 0)",
            (place, START + place * MINUTE, text, body, place % 2),
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, ?)", (place,))
    if photo:
        conn.execute("INSERT INTO attachment VALUES (1, 'photo.jpg', 'image/jpeg', 1024)")
        conn.execute("INSERT INTO message_attachment_join VALUES (1, 1)")
    if reaction:
        conn.execute(
            "INSERT INTO message VALUES (900, ?, 1, NULL, NULL, 0, 'iMessage', 'guid-1', 2000)",
            (START + 900 * MINUTE,),
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 900)")
    conn.commit()


def messages_db(path, texts, **parts):
    """Write an invented database at `path` and give the path back. See `fill` for the parts."""
    conn = sqlite3.connect(str(path))
    try:
        fill(conn, texts, **parts)
    finally:
        conn.close()
    return path
