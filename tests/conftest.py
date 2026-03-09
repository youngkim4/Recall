import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_messages_df():
    """Minimal messages DataFrame for testing."""
    return pd.DataFrame({
        "message_id": [1, 2, 3, 4, 5],
        "timestamp": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-15 10:05:00",
            "2024-02-20 14:30:00",
            "2024-03-10 09:00:00",
            "2024-03-10 09:01:00",
        ]),
        "sender": ["them", "them", "them", "them", "them"],
        "text": ["hello", "how are you", "long time no see", "let's meet", "ok cool"],
        "chat_id": ["+1234", "+1234", "+1234", "+1234", "+1234"],
        "is_from_me": [0, 1, 0, 1, 0],
        "service": ["iMessage"] * 5,
    })


@pytest.fixture
def sample_messages_csv(tmp_path, sample_messages_df):
    """Write sample messages to a CSV and return the path."""
    csv_path = tmp_path / "messages.csv"
    sample_messages_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_attachments_df():
    return pd.DataFrame({
        "message_id": [1, 2, 3],
        "chat_id": ["+1234", "+1234", "+1234"],
        "filename": ["photo.jpg", "video.mp4", "doc.pdf"],
        "mime_type": ["image/jpeg", "video/mp4", "application/pdf"],
        "total_bytes": [1024, 2048, 512],
        "is_from_me": [1, 0, 1],
        "timestamp": ["2024-01-15 10:00", "2024-01-15 10:05", "2024-02-20 14:30"],
        "category": ["photo", "video", "document"],
    })


@pytest.fixture
def sample_attachments_csv(tmp_path, sample_attachments_df):
    csv_path = tmp_path / "messages_attachments.csv"
    sample_attachments_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_reactions_df():
    return pd.DataFrame({
        "reaction_id": [1, 2, 3],
        "associated_message_guid": ["g1", "g2", "g3"],
        "associated_message_type": [2000, 2001, 2003],
        "is_from_me": [1, 0, 1],
        "chat_id": ["+1234", "+1234", "+1234"],
        "timestamp": ["2024-01-15 10:00", "2024-01-15 10:05", "2024-02-20 14:30"],
        "reaction_type": ["love", "like", "laugh"],
        "is_add": [True, True, True],
    })


@pytest.fixture
def sample_reactions_csv(tmp_path, sample_reactions_df):
    csv_path = tmp_path / "messages_reactions.csv"
    sample_reactions_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_chat_db(tmp_path):
    """Create a minimal in-memory-like SQLite DB mimicking chat.db schema."""
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("""CREATE TABLE handle (
        ROWID INTEGER PRIMARY KEY,
        id TEXT
    )""")
    c.execute("""CREATE TABLE chat (
        ROWID INTEGER PRIMARY KEY,
        chat_identifier TEXT
    )""")
    c.execute("""CREATE TABLE message (
        ROWID INTEGER PRIMARY KEY,
        date INTEGER,
        handle_id INTEGER,
        text TEXT,
        attributedBody BLOB,
        is_from_me INTEGER,
        service TEXT,
        associated_message_guid TEXT,
        associated_message_type INTEGER DEFAULT 0
    )""")
    c.execute("""CREATE TABLE chat_message_join (
        chat_id INTEGER,
        message_id INTEGER
    )""")
    c.execute("""CREATE TABLE attachment (
        ROWID INTEGER PRIMARY KEY,
        filename TEXT,
        mime_type TEXT,
        total_bytes INTEGER
    )""")
    c.execute("""CREATE TABLE message_attachment_join (
        message_id INTEGER,
        attachment_id INTEGER
    )""")

    # Insert test data
    # Apple epoch: 2001-01-01. We use nanoseconds.
    # 2024-01-15 10:00:00 UTC = 726915600 seconds since 2001-01-01
    base_ns = 726915600 * 1_000_000_000

    c.execute("INSERT INTO handle VALUES (1, '+15551234567')")
    c.execute("INSERT INTO chat VALUES (1, '+15551234567')")

    c.execute("INSERT INTO message VALUES (1, ?, 1, 'hello there', NULL, 0, 'iMessage', NULL, 0)",
              (base_ns,))
    c.execute("INSERT INTO message VALUES (2, ?, 1, 'hey!', NULL, 1, 'iMessage', NULL, 0)",
              (base_ns + 60_000_000_000,))
    c.execute("INSERT INTO message VALUES (3, ?, 1, NULL, NULL, 0, 'iMessage', 'guid1', 2000)",
              (base_ns + 120_000_000_000,))

    c.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    c.execute("INSERT INTO chat_message_join VALUES (1, 2)")
    c.execute("INSERT INTO chat_message_join VALUES (1, 3)")

    c.execute("INSERT INTO attachment VALUES (1, 'photo.jpg', 'image/jpeg', 1024)")
    c.execute("INSERT INTO message_attachment_join VALUES (1, 1)")

    conn.commit()
    conn.close()
    return str(db_path)
