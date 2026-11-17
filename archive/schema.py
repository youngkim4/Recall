"""Tables of the archive file, version 1.

Rules that the rest of the package relies on:
- `messages.guid` is Apple's message guid. It is the only stable id.
- `messages.sent_ns` is UTC unix nanoseconds and gives the order of messages.
  `messages.sent_at` is the same moment in UTC unix seconds.
- `kind` is `text`, `attachment` or `system`. Tapbacks are rows of `reactions`,
  never of `messages`.
- Nothing here deletes a message. The delete trigger exists only to keep the
  full-text index right if a row is ever removed by hand.
"""

from __future__ import annotations

APPLICATION_ID = int.from_bytes(b"RCAL", "big")

FTS_TOKENIZER = "unicode61 remove_diacritics 2"

SCHEMA_V1: tuple[str, ...] = (
    f"PRAGMA application_id = {APPLICATION_ID}",
    """
    CREATE TABLE import_runs (
        id INTEGER PRIMARY KEY,
        started_at INTEGER NOT NULL,
        finished_at INTEGER,
        source_path TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('full', 'incremental')),
        status TEXT NOT NULL CHECK (status IN ('running', 'ok', 'failed', 'cancelled')),
        error TEXT,
        counts TEXT NOT NULL DEFAULT '{}',
        checkpoints TEXT NOT NULL DEFAULT '[]'
    ) STRICT
    """,
    """
    CREATE TABLE people (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL DEFAULT ''
    ) STRICT
    """,
    """
    CREATE TABLE handles (
        id INTEGER PRIMARY KEY,
        handle TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK (kind IN ('phone', 'email', 'other')),
        match_key TEXT NOT NULL,
        person_id INTEGER NOT NULL REFERENCES people(id)
    ) STRICT
    """,
    "CREATE INDEX handles_by_person ON handles(person_id)",
    """
    CREATE TABLE conversations (
        id INTEGER PRIMARY KEY,
        chat_identifier TEXT NOT NULL UNIQUE,
        is_group INTEGER NOT NULL CHECK (is_group IN (0, 1)),
        title TEXT NOT NULL DEFAULT '',
        service TEXT NOT NULL DEFAULT '',
        person_id INTEGER REFERENCES people(id),
        message_count INTEGER NOT NULL DEFAULT 0,
        sent_count INTEGER NOT NULL DEFAULT 0,
        received_count INTEGER NOT NULL DEFAULT 0,
        first_sent_ns INTEGER,
        last_sent_ns INTEGER
    ) STRICT
    """,
    "CREATE INDEX conversations_by_person ON conversations(person_id)",
    """
    CREATE TABLE conversation_members (
        conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        handle_id INTEGER NOT NULL REFERENCES handles(id),
        PRIMARY KEY (conversation_id, handle_id)
    ) STRICT, WITHOUT ROWID
    """,
    """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        guid TEXT NOT NULL UNIQUE,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        sender_handle_id INTEGER REFERENCES handles(id),
        is_from_me INTEGER NOT NULL CHECK (is_from_me IN (0, 1)),
        sent_ns INTEGER NOT NULL,
        sent_at INTEGER GENERATED ALWAYS AS (sent_ns / 1000000000) VIRTUAL,
        text TEXT NOT NULL DEFAULT '',
        kind TEXT NOT NULL CHECK (kind IN ('text', 'attachment', 'system')),
        service TEXT NOT NULL DEFAULT '',
        reply_to_guid TEXT,
        edited_at INTEGER,
        unsent INTEGER NOT NULL DEFAULT 0 CHECK (unsent IN (0, 1)),
        item_type INTEGER NOT NULL DEFAULT 0,
        text_source TEXT NOT NULL DEFAULT 'none' CHECK (text_source IN ('column', 'blob', 'none')),
        extra TEXT,
        source_rowid INTEGER NOT NULL,
        first_import_id INTEGER NOT NULL REFERENCES import_runs(id),
        last_import_id INTEGER NOT NULL REFERENCES import_runs(id)
    ) STRICT
    """,
    "CREATE INDEX messages_by_conversation_time ON messages(conversation_id, sent_ns, id)",
    """
    CREATE INDEX messages_counted_by_time ON messages(sent_ns, conversation_id, is_from_me)
    WHERE kind != 'system'
    """,
    """
    CREATE TABLE attachments (
        id INTEGER PRIMARY KEY,
        guid TEXT NOT NULL UNIQUE,
        message_id INTEGER NOT NULL REFERENCES messages(id),
        position INTEGER NOT NULL,
        mime_type TEXT,
        file_name TEXT,
        path TEXT,
        total_bytes INTEGER
    ) STRICT
    """,
    "CREATE INDEX attachments_by_message ON attachments(message_id, position, id)",
    """
    CREATE TABLE reactions (
        id INTEGER PRIMARY KEY,
        guid TEXT NOT NULL UNIQUE,
        target_guid TEXT NOT NULL,
        target_part INTEGER,
        conversation_id INTEGER REFERENCES conversations(id),
        sender_handle_id INTEGER REFERENCES handles(id),
        is_from_me INTEGER NOT NULL CHECK (is_from_me IN (0, 1)),
        kind TEXT NOT NULL,
        emoji TEXT,
        removed INTEGER NOT NULL CHECK (removed IN (0, 1)),
        sent_ns INTEGER NOT NULL,
        source_rowid INTEGER NOT NULL,
        first_import_id INTEGER NOT NULL REFERENCES import_runs(id),
        last_import_id INTEGER NOT NULL REFERENCES import_runs(id)
    ) STRICT
    """,
    "CREATE INDEX reactions_by_target ON reactions(target_guid, sent_ns, id)",
    f"""
    CREATE VIRTUAL TABLE messages_fts USING fts5(
        text,
        content='messages',
        content_rowid='id',
        tokenize='{FTS_TOKENIZER}'
    )
    """,
    """
    CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER messages_fts_update AFTER UPDATE OF text ON messages
    WHEN old.text IS NOT new.text BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
)
