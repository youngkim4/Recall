"""Tables of the archive file, version 1.

Rules that the rest of the package relies on:
- `messages.guid` is Apple's message guid. It is the only stable id.
- `messages.sent_ns` is UTC unix nanoseconds and gives the order of messages.
  Messages of one moment are in the order of their source ROWID.
  `messages.sent_at` is the same moment in UTC unix seconds.
- `conversation_members.message_count` is the number of messages that the member
  wrote in that conversation. It gives the order of the members.
- `kind` is `text`, `attachment` or `system`. Tapbacks are rows of `reactions`,
  never of `messages`.
- `messages.text` is the whole text. The full-text index holds `search_text`,
  the first INDEXED_TEXT_CHARS characters, so that a text of many megabytes
  does not fill the index.
- `messages.source_rowid` is the ROWID the row had in the database of the last pull.
- `messages.conversation_id` is NULL for a message that the source links to no chat.
  Such a message belongs to no conversation: no total counts it.
"""

from __future__ import annotations

import unicodedata

APPLICATION_ID = int.from_bytes(b"RCAL", "big")

EMOJI_BLOCKS = range(0x1F000, 0x1FC00)


def _separators() -> str:
    """Every symbol, punctuation mark and emoji above ASCII.

    The unicode61 tokenizer knows Unicode 6.1. It takes a newer emoji for a
    letter, so "ok" followed by such an emoji would be one word that a search
    for "ok" does not find. This list makes each of them a separator. The
    emoji blocks are listed whole, so that an emoji of a later Unicode
    version is a separator too.
    """
    return "".join(
        chr(code)
        for code in range(0x80, 0x20000)
        if not 0xD800 <= code <= 0xDFFF and (unicodedata.category(chr(code))[0] in "SP" or code in EMOJI_BLOCKS)
    )


# The archive file keeps the list it was made with: it is part of the stored table definition.
FTS_TOKENIZER = f"unicode61 remove_diacritics 2 separators '{_separators()}'"
# A text longer than this is an oversize text: only this many characters are searched.
INDEXED_TEXT_CHARS = 10_000

SCHEMA_V1: tuple[str, ...] = (
    f"PRAGMA application_id = {APPLICATION_ID}",
    """
    CREATE TABLE import_runs (
        id INTEGER PRIMARY KEY,
        started_at INTEGER NOT NULL,
        finished_at INTEGER,
        source_path TEXT NOT NULL,
        mode TEXT NOT NULL CHECK (mode IN ('full', 'incremental', 'legacy_csv')),
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
        message_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (conversation_id, handle_id)
    ) STRICT, WITHOUT ROWID
    """,
    f"""
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        guid TEXT NOT NULL UNIQUE,
        conversation_id INTEGER REFERENCES conversations(id),
        sender_handle_id INTEGER REFERENCES handles(id),
        is_from_me INTEGER NOT NULL CHECK (is_from_me IN (0, 1)),
        sent_ns INTEGER NOT NULL,
        sent_at INTEGER GENERATED ALWAYS AS (sent_ns / 1000000000) VIRTUAL,
        text TEXT NOT NULL DEFAULT '',
        search_text TEXT GENERATED ALWAYS AS (substr(text, 1, {INDEXED_TEXT_CHARS})) VIRTUAL,
        kind TEXT NOT NULL CHECK (kind IN ('text', 'attachment', 'system')),
        service TEXT NOT NULL DEFAULT '',
        reply_to_guid TEXT,
        edited_at INTEGER,
        unsent INTEGER NOT NULL DEFAULT 0 CHECK (unsent IN (0, 1)),
        item_type INTEGER NOT NULL DEFAULT 0,
        text_source TEXT NOT NULL DEFAULT 'none' CHECK (text_source IN ('column', 'blob', 'export', 'none')),
        extra TEXT,
        source_rowid INTEGER NOT NULL,
        first_import_id INTEGER NOT NULL REFERENCES import_runs(id),
        last_import_id INTEGER NOT NULL REFERENCES import_runs(id)
    ) STRICT
    """,
    # One conversation in time order: pages, streams, and its figures without a table read.
    # The order of messages is (sent_ns, source_rowid, id): messages of one moment stay in the
    # order of the source. A row-value comparison on that key needs the three columns in the index.
    # The last three columns let a count, and the count of each sender, read the index only.
    """
    CREATE INDEX messages_by_conversation_time ON messages(
        conversation_id, sent_ns, source_rowid, id, is_from_me, kind, sender_handle_id
    )
    """,
    # Every counted message in time order: figures over the whole archive without a table read.
    # SQLite does not pick this index by itself for a query with no time range: name it with INDEXED BY.
    """
    CREATE INDEX messages_counted_by_time ON messages(sent_ns, conversation_id, is_from_me)
    WHERE kind != 'system'
    """,
    # Saved chats of the app hold source ROWIDs: this index resolves them.
    "CREATE INDEX messages_by_source_rowid ON messages(source_rowid)",
    # One source attachment can be joined to more than one message: the guid is unique in a message only.
    """
    CREATE TABLE attachments (
        id INTEGER PRIMARY KEY,
        guid TEXT NOT NULL,
        message_id INTEGER NOT NULL REFERENCES messages(id),
        position INTEGER NOT NULL,
        mime_type TEXT,
        file_name TEXT,
        path TEXT,
        total_bytes INTEGER,
        UNIQUE (message_id, guid)
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
        search_text,
        content='messages',
        content_rowid='id',
        tokenize="{FTS_TOKENIZER}"
    )
    """,
    """
    CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, search_text) VALUES (new.id, new.search_text);
    END
    """,
    # A delete must give the index the same text that was indexed.
    """
    CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
    END
    """,
    """
    CREATE TRIGGER messages_fts_update AFTER UPDATE OF text ON messages
    WHEN old.search_text IS NOT new.search_text BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
        INSERT INTO messages_fts(rowid, search_text) VALUES (new.id, new.search_text);
    END
    """,
)
