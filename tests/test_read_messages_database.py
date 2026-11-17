"""The readers of a Messages database.

A reader gives a frame with all of its columns, also when there is nothing to
give. It opens the database to read and nothing else: a path with no file
makes no file, a copy gets no file beside it, and what is still in the -wal
file is read. An error is the one sqlite3 raises, with no text of a query in
it.
"""
import os
import shutil
import sqlite3

import pandas as pd
import pytest

import setup_status
import ui_server
from conversation import load_attachments, load_reactions
from parse_imessage import (
    extract_attachments,
    extract_messages,
    extract_reactions,
    list_contacts,
    read_frame,
    search_contacts,
)
from setup_status import open_read_only
from tests.invented_db import GROUP, apple_archive, fill, messages_db

MESSAGE_COLUMNS = [
    "message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name",
    "is_from_me", "service", "attachment_types", "attachment_files",
]
ATTACHMENT_COLUMNS = [
    "message_id", "chat_id", "filename", "mime_type", "total_bytes", "is_from_me", "timestamp", "category",
]
REACTION_COLUMNS = [
    "reaction_id", "associated_message_guid", "associated_message_type", "is_from_me",
    "chat_id", "timestamp", "reaction_type", "is_add",
]

READERS = {
    "extract_messages": extract_messages,
    "extract_attachments": extract_attachments,
    "extract_reactions": extract_reactions,
    "list_contacts": list_contacts,
    "search_contacts": lambda path: search_contacts(path, "555"),
}


def read_all(db):
    return extract_messages(str(db)), extract_attachments(str(db)), extract_reactions(str(db))


# ---- a result with nothing in it keeps its columns ----


@pytest.mark.parametrize("photo, reaction", [(True, True), (False, False), (True, False), (False, True)])
def test_a_database_is_read_with_or_without_a_photo_and_a_reaction(tmp_path, photo, reaction):
    db = messages_db(tmp_path / "chat.db", ["one", "two"], photo=photo, reaction=reaction)

    messages, attachments, reactions = read_all(db)

    assert list(messages.columns) == MESSAGE_COLUMNS
    assert list(attachments.columns) == ATTACHMENT_COLUMNS
    assert list(reactions.columns) == REACTION_COLUMNS
    assert (len(messages), len(attachments), len(reactions)) == (2, int(photo), int(reaction))


def test_a_database_with_no_message_gives_three_frames_with_their_columns(tmp_path):
    db = messages_db(tmp_path / "chat.db", [], photo=False, reaction=False)

    messages, attachments, reactions = read_all(db)

    assert list(messages.columns) == MESSAGE_COLUMNS
    assert list(attachments.columns) == ATTACHMENT_COLUMNS
    assert list(reactions.columns) == REACTION_COLUMNS
    assert (len(messages), len(attachments), len(reactions)) == (0, 0, 0)


def test_a_reaction_says_what_it_is_and_whether_it_was_added(tmp_path):
    db = messages_db(tmp_path / "chat.db", ["one"])
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO message VALUES (901, 0, 1, NULL, NULL, 1, 'iMessage', 'guid-1', 3003)")
    conn.commit()
    conn.close()

    reactions = extract_reactions(str(db)).set_index("reaction_id")

    assert reactions["is_add"].dtype == bool
    assert (reactions.loc[900, "reaction_type"], bool(reactions.loc[900, "is_add"])) == ("love", True)
    assert (reactions.loc[901, "reaction_type"], bool(reactions.loc[901, "is_add"])) == ("laugh", False)


def test_a_message_with_only_a_photo_keeps_its_place_as_an_attachment(tmp_path):
    db = messages_db(tmp_path / "chat.db", [], photo=False, reaction=False)
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO message VALUES (1, 0, 1, NULL, NULL, 0, 'iMessage', NULL, 0)")
    conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    conn.execute("INSERT INTO attachment VALUES (1, 'photo.jpg', 'image/jpeg', 1024)")
    conn.execute("INSERT INTO message_attachment_join VALUES (1, 1)")
    conn.commit()
    conn.close()

    assert list(extract_messages(str(db))["text"]) == ["[Attachment] image/jpeg | photo.jpg"]


def test_files_written_from_empty_frames_are_read_back_as_empty(tmp_path):
    db = messages_db(tmp_path / "chat.db", ["one"], photo=False, reaction=False)
    _, attachments, reactions = read_all(db)
    export = tmp_path / "messages.csv"
    attachments.to_csv(tmp_path / "messages_attachments.csv", index=False)
    reactions.to_csv(tmp_path / "messages_reactions.csv", index=False)

    read_attachments, read_reactions = load_attachments(str(export)), load_reactions(str(export))

    assert read_attachments.empty and list(read_attachments.columns) == ATTACHMENT_COLUMNS
    assert read_reactions.empty and list(read_reactions.columns) == REACTION_COLUMNS


# ---- the database is opened to read, and nothing else ----


@pytest.mark.parametrize("reader", READERS)
def test_a_reader_makes_no_file_where_there_is_none(tmp_path, reader):
    ghost = tmp_path / "ghost.db"

    with pytest.raises(sqlite3.OperationalError):
        READERS[reader](str(ghost))

    assert list(tmp_path.iterdir()) == []


def test_a_write_is_refused(tmp_path):
    db = messages_db(tmp_path / "chat.db", ["one"])
    conn = open_read_only(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM message")
    finally:
        conn.close()


@pytest.mark.parametrize("reader", READERS)
def test_a_reader_leaves_the_database_as_it_was(tmp_path, reader):
    db = messages_db(tmp_path / "chat.db", ["one", "two"])
    before = db.read_bytes()

    READERS[reader](str(db))

    assert db.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == ["chat.db"]


@pytest.mark.parametrize(
    "name", ["with space.db", "what?.db", "hash#tag.db", "per%cent.db", "per%20cent.db", "café.db", "a&b=c.db"]
)
def test_a_name_with_marks_in_it_is_the_name_of_the_file(tmp_path, name):
    db = messages_db(tmp_path / name, ["one"])

    assert list(extract_messages(str(db))["text"]) == ["one"]
    assert [path.name for path in tmp_path.iterdir()] == [name]


def test_a_path_that_is_not_whole_is_read_from_the_folder_in_use(tmp_path, monkeypatch):
    messages_db(tmp_path / "chat.db", ["one"])
    monkeypatch.chdir(tmp_path)

    assert list(extract_messages("chat.db")["text"]) == ["one"]


def copy_made_alone(path, texts):
    """A database in WAL mode with no file beside it: what a copy of chat.db alone is."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        fill(conn, texts)
    finally:
        conn.close()
    assert [found.name for found in path.parent.iterdir()] == [path.name]
    return path


def addresses_asked_for(monkeypatch):
    """Each address that sqlite3 is asked to open from now on."""
    asked = []
    connect = sqlite3.connect

    def note_and_connect(address, *args, **kwargs):
        asked.append(str(address))
        return connect(address, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", note_and_connect)
    return asked


@pytest.mark.parametrize("reader", READERS)
def test_a_copy_with_no_wal_beside_it_gets_no_file_beside_it(tmp_path, reader):
    db = copy_made_alone(tmp_path / "chat.db", ["one", "two"])
    before = db.read_bytes()

    READERS[reader](str(db))

    assert [path.name for path in tmp_path.iterdir()] == ["chat.db"]
    assert db.read_bytes() == before


def test_a_copy_made_alone_is_read_whole(tmp_path):
    db = copy_made_alone(tmp_path / "chat.db", ["one", "two"])

    messages, attachments, reactions = read_all(db)

    assert set(messages["text"]) == {"one", "two"}
    assert (len(attachments), len(reactions)) == (1, 1)


def test_a_copy_in_a_folder_that_cannot_be_written_to_is_read(tmp_path):
    folder = tmp_path / "closed"
    folder.mkdir()
    db = copy_made_alone(folder / "chat.db", ["one", "two"])
    folder.chmod(0o555)
    try:
        if os.access(folder, os.W_OK):
            pytest.skip("this user writes to any folder")
        texts = set(extract_messages(str(db))["text"])
    finally:
        folder.chmod(0o755)

    assert texts == {"one", "two"}


def test_a_copy_is_opened_as_a_file_that_does_not_change_only_with_no_wal_beside_it(tmp_path, monkeypatch):
    alone = copy_made_alone(tmp_path / "chat.db", ["one"])
    asked = addresses_asked_for(monkeypatch)

    extract_messages(str(alone))
    (tmp_path / "chat.db-wal").write_bytes(b"")
    extract_messages(str(alone))

    assert ["immutable=1" in address for address in asked] == [True, False]


def test_the_database_of_messages_is_never_opened_as_a_file_that_does_not_change(tmp_path, monkeypatch):
    # Messages can write to it at any moment, also when no -wal file is there yet
    home = tmp_path.resolve() / "Messages"
    home.mkdir()
    monkeypatch.setattr(setup_status, "LIVE_MESSAGES_DIR", home)
    live = copy_made_alone(home / "chat.db", ["one"])
    asked = addresses_asked_for(monkeypatch)

    extract_messages(str(live))

    assert len(asked) == 1 and "mode=ro" in asked[0] and "immutable" not in asked[0]


def test_a_path_through_a_link_and_back_is_the_file_the_system_means(tmp_path):
    (tmp_path / "deep" / "inner").mkdir(parents=True)
    (tmp_path / "side").mkdir()
    messages_db(tmp_path / "deep" / "chat.db", ["the one the path means"])
    (tmp_path / "side" / "link").symlink_to(tmp_path / "deep" / "inner")

    # "side/link/.." is "deep", not "side"
    texts = list(extract_messages(str(tmp_path / "side" / "link" / ".." / "chat.db"))["text"])

    assert texts == ["the one the path means"]


def test_a_path_that_is_not_whole_makes_no_file_where_there_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(sqlite3.OperationalError):
        extract_messages("ghost.db")

    assert list(tmp_path.iterdir()) == []


def tables_in_the_file_alone(path, folder):
    """How many tables the database file holds without its -wal file."""
    folder.mkdir()
    conn = sqlite3.connect(str(shutil.copy(path, folder / path.name)))
    try:
        return conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
    finally:
        conn.close()


def test_what_is_still_in_the_wal_is_read(tmp_path):
    path = tmp_path / "chat.db"
    writer = sqlite3.connect(str(path))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        # nothing moves from the -wal file into the database while the writer holds it
        writer.execute("PRAGMA wal_autocheckpoint=0")
        fill(writer, ["first", "second", "third"])
        assert path.with_name("chat.db-wal").stat().st_size > 0
        assert tables_in_the_file_alone(path, tmp_path / "alone") == 0

        messages, attachments, reactions = read_all(path)
    finally:
        writer.close()

    assert set(messages["text"]) == {"first", "second", "third"}
    assert (len(attachments), len(reactions)) == (1, 1)


def test_a_copy_brought_with_its_wal_is_read_whole(tmp_path):
    path = tmp_path / "chat.db"
    copied = tmp_path / "copied"
    copied.mkdir()
    writer = sqlite3.connect(str(path))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        fill(writer, ["first", "second"])
        for name in ("chat.db", "chat.db-wal", "chat.db-shm"):
            shutil.copy(tmp_path / name, copied / name)
    finally:
        writer.close()

    assert set(extract_messages(str(copied / "chat.db"))["text"]) == {"first", "second"}


def test_the_name_of_a_group_is_read_from_a_file_whose_name_has_marks_in_it(tmp_path):
    db = messages_db(tmp_path / "what?#.db", ["one"], group_name="Project Crew")

    assert ui_server.group_display_names_from_database(db, [GROUP]) == {GROUP: "Project Crew"}
    assert [path.name for path in tmp_path.iterdir()] == ["what?#.db"]


# ---- an error is the one sqlite3 raises ----


@pytest.mark.parametrize("reader", READERS)
def test_an_error_holds_no_text_of_the_query(tmp_path, reader):
    no_tables = tmp_path / "other.db"
    conn = sqlite3.connect(str(no_tables))
    conn.execute("CREATE TABLE other (id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError) as stopped:
        READERS[reader](str(no_tables))

    assert "no such table" in str(stopped.value)
    assert "SELECT" not in str(stopped.value)


@pytest.mark.parametrize("reader", READERS)
def test_a_file_that_is_no_database_gives_an_error_with_no_text_of_the_query(tmp_path, reader):
    not_a_database = tmp_path / "chat.db"
    not_a_database.write_bytes(b"this is not a database, and it is long enough to be read as a page " * 40)

    with pytest.raises(sqlite3.DatabaseError) as stopped:
        READERS[reader](str(not_a_database))

    assert "not a database" in str(stopped.value)
    assert "SELECT" not in str(stopped.value)


@pytest.mark.parametrize("sql", [
    "SELECT ROWID AS message_id, handle_id, is_from_me, text, attributedBody FROM message ORDER BY ROWID",
    "SELECT ROWID AS message_id, date FROM message WHERE ROWID < 0",
    "SELECT COUNT(*) AS messages, MAX(date) AS last FROM message",
    "SELECT filename, total_bytes FROM attachment",
])
def test_a_frame_is_the_one_pandas_gives_for_the_query(tmp_path, sql):
    db = messages_db(tmp_path / "chat.db", ["one", "two", "three"], bodies=[apple_archive("ascii_128")])
    conn = sqlite3.connect(str(db))
    try:
        # a column of whole numbers with a gap in it
        conn.execute("UPDATE message SET handle_id = NULL, is_from_me = NULL WHERE ROWID = 2")
        conn.commit()

        pd.testing.assert_frame_equal(read_frame(conn, sql), pd.read_sql_query(sql, conn))
    finally:
        conn.close()
