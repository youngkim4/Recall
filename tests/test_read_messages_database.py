"""The readers of a Messages database.

A reader gives a frame with all of its columns, also when there is nothing to
give.
"""
import sqlite3

import pytest

from conversation import load_attachments, load_reactions
from parse_imessage import extract_attachments, extract_messages, extract_reactions
from tests.invented_db import messages_db

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
