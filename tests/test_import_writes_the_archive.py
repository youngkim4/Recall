"""What an import writes.

After an import the three files of the archive are those of the database: the
messages, the attachments and the reactions. Each one is written, also when it
has only its header, so no figure of an earlier archive stays beside new
messages. A database with no message to keep is not imported.
"""
import pandas as pd
import pytest

import ui_server
from conversation import load_attachments, load_reactions
from tests.archive_kit import left_behind, messages_csv, run, texts_in, unchanged
from tests.invented_db import PERSON, apple_archive, messages_db

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_saved_names", "messages_home")

NAMES = ["messages.csv", "messages_attachments.csv", "messages_reactions.csv"]


def import_from(copy, archive_file):
    return run("setup_import", {"dbPath": str(copy), "messagesPath": str(archive_file), "includeContacts": False})


@pytest.mark.parametrize("photo, reaction", [(False, False), (True, False), (False, True), (True, True)])
def test_after_an_import_the_three_files_are_those_of_the_database(archive, tmp_path, photo, reaction):
    copy = messages_db(tmp_path / "copy.db", ["one", "two"], photo=photo, reaction=reaction)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    assert job["result"]["messages"] == 2
    assert texts_in(messages_csv(archive)) == {"one", "two"}
    # the photo and the reaction that the earlier archive held are gone with it
    assert len(load_attachments(str(messages_csv(archive)))) == int(photo)
    assert len(load_reactions(str(messages_csv(archive)))) == int(reaction)
    assert sorted(path.name for path in messages_csv(archive).parent.iterdir()) == NAMES


def test_the_figures_of_a_conversation_are_those_of_the_new_archive(archive, tmp_path):
    copy = messages_db(tmp_path / "copy.db", ["one", "two"], photo=False, reaction=False)

    job = import_from(copy, messages_csv(archive))
    stats = ui_server.preview_payload(messages_csv(archive), PERSON, "gpt-5.4-mini")["stats"]

    assert job["status"] == "completed", job["error"]
    assert (stats["totalMessages"], stats["attachments"]["total"], stats["reactions"]["total"]) == (2, 0, 0)


def test_a_first_import_writes_the_three_files(tmp_path):
    copy = messages_db(tmp_path / "copy.db", ["one"], photo=False, reaction=False)
    folder = tmp_path / "new" / "folder"

    job = import_from(copy, folder / "messages.csv")

    assert job["status"] == "completed", job["error"]
    assert sorted(path.name for path in folder.iterdir()) == NAMES


def test_a_long_message_that_is_only_in_its_body_is_whole_in_the_archive(archive, tmp_path):
    long_text = bytes.fromhex(
        "303132333435363738396162636465666768696a6b6c6d6e6f707172737475767778797a"
        "4142434445464748494a4b4c4d4e4f505152535455565758595a"
    ).decode("ascii")
    copy = messages_db(tmp_path / "copy.db", ["short"], bodies=[apple_archive("ascii_300"), apple_archive("spaces_both_ends")])

    job = import_from(copy, messages_csv(archive))

    written = pd.read_csv(messages_csv(archive), dtype=str, keep_default_na=False)["text"].tolist()
    assert job["status"] == "completed", job["error"]
    assert len(written) == 3
    assert len(written[1]) == 300 and written[1].startswith(long_text)
    # the text is kept as Messages has it: its ends are not trimmed
    assert written[2] != written[2].strip()


def test_a_database_with_no_message_to_keep_leaves_the_archive(archive, tmp_path):
    # rows are there, and each one is a reaction: nothing would be left of the archive
    copy = messages_db(tmp_path / "copy.db", [], photo=False, reaction=True)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "holds no messages" in job["error"] and "not changed" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []
