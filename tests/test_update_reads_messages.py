"""An update reads Messages itself and nothing else.

The archive is written from one database. For an update that database is the
one Messages keeps on this Mac: a path the client sends, a path picked in an
earlier setup and a copy in the data folder do not change that. A copy is read
only when the setup guide names it for that one import. Whatever goes wrong,
the archive that was there stays as it was.
"""
import json
import os
from urllib.parse import quote

import pandas as pd
import pytest

import recall_paths
import ui_server
from setup_status import DbProbe
from tests.archive_kit import left_behind, messages_csv, run, texts_in, unchanged
from tests.invented_db import GROUP, messages_db
from tests.loopback import fetch

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_saved_names")


# ---- which database is read ----


def test_an_update_reads_messages_itself_whatever_path_the_client_sends(messages_home, archive, tmp_path):
    messages_db(messages_home / "chat.db", ["from messages", "also from messages"])
    copy = messages_db(tmp_path / "old.db", ["from an old copy"])

    job = run("export", {"dbPath": str(copy), "messagesPath": str(messages_csv(archive))})

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"from messages", "also from messages"}


def test_the_setup_guide_reads_messages_itself_when_it_names_no_copy(messages_home, archive):
    messages_db(messages_home / "chat.db", ["from messages"])

    job = run("setup_import", {"messagesPath": str(messages_csv(archive)), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"from messages"}


def test_the_setup_guide_reads_the_copy_it_names_for_that_import(messages_home, archive, tmp_path):
    messages_db(messages_home / "chat.db", ["from messages"])
    copy = messages_db(tmp_path / "copy.db", ["from the copy"])

    job = run("setup_import", {"dbPath": str(copy), "messagesPath": str(messages_csv(archive)), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"from the copy"}


def test_only_the_setup_guide_can_name_a_database(messages_home, tmp_path):
    live = messages_home / "chat.db"
    copy = tmp_path / "copy.db"
    named = {"dbPath": str(copy), "extractFirst": True}

    assert ui_server.database_for("export", named) == live
    assert ui_server.database_for("setup_import", {}) == live
    assert ui_server.database_for("setup_import", named) == copy.resolve()
    assert ui_server.database_for("analyze", named) is None
    assert ui_server.database_for("semantic", named) is None


def test_a_report_does_not_read_a_database_first(messages_home, archive, tmp_path):
    messages_db(messages_home / "chat.db", ["from messages"])
    copy = messages_db(tmp_path / "old.db", ["from an old copy"])

    # the job stops for want of a conversation; before that it must not have written an archive
    job = run("analyze", {"dbPath": str(copy), "extractFirst": True, "messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert unchanged(archive)


# ---- what is written ----


def test_no_second_source_is_merged_into_an_update(messages_home, archive, tmp_path, monkeypatch):
    messages_db(messages_home / "chat.db", ["still in messages"])
    in_data_folder = messages_db(tmp_path / "chat.db", ["still in messages", "gone from messages"])
    monkeypatch.setattr(recall_paths, "SNAPSHOT_DB", in_data_folder)
    monkeypatch.setattr(ui_server, "SNAPSHOT_DB", in_data_folder, raising=False)

    job = run("setup_import", {"messagesPath": str(messages_csv(archive)), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"still in messages"}
    assert "recovered" not in job["result"]


def test_an_update_writes_the_three_files_and_leaves_nothing_else(messages_home, archive):
    messages_db(messages_home / "chat.db", ["one", "two", "three"])

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "completed", job["error"]
    assert job["result"]["messages"] == len(pd.read_csv(messages_csv(archive)))
    assert "recovered" not in job["result"]
    assert not unchanged(archive)
    assert left_behind(archive) == []


# ---- when the database cannot be read ----


def test_an_update_that_macos_hides_the_database_from_says_so_and_keeps_the_archive(messages_home, archive, monkeypatch):
    # without Full Disk Access macOS can show the file as missing; the check tells the two apart
    monkeypatch.setattr(
        ui_server,
        "probe_database",
        lambda path=None, deep=False: DbProbe(status="fda_blocked", kind="live", path=str(path)),
    )

    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert "Full Disk Access" in job["error"]
    assert "not changed" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_an_update_that_is_refused_the_database_says_so_and_keeps_the_archive(messages_home, archive):
    live = messages_db(messages_home / "chat.db", ["from messages"])
    live.chmod(0)
    try:
        if os.access(live, os.R_OK):
            pytest.skip("this user reads any file")
        job = run("export", {"messagesPath": str(messages_csv(archive))})
    finally:
        live.chmod(0o600)

    assert job["status"] == "failed"
    assert "Full Disk Access" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_an_update_on_a_mac_with_no_messages_database_says_so(messages_home, archive):
    job = run("export", {"messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert "no Messages database" in job["error"]
    assert unchanged(archive)


def test_a_copy_that_is_not_there_is_named_in_what_went_wrong(messages_home, archive, tmp_path):
    messages_db(messages_home / "chat.db", ["from messages"])
    missing = tmp_path / "gone.db"

    job = run("setup_import", {"dbPath": str(missing), "messagesPath": str(messages_csv(archive)), "includeContacts": False})

    # a copy that was named and cannot be read stops the import: Messages is not read in its place
    assert job["status"] == "failed"
    assert str(missing.resolve()) in job["error"]
    assert unchanged(archive)


@pytest.mark.parametrize("reader", ["extract_attachments", "extract_reactions"])
def test_an_update_that_fails_part_way_keeps_all_three_files(messages_home, archive, monkeypatch, reader):
    live = messages_db(messages_home / "chat.db", ["from messages"])

    def fail(_path):
        raise RuntimeError("the table could not be read")

    monkeypatch.setattr(ui_server, reader, fail)

    job = run("export", {"dbPath": str(live), "messagesPath": str(messages_csv(archive))})

    assert job["status"] == "failed"
    assert unchanged(archive)
    assert left_behind(archive) == []


# ---- what is remembered ----


def test_a_path_picked_in_an_earlier_setup_is_not_kept(marker):
    marker.parent.mkdir()
    marker.write_text(json.dumps({"completed": True, "pickedDbPath": "/somewhere/chat.db"}), encoding="utf-8")

    assert ui_server.read_setup_marker() == {"completed": True}
    assert ui_server.write_setup_marker({"firstWordsShown": True}) == {"completed": True, "firstWordsShown": True}
    assert "pickedDbPath" not in json.loads(marker.read_text(encoding="utf-8"))


def test_setup_records_how_far_it_got_and_no_path():
    said = {"completed": True, "skipped": False, "pickedDbPath": "/somewhere/chat.db", "other": True}

    assert ui_server.setup_marker_updates(said) == {"completed": True, "skipped": False}


def test_the_defaults_name_no_database(server, marker):
    status, _, body = fetch(server, "/api/defaults")

    defaults = json.loads(body)
    assert status == 200
    assert "dbPath" not in defaults
    assert "hasDb" not in defaults
    assert defaults["messagesPath"]


# ---- the names of groups ----


@pytest.fixture
def group_export(tmp_path):
    """An archive with one group that has no name of its own in it."""
    path = tmp_path / "group.csv"
    pd.DataFrame({
        "timestamp": ["2025-01-01"],
        "chat_id": [GROUP],
        "text": ["who is driving"],
        "is_from_me": [1],
    }).to_csv(path, index=False)
    return path


def names_of_conversations(server, export, db=None):
    query = f"messagesPath={quote(str(export))}" + (f"&dbPath={quote(str(db))}" if db else "")
    status, _, body = fetch(server, f"/api/contacts?{query}")
    assert status == 200, body
    return {contact["chat_id"]: contact.get("display_name") for contact in json.loads(body)["contacts"]}


def test_the_list_of_conversations_takes_no_database_from_the_client(server, messages_home, group_export, tmp_path):
    copy = messages_db(tmp_path / "old.db", ["anything"], group_name="Named In A Copy")

    names = names_of_conversations(server, group_export, db=copy)

    assert names[GROUP] != "Named In A Copy"


def test_a_group_takes_its_name_from_messages_itself_when_that_can_be_read(server, messages_home, group_export):
    messages_db(messages_home / "chat.db", ["anything"], group_name="Project Crew")

    names = names_of_conversations(server, group_export)

    assert names[GROUP] == "Project Crew"
