import sqlite3

import pytest

import setup_status
from setup_status import DbProbe, derive_state, probe_database


def make_messages_db(path, rows=0, with_chat=True):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, date INTEGER)")
    if with_chat:
        conn.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO chat DEFAULT VALUES")
    for i in range(rows):
        # apple epoch nanoseconds; 7e17 ns ~ 2023
        conn.execute("INSERT INTO message (date) VALUES (?)", (700000000000000000 + i,))
    conn.commit()
    conn.close()


def test_probe_missing_copy(tmp_path):
    probe = probe_database(tmp_path / "nope.db")
    assert probe.status == "missing"
    assert probe.kind == "copy"


def test_probe_not_sqlite(tmp_path):
    bad = tmp_path / "chat.db"
    bad.write_bytes(b"definitely not a database")
    assert probe_database(bad).status == "invalid"


def test_probe_directory_is_invalid(tmp_path):
    assert probe_database(tmp_path).status == "invalid"


def test_probe_sqlite_without_message_table(tmp_path):
    db = tmp_path / "chat.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE other (id INTEGER)")
    conn.commit()
    conn.close()
    probe = probe_database(db)
    assert probe.status == "invalid"
    assert "message table" in probe.detail


def test_probe_empty_db(tmp_path):
    db = tmp_path / "chat.db"
    make_messages_db(db, rows=0)
    assert probe_database(db).status == "empty"


def test_probe_readable_with_counts(tmp_path):
    db = tmp_path / "chat.db"
    make_messages_db(db, rows=12)
    probe = probe_database(db, deep=True)
    assert probe.status == "readable"
    assert probe.approx_messages == 12
    assert probe.conversations == 1
    assert probe.first_year and probe.first_year >= 2020


def test_probe_never_creates_the_file(tmp_path):
    target = tmp_path / "ghost.db"
    probe_database(target)
    assert not target.exists()


@pytest.mark.parametrize("name", ["with space.db", "what?.db", "hash#tag.db", "per%20cent.db"])
def test_probe_reads_a_file_whose_name_has_marks_in_it(tmp_path, name):
    db = tmp_path / name
    make_messages_db(db, rows=3)

    probe = probe_database(db)

    assert (probe.status, probe.approx_messages) == ("readable", 3)
    assert [path.name for path in tmp_path.iterdir()] == [name]


def test_only_a_file_in_the_messages_folder_is_the_database_of_messages(tmp_path, monkeypatch):
    library = tmp_path.resolve() / "Library"
    monkeypatch.setattr(setup_status, "LIVE_MESSAGES_DIR", library / "Messages")
    for folder in ("Messages", "Messages-old", "Messages2", "Messages/Archive"):
        (library / folder).mkdir(parents=True)
        make_messages_db(library / folder / "chat.db", rows=1)

    kinds = {folder: probe_database(library / folder / "chat.db").kind for folder in ("Messages", "Messages/Archive", "Messages-old", "Messages2")}

    # a folder whose name only starts like the folder of Messages holds a copy
    assert kinds == {"Messages": "live", "Messages/Archive": "live", "Messages-old": "copy", "Messages2": "copy"}


def test_a_missing_file_beside_the_messages_folder_is_a_missing_copy(tmp_path, monkeypatch):
    library = tmp_path.resolve() / "Library"
    monkeypatch.setattr(setup_status, "LIVE_MESSAGES_DIR", library / "Messages")
    (library / "Messages-old").mkdir(parents=True)

    probe = probe_database(library / "Messages-old" / "chat.db")

    assert (probe.status, probe.kind) == ("missing", "copy")


def test_the_messages_folder_is_known_through_a_link(tmp_path, monkeypatch):
    real = tmp_path.resolve() / "real" / "Messages"
    real.mkdir(parents=True)
    make_messages_db(real / "chat.db", rows=1)
    linked = tmp_path / "home"
    linked.symlink_to(tmp_path / "real")
    monkeypatch.setattr(setup_status, "LIVE_MESSAGES_DIR", linked / "Messages")

    assert probe_database(real / "chat.db").kind == "live"
    assert probe_database(linked / "Messages" / "chat.db").kind == "live"


def test_a_path_is_looked_at_as_the_file_it_means(tmp_path, monkeypatch):
    library = tmp_path.resolve() / "Library"
    (library / "Messages" / "Archive").mkdir(parents=True)
    monkeypatch.setattr(setup_status, "LIVE_MESSAGES_DIR", library / "Messages")

    # the text of this path starts like the folder of Messages, and the file it means is outside it
    assert setup_status.in_messages_folder(library / "Messages" / ".." / "elsewhere.db") is False
    assert setup_status.in_messages_folder(library / "Messages" / "Archive" / ".." / "chat.db") is True


def test_a_path_that_no_system_can_hold_is_a_copy_that_could_not_be_read():
    probe = probe_database("bad\0name.db")

    assert (probe.kind, probe.status) == ("copy", "error")


def test_a_copy_made_alone_gets_no_file_beside_it_from_the_probe(tmp_path):
    db = tmp_path / "chat.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    make_messages_db(db, rows=4)

    probe = probe_database(db, deep=True)

    assert (probe.status, probe.approx_messages) == ("readable", 4)
    assert [path.name for path in tmp_path.iterdir()] == ["chat.db"]


def test_derive_state_matrix():
    def db(status):
        return DbProbe(status=status, kind="copy", path="x")

    assert derive_state(db("readable"), export_exists=True) == "ready"
    assert derive_state(db("fda_blocked"), export_exists=True) == "ready"
    assert derive_state(db("readable"), export_exists=False) == "needs_export"
    assert derive_state(db("fda_blocked"), export_exists=False) == "needs_permission"
    assert derive_state(db("missing"), export_exists=False) == "no_messages"
    assert derive_state(db("empty"), export_exists=False) == "no_messages"
    assert derive_state(db("locked"), export_exists=False) == "db_locked"
    assert derive_state(db("invalid"), export_exists=False) == "db_invalid"
    assert derive_state(db("error"), export_exists=False) == "db_invalid"
