import sqlite3

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
