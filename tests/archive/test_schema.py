"""Schema, migrations and connection rules of the archive file."""

from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

from archive import connection, migrate
from archive.errors import ArchiveError, ArchiveVersionError, SqliteFeatureError
from archive.schema import APPLICATION_ID

EXPECTED_TABLES = {
    "import_runs",
    "people",
    "handles",
    "conversations",
    "conversation_members",
    "messages",
    "attachments",
    "reactions",
    "messages_fts",
}


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def _seed(conn: sqlite3.Connection) -> int:
    """One run, one person, one handle, one conversation. Returns the conversation id."""
    conn.execute(
        "INSERT INTO import_runs(id, started_at, source_path, mode, status) VALUES (1, 0, 'synthetic', 'full', 'ok')"
    )
    conn.execute("INSERT INTO people(id, name) VALUES (1, 'Rowan Vale')")
    conn.execute(
        "INSERT INTO handles(id, handle, kind, match_key, person_id) VALUES (1, '+15550100', 'phone', 'p:5550100', 1)"
    )
    conn.execute("INSERT INTO conversations(id, chat_identifier, is_group) VALUES (1, '+15550100', 0)")
    return 1


def _add_message(conn: sqlite3.Connection, guid: str, text: str, sent_ns: int = 1_700_000_000_000_000_000) -> int:
    cursor = conn.execute(
        "INSERT INTO messages(guid, conversation_id, sender_handle_id, is_from_me, sent_ns, text, kind,"
        " source_rowid, first_import_id, last_import_id) VALUES (?, 1, 1, 0, ?, ?, 'text', 1, 1, 1)",
        (guid, sent_ns, text),
    )
    return int(cursor.lastrowid or 0)


def _match(conn: sqlite3.Connection, query: str) -> list[int]:
    rows = conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rowid", (query,))
    return [row[0] for row in rows]


@pytest.fixture
def writer(archive_path: Path) -> sqlite3.Connection:
    conn = connection.open_writer(archive_path)
    yield conn
    conn.close()


class TestOpenWriter:
    def test_new_file_has_the_latest_schema(self, writer: sqlite3.Connection) -> None:
        assert _version(writer) == len(migrate.MIGRATIONS) >= 1
        assert EXPECTED_TABLES <= _tables(writer)
        assert writer.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID

    def test_wal_mode_and_foreign_keys(self, writer: sqlite3.Connection) -> None:
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert writer.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_parent_folder_is_created_and_file_is_private(self, archive_path: Path) -> None:
        assert not archive_path.parent.exists()
        connection.open_writer(archive_path).close()
        assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600

    def test_second_open_changes_nothing(self, archive_path: Path) -> None:
        first = connection.open_writer(archive_path)
        _seed(first)
        _add_message(first, "g1", "kept")
        first.close()
        second = connection.open_writer(archive_path)
        assert _version(second) == len(migrate.MIGRATIONS)
        assert second.execute("SELECT text FROM messages").fetchall() == [("kept",)]
        second.close()

    def test_newer_archive_is_a_clear_error(self, archive_path: Path) -> None:
        conn = connection.open_writer(archive_path)
        conn.execute(f"PRAGMA user_version = {len(migrate.MIGRATIONS) + 1}")
        conn.close()
        with pytest.raises(ArchiveVersionError, match="newer"):
            connection.open_writer(archive_path)
        with pytest.raises(ArchiveVersionError, match="newer"):
            connection.open_reader(archive_path)

    def test_another_database_is_refused(self, archive_path: Path) -> None:
        archive_path.parent.mkdir(parents=True)
        other = sqlite3.connect(str(archive_path))
        other.execute("CREATE TABLE notes(body TEXT)")
        other.commit()
        other.close()
        with pytest.raises(ArchiveError, match="not a Recall archive"):
            connection.open_writer(archive_path)
        check = sqlite3.connect(str(archive_path))
        assert _tables(check) == {"notes"}
        check.close()

    def test_a_file_that_is_not_sqlite_is_refused(self, archive_path: Path) -> None:
        archive_path.parent.mkdir(parents=True)
        archive_path.write_bytes(b"this is not a database, it is a text file with enough bytes" * 4)
        with pytest.raises(ArchiveError):
            connection.open_writer(archive_path)


class TestMigrate:
    def test_missing_migrations_run_in_order(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(str(tmp_path / "m.db"), isolation_level=None)
        seen: list[int] = []
        steps = tuple((lambda c, n=n: (seen.append(n), c.execute(f"CREATE TABLE t{n}(x)"))[0]) for n in (1, 2, 3))
        assert migrate.migrate(conn, steps[:1]) == 1
        assert migrate.migrate(conn, steps) == 3
        assert seen == [1, 2, 3]
        assert _tables(conn) == {"t1", "t2", "t3"}
        assert migrate.migrate(conn, steps) == 3
        assert seen == [1, 2, 3]
        conn.close()

    def test_a_failed_migration_rolls_back_every_step(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(str(tmp_path / "m.db"), isolation_level=None)

        def good(c: sqlite3.Connection) -> None:
            c.execute("CREATE TABLE made(x)")

        def bad(c: sqlite3.Connection) -> None:
            c.execute("CREATE TABLE also_made(x)")
            raise RuntimeError("step two failed")

        with pytest.raises(RuntimeError, match="step two failed"):
            migrate.migrate(conn, (good, bad))
        assert _version(conn) == 0
        assert _tables(conn) == set()
        assert not conn.in_transaction
        conn.close()

    def test_newer_version_is_refused_before_any_change(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(str(tmp_path / "m.db"), isolation_level=None)
        conn.execute("PRAGMA user_version = 7")
        with pytest.raises(ArchiveVersionError, match="version 7"):
            migrate.migrate(conn, (lambda c: None,))
        assert _version(conn) == 7
        conn.close()


class TestFullTextIndex:
    def test_insert_update_delete_keep_the_index_in_step(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        first = _add_message(writer, "g1", "the ferry leaves at noon")
        second = _add_message(writer, "g2", "bring the tickets")
        assert _match(writer, '"ferry"') == [first]
        assert _match(writer, '"the"') == [first, second]

        writer.execute("UPDATE messages SET text = 'the train leaves at noon' WHERE id = ?", (first,))
        assert _match(writer, '"ferry"') == []
        assert _match(writer, '"train"') == [first]

        writer.execute("UPDATE messages SET last_import_id = 1 WHERE id = ?", (first,))
        assert _match(writer, '"train"') == [first]

        writer.execute("DELETE FROM messages WHERE id = ?", (second,))
        assert _match(writer, '"tickets"') == []
        writer.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES ('integrity-check', 1)")

    def test_case_and_diacritics_do_not_matter(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        row = _add_message(writer, "g1", "Déjà vu at the CAFÉ")
        assert _match(writer, '"deja"') == [row]
        assert _match(writer, '"cafe"') == [row]
        assert _match(writer, '"DÉJÀ"') == [row]

    def test_index_is_external_content(self, writer: sqlite3.Connection) -> None:
        sql = writer.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts'").fetchone()[0]
        assert "content='messages'" in sql
        assert "unicode61 remove_diacritics 2" in sql


class TestConstraints:
    def test_sent_at_is_derived_from_nanoseconds(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        _add_message(writer, "g1", "hello", sent_ns=1_700_000_123_987_654_321)
        assert writer.execute("SELECT sent_at FROM messages").fetchone()[0] == 1_700_000_123

    def test_guid_is_unique(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        _add_message(writer, "g1", "one")
        with pytest.raises(sqlite3.IntegrityError):
            _add_message(writer, "g1", "two")

    def test_kind_is_checked(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        with pytest.raises(sqlite3.IntegrityError):
            writer.execute(
                "INSERT INTO messages(guid, conversation_id, is_from_me, sent_ns, text, kind, source_rowid,"
                " first_import_id, last_import_id) VALUES ('g', 1, 1, 0, '', 'tapback', 1, 1, 1)"
            )

    def test_strict_tables_refuse_wrong_types(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        with pytest.raises(sqlite3.IntegrityError):
            writer.execute("UPDATE conversations SET message_count = 'many' WHERE id = 1")

    def test_foreign_keys_are_enforced(self, writer: sqlite3.Connection) -> None:
        _seed(writer)
        with pytest.raises(sqlite3.IntegrityError):
            writer.execute(
                "INSERT INTO messages(guid, conversation_id, is_from_me, sent_ns, text, kind, source_rowid,"
                " first_import_id, last_import_id) VALUES ('g', 99, 1, 0, '', 'text', 1, 1, 1)"
            )


class TestOpenReader:
    def test_reader_cannot_write(self, archive_path: Path) -> None:
        connection.open_writer(archive_path).close()
        reader = connection.open_reader(archive_path)
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("INSERT INTO people(name) VALUES ('nobody')")
        reader.close()

    def test_reader_sees_committed_rows_while_a_writer_is_open(self, archive_path: Path) -> None:
        writer = connection.open_writer(archive_path)
        _seed(writer)
        reader = connection.open_reader(archive_path)
        assert reader.execute("SELECT name FROM people").fetchall() == [("Rowan Vale",)]
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO people(name) VALUES ('Mira Okafor')")
        assert reader.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 1
        writer.execute("COMMIT")
        assert reader.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 2
        reader.close()
        writer.close()

    def test_missing_file_is_not_created(self, archive_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            connection.open_reader(archive_path)
        assert not archive_path.exists()


class TestSqliteFeatures:
    def test_this_interpreter_passes(self) -> None:
        connection.require_sqlite_features()

    def test_missing_fts5_is_a_clear_error(self) -> None:
        with pytest.raises(SqliteFeatureError, match="FTS5"):
            connection.require_sqlite_features(has_fts5=False)

    def test_old_sqlite_is_a_clear_error(self) -> None:
        with pytest.raises(SqliteFeatureError, match="3.44"):
            connection.require_sqlite_features(version=(3, 31, 1))

    def test_fts5_probe_reports_a_missing_module(self) -> None:
        class NoFts:
            def execute(self, sql: str) -> None:
                raise sqlite3.OperationalError("no such module: fts5")

            def close(self) -> None:
                return None

        assert connection.fts5_available(lambda: NoFts()) is False
        assert connection.fts5_available() is True

    def test_open_fails_at_once_without_fts5(self, archive_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(connection, "fts5_available", lambda *args: False)
        with pytest.raises(SqliteFeatureError, match="FTS5"):
            connection.open_writer(archive_path)
        assert not archive_path.exists()
