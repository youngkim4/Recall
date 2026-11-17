"""Odd and hostile input: a source with wrong column types, strange names, SQL in text, bad arguments."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from archive import Archive, ImportFailed, import_chat_db, runs
from archive import importer as importer_module
from archive import search as search_module
from archive.compat import messages_frame
from archive.filters import as_id_list
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, apple_time, utc
from tests.archive.conftest import Harbor
from tests.archive.inspect_archive import fts_is_consistent, query, scalar, texts, totals_match_messages
from tests.archive.typedstream_fixtures import blob

WHEN = apple_time(utc(2024, 5, 1, 12, 0))


class TestSourceWithWrongColumnTypes:
    """SQLite lets any value sit in any column. A row with odd types must not stop an import."""

    def insert(self, builder: ChatDbBuilder, chat: int, **values: Any) -> int:
        row: dict[str, Any] = {"guid": f"ODD-{len(values)}-{id(values)}", "date": WHEN, "handle_id": 1, "is_from_me": 0}
        row.update(values)
        names = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        rowid = builder.raw.execute(f"INSERT INTO message ({names}) VALUES ({marks})", tuple(row.values())).lastrowid
        builder.link(chat, int(rowid or 0))
        return int(rowid or 0)

    def test_values_of_the_wrong_type_are_read(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        self.insert(builder, chat, guid="TEXT-AS-BLOB", text=b"bytes in the text column")
        self.insert(builder, chat, guid="FLAG-AS-TEXT", text="flag as text", is_from_me="1")
        self.insert(builder, chat, guid="DATE-AS-REAL", text="date as a real number", date=float(WHEN))
        self.insert(builder, chat, guid="HANDLE-AS-TEXT", text="handle is not a number", handle_id="abc")
        self.insert(builder, chat, guid=12345, text="guid is a number")
        self.insert(builder, chat, guid="BODY-AS-TEXT", attributedBody="not a blob")
        self.insert(builder, chat, guid="TYPE-AS-TEXT", text="item type is text", item_type="x")
        self.insert(
            builder, chat, guid="TAPBACK-AS-TEXT", associated_message_type="2000", associated_message_guid="p:0/12345"
        )
        self.insert(builder, chat, guid="BAD-UTF8", text=b"caf\xc3\xa9 \xff\xfe ok")
        self.insert(builder, chat, guid="DATE-AS-WORD", text="no usable date", date="soon")
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        stored = dict(query(archive_path, "SELECT guid, text FROM messages"))
        assert stored["TEXT-AS-BLOB"] == "bytes in the text column"
        assert stored["DATE-AS-REAL"] == "date as a real number"
        assert stored["12345"] == "guid is a number"
        assert stored["TYPE-AS-TEXT"] == "item type is text"
        assert stored["BAD-UTF8"].startswith("café ") and stored["BAD-UTF8"].endswith(" ok")
        assert "BODY-AS-TEXT" not in stored and "DATE-AS-WORD" not in stored
        assert scalar(archive_path, "SELECT is_from_me FROM messages WHERE guid = 'FLAG-AS-TEXT'") == 1
        assert scalar(archive_path, "SELECT sender_handle_id FROM messages WHERE guid = 'HANDLE-AS-TEXT'") is None
        assert query(archive_path, "SELECT target_guid, kind FROM reactions") == [("12345", "love")]
        assert dict(result.skipped) == {"empty": 1, "no_date": 1}
        assert result.rows_seen == 10
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)

    def test_attachment_with_odd_values(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        message = builder.message(chat, when=utc(2024, 5, 1, 12), handle=rowan)
        builder.raw.execute(
            "INSERT INTO attachment (ROWID, guid, filename, mime_type, transfer_name, total_bytes) VALUES (1, 77, 88, 99, NULL, 'big')"
        )
        builder.raw.execute("INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (?, 1)", (message,))
        builder.raw.execute(
            "INSERT INTO attachment (ROWID, guid, filename, mime_type, transfer_name, total_bytes) VALUES (2, '  ', NULL, '', '', -5)"
        )
        builder.raw.execute("INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (?, 2)", (message,))
        builder.commit()
        import_chat_db(source_path, archive_path)
        rows = query(
            archive_path,
            "SELECT guid, position, mime_type, file_name, path, total_bytes FROM attachments ORDER BY position",
        )
        assert rows == [("77", 0, "99", "88", "88", None), (f"{builder.guid_of(message)}/1", 1, None, None, None, None)]
        with Archive.open(archive_path) as archive:
            assert archive.messages(worlds.ROWAN_PHONE)["messages"][0]["attachments"] == [
                "99",
                "application/octet-stream",
            ]
            frame = messages_frame(archive)
        assert frame.iloc[0]["text"] == "[Attachment] 99|application/octet-stream | 88|"  # one item for each file


class TestHostileSource:
    def test_strange_index_and_table_names(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        worlds.build_lake(builder)
        builder.raw.execute("DROP INDEX chat_message_join_idx_message_id_only")
        builder.raw.execute('CREATE INDEX "odd "" name; DROP TABLE message; --" ON chat_message_join(chat_id)')
        builder.raw.execute('CREATE TABLE "other; table" (x)')
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert result.messages_added == worlds.MESSAGE_ROWS
        check = sqlite3.connect(str(source_path))
        assert check.execute("SELECT COUNT(*) FROM message").fetchone()[0] == worlds.SOURCE_ROWS
        assert [row[1] for row in check.execute("PRAGMA index_list(chat_message_join)") if "recall" in row[1]] == []
        check.close()

    def test_sql_in_text_is_only_text(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle("'; DROP TABLE handles; --")
        chat = builder.chat("x' OR '1'='1", members=[rowan], display_name='"); DELETE FROM messages; --')
        nasty = "Robert'); DROP TABLE messages; -- \" OR 1=1 /* */ \\ %s {0} \x00 end"
        builder.message(chat, when=utc(2024, 5, 1, 12), text=nasty, handle=rowan, guid="g'; DROP TABLE reactions; --")
        builder.commit()
        import_chat_db(
            source_path, archive_path, contacts=[{"name": "'); DROP TABLE people; --", "phones": [], "emails": ["a@b"]}]
        )
        with Archive.open(archive_path) as archive:
            conversation = archive.conversations()[0]
            assert conversation["id"] == "x' OR '1'='1"
            assert conversation["members"] == ["'; DROP TABLE handles; --"]
            message = archive.messages("x' OR '1'='1")["messages"][0]
            assert message["id"] == "g'; DROP TABLE reactions; --"
            assert message["text"] == nasty.replace("\x00", "")
            assert archive.message("g'; DROP TABLE reactions; --") == message
            assert archive.search("DROP TABLE messages")["total"] == 1
            assert archive.search("'); DROP TABLE messages; --")["total"] == 1
            assert archive.count(conversations=["x' OR '1'='1"], text="robert") == 1
            assert archive.reactions(["g'; DROP TABLE reactions; --"]) == {"g'; DROP TABLE reactions; --": []}
            assert archive.stats("x' OR '1'='1")["total_messages"] == 1
            assert archive.context("' OR 1=1 --") == []
            assert archive.conversation("' OR 1=1 --") is None
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)

    def test_unreadable_blobs_do_not_stop_the_import(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        good = blob("ascii_300")
        bodies = [good[:40], good[:-200], b"\x04\x0bstreamtyped" + b"\x84" * 5000, bytes(range(256)) * 4, b"\x00"]
        bodies.append(good.replace(b"\x81\x2c\x01", b"\x82\xff\xff\xff\x7f"))
        for index, body in enumerate(bodies):
            builder.message(chat, when=utc(2024, 5, 1, 12, index), body=body, handle=rowan)
        builder.message(chat, when=utc(2024, 5, 1, 13), body=good, handle=rowan)
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert result.decode_failures == len(bodies)
        assert dict(result.skipped) == {"empty": len(bodies)}
        assert len(texts(archive_path)) == 1


class TestBadArguments:
    @pytest.mark.parametrize("value", [5, None, [1, 2], b"abc", {"a": 1}, ["ok", None]])
    def test_ids_must_be_text(self, value: object) -> None:
        with pytest.raises(TypeError):
            as_id_list("ids", value)

    def test_id_lists_drop_repeats_and_keep_order(self) -> None:
        assert as_id_list("ids", ("b", "a", "b")) == ["b", "a"]
        assert as_id_list("ids", "one") == ["one"]
        assert as_id_list("ids", iter(["x"])) == ["x"]

    @pytest.mark.parametrize("call", ["messages", "iter_messages", "stats", "reactions"])
    def test_methods_refuse_ids_that_are_not_text(self, harbor: Harbor, call: str) -> None:
        with pytest.raises(TypeError):
            result = getattr(harbor.archive, call)(42)
            list(result) if call == "iter_messages" else None

    def test_cancel_must_be_callable_or_have_is_set(self, source_path: Path, archive_path: Path) -> None:
        with pytest.raises(TypeError, match="cancel"):
            import_chat_db(source_path, archive_path, cancel="stop")
        assert not archive_path.exists()

    def test_bad_days_raise_before_any_query(self, harbor: Harbor) -> None:
        for call in (harbor.archive.search, harbor.archive.count):
            with pytest.raises(ValueError, match="YYYY-MM-DD"):
                call("x", since="2024/01/01") if call == harbor.archive.search else call(since="2024/01/01")
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            harbor.archive.stats(worlds.ROWAN_PHONE, until="tomorrow")


class TestSearchFailures:
    def test_an_fts_error_gives_an_empty_result(self, harbor: Harbor, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise sqlite3.OperationalError('fts5: syntax error near "x"')

        monkeypatch.setattr(search_module, "_page_keys", broken)
        assert harbor.archive.search("brunch") == {"total": 0, "hits": []}

    def test_another_database_error_is_not_hidden(self, harbor: Harbor, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*args: Any, **kwargs: Any) -> Any:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(search_module, "_page_keys", broken)
        with pytest.raises(sqlite3.OperationalError, match="disk"):
            harbor.archive.search("brunch")
        assert harbor.archive.summary()["messages"] == 19  # the read transaction was closed

    def test_text_that_holds_every_marker_still_has_a_hit(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        markers = "".join(char for pair in search_module.MARKERS for char in pair if ord(char) > 31)
        builder.message(chat, when=utc(2024, 5, 1, 12), text=f"all {markers} markers \x01 and \x02 here", handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            result = archive.search("markers")
        assert result["total"] == 1 and len(result["hits"]) == 1
        assert result["hits"][0]["ranges"] == []


class TestQueriesInsideOneReadTransaction:
    def test_nested_reads_share_the_transaction(self, harbor: Harbor) -> None:
        with harbor.archive._reading() as conn:
            assert conn.in_transaction
            assert harbor.archive.search("happy")["total"] == 3
            assert harbor.archive.stats(worlds.ROWAN_PHONE)["total_messages"] == 10
            assert conn.in_transaction
        assert not harbor.archive._slot().conn.in_transaction

    def test_an_error_inside_a_read_closes_the_transaction(self, harbor: Harbor) -> None:
        with pytest.raises(ValueError):
            harbor.archive.messages(worlds.ROWAN_PHONE, limit=0)
        assert not harbor.archive._slot().conn.in_transaction

    def test_close_while_another_thread_opens_its_connection(self, harbor_files: tuple[Path, dict[str, str]]) -> None:
        archive = Archive.open(harbor_files[0])
        errors: list[str] = []

        def work() -> None:
            try:
                for _ in range(200):
                    archive.summary()
            except Exception as exc:  # noqa: BLE001 - recorded for the assertion
                errors.append(type(exc).__name__)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        archive.close()
        for thread in threads:
            thread.join()
        assert set(errors) <= {"ArchiveError", "ProgrammingError"}
        assert archive.open_connections() == 0


class TestRunLog:
    def test_broken_checkpoints_of_an_old_run_are_ignored(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        worlds.build_lake(builder)
        import_chat_db(source_path, archive_path)
        conn = sqlite3.connect(str(archive_path))
        conn.execute("UPDATE import_runs SET checkpoints = 'not json'")
        conn.commit()
        conn.close()
        again = import_chat_db(source_path, archive_path, full=False)
        assert (again.mode, again.messages_added) == ("full", 0)
        assert import_chat_db(source_path, archive_path, full=False).mode == "incremental"

    def test_failure_that_cannot_be_logged_still_raises_the_first_error(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worlds.build_lake(builder)

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("first error")

        def cannot_log(*args: Any, **kwargs: Any) -> Any:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(importer_module, "classify", boom)
        monkeypatch.setattr(runs, "record_ended_run", cannot_log)
        with pytest.raises(ImportFailed, match="first error") as caught:
            import_chat_db(source_path, archive_path)
        assert caught.value.run_id is None
        assert scalar(archive_path, "SELECT COUNT(*) FROM import_runs") == 0


class TestWorkFolder:
    def test_a_file_that_is_not_ours_keeps_the_folder(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        worlds.build_lake(builder)
        work = importer_module.work_dir(archive_path)
        work.mkdir(parents=True)
        (work / "notes.txt").write_text("not a source copy", encoding="utf-8")
        import_chat_db(source_path, archive_path)
        assert [path.name for path in work.iterdir()] == ["notes.txt"]

    def test_copy_file_is_private_while_it_exists(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        worlds.build_lake(builder)
        builder.close()
        seen: list[tuple[int, int]] = []
        work = importer_module.work_dir(archive_path)

        def progress(event: dict[str, Any]) -> None:
            if event["phase"] == "messages" and not seen:
                seen.extend((path.stat().st_mode & 0o777, work.stat().st_mode & 0o777) for path in work.iterdir())

        import_chat_db(source_path, archive_path, progress=progress)
        assert seen and all(file_mode & 0o077 == 0 and folder_mode == 0o700 for file_mode, folder_mode in seen)
        assert not work.exists()
