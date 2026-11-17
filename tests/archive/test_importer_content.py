"""What one import writes into the archive, and what it leaves alone in the source."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from archive import ImportResult, SourceError, SourceFormatError, import_chat_db
from archive import importer as importer_module
from tests.archive import worlds
from tests.archive.builders import OLD_MACOS_MISSING, ChatDbBuilder, utc
from tests.archive.inspect_archive import (
    content,
    fts_is_consistent,
    query,
    scalar,
    texts,
    totals_match_messages,
)
from tests.archive.typedstream_fixtures import blob, counter_text

NS = 1_000_000_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def lake(builder: ChatDbBuilder) -> worlds.Lake:
    return worlds.build_lake(builder)


@pytest.fixture
def imported(lake: worlds.Lake, source_path: Path, archive_path: Path) -> ImportResult:
    return import_chat_db(source_path, archive_path)


class TestContent:
    def test_result_counts(self, imported: ImportResult) -> None:
        assert (imported.status, imported.mode) == ("ok", "full")
        assert imported.rows_seen == worlds.SOURCE_ROWS
        assert (imported.messages_seen, imported.reactions_seen, imported.system_seen) == (12, 4, 2)
        assert (imported.messages_added, imported.reactions_added, imported.system_added) == (12, 4, 2)
        assert imported.messages_updated == 0
        assert imported.attachments_added == 2
        assert imported.conversations_added == 3
        assert imported.handles_added == 4
        assert imported.blob_texts == 3
        assert imported.decode_failures == 0
        assert dict(imported.skipped) == {}
        json.dumps(imported.as_dict())

    def test_tapbacks_are_reactions_and_not_messages(
        self, imported: ImportResult, lake: worlds.Lake, archive_path: Path
    ) -> None:
        tapback_guids = {lake.guids[name] for name in ("love_dinner", "like_saturday", "unlike_saturday", "fire_great")}
        message_guids = {row[0] for row in query(archive_path, "SELECT guid FROM messages")}
        assert tapback_guids.isdisjoint(message_guids)
        assert not any("Loved" in text for text in texts(archive_path))
        reactions = query(
            archive_path,
            "SELECT guid, target_guid, target_part, kind, emoji, removed, is_from_me FROM reactions ORDER BY sent_ns",
        )
        assert reactions == [
            (lake.guids["love_dinner"], lake.guids["dinner"], 0, "love", None, 0, 0),
            (lake.guids["like_saturday"], lake.guids["saturday"], 0, "like", None, 0, 1),
            (lake.guids["unlike_saturday"], lake.guids["saturday"], 0, "like", None, 1, 1),
            (lake.guids["fire_great"], lake.guids["great"], None, "emoji", "🔥", 0, 0),
        ]
        assert scalar(archive_path, "SELECT SUM(message_count) FROM conversations") == worlds.MESSAGE_ROWS

    def test_text_comes_from_the_column_or_from_the_blob(
        self, imported: ImportResult, lake: worlds.Lake, archive_path: Path
    ) -> None:
        rows = dict(
            (guid, (text, source))
            for guid, text, source in query(archive_path, "SELECT guid, text, text_source FROM messages")
        )
        assert rows[lake.guids["saturday"]] == ("Are we still on for Saturday?", "column")
        assert rows[lake.guids["dinner"]] == ("Dinner at seven?", "blob")
        assert rows[lake.guids["view"]] == ("Look at this view", "blob")
        assert rows[lake.guids["long"]] == (counter_text(300), "blob")
        assert rows[lake.guids["photo"]] == ("", "none")

    def test_attachment_kind_and_metadata(self, imported: ImportResult, lake: worlds.Lake, archive_path: Path) -> None:
        kinds = dict(query(archive_path, "SELECT guid, kind FROM messages"))
        assert kinds[lake.guids["photo"]] == "attachment"
        assert kinds[lake.guids["view"]] == "text"
        attachments = query(
            archive_path,
            "SELECT m.guid, a.position, a.mime_type, a.file_name, a.total_bytes, a.path FROM attachments a"
            " JOIN messages m ON m.id = a.message_id ORDER BY m.sent_ns",
        )
        assert [row[:5] for row in attachments] == [
            (lake.guids["photo"], 0, "image/heic", "IMG_0042.HEIC", 2_048_000),
            (lake.guids["view"], 0, "image/jpeg", "view.jpeg", 512_000),
        ]
        assert attachments[0][5].endswith("/IMG_0042.HEIC")

    def test_group_events_are_system_rows(self, imported: ImportResult, lake: worlds.Lake, archive_path: Path) -> None:
        rows = query(
            archive_path, "SELECT guid, text, item_type, extra FROM messages WHERE kind = 'system' ORDER BY sent_ns"
        )
        assert [row[:3] for row in rows] == [
            (lake.guids["renamed"], worlds.GROUP_TITLE, 2),
            (lake.guids["added"], "", 1),
        ]
        assert json.loads(rows[1][3]) == {"group_action_type": 0, "other_handle": worlds.TESSA_EMAIL}
        assert (
            scalar(
                archive_path, "SELECT message_count FROM conversations WHERE chat_identifier = ?", (worlds.GROUP_ID,)
            )
            == 5
        )

    def test_own_messages_have_no_sender(self, imported: ImportResult, archive_path: Path) -> None:
        assert (
            scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE is_from_me = 1 AND sender_handle_id IS NOT NULL")
            == 0
        )
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE is_from_me = 1") == 4
        assert (
            scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE is_from_me = 0 AND sender_handle_id IS NULL") == 0
        )

    def test_group_flag_title_and_service(self, imported: ImportResult, archive_path: Path) -> None:
        rows = query(
            archive_path, "SELECT chat_identifier, is_group, title, service FROM conversations ORDER BY chat_identifier"
        )
        assert rows == [
            (worlds.ROWAN_PHONE, 0, "", "iMessage"),
            (worlds.GROUP_ID, 1, worlds.GROUP_TITLE, "iMessage"),
            (worlds.ROWAN_EMAIL, 0, "", "iMessage"),
        ]

    def test_reply_edit_and_unsent_fields(self, imported: ImportResult, lake: worlds.Lake, archive_path: Path) -> None:
        rows = {
            row[0]: row[1:]
            for row in query(archive_path, "SELECT guid, reply_to_guid, edited_at, unsent, text, kind FROM messages")
        }
        assert rows[lake.guids["gotit"]] == (lake.guids["laptop"], None, 0, "Got it", "text")
        assert rows[lake.guids["jackets"]] == (
            None,
            int(utc(2024, 3, 1, 17, 12).timestamp()),
            0,
            "Bring life jackets",
            "text",
        )
        assert rows[lake.guids["unsent"]] == (None, None, 1, "", "text")

    def test_times_are_utc_unix(self, imported: ImportResult, lake: worlds.Lake, archive_path: Path) -> None:
        sent_ns, sent_at = query(
            archive_path, "SELECT sent_ns, sent_at FROM messages WHERE guid = ?", (lake.guids["saturday"],)
        )[0]
        assert sent_at == int(utc(2024, 1, 15, 18, 0).timestamp())
        assert sent_ns == sent_at * NS

    def test_totals_and_full_text_index(self, imported: ImportResult, archive_path: Path) -> None:
        assert totals_match_messages(archive_path)
        assert fts_is_consistent(archive_path)
        totals = query(
            archive_path,
            "SELECT chat_identifier, message_count, sent_count, received_count FROM conversations ORDER BY 1",
        )
        assert totals == [(worlds.ROWAN_PHONE, 5, 2, 3), (worlds.GROUP_ID, 5, 1, 4), (worlds.ROWAN_EMAIL, 2, 1, 1)]

    def test_dates_in_seconds_import_the_same(self, source_path: Path, archive_path: Path) -> None:
        with_seconds = ChatDbBuilder(source_path, date_unit="s")
        lake = worlds.build_lake(with_seconds)
        with_seconds.close()
        import_chat_db(source_path, archive_path)
        assert scalar(archive_path, "SELECT sent_at FROM messages WHERE guid = ?", (lake.guids["saturday"],)) == int(
            utc(2024, 1, 15, 18, 0).timestamp()
        )


class TestMembers:
    def test_members_come_from_the_join_table(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        lake = worlds.build_lake(builder)
        jonas = builder.handle("+15550100789")
        builder.message(
            lake.chats["group"], when=utc(2024, 3, 2, 9, 0), text="I am not on the member list", handle=jonas
        )
        builder.commit()
        import_chat_db(source_path, archive_path)
        members = query(
            archive_path,
            "SELECT h.handle FROM conversation_members cm JOIN handles h ON h.id = cm.handle_id"
            " JOIN conversations c ON c.id = cm.conversation_id WHERE c.chat_identifier = ? ORDER BY 1",
            (worlds.GROUP_ID,),
        )
        # Rowan never wrote a normal message in the group but is a member. Jonas wrote but is not a member.
        assert [row[0] for row in members] == sorted([worlds.ROWAN_PHONE, worlds.MIRA_PHONE, worlds.TESSA_EMAIL])


class TestPeople:
    def person_of(self, archive_path: Path, handle: str) -> tuple[int, str]:
        return query(
            archive_path,
            "SELECT p.id, p.name FROM handles h JOIN people p ON p.id = h.person_id WHERE h.handle = ?",
            (handle,),
        )[0]

    def test_phone_and_email_of_one_record_are_one_person(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT, worlds.MIRA_CONTACT])
        assert self.person_of(archive_path, worlds.ROWAN_PHONE) == self.person_of(archive_path, worlds.ROWAN_EMAIL)
        assert self.person_of(archive_path, worlds.ROWAN_PHONE)[1] == "Rowan Vale"
        assert self.person_of(archive_path, worlds.MIRA_PHONE)[1] == "Mira Okafor"
        linked = dict(query(archive_path, "SELECT chat_identifier, person_id FROM conversations"))
        assert (
            linked[worlds.ROWAN_PHONE]
            == linked[worlds.ROWAN_EMAIL]
            == self.person_of(archive_path, worlds.ROWAN_PHONE)[0]
        )
        assert linked[worlds.GROUP_ID] is None

    def test_same_named_records_stay_two_people(self, lake: worlds.Lake, source_path: Path, archive_path: Path) -> None:
        contacts = [
            {"name": "Sam Lee", "phones": [worlds.ROWAN_PHONE], "emails": []},
            {"name": "Sam Lee", "phones": [worlds.MIRA_PHONE], "emails": []},
        ]
        import_chat_db(source_path, archive_path, contacts=contacts)
        first, second = (
            self.person_of(archive_path, worlds.ROWAN_PHONE),
            self.person_of(archive_path, worlds.MIRA_PHONE),
        )
        assert first[1] == second[1] == "Sam Lee"
        assert first[0] != second[0]

    def test_handle_without_record_is_its_own_unnamed_person(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        tessa, mira = self.person_of(archive_path, worlds.TESSA_EMAIL), self.person_of(archive_path, worlds.MIRA_PHONE)
        assert tessa[1] == mira[1] == ""
        assert tessa[0] != mira[0]
        assert scalar(archive_path, "SELECT COUNT(*) FROM people") == 3
        assert scalar(archive_path, "SELECT COUNT(*) FROM people WHERE id NOT IN (SELECT person_id FROM handles)") == 0

    def test_import_without_contacts_keeps_people(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        before = content(archive_path)
        import_chat_db(source_path, archive_path, full=True)
        assert content(archive_path) == before

    def test_new_contact_list_renames_and_keeps_ids(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        phone_person = self.person_of(archive_path, worlds.ROWAN_PHONE)[0]
        email_person = self.person_of(archive_path, worlds.ROWAN_EMAIL)[0]
        assert phone_person != email_person
        import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        merged = self.person_of(archive_path, worlds.ROWAN_PHONE)
        assert merged == (min(phone_person, email_person), "Rowan Vale")
        assert self.person_of(archive_path, worlds.ROWAN_EMAIL) == merged
        import_chat_db(source_path, archive_path, contacts=[])
        assert self.person_of(archive_path, worlds.ROWAN_PHONE)[1] == ""
        assert (
            self.person_of(archive_path, worlds.ROWAN_PHONE)[0] != self.person_of(archive_path, worlds.ROWAN_EMAIL)[0]
        )
        assert scalar(archive_path, "SELECT COUNT(*) FROM people WHERE id NOT IN (SELECT person_id FROM handles)") == 0

    def test_bad_contact_records_are_counted(self, lake: worlds.Lake, source_path: Path, archive_path: Path) -> None:
        result = import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT, "junk", 7])
        assert result.counts["contacts_skipped"] == 2

    def test_contacts_of_the_wrong_type_fail_before_any_work(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        with pytest.raises(TypeError):
            import_chat_db(source_path, archive_path, contacts={"name": "Rowan Vale"})  # type: ignore[arg-type]
        assert not archive_path.exists()


class TestSourceIsNeverWritten:
    def test_bytes_are_unchanged_and_no_file_appears(
        self, lake: worlds.Lake, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        builder.close()
        before, names = sha(source_path), sorted(p.name for p in source_path.parent.iterdir())
        import_chat_db(source_path, archive_path)
        assert sha(source_path) == before
        assert sorted(p.name for p in source_path.parent.iterdir()) == names == ["chat.db"]

    def test_wal_format_copy_without_side_files_gets_none(self, source_path: Path, archive_path: Path) -> None:
        wal_builder = ChatDbBuilder(source_path, wal=True)
        worlds.build_lake(wal_builder)
        wal_builder.close()
        assert [p.name for p in source_path.parent.iterdir()] == ["chat.db"]
        assert source_path.read_bytes()[18:20] == b"\x02\x02"
        before = sha(source_path)
        result = import_chat_db(source_path, archive_path)
        assert result.messages_added == worlds.MESSAGE_ROWS
        assert sha(source_path) == before
        assert [p.name for p in source_path.parent.iterdir()] == ["chat.db"]

    def test_live_wal_database_is_read_with_its_unflushed_rows(self, source_path: Path, archive_path: Path) -> None:
        live = ChatDbBuilder(source_path, wal=True)
        try:
            live.raw.execute("PRAGMA wal_autocheckpoint=0")
            worlds.build_lake(live)
            assert (source_path.parent / "chat.db-wal").stat().st_size > 0
            main_file = sha(source_path)
            result = import_chat_db(source_path, archive_path)
            assert result.messages_added == worlds.MESSAGE_ROWS
            assert sha(source_path) == main_file
        finally:
            live.close()

    def test_temp_copy_is_removed(self, imported: ImportResult, archive_path: Path) -> None:
        assert sorted(p.name for p in archive_path.parent.iterdir() if "import" in p.name) == []
        assert not (archive_path.parent / importer_module.WORK_DIR_NAME).exists()

    def test_stale_temp_copies_are_removed(self, lake: worlds.Lake, source_path: Path, archive_path: Path) -> None:
        work = archive_path.parent / importer_module.WORK_DIR_NAME
        work.mkdir(parents=True)
        (work / "source-crashed.sqlite3").write_bytes(b"left by a run that was killed")
        import_chat_db(source_path, archive_path)
        assert not work.exists()

    def test_archive_and_its_side_files_are_private(self, imported: ImportResult, archive_path: Path) -> None:
        for path in archive_path.parent.iterdir():
            assert path.stat().st_mode & 0o077 == 0, path.name


class TestOtherSchemas:
    def test_missing_optional_columns_import(self, source_path: Path, archive_path: Path) -> None:
        old = ChatDbBuilder(source_path, without=OLD_MACOS_MISSING)
        lake = worlds.build_lake(old)
        old.close()
        check = sqlite3.connect(str(source_path))
        columns = {row[1] for row in check.execute("PRAGMA table_info(message)")}
        check.close()
        assert columns.isdisjoint(OLD_MACOS_MISSING)
        result = import_chat_db(source_path, archive_path)
        # no date_retracted column: the unsent row has nothing to show and is skipped
        assert result.messages_added == worlds.MESSAGE_ROWS - 1
        assert dict(result.skipped) == {"empty": 1}
        assert result.reactions_added == worlds.REACTION_ROWS
        rows = {
            row[0]: row[1:]
            for row in query(archive_path, "SELECT guid, reply_to_guid, edited_at, unsent FROM messages")
        }
        assert rows[lake.guids["gotit"]] == (None, None, 0)
        assert rows[lake.guids["jackets"]] == (None, None, 0)
        assert (
            scalar(archive_path, "SELECT title FROM conversations WHERE chat_identifier = ?", (worlds.GROUP_ID,)) == ""
        )
        assert scalar(archive_path, "SELECT emoji FROM reactions WHERE kind = 'emoji'") is None

    def test_source_without_attachment_and_member_tables(self, source_path: Path, archive_path: Path) -> None:
        bare = ChatDbBuilder(source_path, skip_tables={"attachment", "message_attachment_join", "chat_handle_join"})
        rowan = bare.handle(worlds.ROWAN_PHONE)
        chat = bare.chat(worlds.ROWAN_PHONE)
        bare.message(chat, when=utc(2024, 1, 1, 12), text="plain", handle=rowan)
        bare.close()
        result = import_chat_db(source_path, archive_path)
        assert (result.messages_added, result.attachments_added) == (1, 0)
        assert scalar(archive_path, "SELECT COUNT(*) FROM conversation_members") == 0

    def test_source_without_join_indexes(self, source_path: Path, archive_path: Path) -> None:
        plain = ChatDbBuilder(source_path, indexes=False)
        worlds.build_lake(plain)
        plain.close()
        before = sha(source_path)
        assert import_chat_db(source_path, archive_path).messages_added == worlds.MESSAGE_ROWS
        assert sha(source_path) == before

    def test_a_database_that_is_not_messages_is_refused(self, source_path: Path, archive_path: Path) -> None:
        other = sqlite3.connect(str(source_path))
        other.execute("CREATE TABLE notes(body TEXT)")
        other.commit()
        other.close()
        with pytest.raises(SourceFormatError, match="message"):
            import_chat_db(source_path, archive_path)

    def test_message_table_without_guid_is_refused(self, source_path: Path, archive_path: Path) -> None:
        ChatDbBuilder(source_path, without={"guid"}).close()
        with pytest.raises(SourceFormatError, match="guid"):
            import_chat_db(source_path, archive_path)

    def test_missing_file_and_non_database_file(self, tmp_path: Path, archive_path: Path) -> None:
        with pytest.raises(SourceError, match="no "):
            import_chat_db(tmp_path / "nowhere" / "chat.db", archive_path)
        not_db = tmp_path / "notes.txt"
        not_db.write_text("this is plain text, not a database " * 10, encoding="utf-8")
        with pytest.raises(SourceError):
            import_chat_db(not_db, archive_path)
        assert not (tmp_path / "nowhere").exists()


class TestOddRows:
    def test_rows_that_cannot_be_stored_are_counted(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        builder.message(chat, when=utc(2024, 1, 1, 12), text="kept", handle=rowan)
        builder.message(None, when=utc(2024, 1, 1, 12, 1), text="no conversation", handle=rowan)
        builder.message(chat, when=0, text="no date", handle=rowan)
        builder.message(chat, when=utc(2024, 1, 1, 12, 3), handle=rowan)
        builder.message(chat, when=utc(2024, 1, 1, 12, 4), body=b"\x04\x0bstreamtyped broken", handle=rowan)
        builder.message(
            chat, when=utc(2024, 1, 1, 12, 5), text="tapback with no target", associated_type=2000, handle=rowan
        )
        builder.message(chat, when=utc(2024, 1, 1, 12, 6), text="", guid="", handle=rowan)
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert texts(archive_path) == ["kept"]
        assert dict(result.skipped) == {"no_conversation": 1, "no_date": 1, "empty": 2, "no_target": 1, "no_guid": 1}
        assert result.decode_failures == 1
        assert result.rows_seen == 7

    def test_sticker_on_a_message_is_an_attachment_message(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        first = builder.message(chat, when=utc(2024, 1, 1, 12), text="hello", handle=rowan)
        sticker = builder.message(
            chat,
            when=utc(2024, 1, 1, 12, 1),
            handle=rowan,
            associated_type=1000,
            associated_guid=f"p:0/{builder.guid_of(first)}",
        )
        builder.attachment(sticker, mime_type="image/png", transfer_name="sticker.png")
        builder.commit()
        import_chat_db(source_path, archive_path)
        kind, extra = query(
            archive_path, "SELECT kind, extra FROM messages WHERE guid = ?", (builder.guid_of(sticker),)
        )[0]
        assert kind == "attachment"
        assert json.loads(extra)["associated_message_type"] == 1000
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 0

    def test_one_identifier_on_two_chat_rows_is_one_conversation(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan_sms = builder.handle(worlds.ROWAN_PHONE, service="SMS")
        rowan = builder.handle(worlds.ROWAN_PHONE)
        sms = builder.chat(worlds.ROWAN_PHONE, service="SMS", members=[rowan_sms])
        imessage = builder.chat(worlds.ROWAN_PHONE, service="iMessage", members=[rowan])
        builder.message(sms, when=utc(2020, 5, 1, 12), text="green bubble", handle=rowan_sms, service="SMS")
        builder.message(imessage, when=utc(2021, 5, 1, 12), text="blue bubble", handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert query(archive_path, "SELECT chat_identifier, service, message_count FROM conversations") == [
            (worlds.ROWAN_PHONE, "iMessage", 2)
        ]
        assert scalar(archive_path, "SELECT COUNT(*) FROM handles") == 1
        assert query(archive_path, "SELECT service FROM messages ORDER BY sent_ns") == [("SMS",), ("iMessage",)]

    def test_mime_type_can_be_missing(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        message = builder.message(chat, when=utc(2024, 1, 1, 12), handle=rowan)
        builder.attachment(
            message,
            mime_type=None,
            transfer_name=None,
            filename="~/Library/Messages/Attachments/aa/file.bin",
            total_bytes=None,
        )
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert query(archive_path, "SELECT mime_type, file_name, total_bytes FROM attachments") == [
            (None, "file.bin", None)
        ]
        assert query(archive_path, "SELECT kind FROM messages") == [("attachment",)]

    def test_big_blob_text_survives(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        builder.message(chat, when=utc(2024, 1, 1, 12), body=blob("ascii_70000"), handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert texts(archive_path) == [counter_text(70_000)]
