"""A full pull makes the archive equal to the source. The source is the truth."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from archive import RemovalLimitError, import_chat_db
from archive import importer as importer_module
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.inspect_archive import content, fts_is_consistent, query, runs, scalar, totals_match_messages

START = utc(2024, 5, 1, 9, 0)
PAT = "+15550100111"


def chat_with_notes(builder: ChatDbBuilder, count: int, identifier: str = PAT) -> tuple[int, int, list[int]]:
    """One conversation with `count` received messages "note 0", "note 1", ... one minute apart."""
    handle = builder.handle(identifier)
    chat = builder.chat(identifier, members=[handle])
    rowids = [
        builder.message(chat, when=START + timedelta(minutes=index), text=f"note {index}", handle=handle)
        for index in range(count)
    ]
    builder.commit()
    return chat, handle, rowids


def drop_message(builder: ChatDbBuilder, rowid: int) -> None:
    builder.raw.execute("DELETE FROM chat_message_join WHERE message_id = ?", (rowid,))
    builder.raw.execute("DELETE FROM message WHERE ROWID = ?", (rowid,))
    builder.commit()


def ids_by_guid(archive_path: Path) -> dict[str, int]:
    return dict(query(archive_path, "SELECT guid, id FROM messages"))


def stored_texts(archive_path: Path) -> list[str]:
    return [row[0] for row in query(archive_path, "SELECT text FROM messages ORDER BY sent_ns, id")]


class TestFullPull:
    def test_full_pull_is_the_default(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        chat_with_notes(builder, 3)
        assert import_chat_db(source_path, archive_path).mode == "full"
        assert import_chat_db(source_path, archive_path).mode == "full"
        assert import_chat_db(source_path, archive_path, full=False).mode == "incremental"

    def test_a_pull_with_no_change_writes_nothing(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat_with_notes(builder, 5)
        import_chat_db(source_path, archive_path)
        before = query(archive_path, "SELECT id, guid, last_import_id FROM messages ORDER BY id")

        again = import_chat_db(source_path, archive_path)

        assert (again.messages_added, again.messages_updated, again.messages_removed) == (0, 0, 0)
        assert query(archive_path, "SELECT id, guid, last_import_id FROM messages ORDER BY id") == before

    def test_rows_gone_from_the_source_are_removed_with_reactions_and_attachments(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 12)
        loved, pictured, kept = rowids[0], rowids[1], rowids[2]
        builder.tapback(chat, builder.guid_of(loved), 2000, when=START + timedelta(hours=1), handle=handle)
        builder.tapback(chat, builder.guid_of(kept), 2001, when=START + timedelta(hours=2), handle=handle)
        builder.attachment(pictured)
        builder.commit()
        gone_guids = {builder.guid_of(loved), builder.guid_of(pictured)}
        first = import_chat_db(source_path, archive_path)
        assert (first.messages_added, first.reactions_added, first.attachments_added) == (12, 2, 1)
        ids_before = ids_by_guid(archive_path)

        # the tapback row on the first message stays in the source: it must go with its target
        drop_message(builder, loved)
        drop_message(builder, pictured)
        second = import_chat_db(source_path, archive_path)

        assert (second.messages_removed, second.reactions_removed, second.attachments_removed) == (2, 1, 1)
        assert second.messages_added == 0
        ids_after = ids_by_guid(archive_path)
        assert set(ids_after) == set(ids_before) - gone_guids
        assert all(ids_before[guid] == message_id for guid, message_id in ids_after.items())
        assert scalar(archive_path, "SELECT COUNT(*) FROM attachments") == 0
        assert [row[0] for row in query(archive_path, "SELECT target_guid FROM reactions")] == [builder.guid_of(kept)]
        assert "note 0" not in stored_texts(archive_path)
        assert fts_is_consistent(archive_path)
        assert totals_match_messages(archive_path)
        assert scalar(archive_path, "SELECT message_count FROM conversations") == 10

    def test_a_third_pull_changes_nothing_and_does_not_add_the_orphan_tapback_again(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 12)
        builder.tapback(chat, builder.guid_of(rowids[0]), 2000, when=START + timedelta(hours=1), handle=handle)
        builder.commit()
        import_chat_db(source_path, archive_path)
        drop_message(builder, rowids[0])
        import_chat_db(source_path, archive_path)
        before = content(archive_path)

        third = import_chat_db(source_path, archive_path)

        assert (third.messages_removed, third.reactions_removed, third.reactions_added) == (0, 0, 0)
        assert third.skipped == {"no_target": 1}
        assert content(archive_path) == before

    def test_an_incremental_run_removes_nothing(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        _, _, rowids = chat_with_notes(builder, 12)
        import_chat_db(source_path, archive_path)
        drop_message(builder, rowids[3])

        result = import_chat_db(source_path, archive_path, full=False)

        assert result.mode == "incremental"
        assert result.messages_removed == 0
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 12

    def test_the_source_wins_when_a_row_differs(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        _, _, rowids = chat_with_notes(builder, 3)
        edited_at = START + timedelta(hours=3)
        builder.raw.execute(
            "UPDATE message SET text = 'note 1, edited', date_edited = ? WHERE ROWID = ?",
            (int((edited_at.timestamp() - 978307200) * 1_000_000_000), rowids[1]),
        )
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert "note 1, edited" in stored_texts(archive_path)

        # a source that shows the message before its edit: the archive follows it
        builder.raw.execute("UPDATE message SET text = 'note 1', date_edited = 0 WHERE ROWID = ?", (rowids[1],))
        builder.commit()
        result = import_chat_db(source_path, archive_path)

        assert result.messages_updated == 1
        assert stored_texts(archive_path) == ["note 0", "note 1", "note 2"]
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE edited_at IS NOT NULL") == 0
        assert fts_is_consistent(archive_path)

    def test_a_row_keeps_its_id_when_the_source_gives_it_a_new_rowid(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path, tmp_path: Path
    ) -> None:
        chat_with_notes(builder, 4)
        guids = [builder.guid_of(rowid) for rowid in (1, 2, 3, 4)]
        import_chat_db(source_path, archive_path)
        before = ids_by_guid(archive_path)

        rebuilt_path = tmp_path / "rebuilt" / "chat.db"
        rebuilt_path.parent.mkdir()
        rebuilt = ChatDbBuilder(rebuilt_path)
        handle = rebuilt.handle(PAT)
        chat = rebuilt.chat(PAT, members=[handle])
        for index, guid in enumerate(guids):
            rebuilt.message(
                chat,
                when=START + timedelta(minutes=index),
                text=f"note {index}",
                handle=handle,
                guid=guid,
                rowid=500 + index,
            )
        rebuilt.close()
        result = import_chat_db(rebuilt_path, archive_path, full=False)

        assert result.mode == "full"  # no row of the old database is at its old ROWID
        assert (result.messages_added, result.messages_removed, result.messages_updated) == (0, 0, 4)
        assert ids_by_guid(archive_path) == before
        assert [row[0] for row in query(archive_path, "SELECT source_rowid FROM messages ORDER BY id")] == [
            500,
            501,
            502,
            503,
        ]

    def test_an_attachment_that_the_source_dropped_is_removed(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        _, _, rowids = chat_with_notes(builder, 3)
        first_file = builder.attachment(rowids[0], transfer_name="one.jpeg")
        builder.attachment(rowids[0], transfer_name="two.jpeg")
        builder.commit()
        import_chat_db(source_path, archive_path)
        builder.raw.execute("DELETE FROM message_attachment_join WHERE attachment_id = ?", (first_file,))
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert result.attachments_removed == 1
        assert [row[0] for row in query(archive_path, "SELECT file_name FROM attachments")] == ["two.jpeg"]
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 3


class TestRowsThatChangeTheirKind:
    def test_a_message_that_becomes_a_tapback_is_stored_one_time(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        _, _, rowids = chat_with_notes(builder, 6)
        target, flipped = builder.guid_of(rowids[0]), builder.guid_of(rowids[1])
        import_chat_db(source_path, archive_path)

        builder.raw.execute(
            "UPDATE message SET associated_message_type = 2000, associated_message_guid = ? WHERE ROWID = ?",
            (f"p:0/{target}", rowids[1]),
        )
        builder.commit()
        result = import_chat_db(source_path, archive_path)

        assert (result.messages_removed, result.reactions_added) == (1, 1)
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE guid = ?", (flipped,)) == 0
        assert query(archive_path, "SELECT guid, target_guid, kind FROM reactions") == [(flipped, target, "love")]
        assert scalar(archive_path, "SELECT message_count FROM conversations") == 5
        assert fts_is_consistent(archive_path) and totals_match_messages(archive_path)
        again = import_chat_db(source_path, archive_path)
        assert (again.messages_removed, again.reactions_added, again.reactions_updated) == (0, 0, 0)

    def test_a_tapback_that_becomes_a_message_is_stored_one_time(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 6)
        tapback_row = builder.tapback(
            chat, builder.guid_of(rowids[0]), 2001, when=START + timedelta(hours=1), handle=handle
        )
        builder.commit()
        flipped = builder.guid_of(tapback_row)
        import_chat_db(source_path, archive_path)
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 1

        builder.raw.execute(
            "UPDATE message SET associated_message_type = 0, associated_message_guid = NULL, text = 'now a text'"
            " WHERE ROWID = ?",
            (tapback_row,),
        )
        builder.commit()
        result = import_chat_db(source_path, archive_path)

        assert (result.reactions_removed, result.messages_added) == (1, 1)
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 0
        assert scalar(archive_path, "SELECT text FROM messages WHERE guid = ?", (flipped,)) == "now a text"
        assert totals_match_messages(archive_path)

    def test_a_new_tapback_on_a_message_that_is_gone_is_never_stored(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 12)
        gone = builder.guid_of(rowids[0])
        import_chat_db(source_path, archive_path)

        drop_message(builder, rowids[0])
        builder.tapback(chat, gone, 2000, when=START + timedelta(hours=1), handle=handle)
        builder.commit()
        first = import_chat_db(source_path, archive_path)
        second = import_chat_db(source_path, archive_path)

        for result in (first, second):  # the same source row is counted the same way on every pull
            assert (result.reactions_added, result.reactions_removed) == (0, 0)
            assert result.skipped == {"no_target": 1}
        assert first.messages_removed == 1
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 0

    def test_group_events_and_tapbacks_have_their_own_update_counters(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 3)
        event = builder.message(
            chat, when=START + timedelta(hours=1), item_type=2, group_title="Old name", handle=handle
        )
        tapback_row = builder.tapback(
            chat, builder.guid_of(rowids[0]), 2000, when=START + timedelta(hours=2), handle=handle
        )
        builder.commit()
        import_chat_db(source_path, archive_path)

        builder.raw.execute("UPDATE message SET group_title = 'New name' WHERE ROWID = ?", (event,))
        builder.raw.execute("UPDATE message SET associated_message_type = 2003 WHERE ROWID = ?", (tapback_row,))
        builder.raw.execute("UPDATE message SET text = 'note 2, changed' WHERE ROWID = ?", (rowids[2],))
        builder.commit()
        result = import_chat_db(source_path, archive_path)

        assert (result.messages_updated, result.system_updated, result.reactions_updated) == (1, 1, 1)
        assert query(archive_path, "SELECT kind FROM reactions") == [("laugh",)]


class TestMarksAndKindChangesOfAnIncrementalRun:
    def test_a_mark_from_before_the_last_full_pull_is_not_trusted(
        self,
        builder: ChatDbBuilder,
        source_path: Path,
        archive_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_with_notes(builder, 12)
        guids = [builder.guid_of(rowid) for rowid in range(1, 13)]
        import_chat_db(source_path, archive_path)
        # an older state of the same database: the same rows, without the last four
        older_path = tmp_path / "older" / "chat.db"
        older_path.parent.mkdir()
        older = ChatDbBuilder(older_path)
        handle = older.handle(PAT)
        chat = older.chat(PAT, members=[handle])
        for index, guid in enumerate(guids[:8]):
            older.message(
                chat,
                when=START + timedelta(minutes=index),
                text=f"note {index}",
                handle=handle,
                guid=guid,
                rowid=index + 1,
            )
        older.close()
        assert import_chat_db(older_path, archive_path, max_removed_share=1).messages_removed == 4
        monkeypatch.setattr(importer_module, "OVERLAP_ROWS", 2)

        result = import_chat_db(source_path, archive_path, full=False)

        assert result.mode == "incremental"
        assert stored_texts(archive_path) == [f"note {index}" for index in range(12)]  # rows 9 to 12 are back

    def test_a_message_that_becomes_a_tapback_leaves_no_tapback_without_a_target(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 6)
        target, flipped = builder.guid_of(rowids[0]), builder.guid_of(rowids[5])
        builder.tapback(chat, flipped, 2000, when=START + timedelta(hours=1), handle=handle)
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 1

        builder.raw.execute(
            "UPDATE message SET associated_message_type = 2001, associated_message_guid = ? WHERE ROWID = ?",
            (f"p:0/{target}", rowids[5]),
        )
        builder.commit()
        result = import_chat_db(source_path, archive_path, full=False)

        assert result.mode == "incremental"
        assert (result.messages_removed, result.reactions_removed, result.reactions_added) == (1, 1, 1)
        assert query(archive_path, "SELECT guid, target_guid FROM reactions") == [(flipped, target)]
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)


class TestSafetyValve:
    def test_a_pull_that_would_remove_too_much_stops_and_changes_nothing(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        _, _, rowids = chat_with_notes(builder, 10)
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        for rowid in rowids[:3]:
            drop_message(builder, rowid)
        builder.message(1, when=START + timedelta(days=1), text="a new note", handle=1)
        builder.commit()

        with pytest.raises(RemovalLimitError) as caught:
            import_chat_db(source_path, archive_path)

        assert (caught.value.would_remove, caught.value.stored, caught.value.max_share) == (3, 10, 0.2)
        assert content(archive_path) == before
        assert fts_is_consistent(archive_path)
        last_run = runs(archive_path)[-1]
        assert last_run[2] == "failed"
        assert "RemovalLimitError" in last_run[3]

    def test_the_caller_can_allow_the_removal(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        _, _, rowids = chat_with_notes(builder, 10)
        import_chat_db(source_path, archive_path)
        for rowid in rowids[:3]:
            drop_message(builder, rowid)

        result = import_chat_db(source_path, archive_path, max_removed_share=0.5)

        assert result.messages_removed == 3
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 7

    def test_a_share_at_the_limit_passes(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        _, _, rowids = chat_with_notes(builder, 10)
        import_chat_db(source_path, archive_path)
        drop_message(builder, rowids[0])
        drop_message(builder, rowids[1])

        assert import_chat_db(source_path, archive_path).messages_removed == 2  # 2 of 10 is not more than 20 %

    def test_a_pull_into_an_empty_archive_never_trips(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat_with_notes(builder, 2)
        result = import_chat_db(source_path, archive_path, max_removed_share=0)
        assert result.messages_added == 2
        assert import_chat_db(source_path, archive_path, max_removed_share=0).status == "ok"

    @pytest.mark.parametrize("share", [-0.1, 1.5, float("nan")])
    def test_a_share_outside_zero_to_one_is_refused(self, share: float, source_path: Path, archive_path: Path) -> None:
        with pytest.raises(ValueError):
            import_chat_db(source_path, archive_path, max_removed_share=share)
        assert not archive_path.exists()

    @pytest.mark.parametrize("share", ["0.2", None, True])
    def test_a_share_that_is_not_a_number_is_refused(self, share: Any, source_path: Path, archive_path: Path) -> None:
        with pytest.raises(TypeError):
            import_chat_db(source_path, archive_path, max_removed_share=share)


class TestTapbackTargets:
    def test_a_tapback_read_before_its_target_is_stored(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handle = builder.handle(PAT)
        chat = builder.chat(PAT, members=[handle])
        builder.tapback(chat, "LATE-TARGET", 2003, when=START + timedelta(minutes=5), handle=handle)
        builder.message(chat, when=START, text="the joke", handle=handle, guid="LATE-TARGET")
        builder.commit()
        monkeypatch.setattr(importer_module, "BATCH_ROWS", 1)

        result = import_chat_db(source_path, archive_path)

        assert (result.reactions_seen, result.reactions_added) == (1, 1)
        assert result.skipped == {}
        assert query(archive_path, "SELECT target_guid, kind FROM reactions") == [("LATE-TARGET", "laugh")]

    def test_a_tapback_on_a_message_the_archive_does_not_hold_is_skipped(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, _ = chat_with_notes(builder, 1)
        builder.tapback(chat, "NOT-IN-THE-SOURCE", 2000, when=START + timedelta(minutes=5), handle=handle)
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert result.skipped == {"no_target": 1}
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 0

    def test_a_tapback_code_of_a_later_version_is_not_stored_as_a_message(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat, handle, rowids = chat_with_notes(builder, 1)
        builder.tapback(
            chat,
            builder.guid_of(rowids[0]),
            2008,
            when=START + timedelta(minutes=5),
            handle=handle,
            text="Did something new to an invented message",
        )
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert result.skipped == {"unknown_tapback": 1}
        assert stored_texts(archive_path) == ["note 0"]
        assert scalar(archive_path, "SELECT COUNT(*) FROM reactions") == 0


class TestDirectoryFollowsTheSource:
    def test_a_member_that_left_is_removed_by_a_full_pull(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        pat, sam = builder.handle(PAT), builder.handle("+15550100122")
        group = builder.chat("chat900100", display_name="Garden club", members=[pat, sam])
        builder.message(group, when=START, text="first", handle=pat)
        builder.message(group, when=START + timedelta(minutes=1), text="second", handle=sam)
        builder.commit()
        import_chat_db(source_path, archive_path)
        builder.raw.execute("DELETE FROM chat_handle_join WHERE handle_id = ?", (sam,))
        builder.commit()

        import_chat_db(source_path, archive_path, full=False)
        assert len(content(archive_path)["members"]) == 2
        import_chat_db(source_path, archive_path)

        assert content(archive_path)["members"] == [("chat900100", PAT)]
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 2  # the messages of that member stay

    def test_a_conversation_gone_from_the_source_is_removed_with_its_handle_and_person(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        chat_with_notes(builder, 10)
        other = builder.handle("gone@example.com")
        small = builder.chat("gone@example.com", members=[other])
        lone = builder.message(small, when=START, text="only one", handle=other)
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert scalar(archive_path, "SELECT COUNT(*) FROM conversations") == 2

        drop_message(builder, lone)
        builder.raw.execute("DELETE FROM chat_handle_join WHERE chat_id = ?", (small,))
        builder.raw.execute("DELETE FROM chat WHERE ROWID = ?", (small,))
        builder.raw.execute("DELETE FROM handle WHERE ROWID = ?", (other,))
        builder.commit()
        import_chat_db(source_path, archive_path)

        found = content(archive_path)
        assert [row[0] for row in found["conversations"]] == [PAT]
        assert [row[0] for row in found["handles"]] == [PAT]
        assert len(found["people"]) == 1
        assert totals_match_messages(archive_path)

    def test_a_title_that_the_source_cleared_is_cleared(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        pat = builder.handle(PAT)
        group = builder.chat("chat900200", display_name="Book club", members=[pat])
        builder.message(group, when=START, text="hello", handle=pat)
        builder.commit()
        import_chat_db(source_path, archive_path)
        assert scalar(archive_path, "SELECT title FROM conversations") == "Book club"

        builder.raw.execute("UPDATE chat SET display_name = NULL WHERE ROWID = ?", (group,))
        builder.commit()
        import_chat_db(source_path, archive_path)

        assert scalar(archive_path, "SELECT title FROM conversations") == ""
