"""Repeat imports: the archive only grows, runs are recorded, a failed run changes nothing."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from archive import ImportBusyError, ImportFailed, SourceError, import_chat_db
from archive import importer as importer_module
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, apple_time, utc
from tests.archive.inspect_archive import (
    content,
    fts_is_consistent,
    query,
    runs,
    scalar,
    texts,
    totals_match_messages,
)
from tests.archive.typedstream_fixtures import blob

NO_CHANGE = {
    "messages_added": 0,
    "messages_updated": 0,
    "reactions_added": 0,
    "system_added": 0,
    "attachments_added": 0,
}


def changes(result: Any) -> dict[str, int]:
    return {name: getattr(result, name) for name in NO_CHANGE}


@pytest.fixture
def lake(builder: ChatDbBuilder) -> worlds.Lake:
    return worlds.build_lake(builder)


def add_many(builder: ChatDbBuilder, chat: int, handle: int, count: int) -> list[int]:
    """Add `count` received messages "extra 0", "extra 1", ... one minute apart, in June 2024."""
    start = utc(2024, 6, 1, 12, 0)
    made = [
        builder.message(chat, when=start + timedelta(minutes=index), text=f"extra {index}", handle=handle)
        for index in range(count)
    ]
    builder.commit()
    return made


class TestTheArchiveOnlyGrows:
    def test_second_import_adds_nothing(self, lake: worlds.Lake, source_path: Path, archive_path: Path) -> None:
        import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        before = content(archive_path)
        second = import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        assert second.mode == "incremental"
        assert changes(second) == NO_CHANGE
        assert content(archive_path) == before
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)

    def test_full_import_of_the_same_source_adds_nothing(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        again = import_chat_db(source_path, archive_path, full=True)
        assert again.mode == "full" and again.rows_seen == worlds.SOURCE_ROWS
        assert changes(again) == NO_CHANGE
        assert content(archive_path) == before

    def test_rows_deleted_from_the_source_stay_in_the_archive(
        self, lake: worlds.Lake, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        builder.raw.execute(
            "DELETE FROM chat_message_join WHERE message_id IN (SELECT ROWID FROM message WHERE ROWID % 2 = 0)"
        )
        builder.raw.execute("DELETE FROM message WHERE ROWID % 2 = 0")
        builder.raw.execute("DELETE FROM message_attachment_join")
        builder.raw.execute("DELETE FROM attachment")
        builder.commit()
        for full in (False, True):
            result = import_chat_db(source_path, archive_path, full=full)
            assert changes(result) == NO_CHANGE
            assert content(archive_path) == before

    def test_smaller_source_loses_nothing(
        self, lake: worlds.Lake, tmp_path: Path, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        small_path = tmp_path / "small" / "chat.db"
        small_path.parent.mkdir()
        small = ChatDbBuilder(small_path)
        rowan = small.handle(worlds.ROWAN_PHONE)
        chat = small.chat(worlds.ROWAN_PHONE, members=[rowan])
        small.message(
            chat,
            when=utc(2024, 1, 15, 18, 0),
            text="Are we still on for Saturday?",
            handle=rowan,
            guid=lake.guids["saturday"],
        )
        small.close()
        result = import_chat_db(small_path, archive_path)
        assert result.mode == "full"
        assert changes(result) == NO_CHANGE
        assert content(archive_path) == before

    def test_older_snapshot_adds_what_the_live_database_lost(
        self, lake: worlds.Lake, tmp_path: Path, source_path: Path, archive_path: Path
    ) -> None:
        old_path = tmp_path / "old" / "chat.db"
        old_path.parent.mkdir()
        old = ChatDbBuilder(old_path)
        rowan = old.handle(worlds.ROWAN_PHONE)
        chat = old.chat(worlds.ROWAN_PHONE, members=[rowan])
        old.message(chat, when=utc(2019, 7, 4, 12, 0), text="offloaded long ago", handle=rowan, guid="OLD-ONLY-1")
        old.message(
            chat,
            when=utc(2024, 1, 15, 18, 0),
            text="Are we still on for Saturday?",
            handle=rowan,
            guid=lake.guids["saturday"],
        )
        old.close()
        import_chat_db(source_path, archive_path)
        result = import_chat_db(old_path, archive_path)
        assert result.messages_added == 1
        assert texts(archive_path)[0] == "offloaded long ago"
        assert (
            scalar(
                archive_path, "SELECT message_count FROM conversations WHERE chat_identifier = ?", (worlds.ROWAN_PHONE,)
            )
            == 6
        )
        assert totals_match_messages(archive_path)

    def test_lost_text_in_the_source_does_not_blank_the_archive(
        self, lake: worlds.Lake, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        builder.raw.execute("UPDATE message SET attributedBody = NULL WHERE guid = ?", (lake.guids["view"],))
        builder.raw.execute("UPDATE message SET text = NULL WHERE guid = ?", (lake.guids["saturday"],))
        builder.commit()
        result = import_chat_db(source_path, archive_path, full=True)
        assert result.messages_updated == 0
        assert content(archive_path) == before


class TestIncremental:
    def test_new_rows_are_added(
        self, lake: worlds.Lake, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        add_many(builder, lake.chats["phone"], lake.handles["rowan_phone"], 3)
        builder.tapback(
            lake.chats["phone"], lake.guids["long"], 2003, when=utc(2024, 6, 2, 9), handle=lake.handles["rowan_phone"]
        )
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert result.mode == "incremental"
        assert (result.messages_added, result.reactions_added, result.messages_updated) == (3, 1, 0)
        assert texts(archive_path)[-3:] == ["extra 0", "extra 1", "extra 2"]
        assert (
            scalar(
                archive_path, "SELECT message_count FROM conversations WHERE chat_identifier = ?", (worlds.ROWAN_PHONE,)
            )
            == 8
        )
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)

    def test_only_the_tail_of_the_source_is_read(
        self,
        lake: worlds.Lake,
        builder: ChatDbBuilder,
        source_path: Path,
        archive_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(importer_module, "OVERLAP_ROWS", 2)
        builder.raw.execute("UPDATE message SET date_edited = 0, date_retracted = 0")
        builder.commit()
        import_chat_db(source_path, archive_path)
        add_many(builder, lake.chats["phone"], lake.handles["rowan_phone"], 5)
        result = import_chat_db(source_path, archive_path)
        assert result.rows_seen == 2 + 5
        assert result.messages_added == 5

    def test_edit_of_an_old_message_is_picked_up(
        self,
        lake: worlds.Lake,
        builder: ChatDbBuilder,
        source_path: Path,
        archive_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(importer_module, "OVERLAP_ROWS", 1)
        import_chat_db(source_path, archive_path)
        edited_at = utc(2024, 1, 15, 18, 9)
        builder.raw.execute(
            "UPDATE message SET text = 'Are we still on for Sunday?', date_edited = ? WHERE guid = ?",
            (apple_time(edited_at), lake.guids["saturday"]),
        )
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert result.mode == "incremental" and result.messages_updated == 1 and result.messages_added == 0
        row = query(archive_path, "SELECT text, edited_at FROM messages WHERE guid = ?", (lake.guids["saturday"],))[0]
        assert row == ("Are we still on for Sunday?", int(edited_at.timestamp()))
        assert fts_is_consistent(archive_path)
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH '\"sunday\"'") == 1
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH '\"saturday\"'") == 0

    def test_unsend_of_an_old_message_is_picked_up(
        self,
        lake: worlds.Lake,
        builder: ChatDbBuilder,
        source_path: Path,
        archive_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(importer_module, "OVERLAP_ROWS", 1)
        import_chat_db(source_path, archive_path)
        builder.raw.execute(
            "UPDATE message SET text = NULL, attributedBody = NULL, date_retracted = ? WHERE guid = ?",
            (apple_time(utc(2024, 2, 1, 20, 6)), lake.guids["gotit"]),
        )
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert result.messages_updated == 1
        assert query(
            archive_path, "SELECT text, unsent, kind FROM messages WHERE guid = ?", (lake.guids["gotit"],)
        ) == [("", 1, "text")]
        assert (
            scalar(
                archive_path, "SELECT message_count FROM conversations WHERE chat_identifier = ?", (worlds.ROWAN_EMAIL,)
            )
            == 2
        )

    def test_attachment_that_arrives_late_is_picked_up(
        self, lake: worlds.Lake, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        builder.attachment(lake.rowids["jackets"], mime_type="application/pdf", transfer_name="packing-list.pdf")
        builder.commit()
        result = import_chat_db(source_path, archive_path)
        assert result.mode == "incremental" and result.attachments_added == 1
        assert query(
            archive_path,
            "SELECT a.mime_type FROM attachments a JOIN messages m ON m.id = a.message_id WHERE m.guid = ?",
            (lake.guids["jackets"],),
        ) == [("application/pdf",)]

    def test_rebuilt_database_with_new_rowids_makes_no_duplicates(
        self, lake: worlds.Lake, tmp_path: Path, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        rebuilt_path = tmp_path / "rebuilt" / "chat.db"
        rebuilt_path.parent.mkdir()
        rebuilt = ChatDbBuilder(rebuilt_path)
        rowan = rebuilt.handle(worlds.ROWAN_PHONE)
        chat = rebuilt.chat(worlds.ROWAN_PHONE, members=[rowan])
        # same guids, other ROWIDs and another order, plus one new message
        rebuilt.message(
            chat,
            when=utc(2024, 6, 9, 12, 0),
            text="written after the rebuild",
            handle=rowan,
            guid="NEW-AFTER-REBUILD",
            rowid=500,
        )
        rebuilt.message(
            chat, when=utc(2024, 1, 16, 2, 0), body=blob("ascii_300"), handle=rowan, guid=lake.guids["long"], rowid=900
        )
        rebuilt.message(
            chat,
            when=utc(2024, 1, 15, 18, 0),
            text="Are we still on for Saturday?",
            handle=rowan,
            guid=lake.guids["saturday"],
            rowid=901,
        )
        rebuilt.close()
        result = import_chat_db(rebuilt_path, archive_path)
        assert result.mode == "full"
        assert (result.messages_added, result.messages_updated) == (1, 0)
        after = content(archive_path)
        assert len(after["messages"]) == len(before["messages"]) + 1
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE guid = ?", (lake.guids["saturday"],)) == 1
        assert (
            scalar(archive_path, "SELECT source_rowid FROM messages WHERE guid = ?", (lake.guids["saturday"],)) == 901
        )

    def test_older_snapshot_does_not_undo_an_edit_or_an_unsend(
        self, lake: worlds.Lake, tmp_path: Path, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        old_path = tmp_path / "old" / "chat.db"
        old_path.parent.mkdir()
        old = ChatDbBuilder(old_path)
        mira = old.handle(worlds.MIRA_PHONE)
        tessa = old.handle(worlds.TESSA_EMAIL)
        group = old.chat(worlds.GROUP_ID, members=[mira, tessa])
        old.message(
            group, when=utc(2024, 3, 1, 17, 10), text="Bring life jakets", handle=mira, guid=lake.guids["jackets"]
        )
        old.message(
            group,
            when=utc(2024, 3, 1, 17, 15),
            text="text before it was unsent",
            handle=tessa,
            guid=lake.guids["unsent"],
        )
        old.close()
        result = import_chat_db(old_path, archive_path)
        assert result.messages_updated == 0
        assert content(archive_path) == before

    def test_full_flag_reads_everything_again(self, lake: worlds.Lake, source_path: Path, archive_path: Path) -> None:
        import_chat_db(source_path, archive_path)
        assert import_chat_db(source_path, archive_path, full=True).rows_seen == worlds.SOURCE_ROWS
        assert import_chat_db(source_path, archive_path).rows_seen < worlds.SOURCE_ROWS + 1

    def test_first_and_last_import_are_tracked(
        self, lake: worlds.Lake, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        first = import_chat_db(source_path, archive_path)
        add_many(builder, lake.chats["phone"], lake.handles["rowan_phone"], 1)
        second = import_chat_db(source_path, archive_path, full=True)
        rows = dict(
            (text, (a, b))
            for text, a, b in query(archive_path, "SELECT text, first_import_id, last_import_id FROM messages")
        )
        assert rows["Got it"] == (first.run_id, second.run_id)
        assert rows["extra 0"] == (second.run_id, second.run_id)


class TestRunRecords:
    def test_each_run_is_recorded(self, lake: worlds.Lake, source_path: Path, archive_path: Path) -> None:
        first = import_chat_db(source_path, archive_path)
        second = import_chat_db(source_path, archive_path)
        assert runs(archive_path) == [(first.run_id, "full", "ok", None), (second.run_id, "incremental", "ok", None)]
        started, finished, source, counts, checkpoints = query(
            archive_path,
            "SELECT started_at, finished_at, source_path, counts, checkpoints FROM import_runs WHERE id = ?",
            (first.run_id,),
        )[0]
        assert finished >= started > 1_700_000_000
        assert source == str(source_path)
        assert json.loads(counts)["messages_added"] == worlds.MESSAGE_ROWS
        assert json.loads(checkpoints)[0] == [lake.rowids["added"], lake.guids["added"]]
        assert (first.started_at, first.finished_at) == (started, finished)

    def test_failed_run_changes_nothing(
        self,
        lake: worlds.Lake,
        builder: ChatDbBuilder,
        source_path: Path,
        archive_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        before = content(archive_path)
        add_many(builder, lake.chats["phone"], lake.handles["rowan_phone"], 40)
        builder.chat("+15550100999", members=[builder.handle("+15550100999")])
        builder.commit()

        monkeypatch.setattr(importer_module, "BATCH_ROWS", 5)
        real_classify = importer_module.classify
        calls = {"count": 0}

        def failing_classify(row: Any, *, has_attachments: bool) -> Any:
            calls["count"] += 1
            if calls["count"] > 23:
                raise RuntimeError("disk on fire")
            return real_classify(row, has_attachments=has_attachments)

        monkeypatch.setattr(importer_module, "classify", failing_classify)
        with pytest.raises(ImportFailed, match="disk on fire") as caught:
            import_chat_db(source_path, archive_path, contacts=[])
        assert content(archive_path) == before
        assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)
        recorded = runs(archive_path)
        assert [row[2] for row in recorded] == ["ok", "failed"]
        assert "disk on fire" in recorded[-1][3]
        assert caught.value.run_id == recorded[-1][0]
        assert isinstance(caught.value.__cause__, RuntimeError)

        monkeypatch.setattr(importer_module, "classify", real_classify)
        healed = import_chat_db(source_path, archive_path, contacts=[worlds.ROWAN_CONTACT])
        assert healed.messages_added == 40
        assert [row[2] for row in runs(archive_path)] == ["ok", "failed", "ok"]

    def test_failure_on_the_first_import_leaves_an_empty_archive(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("first run failed")

        monkeypatch.setattr(importer_module, "classify", boom)
        with pytest.raises(ImportFailed):
            import_chat_db(source_path, archive_path)
        assert all(not rows for rows in content(archive_path).values())
        assert [row[1:3] for row in runs(archive_path)] == [("full", "failed")]

    def test_source_error_is_recorded_and_raised_as_itself(
        self, lake: worlds.Lake, tmp_path: Path, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        with pytest.raises(SourceError):
            import_chat_db(tmp_path / "gone.db", archive_path)
        assert content(archive_path) == before
        assert [row[2] for row in runs(archive_path)] == ["ok", "failed"]

    def test_cancel_leaves_the_archive_as_it_was(
        self,
        lake: worlds.Lake,
        builder: ChatDbBuilder,
        source_path: Path,
        archive_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import_chat_db(source_path, archive_path)
        before = content(archive_path)
        add_many(builder, lake.chats["phone"], lake.handles["rowan_phone"], 30)
        monkeypatch.setattr(importer_module, "BATCH_ROWS", 5)
        seen: list[dict[str, Any]] = []

        def cancel() -> bool:
            return sum(1 for event in seen if event["phase"] == "messages") >= 3

        result = import_chat_db(source_path, archive_path, progress=seen.append, cancel=cancel)
        assert result.status == "cancelled"
        assert changes(result) == NO_CHANGE
        assert content(archive_path) == before
        assert [row[2] for row in runs(archive_path)] == ["ok", "cancelled"]
        assert import_chat_db(source_path, archive_path).messages_added == 30

    def test_cancel_with_an_event_before_the_copy(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        stop = threading.Event()
        stop.set()
        result = import_chat_db(source_path, archive_path, cancel=stop)
        assert result.status == "cancelled"
        assert all(not rows for rows in content(archive_path).values())
        assert not (archive_path.parent / importer_module.WORK_DIR_NAME).exists()

    def test_progress_reports_each_phase(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(importer_module, "BATCH_ROWS", 4)
        events: list[dict[str, Any]] = []
        import_chat_db(source_path, archive_path, progress=events.append)
        phases = [event["phase"] for event in events]
        assert [phase for index, phase in enumerate(phases) if index == 0 or phases[index - 1] != phase] == [
            "copy",
            "messages",
            "finish",
        ]
        for event in events:
            assert set(event) == {"phase", "done", "total"}
            assert 0 <= event["done"] <= event["total"]
        message_events = [event for event in events if event["phase"] == "messages"]
        assert message_events[-1] == {"phase": "messages", "done": worlds.SOURCE_ROWS, "total": worlds.SOURCE_ROWS}
        assert [event["done"] for event in message_events] == sorted(event["done"] for event in message_events)
        json.dumps(events)

    def test_a_progress_callback_that_raises_fails_the_run(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        def progress(event: dict[str, Any]) -> None:
            raise ValueError("ui went away")

        with pytest.raises(ImportFailed, match="ui went away"):
            import_chat_db(source_path, archive_path, progress=progress)
        assert all(not rows for rows in content(archive_path).values())

    def test_second_import_at_the_same_time_is_refused(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import_chat_db(source_path, archive_path)
        monkeypatch.setattr(importer_module, "LOCK_WAIT_SECONDS", 0.05)
        other = sqlite3.connect(str(archive_path), isolation_level=None)
        other.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(ImportBusyError):
                import_chat_db(source_path, archive_path)
        finally:
            other.execute("ROLLBACK")
            other.close()
        assert [row[2] for row in runs(archive_path)] == ["ok"]
        assert import_chat_db(source_path, archive_path).status == "ok"

    def test_readers_see_the_old_state_until_the_run_commits(
        self, lake: worlds.Lake, source_path: Path, archive_path: Path
    ) -> None:
        import_chat_db(source_path, archive_path)
        reader = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
        seen_during: list[int] = []

        def progress(event: dict[str, Any]) -> None:
            if event["phase"] == "finish":
                seen_during.append(reader.execute("SELECT COUNT(*) FROM import_runs").fetchone()[0])

        import_chat_db(source_path, archive_path, full=True, progress=progress)
        assert seen_during and set(seen_during) == {1}
        assert reader.execute("SELECT COUNT(*) FROM import_runs").fetchone()[0] == 2
        reader.close()
