"""The bridge loads one legacy CSV export into an empty archive. All rows are invented."""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

import pytest

from archive import (
    INDEXED_TEXT_CHARS,
    Archive,
    ArchiveNotEmptyError,
    SourceError,
    SourceFormatError,
    import_chat_db,
    import_legacy_csv,
)
from archive.tapbacks import is_reaction_phrase
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.inspect_archive import content, fts_is_consistent, query, runs, scalar, totals_match_messages

COLUMNS = (
    "message_id",
    "timestamp",
    "sender",
    "text",
    "chat_id",
    "chat_display_name",
    "is_from_me",
    "service",
    "attachment_types",
    "attachment_files",
)
ROWAN = "+15550100131"
QUINN = "quinn@example.com"
GROUP = "chat700100"
PLACEHOLDER = chr(0xFFFC)
CURLY_OPEN, CURLY_CLOSE = chr(0x201C), chr(0x201D)


def line(
    number: int, stamp: str, text: str, chat: str = ROWAN, *, sender: str | None = None, **more: Any
) -> dict[str, Any]:
    """One row of the export. A received row names its sender; a sent row names the other person, as chat.db does."""
    from_me = bool(more.pop("from_me", False))
    row = {
        "message_id": number,
        "timestamp": stamp,
        "sender": sender if sender is not None else (chat if not chat.startswith("chat") else ""),
        "text": text,
        "chat_id": chat,
        "is_from_me": int(from_me),
        "service": "iMessage",
    }
    return {**row, **more}


def write_export(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] = COLUMNS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})
    return path


BASIC = [
    line(11, "2024-03-01 09:00:00", "Are we still on for the ferry?"),
    line(12, "2024-03-01 09:02:10", "  Yes, 10:15 from pier two  ", from_me=True),
    line(13, "2024-03-01 09:03:00", "Bring the blue thermos,\nand the map", sender=ROWAN),
    line(21, "2024-03-02 18:30:00", "Dinner on Friday?", GROUP, sender=ROWAN, chat_display_name="Harbor crew"),
    line(22, "2024-03-02 18:31:00", "I can bring bread", GROUP, sender=QUINN, chat_display_name="Harbor crew"),
    line(23, "2024-03-02 18:35:00", "Perfect", GROUP, from_me=True, chat_display_name="Harbor crew"),
]


@pytest.fixture
def export_path(tmp_path: Path) -> Path:
    return tmp_path / "export" / "messages.csv"


@pytest.fixture
def basic_export(export_path: Path) -> Path:
    export_path.parent.mkdir()
    return write_export(export_path, BASIC)


class TestLoad:
    def test_rows_become_messages_with_legacy_ids(self, basic_export: Path, archive_path: Path) -> None:
        result = import_legacy_csv(basic_export, archive_path)

        assert (result.status, result.mode) == ("ok", "legacy_csv")
        assert (result.rows_seen, result.messages_added, result.conversations_added) == (6, 6, 2)
        assert result.skipped == {}
        with Archive.open(archive_path) as archive:
            assert archive.summary()["messages"] == 6
            page = archive.messages(ROWAN)["messages"]
            assert [message["id"] for message in page] == ["legacy:11", "legacy:12", "legacy:13"]
            assert [message["timestamp"] for message in page] == [row["timestamp"] for row in BASIC[:3]]
            assert [message["is_from_me"] for message in page] == [False, True, False]
            assert [message["sender"] for message in page] == [ROWAN, "", ROWAN]
            assert page[1]["text"] == "  Yes, 10:15 from pier two  "  # nothing is trimmed
            assert page[2]["text"] == "Bring the blue thermos,\nand the map"
        assert fts_is_consistent(archive_path)
        assert totals_match_messages(archive_path)
        assert runs(archive_path) == [(1, "legacy_csv", "ok", None)]

    def test_conversations_and_members_come_from_the_rows(self, basic_export: Path, archive_path: Path) -> None:
        import_legacy_csv(basic_export, archive_path)

        with Archive.open(archive_path) as archive:
            group = archive.conversation(GROUP)
            direct = archive.conversation(ROWAN)
        assert group is not None and direct is not None
        assert (group["is_group"], group["name"], group["members"]) == (True, "Harbor crew", sorted([ROWAN, QUINN]))
        assert (direct["is_group"], direct["members"], direct["message_count"]) == (False, [ROWAN], 3)
        # the handle of a sent row names the other person: it is not the sender
        assert query(
            archive_path, "SELECT COUNT(*) FROM messages WHERE is_from_me = 1 AND sender_handle_id IS NOT NULL"
        ) == [(0,)]

    def test_contacts_name_the_people(self, basic_export: Path, archive_path: Path) -> None:
        import_legacy_csv(
            basic_export, archive_path, contacts=[{"name": "Rowan Vale", "phones": [ROWAN], "emails": []}]
        )

        with Archive.open(archive_path) as archive:
            named = [person for person in archive.people() if person["name"]]
        assert [(person["name"], person["handles"]) for person in named] == [("Rowan Vale", [ROWAN])]

    def test_the_source_rowid_resolves(self, basic_export: Path, archive_path: Path) -> None:
        import_legacy_csv(basic_export, archive_path)

        with Archive.open(archive_path) as archive:
            found = archive.message_by_source_rowid(22)
            assert found is not None and found["text"] == "I can bring bread"
            assert archive.message_by_source_rowid(999) is None

    def test_a_recovered_row_keeps_its_large_id(self, export_path: Path, archive_path: Path) -> None:
        export_path.parent.mkdir()
        write_export(export_path, [line(1_000_000_042, "2023-01-05 08:00:00", "recovered from a snapshot")])

        import_legacy_csv(export_path, archive_path)

        with Archive.open(archive_path) as archive:
            found = archive.message_by_source_rowid(1_000_000_042)
        assert found is not None and found["id"] == "legacy:1000000042"


class TestContent:
    def load(self, export_path: Path, archive_path: Path, rows: list[dict[str, Any]]) -> Any:
        export_path.parent.mkdir(exist_ok=True)
        return import_legacy_csv(write_export(export_path, rows), archive_path)

    def test_attachment_rows(self, export_path: Path, archive_path: Path) -> None:
        self.load(
            export_path,
            archive_path,
            [
                line(
                    1,
                    "2024-04-01 10:00:00",
                    PLACEHOLDER,
                    attachment_types="image/jpeg",
                    attachment_files="~/a/IMG_1.jpeg",
                ),
                line(
                    2,
                    "2024-04-01 10:01:00",
                    "[Attachment] image/png | ~/a/shot.png",
                    attachment_types="image/png",
                    attachment_files="~/a/shot.png",
                ),
                line(
                    3,
                    "2024-04-01 10:02:00",
                    f"{PLACEHOLDER}Look at this",
                    attachment_types="image/heic|video/mp4",
                    attachment_files="~/a/one.heic|~/a/two.mov",
                ),
                line(4, "2024-04-01 10:03:00", PLACEHOLDER),  # the file record is gone, the message is not
                line(5, "2024-04-01 10:04:00", "[Attachment]"),
            ],
        )

        with Archive.open(archive_path) as archive:
            found = archive.messages(ROWAN)["messages"]
        assert [(m["kind"], m["text"], m["attachments"]) for m in found] == [
            ("attachment", "", ["image/jpeg"]),
            ("attachment", "", ["image/png"]),
            ("text", "Look at this", ["image/heic", "video/mp4"]),
            ("attachment", "", []),
            ("attachment", "", []),
        ]
        assert query(archive_path, "SELECT file_name FROM attachments ORDER BY id") == [
            ("IMG_1.jpeg",),
            ("shot.png",),
            ("one.heic",),
            ("two.mov",),
        ]

    def test_a_tapback_phrase_is_stored_as_a_hidden_row(self, export_path: Path, archive_path: Path) -> None:
        result = self.load(
            export_path,
            archive_path,
            [
                line(1, "2024-04-02 10:00:00", "See you at the pier"),
                line(2, "2024-04-02 10:01:00", f"Loved {CURLY_OPEN}See you at the pier{CURLY_CLOSE}", from_me=True),
                line(3, "2024-04-02 10:02:00", 'Laughed at "See you at\nthe pier"'),
                line(
                    4,
                    "2024-04-02 10:03:00",
                    f"Removed a heart from {CURLY_OPEN}See you at the pier{CURLY_CLOSE}",
                    from_me=True,
                ),
                line(5, "2024-04-02 10:04:00", "Loved the pier, will go again"),
            ],
        )

        assert (result.messages_added, result.system_added, result.counts["reaction_phrases"]) == (2, 3, 3)
        with Archive.open(archive_path) as archive:
            assert archive.count() == 2
            assert [m["text"] for m in archive.messages(ROWAN)["messages"]] == [
                "See you at the pier",
                "Loved the pier, will go again",
            ]
            assert len(archive.messages(ROWAN, include_system=True)["messages"]) == 5

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('Loved "ok"', True),
            (f"Liked {CURLY_OPEN}ok{CURLY_CLOSE}", True),
            (f"Disliked {CURLY_OPEN}ok{CURLY_CLOSE}", True),
            (f"Emphasized {CURLY_OPEN}ok{CURLY_CLOSE}", True),
            (f"Questioned {CURLY_OPEN}two\nlines{CURLY_CLOSE}", True),
            (f"Removed an exclamation from {CURLY_OPEN}ok{CURLY_CLOSE}", True),
            (f"Removed a question mark from {CURLY_OPEN}ok{CURLY_CLOSE}", True),
            ("Loved an image", True),
            ("Laughed at a movie", True),
            ("Emphasized an audio message", True),
            ("Disliked an attachment", True),
            ("Removed a like from an image", True),
            (f"Reacted {chr(0x1F602)} to {CURLY_OPEN}hi{CURLY_CLOSE}", True),
            ("Liked a contact", True),
            (f"Removed {chr(0x1F602)} from {CURLY_OPEN}hi{CURLY_CLOSE}", True),
            ("Loved an image of you", False),
            (f"Loved {CURLY_OPEN}Dune{CURLY_CLOSE} a lot", False),
            ("Loved a sticker", False),
            ('Removed it from "Groceries"', False),
            ('Removed Mira from "Book club"', False),
            ('Reacted badly to "the news"', False),
            ('Reacted 100 to "the news"', False),
            ('Removed an heart from "ok"', False),
            ('Removed a exclamation from "ok"', False),
            ("Reacted badly to the news", False),
            ("Loved it", False),
            (f"I Loved {CURLY_OPEN}ok{CURLY_CLOSE}", False),
            (f"Loved {CURLY_OPEN}ok{CURLY_CLOSE} so much", False),
            ('loved "ok"', False),
            ("", False),
        ],
    )
    def test_the_tapback_phrase_rule(self, text: str, expected: bool) -> None:
        assert is_reaction_phrase(text) is expected

    def test_rows_that_cannot_be_stored_are_counted(self, export_path: Path, archive_path: Path) -> None:
        result = self.load(
            export_path,
            archive_path,
            [
                line(1, "2024-04-03 10:00:00", "kept"),
                line(2, "2024-04-03 10:01:00", "no chat", ""),
                line(3, "not a time", "bad time"),
                line(4, "2024-04-03 10:03:00", ""),
                {**line(5, "2024-04-03 10:04:00", "bad id"), "message_id": "five"},
                {**line(6, "2024-04-03 10:05:00", "bad flag"), "is_from_me": "maybe"},
            ],
        )

        assert result.messages_added == 2  # the row with no chat is stored: it belongs to no conversation
        assert result.counts["without_conversation"] == 1
        assert result.skipped == {"no_date": 1, "empty": 1, "bad_row": 2}
        assert scalar(archive_path, "SELECT COUNT(*) FROM conversations") == 1  # a skipped row makes no conversation

    def test_one_rowid_twice_keeps_the_last_row(self, export_path: Path, archive_path: Path) -> None:
        result = self.load(
            export_path,
            archive_path,
            [line(7, "2024-04-04 10:00:00", "first copy"), line(7, "2024-04-04 10:00:00", "second copy")],
        )

        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 1
        assert scalar(archive_path, "SELECT text FROM messages") == "second copy"
        assert result.rows_seen == 2

    def test_an_oversize_text_is_kept_whole_and_counted(self, export_path: Path, archive_path: Path) -> None:
        huge = "start " + "filler " * 40_000 + "finish"  # larger than the csv module's default field limit
        result = self.load(export_path, archive_path, [line(1, "2024-04-05 10:00:00", huge)])

        assert len(huge) > 131_072 > INDEXED_TEXT_CHARS
        assert result.oversize_texts == 1
        with Archive.open(archive_path) as archive:
            assert archive.messages(ROWAN)["messages"][0]["text"] == huge
            assert archive.search("start")["total"] == 1
            assert archive.search("finish")["total"] == 0  # only the first part of an oversize text is searched
        assert fts_is_consistent(archive_path)

    def test_fractions_of_a_second_and_other_flag_spellings_are_read(
        self, export_path: Path, archive_path: Path
    ) -> None:
        self.load(
            export_path,
            archive_path,
            [
                {**line(1, "2024-04-06 10:00:00.250000", "with a fraction"), "is_from_me": "True"},
                {**line(2, "2024-04-06 10:00:01", "float flag"), "is_from_me": "0.0"},
            ],
        )

        assert query(archive_path, "SELECT is_from_me, text FROM messages ORDER BY sent_ns") == [
            (1, "with a fraction"),
            (0, "float flag"),
        ]


class TestOddFiles:
    def load(self, export_path: Path, archive_path: Path, rows: list[dict[str, Any]]) -> Any:
        export_path.parent.mkdir(exist_ok=True)
        return import_legacy_csv(write_export(export_path, rows), archive_path)

    def test_a_row_with_too_few_or_too_many_fields_is_counted_and_not_stored(
        self, export_path: Path, archive_path: Path
    ) -> None:
        export_path.parent.mkdir()
        write_export(export_path, [line(1, "2024-04-07 10:00:00", "whole row")])
        with export_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(f"2,2024-04-07 10:01:00,{ROWAN},cut off\r\n")
            handle.write(f"3,2024-04-07 10:02:00,{ROWAN},too long,{ROWAN},,0,iMessage,,,extra\r\n")
            handle.write("\r\n")

        result = import_legacy_csv(export_path, archive_path)

        assert (result.rows_seen, result.messages_added) == (3, 1)
        assert result.skipped == {"bad_row": 2}

    def test_a_row_with_no_direction_or_a_number_too_large_is_a_bad_row(
        self, export_path: Path, archive_path: Path
    ) -> None:
        result = self.load(
            export_path,
            archive_path,
            [
                line(1, "2024-04-07 10:00:00", "kept"),
                {**line(2, "2024-04-07 10:01:00", "no direction"), "is_from_me": ""},
                line(2**63, "2024-04-07 10:02:00", "number too large"),
            ],
        )

        assert result.messages_added == 1
        assert result.skipped == {"bad_row": 2}

    def test_a_row_at_or_before_the_start_of_apple_time_has_no_date(
        self, export_path: Path, archive_path: Path
    ) -> None:
        result = self.load(
            export_path,
            archive_path,
            [
                line(1, "2024-04-07 10:00:00", "dated"),
                line(2, "2000-12-31 16:00:00", "the exporter wrote this time for a row with no date"),
                line(3, "1999-07-04 00:30:00", "before any message"),
            ],
        )

        assert result.messages_added == 1
        assert result.skipped == {"no_date": 2}
        with Archive.open(archive_path) as archive:
            assert archive.summary()["first_timestamp"] == "2024-04-07 10:00:00"

    def test_a_quote_that_is_never_closed_is_an_error_and_not_one_long_text(
        self, export_path: Path, archive_path: Path
    ) -> None:
        export_path.parent.mkdir()
        write_export(export_path, [line(1, "2024-04-07 10:00:00", "fine")])
        with export_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(f'2,2024-04-07 10:01:00,{ROWAN},"never closed,{ROWAN},,0,iMessage,,\r\n')
            handle.write(f"3,2024-04-07 10:02:00,{ROWAN},swallowed,{ROWAN},,0,iMessage,,\r\n")

        with pytest.raises(SourceFormatError) as caught:
            import_legacy_csv(export_path, archive_path)

        assert "never closed" not in str(caught.value) and caught.value.__cause__ is None  # no text of the file
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 0

    def test_a_header_that_names_a_column_twice_is_refused(self, export_path: Path, archive_path: Path) -> None:
        export_path.parent.mkdir()
        write_export(export_path, BASIC, columns=(*COLUMNS, "text"))

        with pytest.raises(SourceFormatError, match="more than one time"):
            import_legacy_csv(export_path, archive_path)

    def test_one_rowid_twice_in_one_batch_keeps_the_files_of_the_last_row_only(
        self, export_path: Path, archive_path: Path
    ) -> None:
        result = self.load(
            export_path,
            archive_path,
            [
                line(
                    7,
                    "2024-04-07 10:00:00",
                    PLACEHOLDER,
                    attachment_types="image/jpeg|image/png",
                    attachment_files="~/a/one.jpeg|~/a/two.png",
                ),
                line(7, "2024-04-07 10:00:00", "the later copy has no file"),
            ],
        )

        assert (result.messages_added, result.messages_updated, result.attachments_added) == (1, 1, 0)
        assert query(archive_path, "SELECT kind, text FROM messages") == [("text", "the later copy has no file")]
        assert scalar(archive_path, "SELECT COUNT(*) FROM attachments") == 0

    def test_lists_of_different_length_are_not_paired(self, export_path: Path, archive_path: Path) -> None:
        self.load(
            export_path,
            archive_path,
            [
                line(
                    1,
                    "2024-04-07 10:00:00",
                    PLACEHOLDER,
                    attachment_types="image/png",
                    attachment_files="~/a/unknown.bin|~/a/shot.png",
                )
            ],
        )

        assert query(archive_path, "SELECT position, mime_type, file_name FROM attachments ORDER BY position") == [
            (0, None, "unknown.bin"),
            (1, None, "shot.png"),
        ]

    def test_two_loads_at_the_same_time_share_the_field_limit(self, tmp_path: Path) -> None:
        big = "x" * 300_000
        paths = []
        for name, count in (("short", 40), ("long", 400)):
            folder = tmp_path / name
            folder.mkdir()
            rows = [
                line(index + 1, "2024-04-08 10:00:00", big if index % 20 == 0 else "small") for index in range(count)
            ]
            paths.append((write_export(folder / "messages.csv", rows), folder / "archive.sqlite3"))
        before = csv.field_size_limit()
        results: list[Any] = []

        def load(export: Path, archive: Path) -> None:
            try:
                results.append(import_legacy_csv(export, archive).status)
            except Exception as error:  # noqa: BLE001 - the test reports it
                results.append(error)

        threads = [threading.Thread(target=load, args=pair) for pair in paths]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)

        assert results == ["ok", "ok"]
        assert csv.field_size_limit() == before


class TestClockChanges:
    """America/Los_Angeles: on 2024-11-03 the hour from 01:00 happens two times, on 2024-03-10 02:00 to 03:00 does not exist."""

    REPEATED = (
        line(101, "2024-11-03 00:50:00", "before the hour"),
        line(102, "2024-11-03 01:10:00", "first pass, ten past"),
        line(103, "2024-11-03 01:40:00", "first pass, twenty to"),
        line(104, "2024-11-03 01:10:00", "second pass, ten past"),
        line(105, "2024-11-03 01:40:00", "second pass, twenty to"),
        line(106, "2024-11-03 02:05:00", "after the hour"),
    )

    def texts(self, archive_path: Path) -> list[str]:
        with Archive.open(archive_path) as archive:
            return [message["text"] for message in archive.messages(ROWAN)["messages"]]

    def test_a_time_that_occurs_twice_is_read_as_its_first_pass_and_keeps_the_rowid_order(self, tmp_path: Path) -> None:
        orders = []
        for name, rows in (("in-order", self.REPEATED), ("shuffled", [self.REPEATED[i] for i in (4, 0, 3, 5, 1, 2)])):
            folder = tmp_path / name
            folder.mkdir()
            import_legacy_csv(write_export(folder / "messages.csv", rows), folder / "archive.sqlite3")
            orders.append(self.texts(folder / "archive.sqlite3"))

        # equal local times are one moment: inside it the source ROWID decides, not the order of the file
        assert (
            orders[0]
            == orders[1]
            == [
                "before the hour",
                "first pass, ten past",
                "second pass, ten past",
                "first pass, twenty to",
                "second pass, twenty to",
                "after the hour",
            ]
        )

    def test_the_time_stamps_of_the_repeated_hour_come_back_as_they_were_written(
        self, export_path: Path, archive_path: Path
    ) -> None:
        export_path.parent.mkdir()
        import_legacy_csv(write_export(export_path, self.REPEATED), archive_path)

        with Archive.open(archive_path) as archive:
            stamps = sorted(message["timestamp"] for message in archive.messages(ROWAN)["messages"])
        assert stamps == sorted(row["timestamp"] for row in self.REPEATED)

    def test_a_time_that_does_not_exist_is_read_one_hour_later(self, export_path: Path, archive_path: Path) -> None:
        export_path.parent.mkdir()
        rows = [
            line(1, "2024-03-10 01:59:00", "before the gap"),
            line(2, "2024-03-10 02:13:00", "inside the gap"),
            line(3, "2024-03-10 03:05:00", "after the gap"),
        ]
        import_legacy_csv(write_export(export_path, rows), archive_path)

        with Archive.open(archive_path) as archive:
            found = [(m["text"], m["timestamp"]) for m in archive.messages(ROWAN)["messages"]]
        assert found == [
            ("before the gap", "2024-03-10 01:59:00"),
            ("after the gap", "2024-03-10 03:05:00"),
            ("inside the gap", "2024-03-10 03:13:00"),
        ]


class TestRefusals:
    def test_a_non_empty_archive_is_refused_and_left_as_it_was(self, basic_export: Path, archive_path: Path) -> None:
        import_legacy_csv(basic_export, archive_path)
        before = content(archive_path)

        with pytest.raises(ArchiveNotEmptyError):
            import_legacy_csv(basic_export, archive_path)

        assert content(archive_path) == before
        assert runs(archive_path)[-1][1:3] == ("legacy_csv", "failed")

    def test_an_archive_filled_from_the_database_is_refused(
        self, basic_export: Path, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        handle = builder.handle(ROWAN)
        chat = builder.chat(ROWAN, members=[handle])
        builder.message(chat, when=utc(2024, 3, 1, 17, 0), text="from the database", handle=handle)
        builder.commit()
        import_chat_db(source_path, archive_path)

        with pytest.raises(ArchiveNotEmptyError):
            import_legacy_csv(basic_export, archive_path)

        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 1

    def test_a_file_without_the_needed_columns_is_refused(self, export_path: Path, archive_path: Path) -> None:
        export_path.parent.mkdir()
        write_export(export_path, BASIC, columns=("message_id", "timestamp", "text"))

        with pytest.raises(SourceFormatError, match="chat_id"):
            import_legacy_csv(export_path, archive_path)

    def test_a_missing_file_is_refused_before_the_archive_is_made(self, export_path: Path, archive_path: Path) -> None:
        with pytest.raises(SourceError):
            import_legacy_csv(export_path, archive_path)
        assert not archive_path.exists()

    def test_a_file_that_is_not_text_is_refused(self, export_path: Path, archive_path: Path) -> None:
        export_path.parent.mkdir()
        export_path.write_bytes(b"message_id,timestamp,text,chat_id,is_from_me\n1,2024-01-01 00:00:00,\xff\xfe,x,0\n")

        with pytest.raises(SourceFormatError):
            import_legacy_csv(export_path, archive_path)

        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 0


class TestRunControl:
    def test_progress_counts_bytes_and_ends_at_the_file_size(self, basic_export: Path, archive_path: Path) -> None:
        seen: list[dict[str, Any]] = []
        import_legacy_csv(basic_export, archive_path, progress=seen.append)

        size = basic_export.stat().st_size
        assert {"phase": "messages", "done": size, "total": size} in seen
        assert seen[-1] == {"phase": "finish", "done": 1, "total": 1}

    def test_a_cancelled_load_leaves_the_archive_empty(self, basic_export: Path, archive_path: Path) -> None:
        stop = threading.Event()
        stop.set()

        result = import_legacy_csv(basic_export, archive_path, cancel=stop)

        assert result.status == "cancelled"
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages") == 0
        assert import_legacy_csv(basic_export, archive_path).messages_added == 6  # still empty: the load can run


class TestThePullThatFollows:
    def source(self, builder: ChatDbBuilder) -> None:
        """The same history as BASIC in a database, with one message more and one message less."""
        rowan, quinn = builder.handle(ROWAN), builder.handle(QUINN)
        direct = builder.chat(ROWAN, members=[rowan])
        group = builder.chat(GROUP, display_name="Harbor crew", members=[rowan, quinn])
        builder.message(
            direct, when=utc(2024, 3, 1, 17, 0), text="Are we still on for the ferry?", handle=rowan, rowid=11
        )
        builder.message(
            direct,
            when=utc(2024, 3, 1, 17, 2, 10),
            text="  Yes, 10:15 from pier two  ",
            from_me=True,
            handle=rowan,
            rowid=12,
        )
        builder.message(group, when=utc(2024, 3, 3, 2, 30), text="Dinner on Friday?", handle=rowan, rowid=21)
        builder.message(group, when=utc(2024, 3, 3, 2, 31), text="I can bring bread", handle=quinn, rowid=22)
        builder.message(group, when=utc(2024, 3, 3, 2, 40), text="See you all then", handle=quinn, rowid=30)
        builder.commit()

    def test_a_full_pull_replaces_everything_the_bridge_loaded(
        self, basic_export: Path, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_legacy_csv(basic_export, archive_path)
        self.source(builder)

        result = import_chat_db(source_path, archive_path)  # the default limit: bridge rows do not trip the valve

        assert result.mode == "full"
        assert result.counts["legacy_rows_replaced"] == 6
        assert (result.messages_added, result.messages_removed) == (5, 0)
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE guid LIKE 'legacy:%'") == 0
        with Archive.open(archive_path) as archive:
            assert archive.summary()["messages"] == 5
            found = archive.message_by_source_rowid(22)
            assert found is not None
            assert (found["text"], found["id"]) == ("I can bring bread", builder.guid_of(22))
            assert archive.message_by_source_rowid(13) is None  # the database no longer has it
            assert archive.conversation(GROUP)["message_count"] == 3  # type: ignore[index]
        assert fts_is_consistent(archive_path)
        assert totals_match_messages(archive_path)

    def test_an_incremental_run_after_the_bridge_is_a_full_pull(
        self, basic_export: Path, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        import_legacy_csv(basic_export, archive_path)
        self.source(builder)

        result = import_chat_db(source_path, archive_path, full=False)

        assert result.mode == "full"
        assert scalar(archive_path, "SELECT COUNT(*) FROM messages WHERE guid LIKE 'legacy:%'") == 0
        assert import_chat_db(source_path, archive_path, full=False).mode == "incremental"
