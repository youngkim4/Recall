"""Pure rules of the importer: Apple dates, handles, tapbacks, row classification."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest

from archive import timeconv
from archive.classify import classify, stored_text
from archive.handles import (
    handle_kind,
    handle_match_key,
    is_group_identifier,
    phone_key,
)
from archive.records import MessageRecord, ReactionRecord, Skipped, SourceMessage
from archive.tapbacks import split_target, tapback
from tests.archive.builders import APPLE_EPOCH_UNIX, apple_time, utc
from tests.archive.typedstream_fixtures import blob, counter_text

NS = 1_000_000_000
PLACEHOLDER = chr(0xFFFC)  # the character that stands for an attachment
MOMENT = utc(2024, 1, 15, 18, 30, 5)
MOMENT_UNIX = int(MOMENT.timestamp())


def source(**changes: object) -> SourceMessage:
    base: dict[str, object] = dict.fromkeys(SourceMessage._fields)
    base.update(rowid=10, guid="G-10", is_from_me=0, handle_rowid=3, chat_rowid=1, date=apple_time(MOMENT))
    base.update(changes)
    return SourceMessage(**base)  # type: ignore[arg-type]


class TestAppleDates:
    def test_nanoseconds(self) -> None:
        assert timeconv.apple_to_unix_ns(apple_time(MOMENT, "ns")) == MOMENT_UNIX * NS

    def test_seconds(self) -> None:
        assert timeconv.apple_to_unix_ns(apple_time(MOMENT, "s")) == MOMENT_UNIX * NS

    def test_sub_second_part_is_kept(self) -> None:
        value = (MOMENT_UNIX - APPLE_EPOCH_UNIX) * NS + 123_456_789
        assert timeconv.apple_to_unix_ns(value) == MOMENT_UNIX * NS + 123_456_789

    @pytest.mark.parametrize("value", [None, 0, -5, "", "not a number", 10**19])
    def test_missing_or_impossible_dates_give_none(self, value: object) -> None:
        assert timeconv.apple_to_unix_ns(value) is None

    def test_unix_seconds_helper(self) -> None:
        assert timeconv.apple_to_unix_seconds(apple_time(MOMENT)) == MOMENT_UNIX
        assert timeconv.apple_to_unix_seconds(0) is None


class TestLocalDays:
    def test_day_bounds_are_local_midnights(self) -> None:
        # the test time zone is America/Los_Angeles
        assert timeconv.day_start("2024-01-15") == int(utc(2024, 1, 15, 8).timestamp())
        assert timeconv.day_end("2024-01-15") == int(utc(2024, 1, 16, 8).timestamp())

    def test_day_with_a_daylight_saving_change_has_23_hours(self) -> None:
        assert timeconv.day_end("2024-03-10") - timeconv.day_start("2024-03-10") == 23 * 3600
        assert timeconv.day_end("2024-11-03") - timeconv.day_start("2024-11-03") == 25 * 3600

    @pytest.mark.parametrize("value", ["2024-1-5", "20240105", "2024-13-01", "yesterday", "", "2024-01-15 10:00"])
    def test_bad_days_raise_value_error(self, value: str) -> None:
        with pytest.raises(ValueError):
            timeconv.day_start(value)

    def test_python_and_sqlite_agree_on_local_time(self) -> None:
        conn = sqlite3.connect(":memory:")
        for unix in (1_710_064_800, 1_710_068_400, 1_730_624_400, 1_730_628_000, MOMENT_UNIX, APPLE_EPOCH_UNIX):
            from_sql = conn.execute(f"SELECT {timeconv.sql_local_timestamp('?')}", (unix,)).fetchone()[0]
            assert from_sql == timeconv.local_timestamp(unix)
            assert from_sql == datetime.fromtimestamp(unix).strftime("%Y-%m-%d %H:%M:%S")
        conn.close()


class TestHandles:
    @pytest.mark.parametrize(
        ("identifier", "expected"),
        [
            ("chat123456789", True),
            ("chat0", True),
            ("chat", False),
            ("chat12ab", False),
            ("xchat123", False),
            ("+15550100", False),
            ("rowan@example.com", False),
            ("chatty@example.com", False),
        ],
    )
    def test_group_rule(self, identifier: str, expected: bool) -> None:
        assert is_group_identifier(identifier) is expected

    @pytest.mark.parametrize(
        ("handle", "kind"),
        [
            ("+15550100", "phone"),
            ("(555) 010-0100", "phone"),
            ("26296", "phone"),
            ("Rowan@Example.com", "email"),
            ("urn:biz:0a1b2c3d", "other"),
            ("INVENTEDBANK", "other"),
        ],
    )
    def test_kind(self, handle: str, kind: str) -> None:
        assert handle_kind(handle) == kind

    def test_phone_key_uses_the_last_ten_digits(self) -> None:
        assert phone_key("+1 (555) 010-0199") == "p:5550100199"
        assert phone_key(f"555{chr(0xA0)}010{chr(0x2011)}0199") == "p:5550100199"  # no-break space and hyphen
        assert phone_key("5550100199") == "p:5550100199"
        assert phone_key("26296") == "p:26296"
        assert phone_key("no digits") is None

    def test_match_keys(self) -> None:
        assert handle_match_key("+15550100199") == "p:5550100199"
        assert handle_match_key(" Rowan@Example.COM ") == "e:rowan@example.com"
        assert handle_match_key("urn:biz:0A1B") == "o:urn:biz:0a1b"


class TestTapbacks:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (2000, ("love", False)),
            (2001, ("like", False)),
            (2002, ("dislike", False)),
            (2003, ("laugh", False)),
            (2004, ("emphasis", False)),
            (2005, ("question", False)),
            (2006, ("emoji", False)),
            (2007, ("sticker", False)),
            (3000, ("love", True)),
            (3007, ("sticker", True)),
            (0, None),
            (None, None),
            (1000, None),
            (1999, None),
            (2008, None),
            (2500, None),
            (3008, None),
        ],
    )
    def test_codes(self, code: int | None, expected: tuple[str, bool] | None) -> None:
        assert tapback(code) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("p:0/ABC-1", ("ABC-1", 0)),
            ("p:12/ABC-1", ("ABC-1", 12)),
            ("bp:ABC-1", ("ABC-1", None)),
            ("ABC-1", ("ABC-1", None)),
            ("  p:1/ABC-1 ", ("ABC-1", 1)),
            ("", None),
            (None, None),
            ("p:0/", None),
        ],
    )
    def test_target_prefixes_are_removed(self, value: str | None, expected: tuple[str, int | None] | None) -> None:
        assert split_target(value) == expected


class TestStoredText:
    def test_only_attachment_placeholders_and_nul_are_removed(self) -> None:
        assert stored_text(PLACEHOLDER) == ""
        assert stored_text(f"{PLACEHOLDER}{PLACEHOLDER} Look at this \n") == " Look at this \n"  # nothing is trimmed
        assert stored_text("line one\nline two") == "line one\nline two"
        assert stored_text("nul\x00byte") == "nulbyte"
        assert stored_text(None) == ""


class TestClassify:
    def test_plain_text_from_the_text_column(self) -> None:
        record = classify(source(text="See you at six", service="iMessage"), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert (record.kind, record.text, record.text_source) == ("text", "See you at six", "column")
        assert record.sent_ns == MOMENT_UNIX * NS
        assert (record.is_from_me, record.handle_rowid, record.chat_rowid) == (False, 3, 1)
        assert (record.guid, record.source_rowid, record.service) == ("G-10", 10, "iMessage")
        assert (record.unsent, record.edited_at, record.reply_to_guid, record.item_type) == (False, None, None, 0)

    def test_text_from_the_blob_when_the_column_is_empty(self) -> None:
        record = classify(source(body=blob("ascii_300")), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert (record.text, record.text_source) == (counter_text(300), "blob")

    def test_own_message_has_no_sender(self) -> None:
        record = classify(source(text="on my way", is_from_me=1, handle_rowid=3), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert record.is_from_me is True and record.handle_rowid is None

    def test_attachment_only_message(self) -> None:
        record = classify(source(body=blob("attachment_only")), has_attachments=True)
        assert isinstance(record, MessageRecord)
        assert (record.kind, record.text, record.text_source) == ("attachment", "", "none")

    def test_caption_with_attachment_is_text(self) -> None:
        record = classify(source(body=blob("attachment_with_caption")), has_attachments=True)
        assert isinstance(record, MessageRecord)
        assert (record.kind, record.text) == ("text", "Look at this view")

    def test_tapback_is_a_reaction(self) -> None:
        record = classify(
            source(text="Loved “hello”", associated_type=2000, associated_guid="p:0/TARGET-1"), has_attachments=False
        )
        assert isinstance(record, ReactionRecord)
        assert (record.kind, record.removed, record.target_guid, record.target_part) == ("love", False, "TARGET-1", 0)
        assert record.emoji is None

    def test_emoji_tapback_and_removal(self) -> None:
        added = classify(
            source(associated_type=2006, associated_guid="bp:TARGET-1", associated_emoji="🔥"), has_attachments=False
        )
        removed = classify(source(associated_type=3001, associated_guid="p:2/TARGET-1"), has_attachments=False)
        assert isinstance(added, ReactionRecord) and isinstance(removed, ReactionRecord)
        assert (added.kind, added.emoji, added.target_part) == ("emoji", "🔥", None)
        assert (removed.kind, removed.removed, removed.target_part) == ("like", True, 2)

    def test_tapback_without_target_is_skipped(self) -> None:
        assert classify(source(associated_type=2000), has_attachments=False) == Skipped("no_target")

    def test_group_event_is_a_system_row(self) -> None:
        record = classify(
            source(item_type=2, group_title="Lake Weekend", group_action_type=0, text=None), has_attachments=False
        )
        assert isinstance(record, MessageRecord)
        assert (record.kind, record.text, record.item_type) == ("system", "Lake Weekend", 2)

    def test_group_event_keeps_its_action_fields(self) -> None:
        record = classify(source(item_type=1, group_action_type=1, other_handle=7), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert record.kind == "system" and record.text == ""
        assert dict(record.extra) == {"group_action_type": 1}
        assert record.other_handle_rowid == 7

    def test_row_with_no_content_is_skipped(self) -> None:
        assert classify(source(), has_attachments=False) == Skipped("empty")

    def test_unreadable_blob_is_reported(self) -> None:
        assert classify(source(body=b"\x01\x02\x03"), has_attachments=False) == Skipped("empty", decode_failed=True)
        record = classify(source(body=b"\x01\x02\x03"), has_attachments=True)
        assert isinstance(record, MessageRecord)
        assert record.kind == "attachment" and record.decode_failed is True

    def test_unsent_message_is_kept_without_text(self) -> None:
        record = classify(source(date_retracted=apple_time(utc(2024, 1, 15, 18, 31))), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert (record.kind, record.text, record.unsent) == ("text", "", True)

    def test_edited_and_reply_fields(self) -> None:
        edited = utc(2024, 1, 15, 18, 40)
        record = classify(
            source(text="fixed typo", date_edited=apple_time(edited), thread_guid="ROOT-1"), has_attachments=False
        )
        assert isinstance(record, MessageRecord)
        assert record.edited_at == int(edited.timestamp()) and record.reply_to_guid == "ROOT-1"

    def test_missing_guid_and_missing_date_are_skipped(self) -> None:
        assert classify(source(guid=None, text="x"), has_attachments=False) == Skipped("no_guid")
        assert classify(source(guid="  ", text="x"), has_attachments=False) == Skipped("no_guid")
        assert classify(source(date=0, text="x"), has_attachments=False) == Skipped("no_date")

    def test_delivery_time_stands_in_for_a_missing_date(self) -> None:
        record = classify(source(date=0, date_delivered=apple_time(MOMENT), text="x"), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert record.sent_ns == MOMENT_UNIX * NS

    def test_rare_fields_go_to_extra(self) -> None:
        record = classify(
            source(
                text="ha",
                subject="Plans",
                balloon_bundle_id="com.example.balloon",
                expressive_style="com.apple.MobileSMS.expressivesend.impact",
                is_audio=1,
                associated_type=1000,
                associated_guid="p:0/TARGET-9",
            ),
            has_attachments=False,
        )
        assert isinstance(record, MessageRecord)
        assert dict(record.extra) == {
            "subject": "Plans",
            "balloon_bundle_id": "com.example.balloon",
            "expressive_send_style_id": "com.apple.MobileSMS.expressivesend.impact",
            "is_audio_message": True,
            "associated_message_type": 1000,
            "associated_message_guid": "p:0/TARGET-9",
        }
        json.dumps(dict(record.extra))

    def test_plain_message_has_no_extra(self) -> None:
        record = classify(source(text="hi", associated_type=0, group_action_type=0), has_attachments=False)
        assert isinstance(record, MessageRecord)
        assert record.extra == ()
