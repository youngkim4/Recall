"""Text is stored as Messages has it. Only the attachment placeholder and NUL are never stored."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from archive import Archive, import_chat_db
from archive.classify import message_text, stored_text
from tests.archive.builders import ChatDbBuilder, apple_time, utc
from tests.archive.inspect_archive import query, scalar
from tests.archive.typedstream_fixtures import blob, expected_text

START = utc(2024, 7, 1, 15, 0)
NORA = "+15550100151"
PLACEHOLDER = chr(0xFFFC)
ZWJ, ZWSP, WORD_JOINER, VS16 = chr(0x200D), chr(0x200B), chr(0x2060), chr(0xFE0F)
FAMILY = ZWJ.join((chr(0x1F468), chr(0x1F469), chr(0x1F467)))  # three people joined into one emoji


def stored(archive_path: Path) -> list[tuple[str, str, str]]:
    return query(archive_path, "SELECT kind, text, text_source FROM messages ORDER BY sent_ns, id")


@pytest.fixture
def chat(builder: ChatDbBuilder) -> tuple[int, int]:
    handle = builder.handle(NORA)
    return builder.chat(NORA, members=[handle]), handle


class TestStoredText:
    def test_outer_white_space_is_kept_from_the_column_and_from_the_blob(
        self, builder: ChatDbBuilder, chat: tuple[int, int], source_path: Path, archive_path: Path
    ) -> None:
        builder.message(chat[0], when=START, text="  padded line \n", handle=chat[1])
        builder.message(chat[0], when=START + timedelta(minutes=1), body=blob("tab_and_spaces"), handle=chat[1])
        builder.commit()

        import_chat_db(source_path, archive_path)

        assert stored(archive_path) == [
            ("text", "  padded line \n", "column"),
            ("text", expected_text("tab_and_spaces"), "blob"),
        ]
        assert expected_text("tab_and_spaces") != expected_text("tab_and_spaces").strip()

    @pytest.mark.parametrize("name", ["zwj_family", "variation_selector", "invisible_marks", "outer_newlines"])
    def test_invisible_characters_survive_apples_encoder_and_the_import(
        self, name: str, builder: ChatDbBuilder, chat: tuple[int, int], source_path: Path, archive_path: Path
    ) -> None:
        text = expected_text(name)
        builder.message(chat[0], when=START, body=blob(name), handle=chat[1])
        builder.message(chat[0], when=START + timedelta(minutes=1), text=text, handle=chat[1])
        builder.commit()

        import_chat_db(source_path, archive_path)

        assert [row[1] for row in stored(archive_path)] == [text, text]
        with Archive.open(archive_path) as archive:
            assert [message["text"] for message in archive.messages(NORA)["messages"]] == [text, text]

    def test_the_fixture_texts_hold_the_characters_they_are_named_for(self) -> None:
        assert ZWJ in expected_text("zwj_family")
        assert VS16 in expected_text("variation_selector")
        assert {ZWSP, WORD_JOINER} <= set(expected_text("invisible_marks"))
        assert expected_text("outer_newlines").startswith("\n") and expected_text("outer_newlines").endswith("\n")

    def test_only_the_placeholder_and_nul_are_removed(self) -> None:
        assert stored_text(f" {PLACEHOLDER}caption\x00 here{ZWSP} ") == f" caption here{ZWSP} "
        assert stored_text(f"a {FAMILY} b{VS16}") == f"a {FAMILY} b{VS16}"
        assert stored_text(None) == ""
        assert stored_text("") == ""

    def test_message_text_gives_the_text_as_the_source_has_it(self) -> None:
        assert message_text("  hi  ", None) == ("  hi  ", "column", False)
        assert message_text(None, blob("tab_and_spaces")) == (expected_text("tab_and_spaces"), "blob", False)
        assert message_text("", blob("attachment_only")) == (PLACEHOLDER, "blob", False)
        assert message_text(None, None) == ("", "none", False)
        assert message_text(None, b"not a typedstream") == ("", "none", True)


class TestKinds:
    def test_a_message_of_white_space_is_what_was_sent(
        self, builder: ChatDbBuilder, chat: tuple[int, int], source_path: Path, archive_path: Path
    ) -> None:
        builder.message(chat[0], when=START, text=" \n", handle=chat[1])
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert result.skipped == {}
        assert stored(archive_path) == [("text", " \n", "column")]

    def test_a_placeholder_with_no_attachment_record_is_still_an_attachment_message(
        self, builder: ChatDbBuilder, chat: tuple[int, int], source_path: Path, archive_path: Path
    ) -> None:
        builder.message(chat[0], when=START, text=PLACEHOLDER, handle=chat[1])
        builder.message(chat[0], when=START + timedelta(minutes=1), text=f"{PLACEHOLDER}\n", handle=chat[1])
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert result.skipped == {}
        assert stored(archive_path) == [("attachment", "", "none"), ("attachment", "\n", "column")]
        with Archive.open(archive_path) as archive:
            assert archive.count() == 2
            assert [message["attachments"] for message in archive.messages(NORA)["messages"]] == [[], []]

    def test_a_caption_keeps_its_text_without_the_placeholder(
        self, builder: ChatDbBuilder, chat: tuple[int, int], source_path: Path, archive_path: Path
    ) -> None:
        with_caption = builder.message(chat[0], when=START, body=blob("attachment_with_caption"), handle=chat[1])
        builder.attachment(with_caption)
        builder.commit()

        import_chat_db(source_path, archive_path)

        assert stored(archive_path) == [
            ("text", expected_text("attachment_with_caption").replace(PLACEHOLDER, ""), "blob")
        ]

    def test_a_row_with_no_send_date_names_the_date_it_used(
        self, builder: ChatDbBuilder, chat: tuple[int, int], source_path: Path, archive_path: Path
    ) -> None:
        builder.message(chat[0], when=START, text="dated", handle=chat[1])
        builder.message(
            chat[0],
            when=0,
            text="delivered only",
            handle=chat[1],
            extra={"date_delivered": apple_time(START + timedelta(hours=1))},
        )
        builder.commit()

        import_chat_db(source_path, archive_path)

        extras = dict(query(archive_path, "SELECT text, extra FROM messages"))
        assert extras["dated"] is None
        assert json.loads(extras["delivered only"]) == {"date_source": "date_delivered"}
        assert scalar(archive_path, "SELECT sent_ns FROM messages WHERE text = 'delivered only'") == int(
            (START + timedelta(hours=1)).timestamp() * 1_000_000_000
        )
