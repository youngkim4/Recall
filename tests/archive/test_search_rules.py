"""Search rules that the review added: words next to emoji, terms with symbols, limits, oversize texts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from archive import INDEXED_TEXT_CHARS, Archive, import_chat_db
from archive.search_query import MAX_QUERY_CHARS, MAX_TERMS, parse_query
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.inspect_archive import fts_is_consistent, scalar

START = utc(2024, 8, 1, 16, 0)
IVY = "+15550100161"
NEW_EMOJI = chr(0x1F979)  # added in Unicode 14: the tokenizer of SQLite does not know it
OLD_EMOJI = chr(0x1F600)  # in Unicode 6.1
Opened = Callable[[list[str]], Archive]


@pytest.fixture
def archive_of(builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> Opened:
    """Import the given texts as one conversation and open the archive."""
    opened: list[Archive] = []

    def make(texts: list[str]) -> Archive:
        handle = builder.handle(IVY)
        chat = builder.chat(IVY, members=[handle])
        for index, text in enumerate(texts):
            builder.message(chat, when=START + timedelta(minutes=index), text=text, handle=handle)
        builder.commit()
        import_chat_db(source_path, archive_path)
        opened.append(Archive.open(archive_path))
        return opened[-1]

    yield make  # type: ignore[misc]
    for archive in opened:
        archive.close()


def found(archive: Archive, query: str) -> list[str]:
    result = archive.search(query, order="oldest")
    assert archive.count(text=query) == result["total"] == len(result["hits"])
    return [hit["message"]["text"] for hit in result["hits"]]


class TestWordsNextToEmoji:
    def test_a_word_glued_to_an_emoji_is_found_by_the_word(self, archive_of: Opened) -> None:
        texts = [f"ok{NEW_EMOJI}", f"great{OLD_EMOJI}", f"caf{chr(0xE9)}{chr(0x1FAE0)}later", "plain ok"]
        archive = archive_of(texts)

        assert found(archive, "ok") == [texts[0], texts[3]]
        assert found(archive, "great") == [texts[1]]
        assert found(archive, "cafe") == [texts[2]]
        assert found(archive, "later") == [texts[2]]

    def test_an_emoji_alone_is_found_as_a_substring_with_its_range(self, archive_of: Opened) -> None:
        archive = archive_of([f"ok{NEW_EMOJI}", "no emoji here"])

        result = archive.search(NEW_EMOJI)

        assert [hit["message"]["text"] for hit in result["hits"]] == [f"ok{NEW_EMOJI}"]
        assert result["hits"][0]["ranges"] == [[2, 3]]

    def test_a_term_with_an_emoji_needs_the_emoji(self, archive_of: Opened) -> None:
        texts = ["hello world", f"hello{OLD_EMOJI}", f"hello {OLD_EMOJI}", f"well hello{NEW_EMOJI} there"]
        archive = archive_of(texts)

        assert found(archive, f"hello{OLD_EMOJI}") == [texts[1]]
        assert found(archive, f'"hello {OLD_EMOJI}"') == [texts[2]]
        assert found(archive, f"hello{NEW_EMOJI}") == [texts[3]]
        assert found(archive, "hello") == texts

    def test_a_term_with_a_symbol_needs_the_symbol(self, archive_of: Opened) -> None:
        texts = ["c++ is fine", "plan c is fine", "a $5 tip", "5 apples"]
        archive = archive_of(texts)

        assert found(archive, "c++") == [texts[0]]
        assert found(archive, "$5") == [texts[2]]
        assert found(archive, "5") == [texts[2], texts[3]]

    def test_an_apostrophe_is_not_a_symbol(self, archive_of: Opened) -> None:
        texts = [f"don{chr(0x2019)}t stop", "don't go"]
        archive = archive_of(texts)

        assert found(archive, "don't") == texts  # a straight or a curly apostrophe: the same words

    def test_a_letter_that_the_index_does_not_know_is_a_substring(self, archive_of: Opened) -> None:
        sign = chr(0x19B0)  # New Tai Lue vowel sign: a letter for Python, a separator for the index
        archive = archive_of([f"mark {sign} here", "plain text"])

        assert found(archive, sign) == [f"mark {sign} here"]
        assert found(archive, f"{sign} plain") == []


class TestRangesOfTermsWithSymbols:
    def test_only_the_whole_term_is_marked(self, archive_of: Opened) -> None:
        text = "I like c and c++ and $5 for 5"
        archive = archive_of([text])

        def marked(query: str) -> list[str]:
            return [text[start:end] for start, end in archive.search(query)["hits"][0]["ranges"]]

        assert marked("c++") == ["c++"]
        assert marked("$5") == ["$5"]
        assert marked("like $5") == ["like", "$5"]
        assert marked("c") == ["c", "c"]


class TestLimits:
    def test_a_query_is_cut_to_its_first_terms(self, archive_of: Opened) -> None:
        words = [f"word{index}" for index in range(MAX_TERMS)]
        archive = archive_of([" ".join(words)])

        assert found(archive, " ".join([*words, "absent"])) == [" ".join(words)]  # term 33 is not read
        assert found(archive, " ".join(["absent", *words])) == []

    def test_a_long_query_is_cut_and_does_not_raise(self, archive_of: Opened) -> None:
        archive = archive_of(["needle in a haystack"])

        assert found(archive, "needle " + "x" * 50_000) == []
        assert found(archive, "needle" + " " * MAX_QUERY_CHARS + "absent") == ["needle in a haystack"]

    def test_thousands_of_substring_terms_do_not_raise(self, archive_of: Opened) -> None:
        archive = archive_of(["plain text"])
        many = " ".join(f"{chr(0x6F22)}{index}" for index in range(2_000))

        assert archive.search(many) == {"total": 0, "hits": []}
        assert archive.count(text=many) == 0
        assert len(parse_query(many).substrings) <= MAX_TERMS

    def test_a_text_with_marker_characters_still_gets_ranges(self, archive_of: Opened) -> None:
        text = f"hello {chr(0xFDD0)}x{chr(0xFDD2)}y{chr(0xFDD4)}z\x01"
        archive = archive_of([text])

        assert archive.search("hello")["hits"][0]["ranges"] == [[0, 5]]


class TestOversizeTexts:
    HEAD = "harbor " * 10
    TAIL = "lighthouse"

    def long_text(self) -> str:
        return self.HEAD + "filler " * 4_000 + self.TAIL

    def test_only_the_first_part_is_searched_and_the_text_stays_whole(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        handle = builder.handle(IVY)
        chat = builder.chat(IVY, members=[handle])
        builder.message(chat, when=START, text=self.long_text(), handle=handle)
        builder.message(chat, when=START + timedelta(minutes=1), text="a short lighthouse note", handle=handle)
        builder.commit()

        result = import_chat_db(source_path, archive_path)

        assert len(self.long_text()) > INDEXED_TEXT_CHARS
        assert result.oversize_texts == 1
        with Archive.open(archive_path) as archive:
            assert archive.messages(IVY, oldest=True, limit=1)["messages"][0]["text"] == self.long_text()
            assert found(archive, "harbor") == [self.long_text()]
            assert found(archive, "lighthouse") == ["a short lighthouse note"]
            ranges = archive.search("harbor")["hits"][0]["ranges"]
            assert ranges[0] == [0, 6] and all(end <= INDEXED_TEXT_CHARS for _, end in ranges)
        assert fts_is_consistent(archive_path)

    def test_a_change_past_the_searched_part_and_a_removal_keep_the_index_right(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        handle = builder.handle(IVY)
        chat = builder.chat(IVY, members=[handle])
        long_row = builder.message(chat, when=START, text=self.long_text(), handle=handle)
        for index in range(1, 9):
            builder.message(chat, when=START + timedelta(minutes=index), text=f"short {index}", handle=handle)
        builder.commit()
        import_chat_db(source_path, archive_path)

        builder.raw.execute("UPDATE message SET text = ? WHERE ROWID = ?", (self.long_text() + " changed", long_row))
        builder.commit()
        assert import_chat_db(source_path, archive_path).messages_updated == 1
        assert fts_is_consistent(archive_path)
        assert (
            scalar(archive_path, "SELECT length(text) FROM messages ORDER BY id LIMIT 1") == len(self.long_text()) + 8
        )

        builder.raw.execute("DELETE FROM chat_message_join WHERE message_id = ?", (long_row,))
        builder.raw.execute("DELETE FROM message WHERE ROWID = ?", (long_row,))
        builder.commit()
        assert import_chat_db(source_path, archive_path).messages_removed == 1
        assert fts_is_consistent(archive_path)
        with Archive.open(archive_path) as archive:
            assert archive.search("harbor")["total"] == 0
