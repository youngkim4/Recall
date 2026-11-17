"""search(): words, phrases, prefixes, diacritics, bad syntax, paging, filters, ranges."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archive import Archive, import_chat_db
from archive.search import parse_query, utf16_ranges
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.conftest import Harbor

A, B, G = worlds.ROWAN_PHONE, worlds.ROWAN_EMAIL, worlds.HARBOR_GROUP_ID


def found(harbor: Harbor, query: str, **options: object) -> list[str]:
    return harbor.labels([hit["message"] for hit in harbor.archive.search(query, **options)["hits"]])  # type: ignore[arg-type]


def marked(hit: dict[str, object]) -> list[str]:
    text = hit["message"]["text"]  # type: ignore[index]
    return [text[start:end] for start, end in hit["ranges"]]  # type: ignore[union-attr]


class TestWords:
    def test_words_are_and_ed(self, harbor: Harbor) -> None:
        assert found(harbor, "clocks change") == ["g16", "a4"]
        assert found(harbor, "clocks tonight") == ["a4"]
        assert found(harbor, "clocks brunch") == []

    def test_case_does_not_matter(self, harbor: Harbor) -> None:
        assert found(harbor, "BRUNCH") == ["a7"]
        assert found(harbor, "bRuNcH") == ["a7"]

    def test_diacritics_do_not_matter(self, harbor: Harbor) -> None:
        assert found(harbor, "cafe") == ["b11"]
        assert found(harbor, "café") == ["b11"]
        assert found(harbor, "deja vu") == ["b12"]
        assert found(harbor, "DÉJÀ") == ["b12"]

    def test_quoted_text_is_a_phrase(self, harbor: Harbor) -> None:
        assert found(harbor, '"clocks change"') == ["g16", "a4"]
        assert found(harbor, '"change clocks"') == []
        assert found(harbor, '"year almost"') == ["g13"]
        assert found(harbor, '"happy new" almost') == ["g13"]

    def test_curly_quotes_are_quotes(self, harbor: Harbor) -> None:
        assert found(harbor, "“change clocks”") == []
        assert found(harbor, "“clocks change”") == ["g16", "a4"]

    def test_trailing_star_is_a_prefix(self, harbor: Harbor) -> None:
        assert found(harbor, "clo*") == ["g16", "a4"]
        assert found(harbor, "happ*") == ["g15", "g14", "g13"]
        assert found(harbor, "clo") == []
        assert found(harbor, '"happy new ye"*') == ["g15", "g14", "g13"]
        assert found(harbor, "bru* rowan") == []

    def test_apostrophes_straight_or_curly(self, harbor: Harbor) -> None:
        assert found(harbor, "can't sleep") == ["a6"]
        assert found(harbor, "can’t sleep") == ["a6"]


class TestTextThatTheIndexDoesNotSplit:
    def test_emoji(self, harbor: Harbor) -> None:
        assert found(harbor, "🎉") == ["g15"]
        assert found(harbor, "☕") == ["b11"]
        assert found(harbor, "year 🎉") == ["g15"]
        assert found(harbor, "🎂") == []

    def test_chinese_japanese_korean(self, harbor: Harbor) -> None:
        assert found(harbor, "寿司") == ["g18"]
        assert found(harbor, "食べました") == ["g18"]
        assert found(harbor, "학교") == ["g19"]
        assert found(harbor, "오늘 만나요") == ["g19"]
        assert found(harbor, "학교 寿司") == []

    def test_punctuation_only(self, harbor: Harbor) -> None:
        assert found(harbor, "?") == ["a7", "a1"]
        assert found(harbor, "!") == ["g14"]


class TestBadSyntaxNeverRaises:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("brunch AND (", []),
            ("brunch OR sleep", []),
            ("NOT brunch", []),
            ('"brunch', ["a7"]),
            ('brunch"', ["a7"]),
            ('"unbalanced quote brunch', []),
            ("NEAR(brunch sleep)", []),
            ("^brunch", []),  # a symbol in a term must be in the text
            ("text:brunch", []),
            ("brunch -sleep", []),
            ("(brunch)", ["a7"]),
            ("brunch*", ["a7"]),
            ("*brunch", ["a7"]),
            ("br*nch", []),
            ("brunch +", []),
            ("{brunch}", ["a7"]),
            ("brunch\x00", ["a7"]),
            ("brunch\n\t", ["a7"]),
        ],
    )
    def test_odd_input_is_plain_words(self, harbor: Harbor, query: str, expected: list[str]) -> None:
        assert found(harbor, query) == expected

    @pytest.mark.parametrize("query", ["", "   ", "*", '""', '"" ""', "***", '"', "“”"])
    def test_query_without_words_finds_nothing(self, harbor: Harbor, query: str) -> None:
        assert harbor.archive.search(query) == {"total": 0, "hits": []}

    def test_every_special_character_alone(self, harbor: Harbor) -> None:
        for char in "\"'()[]{}*^-+:;,.<>~`!@#$%&|\\/=_":
            for query in (char, f"{char}brunch", f"brunch{char}", f"brunch {char} sleep"):
                result = harbor.archive.search(query)
                assert set(result) == {"total", "hits"}
                assert result["total"] >= len(result["hits"])

    def test_query_that_is_not_text_raises_type_error(self, harbor: Harbor) -> None:
        with pytest.raises(TypeError):
            harbor.archive.search(None)  # type: ignore[arg-type]


class TestPaging:
    def test_total_is_every_match_and_hits_is_the_page(self, harbor: Harbor) -> None:
        first = harbor.archive.search("happy", limit=2)
        assert first["total"] == 3 and len(first["hits"]) == 2
        second = harbor.archive.search("happy", limit=2, offset=2)
        assert second["total"] == 3 and len(second["hits"]) == 1
        beyond = harbor.archive.search("happy", limit=2, offset=5)
        assert beyond == {"total": 3, "hits": []}
        only_total = harbor.archive.search("happy", limit=0)
        assert only_total == {"total": 3, "hits": []}

    def test_pages_do_not_overlap(self, harbor: Harbor) -> None:
        everything = found(harbor, "the OR a OR happy")  # plain words: nothing has all of them
        assert everything == []
        pages = [found(harbor, "happy", limit=1, offset=offset) for offset in range(3)]
        assert pages == [["g15"], ["g14"], ["g13"]]

    def test_default_limit_is_fifty(self, builder: ChatDbBuilder, source_path: Path, archive_path: Path) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        for index in range(120):
            builder.message(
                chat, when=utc(2024, 1, 1, 12, index // 60, index % 60), text=f"ping number {index}", handle=rowan
            )
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            result = archive.search("ping")
            assert result["total"] == 120 and len(result["hits"]) == 50
            assert result["hits"][0]["message"]["text"] == "ping number 119"
            assert (
                archive.search("ping", order="oldest", limit=3, offset=118)["hits"][-1]["message"]["text"]
                == "ping number 119"
            )

    @pytest.mark.parametrize("bad", [{"limit": -1}, {"offset": -1}, {"order": "random"}, {"limit": "5"}])
    def test_bad_arguments_raise_value_error(self, harbor: Harbor, bad: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            harbor.archive.search("happy", **bad)  # type: ignore[arg-type]


class TestOrder:
    def test_newest_first_by_default_and_oldest_on_request(self, harbor: Harbor) -> None:
        assert found(harbor, "happy") == ["g15", "g14", "g13"]
        assert found(harbor, "happy", order="newest") == ["g15", "g14", "g13"]
        assert found(harbor, "happy", order="oldest") == ["g13", "g14", "g15"]

    def test_relevance_puts_the_better_match_first(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        builder.message(chat, when=utc(2024, 1, 1, 12), text="tea tea tea", handle=rowan)
        builder.message(
            chat,
            when=utc(2024, 1, 2, 12),
            text="we could have tea with milk and sugar after the long walk home",
            handle=rowan,
        )
        builder.message(chat, when=utc(2024, 1, 3, 12), text="no match here", handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            by_relevance = [hit["message"]["text"] for hit in archive.search("tea", order="relevance")["hits"]]
            newest = [hit["message"]["text"] for hit in archive.search("tea")["hits"]]
        assert by_relevance[0] == "tea tea tea" and len(by_relevance) == 2
        assert newest[0].startswith("we could have tea")

    def test_relevance_without_indexed_words_falls_back_to_newest(self, harbor: Harbor) -> None:
        assert found(harbor, "?", order="relevance") == ["a7", "a1"]


class TestFilters:
    def test_conversations(self, harbor: Harbor) -> None:
        assert found(harbor, "happy", conversations=[G]) == ["g15", "g14", "g13"]
        assert found(harbor, "happy", conversations=[A, B]) == []
        assert found(harbor, "clocks", conversations=[A]) == ["a4"]
        assert found(harbor, "clocks", conversations=[A, G]) == ["g16", "a4"]
        assert found(harbor, "clocks", conversations=[]) == []
        assert found(harbor, "clocks", conversations=["+15550109999"]) == []
        assert found(harbor, "clocks", conversations=A) == ["a4"]

    def test_days(self, harbor: Harbor) -> None:
        assert found(harbor, "happy", since="2024-01-01") == ["g15", "g14"]
        assert found(harbor, "happy", until="2023-12-31") == ["g13"]
        assert found(harbor, "happy", since="2024-01-01", until="2024-01-01") == ["g15", "g14"]
        assert found(harbor, "happy", since="2024-01-02") == []

    def test_direction(self, harbor: Harbor) -> None:
        assert found(harbor, "happy", from_me=True) == ["g15"]
        assert found(harbor, "happy", from_me=False) == ["g14", "g13"]
        assert harbor.archive.search("happy", from_me=False)["total"] == 2

    def test_group_events_only_on_request(self, harbor: Harbor) -> None:
        assert found(harbor, "harbor crew") == []
        assert found(harbor, "harbor crew", include_system=True) == ["g_renamed"]

    def test_tapback_text_is_never_found(self, harbor: Harbor) -> None:
        assert harbor.archive.search("loved")["total"] == 0
        assert harbor.archive.search("invented")["total"] == 0


class TestRanges:
    def test_each_word_is_marked(self, harbor: Harbor) -> None:
        hit = harbor.archive.search("clocks change", conversations=[A])["hits"][0]
        assert hit["ranges"] == [[0, 6], [7, 13]]
        assert marked(hit) == ["Clocks", "change"]

    def test_phrase_is_one_range(self, harbor: Harbor) -> None:
        hit = harbor.archive.search('"clocks change"', conversations=[A])["hits"][0]
        assert hit["ranges"] == [[0, 13]]

    def test_range_covers_the_accented_word(self, harbor: Harbor) -> None:
        hit = harbor.archive.search("cafe")["hits"][0]
        assert hit["ranges"] == [[27, 31]]
        assert marked(hit) == ["café"]

    def test_prefix_marks_the_whole_word(self, harbor: Harbor) -> None:
        assert marked(harbor.archive.search("bru*")["hits"][0]) == ["Brunch"]

    def test_emoji_and_script_ranges(self, harbor: Harbor) -> None:
        assert harbor.archive.search("🎉")["hits"][0]["ranges"] == [[15, 16]]
        assert marked(harbor.archive.search("寿司")["hits"][0]) == ["寿司"]
        assert marked(harbor.archive.search("year 🎉")["hits"][0]) == ["year", "🎉"]
        assert marked(harbor.archive.search("학교 오늘")["hits"][0]) == ["오늘", "학교"]

    def test_ranges_count_code_points_after_an_emoji(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        builder.message(chat, when=utc(2024, 1, 1, 12), text="🎉 party 🎉 time, party time", handle=rowan)
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            hit = archive.search("time")["hits"][0]
            both = archive.search("party time 🎉")["hits"][0]
        text = hit["message"]["text"]
        assert hit["ranges"] == [[10, 14], [22, 26]]
        assert utf16_ranges(text, hit["ranges"]) == [[12, 16], [24, 28]]
        assert marked(both) == ["🎉", "party", "🎉", "time", "party", "time"]
        assert both["ranges"] == sorted(both["ranges"])

    def test_hits_are_json(self, harbor: Harbor) -> None:
        result = harbor.archive.search("happy")
        json.dumps(result)
        assert list(result["hits"][0]) == ["message", "ranges"]
        assert result["hits"][0]["message"]["conversation"] == G

    def test_text_with_the_marker_characters_still_gives_right_ranges(
        self, builder: ChatDbBuilder, source_path: Path, archive_path: Path
    ) -> None:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        builder.message(
            chat, when=utc(2024, 1, 1, 12), text=f"odd {chr(0xFDD0)} text {chr(0xFDD1)} with markers", handle=rowan
        )
        builder.commit()
        import_chat_db(source_path, archive_path)
        with Archive.open(archive_path) as archive:
            hit = archive.search("markers text")["hits"][0]
        assert marked(hit) == ["text", "markers"]


class TestParseQuery:
    def test_words_phrases_and_prefixes(self) -> None:
        parsed = parse_query('ferry "at noon" tick*')
        assert parsed.match == '"ferry" "at noon" "tick" *'
        assert parsed.substrings == ()

    def test_terms_the_index_cannot_split(self) -> None:
        parsed = parse_query("LA맛집 🎉 ferry")
        assert parsed.match == '"ferry"'
        assert parsed.substrings == ("la맛집", "🎉")

    def test_empty(self) -> None:
        for text in ("", "  ", "*", '""'):
            assert parse_query(text).is_empty

    def test_quote_inside_a_term_cannot_break_out(self) -> None:
        parsed = parse_query('say "hi" OR 1=1 --')
        assert parsed.match == '"say" "hi" "OR" "1=1"'
        assert parsed.substrings == ("1=1", "--")  # a term with a symbol is also a substring


class TestUtf16Ranges:
    def test_plain_text_is_unchanged(self) -> None:
        assert utf16_ranges("hello world", [[0, 5], [6, 11]]) == [[0, 5], [6, 11]]

    def test_astral_characters_count_twice(self) -> None:
        assert utf16_ranges("🎉a🎉b", [[1, 2], [3, 4]]) == [[2, 3], [5, 6]]
        assert utf16_ranges("🎉", [[0, 1]]) == [[0, 2]]

    def test_empty(self) -> None:
        assert utf16_ranges("", []) == []
