import sqlite3

import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from parse_imessage import (
    extract_text_from_attributed_body,
    extract_messages,
    list_contacts,
    extract_attachments,
    extract_reactions,
    search_contacts,
)


class TestExtractTextFromAttributedBody:
    """The archives in tests/fixtures/typedstream come from Apple's NSArchiver
    (scripts/make_attributed_body_fixtures.swift), not from a writer in this repo:
    a test that builds its own archive repeats the mistake of the decoder."""

    # each of these texts must come back exactly as it was archived
    WHOLE = [
        "ascii_1", "ascii_127", "ascii_128", "ascii_255", "ascii_256", "ascii_300",
        "ascii_32767", "ascii_32768", "ascii_65535", "ascii_70000",
        "emoji_end", "emoji_end_130", "zwj_emoji", "emoji_sequences", "accented", "cjk",
        "newline", "spaces_both_ends", "newline_ends", "whitespace_only", "format_chars",
        "attachment_char", "multibyte_128", "multibyte_300",
        "mutable_ascii_1", "mutable_ascii_128", "mutable_ascii_300",
        "mutable_zwj_emoji", "mutable_spaces_both_ends",
    ]
    # these have a result of their own, each in its own test
    OTHER = ["control_chars", "empty", "not_text"]

    @staticmethod
    def _fixture_dir():
        from pathlib import Path

        return Path(__file__).parent / "fixtures" / "typedstream"

    @classmethod
    def _manifest(cls):
        import json

        text = (cls._fixture_dir() / "manifest.json").read_text(encoding="utf-8")
        return {entry["name"]: entry for entry in json.loads(text)["fixtures"]}

    @classmethod
    def _blob(cls, name):
        return bytes.fromhex((cls._fixture_dir() / f"{name}.hex").read_text(encoding="ascii"))

    @classmethod
    def _stored(cls, name):
        """Text of a short fixture, as it went into the archive."""
        return bytes.fromhex(cls._manifest()[name]["utf8_hex"]).decode("utf-8")

    @staticmethod
    def _length_field_start(blob):
        return blob.index(b"+", blob.index(b"NSString")) + 1

    def test_none_returns_none(self):
        assert extract_text_from_attributed_body(None) is None

    def test_empty_bytes_returns_none(self):
        assert extract_text_from_attributed_body(b"") is None

    def test_nsstring_short_length(self):
        # Build a blob with NSString marker + length byte + text
        text = b"Hello world"
        blob = b"\x00\x00NSString\x01+" + bytes([len(text)]) + text + b"\x00\x00"
        result = extract_text_from_attributed_body(blob)
        assert result == "Hello world"

    def test_every_fixture_has_a_test(self):
        names = set(self._manifest())
        assert names == {path.stem for path in self._fixture_dir().glob("*.hex")}
        assert names == set(self.WHOLE) | set(self.OTHER)

    @pytest.mark.parametrize("name", WHOLE)
    def test_apple_archive_comes_back_whole(self, name):
        import hashlib

        entry = self._manifest()[name]
        result = extract_text_from_attributed_body(self._blob(name))
        assert result is not None
        if entry["utf8_hex"] is not None:
            assert result == self._stored(name)
        data = result.encode("utf-8")
        assert len(data) == entry["utf8_bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]

    @pytest.mark.parametrize("name, length_field", [
        ("ascii_127", "7f"),
        ("ascii_128", "818000"),
        ("ascii_255", "81ff00"),
        ("ascii_256", "810001"),
        ("ascii_300", "812c01"),
        ("ascii_32767", "81ff7f"),
        ("ascii_32768", "8200800000"),
        ("ascii_65535", "82ffff0000"),
        ("ascii_70000", "8270110100"),
    ])
    def test_apple_writes_the_length_low_byte_first(self, name, length_field):
        """The bytes after "+" in Apple's archive. 0x81 is followed by two bytes
        and 0x82 by four: it is not "this many length bytes follow"."""
        blob = self._blob(name)
        start = self._length_field_start(blob)
        assert blob[start:start + len(length_field) // 2].hex() == length_field

    def test_a_300_byte_text_is_not_cut_to_43_bytes(self):
        """The old decoder read 0x81 as "one length byte follows". It took the
        length from the low byte (0x2c = 44), started one byte early, and kept
        (300 mod 256) - 1 = 43 bytes. From 128 to 255 bytes it lost the last byte."""
        stored = self._stored("ascii_300")
        result = extract_text_from_attributed_body(self._blob("ascii_300"))
        assert result != stored[:43]
        assert result == stored
        assert len(result.encode("utf-8")) == 300

    def test_joiners_and_selectors_are_kept(self):
        result = extract_text_from_attributed_body(self._blob("zwj_emoji"))
        assert result == self._stored("zwj_emoji")
        assert result.count(chr(0x200D)) == 4  # zero-width joiner
        assert result.count(chr(0xFE0F)) == 2  # variation selector-16

    def test_format_characters_are_kept(self):
        result = extract_text_from_attributed_body(self._blob("format_chars"))
        kept = [0x200B, 0x200E, 0x200F, 0x2060, 0x00AD, 0x00A0, 0x2066, 0x2069, 0x2028, 0x202B, 0x202C]
        for mark in map(chr, kept):
            assert mark in result, f"U+{ord(mark):04X} was dropped"

    @pytest.mark.parametrize(
        "name", ["spaces_both_ends", "newline_ends", "whitespace_only", "mutable_spaces_both_ends"]
    )
    def test_ends_are_not_trimmed(self, name):
        stored = self._stored(name)
        assert stored != stored.strip()
        assert extract_text_from_attributed_body(self._blob(name)) == stored

    def test_only_c0_controls_are_dropped(self):
        """NUL, U+0001, U+001F and the carriage return go. Tab, newline and U+007F stay."""
        assert self._stored("control_chars") == "a\x00b\x01c\x1fd\te\nf\rg\x7fh"
        assert extract_text_from_attributed_body(self._blob("control_chars")) == "abcd\te\nfg\x7fh"

    def test_empty_text_is_none(self):
        assert extract_text_from_attributed_body(self._blob("empty")) is None

    def test_blob_path_gives_the_same_text_as_the_text_column(self, mock_chat_db):
        names = [
            "ascii_300", "spaces_both_ends", "newline_ends", "whitespace_only",
            "zwj_emoji", "format_chars", "attachment_char", "cjk",
        ]
        base_ns = 726915600 * 1_000_000_000
        conn = sqlite3.connect(mock_chat_db)
        for index, name in enumerate(names):
            plain_id, blob_id = 100 + 2 * index, 101 + 2 * index
            date = base_ns + (10 + index) * 60_000_000_000
            conn.execute(
                "INSERT INTO message VALUES (?, ?, 1, ?, NULL, 0, 'iMessage', NULL, 0)",
                (plain_id, date, self._stored(name)),
            )
            conn.execute(
                "INSERT INTO message VALUES (?, ?, 1, NULL, ?, 0, 'iMessage', NULL, 0)",
                (blob_id, date, self._blob(name)),
            )
            conn.executemany("INSERT INTO chat_message_join VALUES (1, ?)", [(plain_id,), (blob_id,)])
        conn.commit()
        conn.close()

        texts = extract_messages(mock_chat_db).set_index("message_id")["text"]

        for index, name in enumerate(names):
            assert texts[101 + 2 * index] == texts[100 + 2 * index] == self._stored(name)

    def test_decoded_text_survives_the_export_csv(self, tmp_path):
        """merge_export compares text read from the CSV with text from the
        decoder, so what the decoder returns must come back from the CSV as it was."""
        names = [name for name in self.WHOLE if self._manifest()[name]["utf8_hex"] is not None]
        names.append("control_chars")
        decoded = [extract_text_from_attributed_body(self._blob(name)) for name in names]
        csv_path = tmp_path / "messages.csv"
        pd.DataFrame({"message_id": range(len(decoded)), "text": decoded}).to_csv(csv_path, index=False)

        read_back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)["text"].tolist()

        assert read_back == decoded

    def test_fallback_reads_a_blob_that_is_plain_text(self):
        # No NSString marker: the whole blob is the text, and it comes back as it is
        assert extract_text_from_attributed_body(b"This is a fallback test") == "This is a fallback test"
        text = " d\xe9j\xe0 vu \U00002014 \U00004eca\U00005929 \U0001f600 ok\n"
        assert extract_text_from_attributed_body(text.encode("utf-8")) == text

    def test_fallback_never_takes_a_run_of_text_out_of_a_blob(self):
        """The old decoder returned the longest run of ASCII in blobs like these.
        A part of a blob reads as a whole message, so the answer is None."""
        framed = b"\x00\x01streamtyped\x00\x02This is a fallback test\x00\x03"
        cut_in_a_character = b"hello\xc3"
        mostly_binary = bytes(range(128, 256)) * 2 + b"only a small readable part"
        text_with_a_control = b"first part\x0bsecond part"
        for blob in (framed, cut_in_a_character, mostly_binary, text_with_a_control):
            assert extract_text_from_attributed_body(blob) is None

    def test_fallback_refuses_text_over_the_bound(self):
        # 4,096 bytes pass. A longer blob is not cut to fit.
        at_the_bound = b"word " * 819 + b"x"
        assert len(at_the_bound) == 4096
        assert extract_text_from_attributed_body(at_the_bound) == at_the_bound.decode("ascii")
        assert extract_text_from_attributed_body(at_the_bound + b"x") is None
        assert extract_text_from_attributed_body(b"word " * 20000) is None

    def test_fallback_refuses_text_with_no_letter_or_digit_or_one_character(self):
        for blob in (b"?!... --", b"   \n  ", b"a", b" a "):
            assert extract_text_from_attributed_body(blob) is None

    def test_unparseable_returns_none(self):
        blob = b"\x00\x01\x02\x03\x04\x05"
        result = extract_text_from_attributed_body(blob)
        assert result is None

    def test_a_typedstream_with_no_string_is_none(self):
        """Apple's archive of two numbers: a class name is not the text of a message."""
        assert extract_text_from_attributed_body(self._blob("not_text")) is None

    def test_fallback_is_not_used_when_the_blob_has_the_marker(self):
        """A blob with the marker gives its text or nothing. The old decoder
        returned the longest run of ASCII from anywhere in such a blob."""
        blob = self._blob("ascii_300")
        start = self._length_field_start(blob)
        cut_in_the_text = blob[:start + 150]
        bad_length_tag = blob[:start] + b"\x83" + blob[start + 1:]
        text_is_not_utf8 = blob[:start + 3] + b"\xff\xfe" + blob[start + 5:]

        assert extract_text_from_attributed_body(cut_in_the_text) is None
        assert extract_text_from_attributed_body(bad_length_tag) is None
        assert extract_text_from_attributed_body(text_is_not_utf8) is None

    @pytest.mark.parametrize(
        "name", ["ascii_1", "zwj_emoji", "ascii_300", "multibyte_300", "mutable_ascii_300"]
    )
    def test_a_cut_archive_gives_the_whole_text_or_none(self, name):
        blob = self._blob(name)
        results = {extract_text_from_attributed_body(blob[:size]) for size in range(len(blob) + 1)}
        assert results == {None, self._stored(name)}

    @pytest.mark.parametrize("bad", ["not bytes", 12345, 10**12, 3.5, float("nan"), object(), ["a", "list"]])
    def test_wrong_input_type_is_none(self, bad):
        assert extract_text_from_attributed_body(bad) is None

    def test_bytearray_and_memoryview_are_read(self):
        blob = self._blob("zwj_emoji")
        assert extract_text_from_attributed_body(bytearray(blob)) == self._stored("zwj_emoji")
        assert extract_text_from_attributed_body(memoryview(blob)) == self._stored("zwj_emoji")

    def test_random_bytes_give_no_text(self):
        import random

        rng = random.Random(20261116)
        for size in (64, 300, 5000, 20000):
            for _ in range(40):
                assert extract_text_from_attributed_body(rng.randbytes(size)) is None

    def test_a_damaged_archive_never_raises(self):
        import random

        rng = random.Random(7)
        blob = self._blob("mutable_ascii_300")
        for _ in range(500):
            damaged = bytearray(blob)
            for _ in range(rng.randint(1, 6)):
                damaged[rng.randrange(len(damaged))] = rng.randrange(256)
            result = extract_text_from_attributed_body(bytes(damaged))
            assert result is None or isinstance(result, str)

    def test_a_buffer_that_cannot_be_read_is_none(self):
        view = memoryview(self._blob("ascii_1"))
        view.release()
        assert extract_text_from_attributed_body(view) is None

    def test_owner_check_counts_where_the_old_rule_differs(self, tmp_path, capsys):
        """scripts/verify_decoder_counts.py keeps a copy of the old rule and
        prints numbers only. Here it reads a database of Apple's archives."""
        import importlib.util
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "verify_decoder_counts.py"
        spec = importlib.util.spec_from_file_location("verify_decoder_counts", script)
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)

        names = [
            "ascii_127",         # the two rules agree
            "spaces_both_ends",  # the old rule trimmed the ends
            "zwj_emoji",         # the old rule dropped the joiners
            "whitespace_only",   # the old rule gave a name from inside the archive
            "ascii_128",         # the old rule lost the last byte
            "emoji_end_130",     # the same, and that split the emoji
            "ascii_300",         # the old rule kept 43 bytes
            "ascii_70000",       # the old rule kept 28,687 bytes
            "empty",             # None from the two rules
            "not_text",          # no marker: the old rule gave a class name
        ]
        db_path = tmp_path / "chat.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB)")
        conn.executemany(
            "INSERT INTO message (text, attributedBody) VALUES (NULL, ?)",
            [(self._blob(name),) for name in names],
        )
        # not checked: a row that has text, and a row that has no blob
        conn.execute("INSERT INTO message (text, attributedBody) VALUES ('has text', ?)", (self._blob("ascii_300"),))
        conn.execute("INSERT INTO message (text, attributedBody) VALUES (NULL, NULL)")
        conn.commit()
        conn.close()
        before = db_path.read_bytes()

        assert check.main(["--db", str(db_path)]) == 0
        printed = capsys.readouterr().out
        conn = check.open_read_only(db_path)
        counts = check.tally(check.read_blobs(conn))
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM message")
        conn.close()

        assert check.old_extract_text(self._blob("ascii_300")) == self._stored("ascii_300")[:43]
        assert {key: count for key, count in counts.items() if count} == {
            "checked": 10,
            "differ": 8,
            "old none": 1,
            "new none": 2,
            "no marker": 1,
            "differ, new none": 1,
            "under 128": 3,
            ("under 128", "only the ends"): 1,
            ("under 128", "characters dropped"): 1,
            ("under 128", "cut or fragment"): 1,
            "128 to 255": 2,
            ("128 to 255", "cut or fragment"): 2,
            "256 and more": 2,
            ("256 and more", "cut or fragment"): 2,
        }
        assert "Rows with text NULL and attributedBody not NULL: 10" in printed
        assert "Rows where the old rule and the new rule differ: 8" in printed
        assert "Rows where the new rule gives None: 2" in printed
        # numbers only: no text of a message, and the database is as it was
        archived = [self._stored(name).strip() for name in names if self._manifest()[name]["utf8_hex"]]
        starts = [text[:8] for text in archived if len(text) >= 8]
        assert len(starts) == 6
        assert not any(start in printed for start in starts)
        assert db_path.read_bytes() == before

        assert check.main(["--db", str(tmp_path / "missing.db")]) == 1


class TestExtractMessages:
    def test_returns_dataframe(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        for col in ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me", "service"]:
            assert col in df.columns

    def test_attributedBody_column_dropped(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        assert "attributedBody" not in df.columns

    def test_text_messages_extracted(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        texts = df["text"].tolist()
        assert "hello there" in texts
        assert "hey!" in texts

    def test_seconds_timestamp_supported(self, mock_chat_db):
        conn = sqlite3.connect(mock_chat_db)
        conn.execute("UPDATE message SET date = ? WHERE ROWID = 1", (726915600,))
        conn.commit()
        conn.close()

        df = extract_messages(mock_chat_db)
        row = df[df["message_id"] == 1].iloc[0]
        assert pd.to_datetime(row["timestamp"]).year == 2024

    def test_connection_closed_on_error(self, tmp_path):
        bad_db = str(tmp_path / "bad.db")
        # Create empty DB (no tables)
        import sqlite3
        conn = sqlite3.connect(bad_db)
        conn.close()
        with pytest.raises(Exception):
            extract_messages(bad_db)


class TestListContacts:
    def test_returns_dataframe(self, mock_chat_db):
        df = list_contacts(mock_chat_db)
        assert isinstance(df, pd.DataFrame)

    def test_contact_found(self, mock_chat_db):
        df = list_contacts(mock_chat_db)
        assert not df.empty
        assert "+15551234567" in df["chat_id"].values

    def test_limit_works(self, mock_chat_db):
        df = list_contacts(mock_chat_db, limit=1)
        assert len(df) <= 1

    def test_a_group_is_an_id_made_of_chat_and_digits(self, mock_chat_db):
        conn = sqlite3.connect(mock_chat_db)
        base_ns = 726915600 * 1_000_000_000
        chats = ["chat4471", "chatterbox@example.com", "chat12a", "chat", "Chat99"]
        for offset, identifier in enumerate(chats, start=2):
            conn.execute("INSERT INTO chat VALUES (?, ?)", (offset, identifier))
            conn.execute(
                "INSERT INTO message VALUES (?, ?, 1, 'hi', NULL, 0, 'iMessage', NULL, 0)",
                (100 + offset, base_ns + offset * 60_000_000_000),
            )
            conn.execute("INSERT INTO chat_message_join VALUES (?, ?)", (offset, 100 + offset))
        conn.commit()
        conn.close()

        groups = list_contacts(mock_chat_db, limit=20).set_index("chat_id")["is_group"].to_dict()

        # an address that only starts with "chat" is one person, the same as everywhere else
        assert groups == {
            "+15551234567": 0,
            "chat4471": 1,
            "chatterbox@example.com": 0,
            "chat12a": 0,
            "chat": 0,
            "Chat99": 0,
        }

    def test_columns_present(self, mock_chat_db):
        df = list_contacts(mock_chat_db)
        for col in ["chat_id", "display_name", "message_count", "first_msg", "last_msg", "is_group"]:
            assert col in df.columns

    def test_group_display_name_included(self, mock_chat_db):
        conn = sqlite3.connect(mock_chat_db)
        conn.execute("ALTER TABLE chat ADD COLUMN display_name TEXT")
        base_ns = 726915600 * 1_000_000_000
        conn.execute("INSERT INTO chat (ROWID, chat_identifier, display_name) VALUES (2, 'chat123', 'Road Trip')")
        conn.execute(
            "INSERT INTO message VALUES (4, ?, 1, 'group hello', NULL, 0, 'iMessage', NULL, 0)",
            (base_ns + 180_000_000_000,),
        )
        conn.execute("INSERT INTO chat_message_join VALUES (2, 4)")
        conn.commit()
        conn.close()

        df = list_contacts(mock_chat_db, limit=10)

        row = df[df["chat_id"] == "chat123"].iloc[0]
        assert row["display_name"] == "Road Trip"
        assert row["is_group"] == 1


class TestExtractAttachments:
    def test_returns_dataframe(self, mock_chat_db):
        df = extract_attachments(mock_chat_db)
        assert isinstance(df, pd.DataFrame)

    def test_category_column_added(self, mock_chat_db):
        df = extract_attachments(mock_chat_db)
        if not df.empty:
            assert "category" in df.columns

    def test_photo_categorized(self, mock_chat_db):
        df = extract_attachments(mock_chat_db)
        if not df.empty:
            assert "photo" in df["category"].values


class TestExtractReactions:
    def test_returns_dataframe(self, mock_chat_db):
        df = extract_reactions(mock_chat_db)
        assert isinstance(df, pd.DataFrame)

    def test_reaction_columns(self, mock_chat_db):
        df = extract_reactions(mock_chat_db)
        if not df.empty:
            assert "reaction_type" in df.columns
            assert "is_add" in df.columns

    def test_love_reaction_decoded(self, mock_chat_db):
        df = extract_reactions(mock_chat_db)
        if not df.empty:
            assert "love" in df["reaction_type"].values

    def test_non_tapback_associated_message_ignored(self, mock_chat_db):
        conn = sqlite3.connect(mock_chat_db)
        base_ns = 726915600 * 1_000_000_000
        conn.execute(
            "INSERT INTO message VALUES (4, ?, 1, NULL, NULL, 0, 'iMessage', 'guid2', 2500)",
            (base_ns + 180_000_000_000,),
        )
        conn.execute("INSERT INTO chat_message_join VALUES (1, 4)")
        conn.commit()
        conn.close()

        df = extract_reactions(mock_chat_db)
        assert 2500 not in set(df["associated_message_type"].astype(int))


class TestSearchContacts:
    def test_search_by_partial(self, mock_chat_db):
        df = search_contacts(mock_chat_db, "5551234")
        assert not df.empty

    def test_search_no_match(self, mock_chat_db):
        df = search_contacts(mock_chat_db, "9999999999")
        assert df.empty
