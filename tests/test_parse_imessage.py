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

    def test_nsstring_long_length_ber(self):
        # BER long form: 0x81 means 1 length byte follows
        text = b"A longer message for testing"
        length_byte = 0x81
        blob = b"\x00NSString\x01+" + bytes([length_byte, len(text)]) + text + b"\x00"
        result = extract_text_from_attributed_body(blob)
        assert result == "A longer message for testing"

    def test_fallback_to_printable_text(self):
        # No NSString marker, but has readable text
        blob = b"\x00\x01streamtyped\x00\x02This is a fallback test\x00\x03"
        result = extract_text_from_attributed_body(blob)
        assert result is not None
        assert "fallback" in result.lower() or "test" in result.lower()

    def test_unparseable_returns_none(self):
        blob = b"\x00\x01\x02\x03\x04\x05"
        result = extract_text_from_attributed_body(blob)
        assert result is None


class TestExtractMessages:
    def test_returns_dataframe(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        for col in ["message_id", "timestamp", "sender", "text", "chat_id", "is_from_me", "service"]:
            assert col in df.columns

    def test_attributedBody_column_dropped(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        assert "attributedBody" not in df.columns

    def test_text_messages_extracted(self, mock_chat_db):
        df = extract_messages(mock_chat_db)
        texts = df["text"].tolist()
        assert "hello there" in texts
        assert "hey!" in texts

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

    def test_columns_present(self, mock_chat_db):
        df = list_contacts(mock_chat_db)
        for col in ["chat_id", "message_count", "first_msg", "last_msg", "is_group"]:
            assert col in df.columns


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


class TestSearchContacts:
    def test_search_by_partial(self, mock_chat_db):
        df = search_contacts(mock_chat_db, "5551234")
        assert not df.empty

    def test_search_no_match(self, mock_chat_db):
        df = search_contacts(mock_chat_db, "9999999999")
        assert df.empty
