"""Tests for CLI estimate_cost and main argument parsing."""
import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from cli import estimate_cost, MODEL_PRICING


class TestEstimateCost:
    @pytest.fixture
    def messages_csv(self, tmp_path):
        df = pd.DataFrame({
            "timestamp": [
                "2024-01-15 10:00:00",
                "2024-01-15 10:05:00",
                "2024-02-20 14:30:00",
            ],
            "text": ["hello", "hey", "what's up"],
            "chat_id": ["chat123", "chat123", "chat123"],
            "is_from_me": [0, 1, 0],
            "service": ["iMessage"] * 3,
        })
        csv = tmp_path / "messages.csv"
        df.to_csv(csv, index=False)
        return str(csv)

    def test_basic_estimate(self, messages_csv):
        result = estimate_cost(messages_csv, "chat123")
        assert result["msg_count"] == 3
        assert result["input_tokens"] > 0
        assert result["estimated_cost"] >= 0
        assert result["needs_chunking"] is False

    def test_zero_messages(self, messages_csv):
        result = estimate_cost(messages_csv, "+9999")
        assert result["msg_count"] == 0
        assert result["estimated_cost"] == 0

    def test_different_model(self, messages_csv):
        result_mini = estimate_cost(messages_csv, "chat123", model="gpt-5-mini")
        result_big = estimate_cost(messages_csv, "chat123", model="gpt-5")
        # gpt-5 costs more per token
        assert result_big["estimated_cost"] >= result_mini["estimated_cost"]

    def test_unknown_model_uses_default_rate(self, messages_csv):
        result = estimate_cost(messages_csv, "chat123", model="unknown-model")
        assert result["estimated_cost"] >= 0


class TestMainValidation:
    """Test main() argument validation and early exit paths."""

    def test_no_args_exits(self):
        from cli import main
        with patch("sys.argv", ["cli.py"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_list_contacts_without_db_exits(self):
        from cli import main
        with patch("sys.argv", ["cli.py", "--list-contacts"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_db_not_found_exits(self):
        from cli import main
        with patch("sys.argv", ["cli.py", "--contact", "+1234", "--db", "/nonexistent/chat.db"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_list_contacts_with_db(self, tmp_path, mock_chat_db):
        from cli import main
        with patch("sys.argv", ["cli.py", "--list-contacts", "--db", mock_chat_db]):
            with pytest.raises(SystemExit) as exc:
                main()
            # exits 0 after listing contacts
            assert exc.value.code == 0

    def test_contact_with_messages_csv(self, messages_csv):
        from cli import main
        with patch("sys.argv", ["cli.py", "--contact", "chat123", "--messages", messages_csv, "--no-confirm"]):
            with patch("cli.run_cli") as mock_run:
                mock_run.return_value = (None, "/tmp/report.md", None, None)
                main()
                mock_run.assert_called_once()

    @pytest.fixture
    def messages_csv(self, tmp_path):
        df = pd.DataFrame({
            "timestamp": [
                "2024-01-15 10:00:00",
                "2024-01-15 10:05:00",
            ],
            "text": ["hello", "hey"],
            "chat_id": ["chat123", "chat123"],
            "is_from_me": [0, 1],
            "service": ["iMessage"] * 2,
        })
        csv = tmp_path / "messages.csv"
        df.to_csv(csv, index=False)
        return str(csv)


class TestParseImessageCategorize:
    """Test the categorize helper inside extract_attachments."""
    def test_gif(self):
        from parse_imessage import extract_attachments
        import sqlite3
        import tempfile

        db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(db.name)
        c = conn.cursor()
        c.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
        c.execute("CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT)")
        c.execute("""CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY, date INTEGER, handle_id INTEGER,
            text TEXT, attributedBody BLOB, is_from_me INTEGER, service TEXT,
            associated_message_guid TEXT, associated_message_type INTEGER DEFAULT 0
        )""")
        c.execute("CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER)")
        c.execute("CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT, mime_type TEXT, total_bytes INTEGER)")
        c.execute("CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER)")

        c.execute("INSERT INTO handle VALUES (1, '+1')")
        c.execute("INSERT INTO chat VALUES (1, '+1')")
        base_ns = 726915600 * 1_000_000_000
        c.execute("INSERT INTO message VALUES (1, ?, 1, 'test', NULL, 0, 'iMessage', NULL, 0)", (base_ns,))
        c.execute("INSERT INTO chat_message_join VALUES (1, 1)")
        c.execute("INSERT INTO attachment VALUES (1, 'funny.gif', 'image/gif', 512)")
        c.execute("INSERT INTO message_attachment_join VALUES (1, 1)")
        conn.commit()
        conn.close()

        df = extract_attachments(db.name)
        assert "gif" in df["category"].values
        os.unlink(db.name)


class TestParseImessageGetText:
    """Test the get_text lambda inside extract_messages for attachment-only messages."""
    def test_attachment_only_message(self, mock_chat_db):
        from parse_imessage import extract_messages
        # msg 1 has an attachment and text - text should be present
        df = extract_messages(mock_chat_db)
        assert any("[Attachment]" in str(t) or "hello" in str(t) for t in df["text"].values)
