import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from analysis import (
    sanitize_filename,
    load_messages,
    load_attachments,
    load_reactions,
    filter_conversation,
    compute_stats,
    compute_attachment_stats,
    compute_reaction_stats,
    progression_series,
    format_all_messages,
    estimate_tokens,
    truncate_to_tokens,
    chunk_by_year,
    _parse_json_events,
    write_markdown_report,
    ConversationStats,
    AttachmentStats,
    ReactionStats,
    TOKEN_BUDGET,
    SECONDS_PER_DAY,
)


class TestSanitizeFilename:
    def test_phone_number(self):
        assert sanitize_filename("+12165551234") == "+12165551234"

    def test_strips_special_chars(self):
        assert sanitize_filename("hello world!@#") == "helloworld"

    def test_empty_returns_conversation(self):
        assert sanitize_filename("!@#$%") == "conversation"

    def test_preserves_hyphens_underscores(self):
        assert sanitize_filename("my-chat_2024") == "my-chat_2024"


class TestLoadMessages:
    def test_loads_csv(self, sample_messages_csv):
        df = load_messages(sample_messages_csv)
        assert len(df) == 5
        assert "timestamp" in df.columns

    def test_timestamp_parsed(self, sample_messages_csv):
        df = load_messages(sample_messages_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_since_filter(self, sample_messages_csv):
        from datetime import datetime
        df = load_messages(sample_messages_csv, since=datetime(2024, 2, 1))
        assert len(df) == 3  # Feb + 2x Mar

    def test_until_filter(self, sample_messages_csv):
        from datetime import datetime
        df = load_messages(sample_messages_csv, until=datetime(2024, 1, 31))
        assert len(df) == 2  # 2x Jan

    def test_missing_timestamp_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"text": ["hi"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_messages(str(csv))

    def test_adds_missing_columns(self, tmp_path):
        csv = tmp_path / "minimal.csv"
        pd.DataFrame({
            "timestamp": ["2024-01-01 10:00"],
            "text": ["hi"],
        }).to_csv(csv, index=False)
        df = load_messages(str(csv))
        assert "is_from_me" in df.columns
        assert "service" in df.columns


class TestLoadAttachments:
    def test_loads_existing(self, tmp_path, sample_attachments_df):
        csv = tmp_path / "messages.csv"
        csv.touch()
        att_csv = tmp_path / "messages_attachments.csv"
        sample_attachments_df.to_csv(att_csv, index=False)
        df = load_attachments(str(csv))
        assert len(df) == 3

    def test_returns_empty_when_missing(self, tmp_path):
        csv = tmp_path / "messages.csv"
        csv.touch()
        df = load_attachments(str(csv))
        assert df.empty


class TestLoadReactions:
    def test_loads_existing(self, tmp_path, sample_reactions_df):
        csv = tmp_path / "messages.csv"
        csv.touch()
        react_csv = tmp_path / "messages_reactions.csv"
        sample_reactions_df.to_csv(react_csv, index=False)
        df = load_reactions(str(csv))
        assert len(df) == 3

    def test_returns_empty_when_missing(self, tmp_path):
        csv = tmp_path / "messages.csv"
        csv.touch()
        df = load_reactions(str(csv))
        assert df.empty


class TestFilterConversation:
    def test_exact_match(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "+1234")
        assert len(result) == 5

    def test_partial_match(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "1234")
        assert len(result) == 5

    def test_no_match(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "+9999")
        assert result.empty

    def test_sorted_by_timestamp(self, sample_messages_df):
        result = filter_conversation(sample_messages_df, "+1234")
        timestamps = result["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_missing_chat_id_raises(self):
        df = pd.DataFrame({"text": ["hi"]})
        with pytest.raises(ValueError, match="chat_id"):
            filter_conversation(df, "+1234")


class TestComputeStats:
    def test_basic_stats(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.total_messages == 5
        assert stats.sent_count == 2
        assert stats.received_count == 3

    def test_empty_df(self):
        empty = pd.DataFrame(columns=["timestamp", "text", "chat_id", "is_from_me"])
        stats = compute_stats(empty)
        assert stats.total_messages == 0
        assert stats.active_days == 0

    def test_active_days(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.active_days == 3  # Jan 15, Feb 20, Mar 10

    def test_busiest_day(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.busiest_day_count == 2  # Jan 15 or Mar 10

    def test_longest_gap(self, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        assert stats.longest_gap_days > 0

    def test_with_attachments(self, sample_messages_df, sample_attachments_df):
        stats = compute_stats(sample_messages_df, sample_attachments_df)
        assert stats.attachments.total == 3
        assert stats.attachments.photos == 1

    def test_with_reactions(self, sample_messages_df, sample_reactions_df):
        stats = compute_stats(sample_messages_df, reactions_df=sample_reactions_df)
        assert stats.reactions.total == 3


class TestComputeAttachmentStats:
    def test_basic(self, sample_attachments_df):
        stats = compute_attachment_stats(sample_attachments_df, "+1234")
        assert stats.total == 3
        assert stats.photos == 1
        assert stats.videos == 1
        assert stats.documents == 1

    def test_sent_received(self, sample_attachments_df):
        stats = compute_attachment_stats(sample_attachments_df, "+1234")
        assert stats.sent_attachments == 2
        assert stats.received_attachments == 1

    def test_empty_df(self):
        stats = compute_attachment_stats(pd.DataFrame(), "+1234")
        assert stats.total == 0

    def test_no_match(self, sample_attachments_df):
        stats = compute_attachment_stats(sample_attachments_df, "+9999")
        assert stats.total == 0


class TestComputeReactionStats:
    def test_basic(self, sample_reactions_df):
        stats = compute_reaction_stats(sample_reactions_df, "+1234")
        assert stats.total == 3
        assert stats.loves == 1
        assert stats.likes == 1
        assert stats.laughs == 1

    def test_sent_received(self, sample_reactions_df):
        stats = compute_reaction_stats(sample_reactions_df, "+1234")
        assert stats.sent_reactions == 2
        assert stats.received_reactions == 1

    def test_empty_df(self):
        stats = compute_reaction_stats(pd.DataFrame(), "+1234")
        assert stats.total == 0


class TestProgressionSeries:
    def test_returns_monthly(self, sample_messages_df):
        result = progression_series(sample_messages_df)
        assert "month" in result.columns
        assert "total" in result.columns
        assert "sent" in result.columns
        assert "received" in result.columns
        assert "sent_ratio" in result.columns

    def test_month_count(self, sample_messages_df):
        result = progression_series(sample_messages_df)
        assert len(result) == 3  # Jan, Feb, Mar


class TestFormatAllMessages:
    def test_format(self, sample_messages_df):
        result = format_all_messages(sample_messages_df)
        assert "ME:" in result
        assert "THEM:" in result
        assert "hello" in result

    def test_missing_is_from_me(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
            "text": ["hi"],
        })
        result = format_all_messages(df)
        assert "??:" in result


class TestEstimateTokens:
    def test_basic(self):
        tokens = estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        result = truncate_to_tokens("Hello", 100)
        assert result == "Hello"

    def test_truncates_long_text(self):
        long_text = "word " * 1000
        result = truncate_to_tokens(long_text, 10)
        assert estimate_tokens(result) <= 10

    def test_empty_returns_empty(self):
        assert truncate_to_tokens("", 100) == ""

    def test_zero_tokens(self):
        assert truncate_to_tokens("Hello", 0) == ""


class TestChunkByYear:
    def test_single_year(self, sample_messages_df):
        chunks = chunk_by_year(sample_messages_df)
        assert len(chunks) == 1
        label, df, text = chunks[0]
        assert label == "2024"
        assert len(df) == 5

    def test_multi_year(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2023-06-01 10:00",
                "2024-01-15 10:00",
                "2024-06-15 10:00",
            ]),
            "text": ["a", "b", "c"],
            "is_from_me": [0, 1, 0],
            "chat_id": ["+1234"] * 3,
        })
        chunks = chunk_by_year(df)
        labels = [c[0] for c in chunks]
        assert "2023" in labels
        assert "2024" in labels

    def test_all_messages_included(self, sample_messages_df):
        chunks = chunk_by_year(sample_messages_df)
        total = sum(len(df) for _, df, _ in chunks)
        assert total == 5


class TestParseJsonEvents:
    def test_valid_json_object(self):
        content = json.dumps({"events": [{"date": "2024-01-15", "title": "Test"}]})
        result = _parse_json_events(content)
        assert len(result) == 1

    def test_valid_json_array(self):
        content = json.dumps([{"date": "2024-01-15", "title": "Test"}])
        result = _parse_json_events(content)
        assert len(result) == 1

    def test_markdown_fenced(self):
        content = "```json\n" + json.dumps({"events": [{"date": "2024-01-15"}]}) + "\n```"
        result = _parse_json_events(content)
        assert len(result) == 1

    def test_invalid_json(self):
        result = _parse_json_events("not json at all")
        assert result == []

    def test_missing_events_key(self):
        content = json.dumps({"data": []})
        result = _parse_json_events(content)
        assert result == []


class TestWriteMarkdownReport:
    def test_creates_file(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        path = write_markdown_report(str(tmp_path), "+1234", stats, monthly, "Test summary")
        assert os.path.exists(path)

    def test_content_includes_stats(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        path = write_markdown_report(str(tmp_path), "+1234", stats, monthly, "Test summary")
        content = Path(path).read_text()
        assert "+1234" in content
        assert "Test summary" in content
        assert "5" in content  # total messages

    def test_with_events(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        events = pd.DataFrame({
            "date": ["2024-01-15"],
            "title": ["First contact"],
            "detail": ["They said hello"],
            "category": ["milestone"],
            "score": [0.9],
            "quote": ["hello"],
        })
        path = write_markdown_report(str(tmp_path), "+1234", stats, monthly, "Summary", events)
        content = Path(path).read_text()
        assert "First contact" in content
        assert "Key Events" in content


class TestConstants:
    def test_token_budget(self):
        assert TOKEN_BUDGET == 170_000

    def test_seconds_per_day(self):
        assert SECONDS_PER_DAY == 86_400.0
