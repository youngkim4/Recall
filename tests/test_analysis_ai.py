"""Tests for AI-dependent analysis functions using mocked OpenAI client."""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from analysis import (
    _call_openai,
    ai_extract_events,
    ai_summary,
    _summarize_period,
    _extract_events_for_period,
    compute_stats,
    format_all_messages,
    estimate_tokens,
    chunk_by_year,
)


def _mock_openai_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_client(content):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response(content)
    return client


class TestCallOpenai:
    def test_returns_content(self):
        client = _make_client("Hello response")
        result = _call_openai(client, [{"role": "user", "content": "hi"}])
        assert result == "Hello response"

    def test_strips_whitespace(self):
        client = _make_client("  trimmed  \n")
        result = _call_openai(client, [{"role": "user", "content": "hi"}])
        assert result == "trimmed"

    def test_none_content_returns_empty(self):
        client = _make_client(None)
        # Override to return None
        client.chat.completions.create.return_value.choices[0].message.content = None
        result = _call_openai(client, [{"role": "user", "content": "hi"}])
        assert result == ""

    def test_rate_limit_retry(self):
        from openai import RateLimitError
        client = MagicMock()
        resp = _mock_openai_response("ok")
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        client.chat.completions.create.side_effect = [
            RateLimitError("rate limited", response=mock_response, body=None),
            resp,
        ]
        with patch("analysis.time.sleep"):
            result = _call_openai(client, [{"role": "user", "content": "hi"}])
        assert result == "ok"

    def test_api_error_raises(self):
        from openai import APIError
        client = MagicMock()
        # Create a real APIError via constructor matching the installed version
        try:
            err = APIError("server error", response=MagicMock(), body=None)
        except TypeError:
            err = APIError(message="server error", request=MagicMock(), body=None)
        client.chat.completions.create.side_effect = err
        with pytest.raises(APIError):
            _call_openai(client, [{"role": "user", "content": "hi"}])


class TestAiExtractEvents:
    @pytest.fixture
    def conv_df(self):
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-15 10:00",
                "2024-01-15 10:05",
                "2024-02-20 14:30",
            ]),
            "text": ["hello", "hey", "what's up"],
            "is_from_me": [0, 1, 0],
            "chat_id": ["+1234"] * 3,
        })

    @patch("analysis.OpenAI")
    def test_small_conversation(self, mock_openai_cls, conv_df):
        events_json = json.dumps({
            "events": [{
                "date": "2024-01-15",
                "title": "First hello",
                "detail": "They said hello",
                "category": "milestone",
                "score": 0.8,
                "quote": "hello",
            }]
        })
        client = _make_client(events_json)
        mock_openai_cls.return_value = client

        stats = compute_stats(conv_df)
        result = ai_extract_events("+1234", stats, conv_df, target_events=5)
        assert not result.empty
        assert "title" in result.columns

    @patch("analysis.OpenAI")
    def test_empty_response(self, mock_openai_cls, conv_df):
        client = _make_client(json.dumps({"events": []}))
        mock_openai_cls.return_value = client

        stats = compute_stats(conv_df)
        result = ai_extract_events("+1234", stats, conv_df, target_events=5)
        assert result.empty

    @patch("analysis.OpenAI")
    def test_chunked_conversation(self, mock_openai_cls, conv_df):
        events_json = json.dumps({
            "events": [{
                "date": "2024-01-15",
                "title": "Event",
                "detail": "Detail",
                "category": "milestone",
                "score": 0.7,
                "quote": "hi",
            }]
        })
        client = _make_client(events_json)
        mock_openai_cls.return_value = client

        stats = compute_stats(conv_df)
        all_messages = format_all_messages(conv_df)
        # Force chunking by setting total_tokens above budget
        result = ai_extract_events(
            "+1234", stats, conv_df, target_events=5,
            total_tokens=200_000, all_messages=all_messages,
        )
        assert not result.empty


class TestAiSummary:
    @pytest.fixture
    def conv_df(self):
        return pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-15 10:00",
                "2024-01-15 10:05",
            ]),
            "text": ["hello", "hey"],
            "is_from_me": [0, 1],
            "chat_id": ["+1234"] * 2,
        })

    @patch("analysis.OpenAI")
    def test_small_conversation(self, mock_openai_cls, conv_df):
        client = _make_client("This is a great relationship summary.")
        mock_openai_cls.return_value = client

        stats = compute_stats(conv_df)
        result = ai_summary("+1234", stats, conv_df)
        assert "relationship" in result.lower() or "summary" in result.lower()

    @patch("analysis.OpenAI")
    def test_chunked_summary(self, mock_openai_cls, conv_df):
        client = _make_client(
            "### SUMMARY\nGood period.\n\n### CONTEXT FOR NEXT PERIOD\nThey were close."
        )
        mock_openai_cls.return_value = client

        stats = compute_stats(conv_df)
        all_messages = format_all_messages(conv_df)
        result = ai_summary(
            "+1234", stats, conv_df,
            total_tokens=200_000, all_messages=all_messages,
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestSummarizePeriod:
    def test_parses_context(self):
        client = _make_client(
            "Great period.\n\n### CONTEXT FOR NEXT PERIOD\nThey talked a lot about work."
        )
        summary, context = _summarize_period(client, "+1234", "2024", "messages", 100)
        assert "Great period" in summary
        assert "work" in context

    def test_no_context_marker(self):
        client = _make_client("Just a summary with no context section.")
        summary, context = _summarize_period(client, "+1234", "2024", "messages", 100)
        assert "Just a summary" in summary
        assert context == ""


class TestExtractEventsForPeriod:
    def test_returns_events(self):
        events_json = json.dumps({
            "events": [{
                "date": "2024-01-15",
                "title": "Something",
                "detail": "Happened",
                "category": "milestone",
                "score": 0.5,
                "quote": "wow",
            }]
        })
        client = _make_client(events_json)
        result = _extract_events_for_period(client, "+1234", "2024", "messages", 5)
        assert len(result) == 1
        assert result[0]["title"] == "Something"

    def test_with_prior_context(self):
        events_json = json.dumps({"events": []})
        client = _make_client(events_json)
        result = _extract_events_for_period(
            client, "+1234", "2024", "messages", 5,
            prior_context="They argued in 2023."
        )
        assert result == []
