import os
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from analysis import compute_stats, progression_series
from html_report import write_html_report
from tests.html_shape import assert_only_allowed


@pytest.fixture
def basic_report_args(tmp_path, sample_messages_df):
    stats = compute_stats(sample_messages_df)
    monthly = progression_series(sample_messages_df)
    return {
        "out_dir": str(tmp_path),
        "contact": "+1234",
        "stats": stats,
        "monthly": monthly,
        "summary_text": "This is the **summary**.",
    }


class TestWriteHtmlReport:
    def test_creates_file(self, basic_report_args):
        path = write_html_report(**basic_report_args)
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_contains_contact(self, basic_report_args):
        path = write_html_report(**basic_report_args)
        content = Path(path).read_text()
        assert "+1234" in content

    def test_a_phone_number_is_not_written_out_in_the_page(self, basic_report_args):
        path = write_html_report(**{**basic_report_args, "contact": "+15550000009"})
        content = Path(path).read_text()
        assert "<h1>Number ending in 0009</h1>" in content
        assert "<title>Conversation Analysis — Number ending in 0009</title>" in content
        assert "5550000009" not in content

    def test_the_page_is_titled_with_the_name_it_is_given(self, basic_report_args):
        path = write_html_report(**{**basic_report_args, "contact": "+15550000009", "title": "Avery <Stone>"})
        content = Path(path).read_text()
        assert "<h1>Avery &lt;Stone&gt;</h1>" in content
        assert "0009" not in content

    def test_the_page_runs_no_script_and_loads_nothing(self, basic_report_args):
        # the report is someone's private messages: opening it must not run
        # code from anywhere, or fetch anything that says it was opened
        path = write_html_report(**basic_report_args)
        content = Path(path).read_text().lower()
        assert "<script" not in content
        assert "<link" not in content
        assert "src=" not in content
        assert "@import" not in content

    def test_the_months_are_drawn_in_the_page(self, basic_report_args):
        path = write_html_report(**basic_report_args)
        content = Path(path).read_text()
        assert "<svg" in content
        assert content.count("<polyline") == 2

    def test_one_month_is_drawn_as_a_mark_that_can_be_seen(self, tmp_path, sample_messages_df):
        january = sample_messages_df[sample_messages_df["timestamp"] < "2024-02-01"]
        path = write_html_report(str(tmp_path), "+1234", compute_stats(january), progression_series(january), "story")
        content = Path(path).read_text()
        # a line through one point draws nothing
        assert "<polyline" not in content
        assert content.count("<circle") == 2
        assert content.count("2024-01-01") == 1

    def test_summary_rendered_as_html(self, basic_report_args):
        path = write_html_report(**basic_report_args)
        content = Path(path).read_text()
        assert "<strong>summary</strong>" in content

    def test_xss_prevention(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        malicious = '<script>alert("xss")</script>'
        path = write_html_report(
            str(tmp_path), malicious, stats, monthly, "safe summary"
        )
        content = Path(path).read_text()
        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content

    def test_summary_xss_prevention(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        malicious_summary = '**safe** <script>alert("xss")</script>'
        path = write_html_report(
            str(tmp_path), "+1234", stats, monthly, malicious_summary
        )
        content = Path(path).read_text()
        assert "<strong>safe</strong>" in content
        assert "<script>alert" not in content
        assert "&lt;script&gt;" in content

    def test_events_rendered(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        events = pd.DataFrame({
            "date": ["2024-01-15"],
            "title": ["Big day"],
            "detail": ["Something happened"],
            "category": ["milestone"],
            "score": [0.9],
            "quote": ["wow"],
        })
        path = write_html_report(
            str(tmp_path), "+1234", stats, monthly, "summary",
            events_df=events,
        )
        content = Path(path).read_text()
        assert "Big day" in content
        assert "Key Events" in content

    def test_events_xss_prevention(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        events = pd.DataFrame({
            "date": ["2024-01-15"],
            "title": ['<img src=x onerror="alert(1)">'],
            "detail": ["safe"],
            "category": ["milestone"],
            "score": [0.9],
            "quote": ['"; DROP TABLE--'],
        })
        path = write_html_report(
            str(tmp_path), "+1234", stats, monthly, "summary",
            events_df=events,
        )
        content = Path(path).read_text()
        assert 'onerror="alert' not in content
        assert "&lt;img" in content

    def test_attachments_section(self, tmp_path, sample_messages_df, sample_attachments_df):
        stats = compute_stats(sample_messages_df, sample_attachments_df)
        monthly = progression_series(sample_messages_df)
        path = write_html_report(
            str(tmp_path), "+1234", stats, monthly, "summary"
        )
        content = Path(path).read_text()
        assert "Attachments" in content
        assert "Photos" in content

    def test_reactions_section(self, tmp_path, sample_messages_df, sample_reactions_df):
        stats = compute_stats(sample_messages_df, reactions_df=sample_reactions_df)
        monthly = progression_series(sample_messages_df)
        path = write_html_report(
            str(tmp_path), "+1234", stats, monthly, "summary"
        )
        content = Path(path).read_text()
        assert "Reactions" in content
        assert "Loves" in content

    def test_the_summary_cannot_name_an_attribute(self, tmp_path, sample_messages_df):
        # the summary is written by a model from other people's messages; an
        # attribute list in it must not become a handler in the saved page
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        text = (
            "Hello\n{: onmouseover=alert(1) }\n\n"
            "[a link](http://a.example){: onclick=alert(2) }\n\n"
            "## A heading {: onfocus=alert(3) tabindex=0 }\n\n"
            "![a picture](http://a.example/p.png)\n\n"
            "[a bad link](javascript:alert(4))\n"
        )
        path = write_html_report(str(tmp_path), "+1234", stats, monthly, text)
        content = Path(path).read_text()
        summary = content.split('<div class="summary">', 1)[1].split("</div>", 1)[0]
        assert_only_allowed(summary)
        assert "<img" not in summary
        assert "a picture" in summary

    def test_the_summary_keeps_its_quotes_and_tables(self, tmp_path, sample_messages_df):
        stats = compute_stats(sample_messages_df)
        monthly = progression_series(sample_messages_df)
        text = "> a quoted line\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        path = write_html_report(str(tmp_path), "+1234", stats, monthly, text)
        summary = Path(path).read_text().split('<div class="summary">', 1)[1].split("</div>", 1)[0]
        assert "<blockquote>" in summary
        assert "<table>" in summary

    def test_footer_has_timestamp(self, basic_report_args):
        path = write_html_report(**basic_report_args)
        content = Path(path).read_text()
        assert "Generated by" in content
