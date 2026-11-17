"""Tests for the files a report run leaves behind, with the model's part replaced."""
from datetime import datetime

import pandas as pd
import pytest

import analysis
import contact_names

AVERY = "+15550000001"
EVENT = {
    "date": "2024-06-10",
    "title": "Plans",
    "detail": "They made plans.",
    "category": "milestone",
    "score": 0.7,
    "quote": "see you at 8",
}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs a report over a small made-up export. The events the model would
    find are whatever the test hands in, and nothing is sent anywhere."""
    csv = tmp_path / "messages.csv"
    pd.DataFrame({
        "message_id": [1, 2, 3, 4],
        # nothing in August
        "timestamp": ["2024-05-20 09:00:00", "2024-06-10 20:00:00", "2024-07-04 12:00:00", "2024-09-02 08:00:00"],
        "sender": [AVERY, AVERY, AVERY, AVERY],
        "text": ["long before", "see you at 8", "long after", "after the summer"],
        "chat_id": [AVERY, AVERY, AVERY, AVERY],
        "is_from_me": [0, 0, 1, 0],
    }).to_csv(csv, index=False)
    out_dir = tmp_path / "out"
    found = {"events": pd.DataFrame()}

    monkeypatch.setattr(analysis, "ai_extract_events", lambda *args, **kwargs: found["events"])
    monkeypatch.setattr(analysis, "ai_summary", lambda *args, **kwargs: "A story.")
    monkeypatch.setattr(contact_names, "resolve_contact_names", lambda handles, **_: {})

    def go(events=(), since=None, until=None):
        found["events"] = pd.DataFrame(list(events))
        _, report_path, events_path, _html = analysis.run_cli(str(csv), AVERY, str(out_dir), since=since, until=until)
        return report_path, events_path

    go.out_dir = out_dir
    return go


def events_file(run):
    return run.out_dir / "events_timeline_+15550000001.csv"


def test_a_run_that_finds_events_writes_them_beside_the_report(run):
    _report, events_path = run(events=[EVENT])

    assert events_path == str(events_file(run))
    assert pd.read_csv(events_path)["title"].tolist() == ["Plans"]


def test_a_run_that_finds_no_events_leaves_none_from_an_earlier_run(run):
    run(events=[EVENT])
    assert events_file(run).exists()

    _report, events_path = run(events=[])

    # the older file would be shown beside the new report as its own
    assert events_path is None
    assert not events_file(run).exists()


def test_a_first_run_that_finds_no_events_writes_no_file(run):
    _report, events_path = run(events=[])

    assert events_path is None
    assert not events_file(run).exists()


def overview(report_path):
    text = open(report_path, encoding="utf-8").read()
    return text.split("## Overview", 1)[1].split("\n## ", 1)[0]


def test_a_report_states_the_dates_it_was_asked_for(run):
    report, _events = run(since=datetime(2024, 6, 1), until=datetime(2024, 6, 30))

    assert "- Scope: 2024-06-01 to 2024-06-30\n" in overview(report)


def test_a_report_on_everything_says_so(run):
    report, _events = run()

    assert "- Scope: all dates\n" in overview(report)


def test_a_report_open_on_one_side_states_the_side_it_was_given(run):
    since_only, _events = run(since=datetime(2024, 6, 1))
    assert "- Scope: from 2024-06-01\n" in overview(since_only)

    until_only, _events = run(until=datetime(2024, 6, 30))
    assert "- Scope: through 2024-06-30\n" in overview(until_only)


def months_of(report_path):
    text = open(report_path, encoding="utf-8").read()
    table = text.split("## Monthly Progression\n", 1)[1].split("\n\n", 1)[0]
    return table.splitlines()[2:]


def test_a_report_keeps_a_month_with_no_messages_in_its_place(run):
    report, _events = run()

    assert months_of(report) == [
        "| 2024-05-01 | 1 | 0 | 1 | 0% |",
        "| 2024-06-01 | 1 | 0 | 1 | 0% |",
        "| 2024-07-01 | 1 | 1 | 0 | 100% |",
        "| 2024-08-01 | 0 | 0 | 0 | 0% |",
        "| 2024-09-01 | 1 | 0 | 1 | 0% |",
    ]
