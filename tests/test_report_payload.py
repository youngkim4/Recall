"""Tests for what the Reports reader is given when a saved report is opened."""
import pandas as pd
import pytest

import ui_server
from tests.html_shape import assert_only_allowed, shape
from ui_server import build_analysis_payload, markdown_to_html, report_scope

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache")

COLUMNS = ["message_id", "timestamp", "sender", "text", "chat_id", "chat_display_name", "is_from_me"]
AVERY = "+15550000001"
BLAKE = "+15550000002"
GROUP = "chat77"
NAMES = {AVERY: "Avery Stone", BLAKE: "Blake Reyes"}

OVERVIEW = """# Conversation Analysis - {contact}

## Overview
- Total messages: 2
- Sent / Received: 1 / 1
- First / Last: {first} / {last}

## Summary
They made plans.
"""


def write_export(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["message_id"] = range(1, len(frame) + 1)
    frame.to_csv(path, index=False)
    return path


def row(chat_id, timestamp, text, from_me=0, sender=None):
    return (0, timestamp, sender if sender is not None else chat_id, text, chat_id, "", from_me)


@pytest.fixture
def saved_names(monkeypatch):
    monkeypatch.setattr(
        ui_server,
        "resolve_contact_names",
        lambda handles, **_: {handle: NAMES[handle] for handle in handles if handle in NAMES},
    )


@pytest.fixture
def export(tmp_path):
    return write_export(tmp_path / "messages.csv", [
        row(AVERY, "2024-11-05 09:00:00", "long before"),
        row(AVERY, "2024-12-06 09:00:00", "are we still on"),
        row(AVERY, "2024-12-17 21:00:00", "yes", from_me=1),
        row(AVERY, "2025-02-01 10:00:00", "long after"),
    ])


def test_the_days_a_report_covers_are_read_from_the_report():
    text = OVERVIEW.format(contact=AVERY, first="2024-12-06 09:00:00", last="2024-12-17 21:00:00")

    assert report_scope(text) == ("2024-12-06", "2024-12-17")


def test_a_report_that_does_not_say_its_days_covers_everything():
    assert report_scope("# Report\n\n## Summary\nA story.\n") == (None, None)
    assert report_scope("## Overview\n- First / Last: None / None\n") == (None, None)


def test_a_report_shows_the_figures_of_the_days_it_covers(tmp_path, export, saved_names):
    report = tmp_path / "analysis.md"
    report.write_text(OVERVIEW.format(contact=AVERY, first="2024-12-06 09:00:00", last="2024-12-17 21:00:00"))

    payload = build_analysis_payload(export, AVERY, "gpt-5.5", report_path=str(report))

    # the story was written about these two messages, so the figures beside it are theirs
    assert payload["stats"]["totalMessages"] == 2
    assert payload["activity"]["unit"] == "day"
    assert payload["activity"]["points"][0]["key"] == "2024-12-06"
    assert payload["activity"]["points"][-1]["key"] == "2024-12-17"


def test_a_report_with_no_days_in_it_shows_the_whole_conversation(tmp_path, export, saved_names):
    report = tmp_path / "analysis.md"
    report.write_text("# Report\n\n## Summary\nA story.\n")

    payload = build_analysis_payload(export, AVERY, "gpt-5.5", report_path=str(report))

    assert payload["stats"]["totalMessages"] == 4


def test_a_report_carries_everything_the_conversation_view_reads(tmp_path, export, saved_names):
    report = tmp_path / "analysis.md"
    report.write_text("# Report\n\n## Summary\nA story.\n")

    payload = build_analysis_payload(export, AVERY, "gpt-5.5", report_path=str(report))

    # the reader draws the same chart and figures as Analyze, from the same fields
    assert {"stats", "activity", "daily", "dynamics", "monthly", "recentMessages"} <= set(payload)
    assert payload["contactDisplayName"] == "Avery Stone"


def test_a_report_on_an_unnamed_group_is_named_after_its_people(tmp_path, saved_names):
    csv = write_export(tmp_path / "messages.csv", [
        row(GROUP, "2025-01-01 09:00:00", "who is in", sender=AVERY),
        row(GROUP, "2025-01-01 09:01:00", "me", sender=BLAKE),
    ])
    report = tmp_path / "analysis.md"
    report.write_text("# Report\n\n## Summary\nA story.\n")

    payload = build_analysis_payload(csv, GROUP, "gpt-5.5", report_path=str(report))

    assert payload["contactDisplayName"] == "Avery Stone and Blake Reyes"


def test_a_summary_keeps_ordinary_links():
    html = markdown_to_html("See [the place](https://example.com/menu) or [write](mailto:hi@example.com).")

    assert 'href="https://example.com/menu"' in html
    assert 'href="mailto:hi@example.com"' in html


def test_a_summary_cannot_carry_a_link_that_runs_code():
    html = markdown_to_html("[open](javascript:alert(1)) and [this](data:text/html;base64,PHNjcmlwdD4=)")

    assert "javascript:" not in html
    assert "data:" not in html
    # the words stay; only the address goes
    assert "open" in html and "this" in html


def test_a_summary_loads_nothing_from_the_network():
    html = markdown_to_html("Before ![a photo](https://example.com/pixel.png) after")

    assert "<img" not in html
    assert "a photo" in html


def test_a_summary_shows_markup_as_text():
    html = markdown_to_html("<script>alert(1)</script> and <b>bold</b>")

    assert "<script" not in html
    assert "&lt;script&gt;" in html


def test_a_saved_report_knows_the_conversation_it_is_about(tmp_path, monkeypatch, saved_names):
    monkeypatch.setattr(ui_server, "OUT_DIR", tmp_path)
    # a file name cannot hold an address, so the address is read from the report itself
    (tmp_path / "analysis_averyexamplecom.md").write_text(
        "# Conversation Analysis - avery@example.com\n\n## Summary\nA story.\n", encoding="utf-8"
    )
    (tmp_path / "analysis_averyexamplecom.html").write_text("<html></html>", encoding="utf-8")

    reports = ui_server.list_report_files()

    assert {report["kind"] for report in reports} == {"md", "html"}
    assert {report["contact"] for report in reports} == {"avery@example.com"}


def test_a_report_with_no_heading_is_known_by_its_file_name(tmp_path, monkeypatch, saved_names):
    monkeypatch.setattr(ui_server, "OUT_DIR", tmp_path)
    (tmp_path / "analysis_+15550000001.md").write_text("## Summary\nA story.\n", encoding="utf-8")

    (report,) = ui_server.list_report_files()

    assert report["contact"] == AVERY
    assert report["displayName"] == "Avery Stone"


def test_a_job_remembers_the_conversation_it_is_for():
    job = ui_server.job_record("analyze", {"contact": AVERY, "model": "gpt-5.5"})

    # the list of jobs can then say whose report is being written
    assert job["contact"] == AVERY
    assert job["action"] == "analyze"
    assert job["status"] == "queued"


def test_a_job_with_no_conversation_has_none():
    assert ui_server.job_record("export", {})["contact"] == ""


def test_a_summary_link_with_a_title_keeps_its_address():
    html = markdown_to_html('See [the menu](https://example.com/menu "Tonight") for more.')

    assert 'href="https://example.com/menu"' in html
    assert "the menu" in html


@pytest.mark.parametrize("text", [
    "[x](JaVaScRiPt:alert(1))",
    "[x](&#106;avascript:alert(1))",
    "[x](java&#10;script:alert(1))",
    "[x][ref]\n\n[ref]: javascript:alert(1)",
    "<javascript:alert(1)>",
    "[x](vbscript:msgbox(1))",
    "[x](//example.com/path)",
    "![x][pic]\n\n[pic]: https://example.com/pixel.png",
    '<img src="https://example.com/pixel.png" onerror="alert(1)">',
    '<a href="javascript:alert(1)">x</a>',
])
def test_a_summary_never_carries_a_link_that_runs_code_or_a_picture(text):
    html = markdown_to_html(text).lower()

    assert "<img" not in html
    assert "<script" not in html
    for href in __import__("re").findall(r'href="([^"]*)"', html):
        assert href.startswith(("http://", "https://", "mailto:")), href


@pytest.mark.parametrize("text", [
    "Hello\n{: onmouseover=alert(1) }",
    "[x](http://a.com){: onclick=alert(2) }",
    "## h {: onfocus=alert(3) tabindex=0 autofocus }",
    'para\n{: style="background:url(javascript:alert(1))" }',
    '``` { .py #x onclick=alert(1) style="color:red" }\ncode\n```',
    '```python hl_lines="1"\nprint(1)\n```',
    '<div markdown="1" onclick="alert(1)">\n*hi*\n</div>',
    "Text[^1]\n\n[^1]: note",
    'The HTML spec\n\n*[HTML]: Hyper " onmouseover="alert(1)',
    "| a | b |\n|:--|--:|\n| 1 | 2 |",
    "Term\n:   Definition",
    '[x](https://example.com "t" ){: #id .cls key=value }',
    "![cap](https://example.com/p.png){: onerror=alert(1) }",
])
def test_a_summary_cannot_put_an_attribute_on_what_it_writes(text):
    # the story is written by a model from other people's words, so nothing in
    # it may carry a handler, a style or a name of its own
    assert_only_allowed(markdown_to_html(text))


def test_a_summary_keeps_its_words_when_an_attribute_list_is_refused():
    html = markdown_to_html("## Dinner {: onfocus=alert(3) tabindex=0 autofocus }")

    assert shape(html).tags == ["h2"]
    assert "Dinner" in html


def test_a_summary_table_keeps_its_cells():
    html = markdown_to_html("| Month | Total |\n|:------|------:|\n| June | 12 |")

    assert shape(html).tags == ["table", "thead", "tr", "th", "th", "tbody", "tr", "td", "td"]
    assert "<th>Month</th>" in html and "<td>12</td>" in html


def test_a_summary_list_keeps_the_number_it_starts_at():
    assert '<ol start="3">' in markdown_to_html("3. three\n4. four")


def test_a_summary_shows_code_as_it_was_typed():
    html = markdown_to_html("She wrote `don't \"wait\" <3` and left.")

    # what the browser decodes is exactly what was between the backticks
    assert "<code>don&#x27;t &quot;wait&quot; &lt;3</code>" in html


def test_a_summary_never_runs_markup_typed_inside_code():
    html = markdown_to_html("`<script>alert(1)</script>` and\n\n    <img src=x onerror=alert(1)>\n")

    assert set(shape(html).tags) <= {"p", "code", "pre"}
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


SCOPED = """# Conversation Analysis - {contact}

## Overview
- Scope: {scope}
- Total messages: 2
- Sent / Received: 1 / 1
- First / Last: 2024-12-06 09:00:00 / 2024-12-17 21:00:00

## Summary
They made plans.
"""


@pytest.mark.parametrize("scope, expected", [
    ("2024-12-01 to 2024-12-31", ("2024-12-01", "2024-12-31")),
    ("from 2024-12-01", ("2024-12-01", None)),
    ("through 2024-12-31", (None, "2024-12-31")),
    ("all dates", (None, None)),
])
def test_the_dates_a_report_was_asked_for_are_read_before_the_days_of_its_messages(scope, expected):
    assert report_scope(SCOPED.format(contact=AVERY, scope=scope)) == expected


@pytest.mark.parametrize("scope", ["", "sometime in December", "2024-12-01", "2024-12-01 to", "to 2024-12-31"])
def test_a_scope_that_cannot_be_read_falls_back_to_the_days_of_the_messages(scope):
    assert report_scope(SCOPED.format(contact=AVERY, scope=scope)) == ("2024-12-06", "2024-12-17")


def test_a_scope_written_in_the_story_is_not_the_scope_of_the_report():
    old_report = OVERVIEW.format(contact=AVERY, first="2024-12-06 09:00:00", last="2024-12-17 21:00:00")
    text = old_report + "\n- Scope: 2019-01-01 to 2019-01-02\n- Scope: all dates\n"

    # a report written before the line existed, with a story that happens to hold one
    assert report_scope(text) == ("2024-12-06", "2024-12-17")


def test_a_scope_among_the_events_is_not_the_scope_of_the_report():
    text = (
        "# Conversation Analysis - x\n\n## Overview\n- First / Last: 2024-12-06 09:00:00 / 2024-12-17 21:00:00\n\n"
        "## Key Events\n### 2024-12-06 - Plans\n- Scope: all dates\n\n## Summary\nA story.\n"
    )

    assert report_scope(text) == ("2024-12-06", "2024-12-17")


def test_a_reopened_report_counts_what_fell_in_its_dates_after_the_last_message(tmp_path, export, saved_names):
    # asked for all of December: the last message is on the 17th, a reaction came on the 20th
    pd.DataFrame({
        "reaction_id": [1, 2],
        "is_from_me": [1, 0],
        "chat_id": [AVERY, AVERY],
        "timestamp": ["2024-12-20 08:00:00", "2025-01-05 08:00:00"],
        "reaction_type": ["love", "like"],
        "is_add": [True, True],
    }).to_csv(tmp_path / "messages_reactions.csv", index=False)
    report = tmp_path / "analysis.md"
    report.write_text(SCOPED.format(contact=AVERY, scope="2024-12-01 to 2024-12-31"))

    payload = build_analysis_payload(export, AVERY, "gpt-5.5", report_path=str(report))

    assert payload["stats"]["totalMessages"] == 2
    assert payload["stats"]["reactions"]["total"] == 1


def test_a_report_on_all_dates_shows_the_whole_conversation(tmp_path, export, saved_names):
    report = tmp_path / "analysis.md"
    report.write_text(SCOPED.format(contact=AVERY, scope="all dates"))

    payload = build_analysis_payload(export, AVERY, "gpt-5.5", report_path=str(report))

    # the days of its messages are not taken for its scope
    assert payload["stats"]["totalMessages"] == 4


def test_the_scope_line_is_not_part_of_the_story(tmp_path, export, saved_names):
    report = tmp_path / "analysis.md"
    report.write_text(SCOPED.format(contact=AVERY, scope="2024-12-01 to 2024-12-31"))

    payload = build_analysis_payload(export, AVERY, "gpt-5.5", report_path=str(report))

    assert payload["summary"] == "They made plans."
    assert "Scope" not in payload["summaryHtml"]
