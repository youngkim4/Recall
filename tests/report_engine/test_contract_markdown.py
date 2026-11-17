"""The report file against the code that reads it in the redesign (commit ab20b5e).

The readers below are copies of `report_contact`, `markdown_section` and
`report_scope` in `ui_server.py`, and of the steps of `story_html` in
`safe_html.py`, as CONTRACTS.md at that commit names them. When `safe_html`
can be imported, the real `story_html` is used too.
"""

import re
from html import escape
from html.parser import HTMLParser
from typing import Any

import markdown as markdown_lib
import pytest

from report_engine.markdown import clean_section, render_markdown
from report_engine.models import SECTION_KEYS, SECTION_TITLES, Scope, Section

from .test_models import report as make_report

REPORT_HEADING = re.compile(r"^# Conversation Analysis - (.+)$", re.MULTILINE)
REPORT_SCOPE = re.compile(r"^- Scope: (.*)$", re.MULTILINE)
QUOTE_MARK = re.compile(r"(?m)^([ \t]*)&gt; ?")
STORY_EXTENSIONS = ["tables", "sane_lists"]
TITLES = [SECTION_TITLES[key] for key in SECTION_KEYS]


def report_contact(text: str) -> str:
    match = REPORT_HEADING.search(text.splitlines(keepends=True)[0])
    return match.group(1).strip() if match else ""


def markdown_section(text: str, heading: str, stop_at_next_heading: bool = True) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    start = text.find("\n", start)
    if start < 0:
        return ""
    next_heading = text.find("\n## ", start + 1) if stop_at_next_heading else -1
    section = text[start : next_heading if next_heading >= 0 else len(text)]
    return section.strip()


def stated_scope(text: str) -> str:
    line = REPORT_SCOPE.search(markdown_section(text, "Overview"))
    return line.group(1).strip() if line else ""


def story_html_steps(text: str) -> str:
    """What story_html does to a story before it is cleaned of tags."""
    escaped = QUOTE_MARK.sub(r"\1> ", escape(text))
    return markdown_lib.markdown(escaped, extensions=STORY_EXTENSIONS, output_format="html5")


class Headings(HTMLParser):
    """Every heading of a page, with its level, its depth in other tags and its words."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[tuple[int, int, str]] = []
        self.tags: list[str] = []
        self._depth = 0
        self._open: list[Any] | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self.tags.append(tag)
        if re.fullmatch(r"h[1-6]", tag):
            self._open = [int(tag[1]), self._depth, ""]
        if tag not in ("br", "hr"):
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        self._depth = max(0, self._depth - 1)
        if self._open is not None and re.fullmatch(r"h[1-6]", tag):
            self.found.append((self._open[0], self._open[1], self._open[2].strip()))
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._open[2] += data


def headings_of(html: str) -> Headings:
    reader = Headings()
    reader.feed(html)
    reader.close()
    return reader


def with_bodies(body: str, **fields: Any) -> Any:
    sections = tuple(
        Section(key=key, title=SECTION_TITLES[key], markdown=clean_section(body))
        for key in SECTION_KEYS
    )
    return make_report(sections=sections, **fields)


@pytest.mark.parametrize(
    "conversation",
    [
        "+12025550143",
        "theo@example.com",
        "chat000000000000000042",
        "odd#name[1]@example.com",
        "back\\slash@example.com",
    ],
)
def test_the_heading_gives_the_conversation_id_back_as_it_is(conversation: str) -> None:
    text = render_markdown(make_report(conversation=conversation))
    assert report_contact(text) == conversation
    assert text.splitlines()[0] == f"# Conversation Analysis - {conversation}"


def test_a_conversation_id_stays_on_the_first_line() -> None:
    text = render_markdown(make_report(conversation="first\n## Summary\nsecond"))
    assert text.splitlines()[0] == "# Conversation Analysis - first ## Summary second"
    assert text.splitlines()[1] == ""


@pytest.mark.parametrize(
    "scope, stated",
    [
        (Scope(since="2024-06-01", until="2024-06-30"), "2024-06-01 to 2024-06-30"),
        (Scope(since="2024-06-01"), "from 2024-06-01"),
        (Scope(until="2024-06-30"), "through 2024-06-30"),
        (Scope(), "all dates"),
    ],
)
def test_the_reader_of_the_scope_finds_the_line_that_was_written(scope: Scope, stated: str) -> None:
    text = render_markdown(
        make_report(scope=scope, warnings=("- Scope: from 1999-01-01", "x\n- Scope: all dates"))
    )
    assert stated_scope(text) == stated


def test_the_story_is_everything_after_the_summary_marker_and_nothing_else() -> None:
    text = render_markdown(with_bodies("One line.\n\nplain ## Summary and ## Overview inside"))
    story = markdown_section(text, "Summary", stop_at_next_heading=False)
    assert story.startswith(f"### {TITLES[0]}\nOne line.")
    assert [line[4:] for line in story.splitlines() if line.startswith("### ")] == TITLES
    assert markdown_section(text, "Overview").startswith("- Scope: ")


HOSTILE_BODIES = [
    "#Injected",
    "#Injected\n#Twice",
    "####### seven",
    "# A heading\n## Another",
    "Title\n=====",
    "Title\n-----\nmore\nSecond\n---",
    "Title\n-",
    "Title\n= = =\nTitle\n=-=",
    "text\n#tag at the start of a line",
    " #one space\n   #three spaces",
    "- #in a list\n- # also in a list\n1. #numbered",
    "> #in a quote\n> # also in a quote\n> Title\n> ===",
    "```\n# in a fence\n```",
    "\\#escaped already",
    "&#35; entity\n&num; entity",
    "<h1>raw</h1>\n<h2 onclick=x>raw</h2>",
]


@pytest.mark.parametrize("body", HOSTILE_BODIES)
def test_no_section_text_makes_a_heading_in_the_story(body: str) -> None:
    """The page cuts the story into chapters at its headings (readStory). A
    heading that the model or a message could add would move the chapters."""
    text = render_markdown(with_bodies(f"Before.\n\n{body}\n\nAfter."))
    story = markdown_section(text, "Summary", stop_at_next_heading=False)
    found = headings_of(story_html_steps(story)).found
    assert found == [(3, 0, title) for title in TITLES]


@pytest.mark.parametrize("body", HOSTILE_BODIES)
def test_stored_section_text_makes_no_heading_when_it_is_rendered_alone(body: str) -> None:
    """A page that reads the JSON report renders `Section.markdown` itself."""
    cleaned = clean_section(f"Before.\n\n{body}\n\nAfter.")
    assert headings_of(story_html_steps(cleaned)).found == []
    assert clean_section(cleaned) == cleaned


def test_lists_quotes_tables_and_emphasis_survive_the_story_renderer() -> None:
    body = (
        "It **began** in *June*.\n\n"
        "- first\n- second\n\n"
        "1. one\n2. two\n\n"
        "> a quoted line\n\n"
        "| Month | Messages |\n|-------|----------|\n| June | 120 |"
    )
    text = render_markdown(with_bodies(body))
    story = markdown_section(text, "Summary", stop_at_next_heading=False)
    tags = headings_of(story_html_steps(story)).tags
    for tag in ("strong", "em", "ul", "ol", "blockquote", "table"):
        assert tags.count(tag) == len(SECTION_KEYS), tag


def test_a_hash_that_is_not_at_the_start_of_a_line_reads_as_it_was_written() -> None:
    cleaned = clean_section("You were #1 in June.\n#1 again in July.")
    html = story_html_steps(cleaned)
    assert "You were #1 in June." in html
    assert "#1 again in July." in html
    assert "\\" not in html


@pytest.mark.parametrize("body", HOSTILE_BODIES)
def test_the_real_story_renderer_agrees(body: str) -> None:
    safe_html = pytest.importorskip("safe_html")
    text = render_markdown(with_bodies(f"Before.\n\n{body}\n\nAfter."))
    story = markdown_section(text, "Summary", stop_at_next_heading=False)
    found = headings_of(safe_html.story_html(story)).found
    assert found == [(3, 0, title) for title in TITLES]
