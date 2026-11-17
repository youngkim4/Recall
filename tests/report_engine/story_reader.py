"""Copies of the code that reads a report file in the redesign (commit ab20b5e).

`report_contact`, `markdown_section` and the scope reader are in `ui_server.py`.
`tame` and `story_html_steps` are the steps of `story_html` in `safe_html.py`,
in its order, without the last step, which takes tags out of the HTML.

`tame` matters as much as the renderer: it cuts the story before the renderer
reads it. A cut that takes one backslash away can free the bracket after it.
"""

import re
from html import escape
from html.parser import HTMLParser
from typing import Any

import markdown as markdown_lib

REPORT_HEADING = re.compile(r"^# Conversation Analysis - (.+)$", re.MULTILINE)
REPORT_SCOPE = re.compile(r"^- Scope: (.*)$", re.MULTILINE)
STORY_EXTENSIONS = ["tables", "sane_lists"]
STORY_LIMIT = 200_000
LONG_RUN = re.compile(r"([^\w\s])\1{16,}")
LONG_RUN_KEPT = 16
LONG_BREAK = re.compile(r"\n{4,}")
QUOTE_MARK = re.compile(r"(?m)^([ \t]*)&gt; ?")
LINK_TITLE = re.compile(r"(\]\([^()\s]+)\s+(?:&quot;.*?&quot;|&#x27;.*?&#x27;)\)")


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


def story_of(report_text: str) -> str:
    """The text that the server gives to story_html."""
    return markdown_section(report_text, "Summary", stop_at_next_heading=False)


def tame(text: str) -> str:
    """What story_html cuts before it reads a story: the whole to a limit, a
    run of one mark to 16, and a run of empty lines to two."""
    cut = LONG_RUN.sub(lambda run: run.group(1) * LONG_RUN_KEPT, text[:STORY_LIMIT])
    return LONG_BREAK.sub("\n\n\n", cut)


def story_html_steps(text: str) -> str:
    """What story_html does to a story before it is cleaned of tags."""
    escaped = QUOTE_MARK.sub(r"\1> ", escape(tame(text)))
    escaped = LINK_TITLE.sub(r"\1)", escaped)
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
