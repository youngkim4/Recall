"""Copies of the code that reads a report file in the redesign (commit 3563549).

`report_contact` and `markdown_section` are in `ui_server.py`. `report_contact`
reads a text here and a file there. `stated_scope` gives the words of the
scope line; the server goes on and reads the two dates from them.
The rest is `story_html` in `safe_html.py`, step by step:

- `tame`: the cuts before anything is read (the length, a run of one mark, a
  run of empty lines). A cut that takes one backslash away frees the bracket
  after it, and a cut at the length limit can leave a line of `=` under a
  line of text.
- `prepared`: the text as the renderer gets it.
- `story_work`: the estimate of the work. A story past the limit is not
  rendered: the page shows `plain` paragraphs.
- `story_html_steps`: the renderer, always. A test of hostile text uses it,
  so that the text is proved safe when it is rendered.
- `story_page`: what the server shows, the renderer or the plain paragraphs.

The server also stops a render after 0.3 s of processor time. That step is
not copied: a test must not depend on a clock.

`STORY_LIMIT` and `STORY_WORK_LIMIT` come from the real `safe_html` when it
can be imported, so a change of one of them there shows in these tests.
"""

import importlib
import re
from html import escape
from html.parser import HTMLParser
from typing import Any

import markdown as markdown_lib

REPORT_HEADING = re.compile(r"^# Conversation Analysis - (.+)$", re.MULTILINE)
REPORT_SCOPE = re.compile(r"^- Scope: (.*)$", re.MULTILINE)
STORY_EXTENSIONS = ["tables", "sane_lists"]
COPIED_STORY_LIMIT = 60_000
COPIED_STORY_WORK_LIMIT = 60_000_000
LONG_RUN = re.compile(r"([^\w\s])\1{16,}")
LONG_RUN_KEPT = 16
LONG_BREAK = re.compile(r"\n{4,}")
QUOTE_MARK = re.compile(r"(?m)^([ \t]*)&gt; ?")
LINK_TITLE = re.compile(
    r"(\]\([^()\s]{1,2000})\s+(?:&quot;.{0,200}?&quot;|&#x27;.{0,200}?&#x27;)\)"
)
BLANK_LINE = re.compile(r"(?<=\n)[ \t]+\n")
MARK = re.compile(r"[^\w \t]|_")
EMPHASIS_MARK = re.compile(r"[*_]")
EMPHASIS_WEIGHT = 3
OPENING_MARK = re.compile(r"[\[\]`]")
OPENING_WEIGHT = 10
GROWS = re.compile(r"[\\&]")
GROWTH = 14
PIECE_WORK = 20_000
LIST_LINES = (re.compile(r" {0,3}[*+-] +\S"), re.compile(r" {0,3}\d+\. +\S"))


def real_safe_html() -> Any:
    """The server's own module, or None where this tree does not hold it."""
    try:
        return importlib.import_module("safe_html")
    except ImportError:
        return None


_REAL = real_safe_html()
STORY_LIMIT: int = getattr(_REAL, "STORY_LIMIT", COPIED_STORY_LIMIT)
STORY_WORK_LIMIT: int = getattr(_REAL, "STORY_WORK_LIMIT", COPIED_STORY_WORK_LIMIT)


def report_contact(text: str) -> str:
    lines = text.splitlines(keepends=True)
    match = REPORT_HEADING.search(lines[0]) if lines else None
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


def prepared(text: str) -> str:
    """The story as the renderer is given it: cut, and with all of it escaped."""
    escaped = QUOTE_MARK.sub(r"\1> ", escape(tame(text)))
    return LINK_TITLE.sub(r"\1)", escaped)


def _read_together(block: str) -> list[str]:
    lines = block.strip("\n").split("\n")
    if len(lines) > 1 and any(all(item.match(line) for line in lines) for item in LIST_LINES):
        return lines
    return [block]


def _piece_work(piece: str) -> int:
    marks = (
        len(MARK.findall(piece))
        + (EMPHASIS_WEIGHT - 1) * len(EMPHASIS_MARK.findall(piece))
        + (OPENING_WEIGHT - 1) * len(OPENING_MARK.findall(piece))
    )
    length = len(piece) + GROWTH * len(GROWS.findall(piece))
    return PIECE_WORK + length * marks


def story_work(ready: str) -> int:
    """The server's estimate of the work to render a prepared story: for each
    paragraph, its length times the marks in it."""
    blocks = BLANK_LINE.sub("\n", ready).split("\n\n")
    return sum(_piece_work(piece) for block in blocks for piece in _read_together(block))


def work_share(text: str) -> float:
    """The work of a story as a part of the limit. Above 1.0 the server does
    not render the story."""
    return story_work(prepared(text)) / STORY_WORK_LIMIT


def plain(text: str) -> str:
    """The words of a story as they are, a paragraph for each."""
    parts = escape(tame(text)).split("\n\n")
    return "".join(f"<p>{part}</p>" for part in parts if part.strip())


def story_html_steps(text: str) -> str:
    """What the renderer makes of a story, before the result is cleaned of
    tags. The work limit is not asked: the text must be safe when rendered."""
    return markdown_lib.markdown(prepared(text), extensions=STORY_EXTENSIONS, output_format="html5")


def story_page(text: str) -> str:
    """What the server shows for a story: the rendered story, or its plain
    paragraphs when the work is past the limit or the nesting is too deep."""
    if not text:
        return ""
    if story_work(prepared(text)) > STORY_WORK_LIMIT:
        return plain(text)
    try:
        return story_html_steps(text)
    except RecursionError:
        return plain(text)


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
