"""Turns the story of a report into HTML in which nothing can run.

The story is written by a model from other people's messages, and it is put
into a page as HTML: the app's own page, and the report file that is saved.
Both go through story_html, so there is one place where this is decided.

Nothing here strips what looks dangerous. The rendered HTML is read, and only
the tags and attributes named below are written back, with every piece of
text escaped. What is not named is left out.

No allowed tag switches a browser into raw text or another namespace, so the
result reads back as the same tags it was written with.
"""

from __future__ import annotations

import re
import time
from html import escape, unescape
from html.parser import HTMLParser
from typing import Callable

import markdown as markdown_lib
from markdown.extensions import Extension

ALLOWED_TAGS = frozenset({
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "em", "strong", "del", "code", "pre",
    "blockquote", "ul", "ol", "li", "a", "table", "thead", "tbody", "tr", "th", "td", "sup", "sub",
})
VOID_TAGS = frozenset({"br", "hr"})
# left out together with everything inside them
SILENCED_TAGS = frozenset({"script", "style"})
CODE_TAGS = frozenset({"code", "pre"})
ORDINARY_ADDRESSES = ("http://", "https://", "mailto:")

# a browser drops these from the ends of an address, and from inside it
_EDGE = "".join(chr(code) for code in range(0x21))
_INSIDE = re.compile(r"[\x00-\x1f\x7f]")
_WHOLE_NUMBER = re.compile(r"[0-9]{1,9}")

# Tables and tidy lists are all a story needs. Neither lets the text name an
# attribute, which the "extra" set did through attribute lists.
STORY_EXTENSIONS = ("tables", "sane_lists")
# several times what a model writes about one conversation; what is past it is not read
STORY_LIMIT = 60_000
# a run of one mark longer than any writing uses, which the renderer is slow to read
_LONG_RUN = re.compile(r"([^\w\s])\1{16,}")
_LONG_BREAK = re.compile(r"\n{4,}")
_QUOTE_MARK = re.compile(r"(?m)^([ \t]*)&gt; ?")
# Each part has an end, so no text makes the search for a title long: an
# address and a title longer than these are not an address and a title.
_LINK_TITLE = re.compile(r"(\]\([^()\s]{1,2000})\s+(?:&quot;.{0,200}?&quot;|&#x27;.{0,200}?&#x27;)\)")

# Two limits keep the renderer quick. A story past either one is shown as
# plain paragraphs.
#
# The renderer is stopped when it has used STORY_SECONDS of processor time.
# It can only be stopped between two of its steps, so no step may be long.
#
# What makes a step long is a paragraph with many marks in it: the renderer
# reads the paragraph again for each mark. story_work adds up, over the
# paragraphs of a story, the length of each times the marks it holds, and a
# story that asks for more than STORY_WORK_LIMIT is not given to the renderer.
# Ordinary writing as long as a story can be asks for a small part of it and
# takes a small part of the time.
STORY_SECONDS = 0.3
STORY_WORK_LIMIT = 60_000_000
# A story is read chapter by chapter (story_page). A report has four chapters;
# a story with more parts than this is read as one piece.
STORY_PARTS = 8
# All the chapters of one page share this much processor time. A chapter that
# starts when it is spent is shown as plain paragraphs.
STORY_PAGE_SECONDS = 1.0
_PART_START = re.compile(r"(?m)^(?=### )")
_BLANK_LINE = re.compile(r"(?<=\n)[ \t]+\n")
# a mark is anything that is not a letter, a digit, a space or a tab
_MARK = re.compile(r"[^\w \t]|_")
# The marks of emphasis are looked for by the slowest rules. The marks that
# open a link or code make the renderer start the paragraph again.
_EMPHASIS_MARK = re.compile(r"[*_]")
_EMPHASIS_WEIGHT = 3
_OPENING_MARK = re.compile(r"[\[\]`]")
_OPENING_WEIGHT = 10
# the renderer puts a longer token in the place of an escaped mark and of an
# entity, so a paragraph that holds them is longer when it is read again
_GROWS = re.compile(r"[\\&]")
_GROWTH = 14
# what a paragraph or a list item costs before anything in it is read
_PIECE_WORK = 20_000
# The renderer reads a list an item at a time. That is only sure when each
# line of the paragraph is an item of one kind, so only then are the lines
# counted by themselves.
_LIST_LINES = (re.compile(r" {0,3}[*+-] +\S"), re.compile(r" {0,3}\d+\. +\S"))

Attributes = list[tuple[str, str | None]]


class _TooSlow(Exception):
    """The renderer has used the processor time a story is given."""


class _Deadline(Extension):
    """Stops a render that has run too long. The renderer cannot be stopped
    from outside, so each of its steps asks first whether there is time left:
    a step is one use of a rule on a paragraph, or one block read. The time is
    the processor time of this thread, so a busy machine does not use it up."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self._seconds = seconds

    def extendMarkdown(self, md: markdown_lib.Markdown) -> None:
        stop_at = time.thread_time() + self._seconds

        def checked(step: Callable) -> Callable:
            def run(*args, **kwargs):
                if time.thread_time() > stop_at:
                    raise _TooSlow
                return step(*args, **kwargs)

            return run

        for rule in md.inlinePatterns:
            rule.handleMatch = checked(rule.handleMatch)
        for reader in md.parser.blockprocessors:
            reader.run = checked(reader.run)


def ordinary_address(value: str | None) -> str:
    """The address as a browser reads it, or "" unless it is a web or mail address."""
    cleaned = _INSIDE.sub("", (value or "").strip(_EDGE))
    return cleaned if cleaned.lower().startswith(ORDINARY_ADDRESSES) else ""


def _first(attrs: Attributes, name: str) -> str:
    # a browser keeps the first of two attributes with one name
    return next((value or "" for key, value in attrs if key == name), "")


def _opening(tag: str, attrs: Attributes) -> str:
    """The opening tag as it is written back, or "" when the tag is left out."""
    if tag == "a":
        address = ordinary_address(_first(attrs, "href"))
        # a link that goes nowhere allowed is only its words; one that goes
        # somewhere takes no word of this page with it
        return f'<a href="{escape(address)}" rel="noopener noreferrer">' if address else ""
    if tag == "ol":
        start = _first(attrs, "start")
        return f'<ol start="{start}">' if _WHOLE_NUMBER.fullmatch(start) else "<ol>"
    return f"<{tag}>"


class _Rebuilder(HTMLParser):
    def __init__(self, code_escaped_twice: bool) -> None:
        super().__init__(convert_charrefs=True)
        self._code_escaped_twice = code_escaped_twice
        self._written: list[str] = []
        self._open: list[str] = []
        self._silenced = 0

    def handle_starttag(self, tag: str, attrs: Attributes) -> None:
        if tag in SILENCED_TAGS:
            self._silenced += 1
            return
        if self._silenced:
            return
        if tag == "img":
            # a picture loads from the network; its caption does not
            self._written.append(escape(_first(attrs, "alt")))
            return
        if tag not in ALLOWED_TAGS:
            return
        opening = _opening(tag, attrs)
        if not opening:
            return
        self._written.append(opening)
        if tag not in VOID_TAGS:
            self._open.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in SILENCED_TAGS:
            self._silenced = max(0, self._silenced - 1)
            return
        if self._silenced or tag not in self._open:
            return
        # whatever was opened inside it and never closed is closed first
        while self._open:
            closing = self._open.pop()
            self._written.append(f"</{closing}>")
            if closing == tag:
                return

    def handle_data(self, data: str) -> None:
        if self._silenced:
            return
        in_code = any(tag in CODE_TAGS for tag in self._open)
        text = unescape(data) if self._code_escaped_twice and in_code else data
        self._written.append(escape(text))

    def result(self) -> str:
        return "".join(self._written) + "".join(f"</{tag}>" for tag in reversed(self._open))


def sanitize(html: str | None, *, code_escaped_twice: bool = False) -> str:
    """The HTML written again with allowed tags only and all of its text escaped.

    code_escaped_twice: the caller escaped the text before it was rendered, so
    the renderer escaped what is inside code a second time. One layer is taken
    off there, and the reader sees the characters that were typed."""
    if not html:
        return ""
    rebuilder = _Rebuilder(code_escaped_twice)
    try:
        rebuilder.feed(html)
        rebuilder.close()
    except Exception:
        # text the parser cannot read is shown as plain text, never passed on
        return escape(html)
    return rebuilder.result()


def _tamed(text: str) -> str:
    """The story cut to the length one can have, with runs of one mark and of
    empty lines cut to a length no writing needs."""
    return _LONG_BREAK.sub("\n\n\n", _LONG_RUN.sub(lambda run: run.group(1) * 16, text[:STORY_LIMIT]))


def prepared(text: str) -> str:
    """The story as the renderer is given it: cut, and with all of it escaped."""
    escaped = escape(_tamed(text))
    # escaping turns the marker of a quoted line into &gt; before markdown can
    # see it; the ones that start a line are put back, so quotes stay quotes
    escaped = _QUOTE_MARK.sub(r"\1> ", escaped)
    # escaping also hides the quotes around a link's title, which would then be
    # read as part of its address; the title is dropped and the address kept
    return _LINK_TITLE.sub(r"\1)", escaped)


def _read_together(block: str) -> list[str]:
    """The pieces of a paragraph that the renderer reads by themselves."""
    lines = block.strip("\n").split("\n")
    if len(lines) > 1 and any(all(item.match(line) for line in lines) for item in _LIST_LINES):
        return lines
    return [block]


def _piece_work(piece: str) -> int:
    marks = (
        len(_MARK.findall(piece))
        + (_EMPHASIS_WEIGHT - 1) * len(_EMPHASIS_MARK.findall(piece))
        + (_OPENING_WEIGHT - 1) * len(_OPENING_MARK.findall(piece))
    )
    length = len(piece) + _GROWTH * len(_GROWS.findall(piece))
    return _PIECE_WORK + length * marks


def story_work(ready: str) -> int:
    """How much the renderer has to do to read a prepared story: for each
    paragraph, its length times the marks in it. The paragraphs are split
    where the renderer splits them, at empty lines."""
    blocks = _BLANK_LINE.sub("\n", ready).split("\n\n")
    return sum(_piece_work(piece) for block in blocks for piece in _read_together(block))


def _plain(text: str) -> str:
    """The words of a story as they are, a paragraph for each."""
    return "".join(f"<p>{part}</p>" for part in escape(_tamed(text)).split("\n\n") if part.strip())


def _rendered(text: str, seconds: float | None = None) -> tuple[str, bool]:
    """One piece of a story as HTML, and whether it is shown as plain paragraphs.

    `seconds` is the processor time that the renderer may use, STORY_SECONDS when not given."""
    allowed = STORY_SECONDS if seconds is None else seconds
    ready = prepared(text)
    if allowed <= 0 or story_work(ready) > STORY_WORK_LIMIT:
        return _plain(text), True
    try:
        # the deadline is named last, so it holds every rule the others added
        rules = [*STORY_EXTENSIONS, _Deadline(allowed)]
        rendered = markdown_lib.markdown(ready, extensions=rules, output_format="html5")
    except (RecursionError, _TooSlow):
        # nested deeper than the renderer can follow, or slow to read: the
        # words are shown as they are
        return _plain(text), True
    return sanitize(rendered, code_escaped_twice=True), False


def story_html(text: str | None) -> str:
    """Markdown written by a model, as HTML that is safe to put into a page.

    A link goes to an ordinary address or is only its words, a picture is only
    its caption, and no tag carries a handler, a style or a name. Showing the
    result runs nothing and loads nothing.

    The text can be steered by whoever wrote the messages, so reading it is
    kept quick: it is cut to the length a story can have, and a story that
    would take the renderer long is shown as plain paragraphs."""
    if not text:
        return ""
    return _rendered(text)[0]


def story_parts(text: str) -> list[str]:
    """The story cut where a chapter starts: a line that begins with `### `.

    The whole story is cut to the length one can have first. A story with no
    such line is one part, and so is a story with more chapters than
    STORY_PARTS: a report has four."""
    whole = _tamed(text)
    parts = [part for part in _PART_START.split(whole) if part.strip()]
    return parts if 1 < len(parts) <= STORY_PARTS else [whole]


def story_page(text: str | None) -> tuple[str, bool]:
    """The story of a report as HTML, and whether a part of it is shown as plain paragraphs.

    Each chapter is read by itself with the rules of story_html, so one dense
    chapter does not make plain paragraphs of the whole story. The work limit
    and the time limit hold for each chapter, and all the chapters share
    STORY_PAGE_SECONDS: a hostile story cannot use the time of a piece once
    for each chapter.

    A chapter is cut at its line, whatever stands around it: a `### ` line
    inside a code fence starts a chapter too. Each piece is made safe by
    itself, so the page is safe; only its form can differ from story_html."""
    if not text:
        return "", False
    stop_at = time.thread_time() + STORY_PAGE_SECONDS
    pieces = [_rendered(part, min(STORY_SECONDS, stop_at - time.thread_time())) for part in story_parts(text)]
    return "\n".join(html for html, _ in pieces), any(plain for _, plain in pieces)
