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
from html import escape, unescape
from html.parser import HTMLParser

import markdown as markdown_lib

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
# far more than a model writes about one conversation; what is past it is not read
STORY_LIMIT = 200_000
# a run of one mark longer than any writing uses, which the renderer is slow to read
_LONG_RUN = re.compile(r"([^\w\s])\1{16,}")
_LONG_BREAK = re.compile(r"\n{4,}")
_QUOTE_MARK = re.compile(r"(?m)^([ \t]*)&gt; ?")
_LINK_TITLE = re.compile(r"(\]\([^()\s]+)\s+(?:&quot;.*?&quot;|&#x27;.*?&#x27;)\)")

Attributes = list[tuple[str, str | None]]


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


def story_html(text: str | None) -> str:
    """Markdown written by a model, as HTML that is safe to put into a page.

    A link goes to an ordinary address or is only its words, a picture is only
    its caption, and no tag carries a handler, a style or a name. Showing the
    result runs nothing and loads nothing."""
    if not text:
        return ""
    # The text can be steered by whoever wrote the messages. Runs of one mark
    # and of empty lines are cut to a length no writing needs, and the whole
    # to a length no report reaches, so that reading it stays quick.
    tamed = _LONG_BREAK.sub("\n\n\n", _LONG_RUN.sub(lambda run: run.group(1) * 16, text[:STORY_LIMIT]))
    escaped = escape(tamed)
    # escaping turns the marker of a quoted line into &gt; before markdown can
    # see it; the ones that start a line are put back, so quotes stay quotes
    escaped = _QUOTE_MARK.sub(r"\1> ", escaped)
    # escaping also hides the quotes around a link's title, which would then be
    # read as part of its address; the title is dropped and the address kept
    escaped = _LINK_TITLE.sub(r"\1)", escaped)
    try:
        rendered = markdown_lib.markdown(escaped, extensions=list(STORY_EXTENSIONS), output_format="html5")
    except RecursionError:
        # nested deeper than the renderer can follow: the words are shown as they are
        return "".join(f"<p>{part}</p>" for part in escape(tamed).split("\n\n") if part.strip())
    return sanitize(rendered, code_escaped_twice=True)
