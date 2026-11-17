"""Render a report in the markdown layout that today's Reports page and server read.

The layout has three level-two sections, in this order: Overview (its first
line is the scope line), Key Events, Summary. Summary is last and holds the
four report sections under level-three headings.

The first line carries the conversation id as it is: the server reads the
id back from that line.

Text that comes from the model or from messages can not add a heading, a
link or an image. The server finds sections by their `## ` markers, the page
cuts the story into chapters at its headings, and a report must not load
anything from the network when it is shown.

The story is rendered by Python-Markdown. There a line that starts with `#`
is a heading with or without a space after it, and a line of `=` or `-`
under a line of text makes that text a heading. Section text has neither.
It also has no `[` or `]` without a backslash, so no form of a link or an
image can be read from it.
"""

from __future__ import annotations

import re

from report_engine.dates import DAY_LENGTH
from report_engine.errors import StoreError
from report_engine.models import Moment, Report, Scope, Section

# what can stand before the text of a line: indent, quote marks, list marks
_LINE_MARKS = r"(?:\s|>|[-*+](?=\s)|\d+[.)](?=\s))*"
_LEADING_HASH = re.compile(rf"^({_LINE_MARKS})#")
_UNDERLINE = re.compile(r"^(?:\s|>)*[=-]+\s*$")
_SECOND_HASH = re.compile(r"(?<=#)#")
# a link or an image of the plain kind: words with no bracket, an address with no space
_PLAIN_LINK = re.compile(r'(?<!\\)!?\[([^\[\]\\]*)\]\([^()\s]*(?:\s+"[^"()]*")?\)')
_LINK_DEFINITION = re.compile(r"^\s{0,3}\[[^\[\]]*\]:")
_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")
MAX_CLEAN_PASSES = 30
MAX_HEADING_INDENT = 3
MAX_HEADING_LEVEL = 6
SECTION_MARK = "## "
_NO_BRACKETS = str.maketrans("", "", "<>[]")
_INLINE_ESCAPES = str.maketrans({"\\": "\\\\", "#": "\\#", "[": "\\[", "]": "\\]"})


def scope_line(scope: Scope) -> str:
    since = scope.since[:DAY_LENGTH] if scope.since else None
    until = scope.until[:DAY_LENGTH] if scope.until else None
    if since and until:
        return f"- Scope: {since} to {until}"
    if since:
        return f"- Scope: from {since}"
    if until:
        return f"- Scope: through {until}"
    return "- Scope: all dates"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _inline(text: str) -> str:
    """Plain text on one line. No character in it starts a heading or a link."""
    return _one_line(text).translate(_INLINE_ESCAPES)


def _strip_markup(line: str) -> str:
    """Take tags, and links and images of the plain kind, out of one line. Taking
    one out can uncover another, so this repeats until nothing changes. A line
    that still changes after many passes loses every bracket."""
    for _ in range(MAX_CLEAN_PASSES):
        cleaned = _HTML_TAG.sub("", _PLAIN_LINK.sub(r"\1", line))
        if cleaned == line:
            return line
        line = cleaned
    return line.translate(_NO_BRACKETS)


def _heading_words(line: str) -> str | None:
    """The words of a heading line, or None when the line is not one. A line of
    hashes only is a heading with no words. This reads the line one time from
    each end: a pattern for the same thing took minutes on one long line."""
    body = line.lstrip()
    if len(line) - len(body) > MAX_HEADING_INDENT or not body.startswith("#"):
        return None
    words = body.lstrip("#")
    if len(body) - len(words) > MAX_HEADING_LEVEL or not (words[:1].isspace() or not words):
        return None
    return words.strip().rstrip("#").rstrip()


def _without_heading(line: str) -> str:
    """A heading line as a bold line. Any other line that starts with `#`, also
    after a quote or list mark, gets a backslash before the `#`."""
    words = _heading_words(line)
    if words is None:
        return _LEADING_HASH.sub(r"\1\\#", line)
    return f"**{words}**" if words else ""


def _escape_brackets(line: str) -> str:
    """A backslash before each `[` and `]` that has none. No renderer makes a
    link or an image of a line with no open bracket, however the link is
    written: across lines, with a definition somewhere else, or nested."""
    if "[" not in line and "]" not in line:
        return line
    parts: list[str] = []
    slashes = 0
    for char in line:
        if char in "[]" and slashes % 2 == 0:
            parts.append("\\")
        parts.append(char)
        slashes = slashes + 1 if char == "\\" else 0
    return "".join(parts)


def _clean_once(markdown: str) -> str:
    lines: list[str] = []
    for raw in markdown.strip().splitlines():
        line = _strip_markup(raw)
        if _LINK_DEFINITION.match(line) or _UNDERLINE.match(line):
            continue
        lines.append(_escape_brackets(_without_heading(line)).rstrip())
    return "\n".join(lines).strip()


def clean_section(markdown: str) -> str:
    """Section text with its formatting kept and its active parts removed.

    - A heading line becomes a bold line. Any other line that starts with `#`
      gets a backslash before it, and a line of only `=` or `-` goes.
    - A tag goes. A link or an image of the plain kind becomes its words.
    - Every `[` and `]` that is left gets a backslash, so no other form of a
      link or an image can be read from the text.

    Cleaning the result again changes nothing. The time is in proportion to
    the length of the text."""
    for _ in range(MAX_CLEAN_PASSES):
        cleaned = _clean_once(markdown)
        if cleaned == markdown:
            return cleaned
        markdown = cleaned
    return _clean_once(markdown.translate(_NO_BRACKETS))


def _body(markdown: str) -> str:
    """Cleaned section text in which no run of hashes can read as a marker."""
    return _SECOND_HASH.sub(r"\\#", clean_section(markdown))


def _overview(report: Report) -> list[str]:
    usage = report.usage
    rows = [
        "## Overview",
        scope_line(report.scope),
        f"- Model: {_inline(report.model)}",
        f"- Created: {report.created_at}",
        f"- Status: {report.status}",
        f"- Periods: {len(report.periods):,}",
        f"- Moments: {len(report.moments):,}",
        f"- Cost: ${usage.cost_usd:.4f} ({usage.input_tokens:,} input tokens, "
        f"{usage.output_tokens:,} output tokens)",
    ]
    return rows + [f"- Warning: {_inline(warning)}" for warning in report.warnings]


def _event(moment: Moment) -> list[str]:
    rows = [f"### {moment.date} - {_inline(moment.title)}", _inline(moment.detail)]
    if moment.quote:
        rows.append(f'> "{_inline(moment.quote)}"')
    rows.append(f"*Category: {moment.category}*")
    return rows + [""]


def _section(section: Section) -> list[str]:
    return [f"### {_inline(section.title)}", _body(section.markdown), ""]


def _first_line(conversation: str) -> str:
    """The heading. The id stands in it as it is, because the server reads it
    back from this line. The server finds a section with a search for its
    marker, so an id that holds the start of a marker can not be written."""
    name = _one_line(conversation)
    if SECTION_MARK in f"{name} ":
        raise StoreError(
            "a conversation id that holds a section marker can not be written to a report file"
        )
    return f"# Conversation Analysis - {name}"


def render_markdown(report: Report) -> str:
    """The whole report as one markdown text that ends with a newline."""
    rows = [_first_line(report.conversation), ""]
    rows += _overview(report) + [""]
    rows += ["## Key Events"]
    rows += [row for moment in report.moments for row in _event(moment)] or [""]
    rows += ["## Summary"]
    rows += [row for section in report.sections for row in _section(section)]
    return "\n".join(rows[:-1]) + "\n"
