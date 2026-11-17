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

The server cuts a story before the renderer reads it: a run of 17 of one mark
to 16, and four line breaks in a row to three. The renderer takes U+0002 and
U+0003 out. A cut or a removal next to a backslash can take the guard off the
mark after it. So the text of a report has nothing that is cut or removed: no
run of one mark longer than 15, no more than two empty lines in a row, and no
control character but a tab and a line break.
"""

from __future__ import annotations

import re

from report_engine.dates import DAY_LENGTH
from report_engine.errors import StoreError
from report_engine.models import Moment, Report, Scope, Section

# what can stand before the text of a line: indent, quote marks, list marks, and
# the colon of a definition list
_LINE_MARKS = r"(?:\s|>|[-*+:](?=\s)|\d+[.)](?=\s))*"
_LEADING_HASH = re.compile(rf"^({_LINE_MARKS})#")
_UNDERLINE = re.compile(r"^(?:\s|>)*[=-]+\s*$")
_SECOND_HASH = re.compile(r"(?<=#)#")
# One less than the run that the server keeps: the cleaning can put one
# backslash before a run of backslashes, and that run must not be cut either.
MAX_MARK_RUN = 15
# Inline text gets every backslash twice, and one more before a guarded mark.
MAX_INLINE_SLASHES = (MAX_MARK_RUN - 1) // 2
_MARK_RUN = re.compile(rf"([^\w\s])\1{{{MAX_MARK_RUN},}}")
_SLASH_RUN = re.compile(rf"\\{{{MAX_INLINE_SLASHES + 1},}}")
_EMPTY_LINES = re.compile(r"\n{4,}")
# Control characters that are not a tab or a line break go. U+001F is white
# space to Python and becomes a space.
_CONTROLS: dict[int, str | None] = {
    **dict.fromkeys(
        (*range(0x00, 0x09), *range(0x0E, 0x1C), *range(0x7F, 0x85), *range(0x86, 0xA0))
    ),
    0x1F: " ",
}
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


def _short_runs(text: str) -> str:
    """The text with no run of one mark longer than MAX_MARK_RUN."""
    return _MARK_RUN.sub(lambda run: run.group(1) * MAX_MARK_RUN, text)


def _inline(text: str) -> str:
    """Plain text on one line. No character in it starts a heading or a link.
    A run of backslashes is cut before each one is written twice, so the run
    in the file stays shorter than a run that the server cuts."""
    flat = _short_runs(_one_line(text.translate(_CONTROLS)))
    flat = _SLASH_RUN.sub(lambda _run: "\\" * MAX_INLINE_SLASHES, flat)
    return flat.translate(_INLINE_ESCAPES)


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
    """One pass. The runs are cut after the heading step, which can add marks,
    and before the bracket step, which adds at most one backslash to a run:
    the result of one pass has no run that the server cuts."""
    lines: list[str] = []
    for raw in markdown.translate(_CONTROLS).strip().splitlines():
        line = _strip_markup(raw)
        if _LINK_DEFINITION.match(line) or _UNDERLINE.match(line):
            continue
        lines.append(_escape_brackets(_short_runs(_without_heading(line))).rstrip())
    return _EMPTY_LINES.sub("\n\n\n", "\n".join(lines).strip())


def clean_section(markdown: str) -> str:
    """Section text with its formatting kept and its active parts removed.

    - A control character goes. A tab and a line break stay.
    - A heading line becomes a bold line. Any other line that starts with `#`
      gets a backslash before it, and a line of only `=` or `-` goes.
    - A tag goes. A link or an image of the plain kind becomes its words.
    - A run of one mark keeps 15 of them, and a run of empty lines keeps two.
    - Every `[` and `]` that is left gets a backslash, so no other form of a
      link or an image can be read from the text.

    Cleaning the result again changes nothing, and the cuts that the server
    makes before it renders a story change nothing in it. The time is in
    proportion to the length of the text."""
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
        # the quote marks of the row can make a run of quote marks longer
        rows.append(_short_runs(f'> "{_inline(moment.quote)}"'))
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
