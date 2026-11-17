"""Render a report in the markdown layout that today's Reports page and server read.

The layout has three level-two sections, in this order: Overview (its first
line is the scope line), Key Events, Summary. Summary is last and holds the
four report sections under level-three headings.

Text that comes from the model or from messages can not add a heading: the
readers find sections by their `## ` markers.
"""

from __future__ import annotations

import re

from report_engine.dates import DAY_LENGTH
from report_engine.models import Moment, Report, Scope, Section

_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_SECOND_HASH = re.compile(r"(?<=#)#")


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


def _inline(text: str) -> str:
    """One line, with no run of hashes that could read as a section marker."""
    return _SECOND_HASH.sub(r"\\#", " ".join(text.split()))


def _body(markdown: str) -> str:
    """Section text. A heading line becomes a bold line."""
    lines = []
    for line in markdown.strip().splitlines():
        match = _HEADING_LINE.match(line)
        lines.append(f"**{match.group(1)}**" if match and match.group(1) else line.rstrip())
    return "\n".join(lines)


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
    if moment.quote_verified and moment.quote:
        rows.append(f'> "{_inline(moment.quote)}"')
    rows.append(f"*Category: {moment.category}*")
    return rows + [""]


def _section(section: Section) -> list[str]:
    return [f"### {section.title}", _body(section.markdown), ""]


def render_markdown(report: Report) -> str:
    """The whole report as one markdown text that ends with a newline."""
    rows = [f"# Conversation Analysis - {_inline(report.conversation)}", ""]
    rows += _overview(report) + [""]
    rows += ["## Key Events"]
    rows += [row for moment in report.moments for row in _event(moment)] or [""]
    rows += ["## Summary"]
    rows += [row for section in report.sections for row in _section(section)]
    return "\n".join(rows[:-1]) + "\n"
