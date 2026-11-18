"""The text of a conversation digest: periods, summaries, moments and the messages that back them.

Every text of the source is masked before it is measured, so the result does
not grow after the plan is made. A message of a moment is registered as
evidence only when its line is in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .digest_fit import Fit, cut_summary, fit_digest
from .ports import Message
from .render import NOTE_TOKENS, ResultWriter, render_block
from .tool_context import ToolContext, plural

MOMENT_MESSAGE_CHARS = 300  # a message under a moment is a pointer: read_around shows the whole of it
_NO_QUOTE = "A summary is not a quote: quote and cite only the message lines."
_NARROW = "Narrow the days with since and until to read more."


@dataclass(frozen=True)
class Moment:
    date: str
    title: str
    detail: str
    quote: str
    message_ids: tuple[str, ...] = ()  # as the source named them
    messages: tuple[Message, ...] = ()  # those the archive holds inside the scope, in time order


@dataclass(frozen=True)
class Period:
    label: str
    first_day: str
    last_day: str
    messages: int
    summary: str
    moments: tuple[Moment, ...] = ()  # the strongest first


@dataclass(frozen=True)
class Notes:
    """What the digest leaves out before any cut: periods that failed, periods outside the scope."""

    failed: int = 0
    outside: int = 0

    def lines(self) -> list[str]:
        lines = []
        if self.failed:
            verb = "is" if self.failed == 1 else "are"
            lines.append(f"[{plural(self.failed, 'period')} could not be built and {verb} missing.]")
        if self.outside:
            reach, verb = ("reaches", "is") if self.outside == 1 else ("reach", "are")
            lines.append(
                f"[{plural(self.outside, 'period')} {reach} outside the days of the scope and {verb} not shown.]"
            )
        return lines


def write_digest(
    ctx: ToolContext, title: str, periods: Sequence[Period], notes: Notes, budget: int
) -> tuple[str, int]:
    """The digest under `budget` tokens, and the number of periods in it."""
    count = ctx.limits.count_tokens
    masked = [_masked(ctx, period) for period in periods]
    total = sum(period.messages for period in masked)
    found = f"{plural(len(masked), 'period')}, {plural(total, 'message')}. {_NO_QUOTE}"
    header = "\n".join([f"Digest of {title}: {found if masked else 'nothing to show.'}", *notes.lines()])
    heads = [_head(period) for period in masked]
    costs = _moment_costs(ctx, masked)
    fit = fit_digest(
        fixed=count(header) + sum(count(head) + 1 for head in heads) + len(masked),
        summaries=[count(period.summary) for period in masked],
        moments=costs,
        room=max(0, budget - NOTE_TOKENS),
    )
    writer = ResultWriter(budget, count)
    writer.add(header, force=True)
    dropped = fit.dropped(costs)
    for period, head, kept in zip(masked, heads, fit.kept, strict=True):
        writer.add(head, force=True)
        writer.add(cut_summary(period.summary, fit.summary_tokens(count(period.summary)), count), force=True)
        dropped += sum(1 for moment in period.moments[:kept] if not _add_moment(ctx, writer, moment))
    if fit.share < 1.0 or dropped:
        writer.add(_cut_note(fit, dropped), force=True)
    return writer.text(), len(masked)


def _head(period: Period) -> str:
    days = f"{period.first_day} to {period.last_day}"
    return f"\n== {period.label} · {days} · {plural(period.messages, 'message')} =="


def _masked(ctx: ToolContext, period: Period) -> Period:
    """The period with every text of the source on one line and with no raw handle."""

    def clean(text: str) -> str:
        return " ".join(ctx.book.redact(text).split())

    moments = tuple(
        replace(m, title=clean(m.title), detail=clean(m.detail), quote=clean(m.quote)) for m in period.moments
    )
    return replace(period, label=clean(period.label), summary=clean(period.summary), moments=moments)


def _moment_lines(ctx: ToolContext, moment: Moment) -> list[str]:
    """One moment and the lines of its messages. The messages are registered here."""
    named = f"{moment.title}: {moment.detail}" if moment.title and moment.detail else moment.title or moment.detail
    line = " · ".join(part for part in (moment.date, named) if part)
    if moment.quote and moment.messages:
        line += f' Quote: "{moment.quote}"'
    if not moment.messages:
        return [f"- {line}"]
    limits = replace(ctx.limits, max_message_chars=min(ctx.limits.max_message_chars, MOMENT_MESSAGE_CHARS))
    block = render_block(moment.messages, book=ctx.book, registry=ctx.registry, limits=limits, marks={})
    return [f"- {line}", *block[1:]]


def _moment_costs(ctx: ToolContext, periods: Sequence[Period]) -> list[list[int]]:
    """The tokens of each moment. The lines are written to be counted, and then forgotten."""
    count = ctx.limits.count_tokens
    mark = ctx.registry.checkpoint()
    costs = [[count("\n".join(_moment_lines(ctx, m))) + 1 for m in period.moments] for period in periods]
    ctx.registry.rollback(mark)
    return costs


def _add_moment(ctx: ToolContext, writer: ResultWriter, moment: Moment) -> bool:
    mark = ctx.registry.checkpoint()
    if writer.add("\n".join(_moment_lines(ctx, moment))):
        return True
    ctx.registry.rollback(mark)  # no message stays registered without its line
    return False


def _cut_note(fit: Fit, dropped: int) -> str:
    parts = []
    if fit.share < 1.0:
        parts.append(f"each summary keeps about {int(fit.share * 100)}% of its text")
    if dropped:
        parts.append(f"{plural(dropped, 'moment')} {'is' if dropped == 1 else 'are'} not shown")
    return f"[Cut to fit: {', and '.join(parts)}. {_NARROW}]"
