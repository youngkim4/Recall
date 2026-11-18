"""The text of a conversation digest: periods, summaries, moments and the messages that back them.

Every text of the source is masked before it is measured, so the result does
not grow after the plan is made. A summary can hold the handle of anyone, so
every phone number and every address in it goes (`handle_mask.py`). A message
of a moment is registered as evidence only when its line is in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .digest_fit import Fit, cut_summary, fit_digest, periods_that_fit
from .handle_mask import mask_every_handle
from .ports import Message
from .render import NOTE_TOKENS, ResultWriter, render_block
from .tool_context import ToolContext, plural

MOMENT_MESSAGE_CHARS = 300  # a message under a moment is a pointer: read_around shows the whole of it
LABEL_CHARS = 60  # the most characters of the label of a period
_NO_QUOTE = "A summary is not a quote: quote and cite only the message lines."
_NARROW = "Narrow the days with since and until to read more."
_NOT_SHOWN = "[{periods} not shown. Narrow the days with since and until.]"


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
    """What the digest leaves out before any cut: periods that failed, periods and moments outside the scope."""

    failed: int = 0
    periods_outside: int = 0
    moments_outside: int = 0

    def lines(self) -> list[str]:
        lines = []
        if self.failed:
            verb = "is" if self.failed == 1 else "are"
            lines.append(f"[{plural(self.failed, 'period')} could not be built and {verb} missing.]")
        outside = [
            plural(count, word)
            for count, word in ((self.periods_outside, "period"), (self.moments_outside, "moment"))
            if count
        ]
        if outside:
            lines.append(f"[Outside the scope and not shown: {', '.join(outside)}.]")
        return lines


@dataclass(frozen=True)
class Written:
    """A digest as text, the periods in it, and the periods of the digest."""

    text: str
    shown: int
    periods: int


def write_digest(ctx: ToolContext, title: str, periods: Sequence[Period], notes: Notes, budget: int) -> Written:
    """The digest under `budget` tokens. Periods that do not fit are counted in the header and in a last line."""
    count = ctx.limits.count_tokens
    masked = [_masked(ctx, period) for period in periods]
    heads = [_head(period) for period in masked]
    room = max(0, budget - NOTE_TOKENS)
    left_out = _NOT_SHOWN.format(periods=plural(len(masked), "period"))
    shown = _fitting(ctx, title, masked, heads, notes, room)
    if shown < len(masked):  # the line that counts the rest needs room too
        room = max(0, room - count(left_out) - 1)
        shown = _fitting(ctx, title, masked, heads, notes, room)
    kept, heads = masked[:shown], heads[:shown]
    header = _header(title, kept, len(masked), notes)
    costs = _moment_costs(ctx, kept)
    fit = fit_digest(
        fixed=count(header) + sum(count(head) + 1 for head in heads) + len(kept),
        summaries=[count(period.summary) for period in kept],
        moments=costs,
        room=room,
    )
    writer = ResultWriter(budget, count)
    writer.add(header, force=True)
    dropped = fit.dropped(costs)
    for period, head, moments in zip(kept, heads, fit.kept, strict=True):
        writer.add(head, force=True)
        writer.add(cut_summary(period.summary, fit.summary_tokens(count(period.summary)), count), force=True)
        dropped += sum(1 for moment in period.moments[:moments] if not _add_moment(ctx, writer, moment))
    if fit.share < 1.0 or dropped:
        writer.add(_cut_note(fit, dropped), force=True)
    if shown < len(masked):
        writer.add(_NOT_SHOWN.format(periods=plural(len(masked) - shown, "period")), force=True)
    return Written(writer.text(), shown, len(masked))


def _fitting(
    ctx: ToolContext, title: str, periods: Sequence[Period], heads: Sequence[str], notes: Notes, room: int
) -> int:
    """How many periods fit in `room`, counted from the first."""
    count = ctx.limits.count_tokens
    return periods_that_fit(
        header=count(_header(title, periods, len(periods) + 1, notes)),  # the longest header there can be
        heads=[count(head) + 1 for head in heads],
        summaries=[count(period.summary) for period in periods],
        room=room,
    )


def _header(title: str, shown: Sequence[Period], periods: int, notes: Notes) -> str:
    messages = plural(sum(period.messages for period in shown), "message")
    if not periods:
        found = "nothing to show."
    elif len(shown) == periods:
        found = f"{plural(periods, 'period')}, {messages}. {_NO_QUOTE}"
    else:
        found = f"{len(shown):,} of {plural(periods, 'period')} shown, {messages} in them. {_NO_QUOTE}"
    return "\n".join([f"Digest of {title}: {found}", *notes.lines()])


def _head(period: Period) -> str:
    days = f"{period.first_day} to {period.last_day}"
    return f"\n== {period.label} · {days} · {plural(period.messages, 'message')} =="


def _masked(ctx: ToolContext, period: Period) -> Period:
    """The period with every text of the source on one line and with no phone number and no address."""

    def clean(text: str) -> str:
        return " ".join(mask_every_handle(text, ctx.labeler).split())

    def quote(text: str) -> str:
        """A quote stays only when it reads as the message line reads: the answer check knows no other form."""
        as_message = " ".join(ctx.book.redact(text).split())
        return as_message if clean(text) == as_message else ""

    label = clean(period.label)
    if len(label) > LABEL_CHARS:
        label = f"{label[:LABEL_CHARS].rstrip()}..."
    moments = tuple(
        replace(m, title=clean(m.title), detail=clean(m.detail), quote=quote(m.quote)) for m in period.moments
    )
    return replace(period, label=label, summary=clean(period.summary), moments=moments)


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
