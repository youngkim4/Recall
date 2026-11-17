"""search_messages with no search word: read one conversation from its start or its end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .browse import browse
from .render import fit_messages
from .tool_args import ToolInputError, bool_arg, choice_arg, day_arg, int_arg, list_arg
from .tool_context import Prepared, ToolContext, ToolOutcome, plural
from .tool_limits import ORDERS, SEARCH_LIMIT_DEFAULT, SEARCH_LIMIT_MAX

_NO_WAY_IN = (
    "Cannot read {title} without a search word: no way into that period was found. "
    'Use count_messages with group_by "day" to see the active days, or search for a word.'
)


@dataclass(frozen=True)
class BrowseRequest:
    conversation: str
    edge: str  # "oldest" reads from the start, "newest" from the end
    limit: int
    since: Optional[str]
    until: Optional[str]

    @property
    def which(self) -> str:
        return "first" if self.edge == "oldest" else "latest"


def prepare_browse(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    if bool_arg(arguments, "from_me") is not None:
        raise ToolInputError("from_me needs a search word in query.")
    edge = "oldest" if choice_arg(arguments, "order", ORDERS, "newest") == "oldest" else "newest"
    since, until = day_arg(arguments, "since"), day_arg(arguments, "until")
    limit = int_arg(arguments, "limit", default=SEARCH_LIMIT_DEFAULT, low=1, high=SEARCH_LIMIT_MAX)
    conversations = ctx.conversations(list_arg(arguments, "conversations"))
    if conversations is None or len(conversations) != 1:
        raise ToolInputError(
            "without a search word, give exactly one conversation to read. Or put a word in query."
        )
    since, until = ctx.days(since, until)
    request = BrowseRequest(conversations[0], edge, limit, since, until)
    return Prepared(f"Reading {_words(ctx, request)}", lambda budget: run_browse(ctx, request, budget))


def _words(ctx: ToolContext, request: BrowseRequest, count: Optional[int] = None) -> str:
    """ "the first messages with Maya Okafor", for the status line and the step."""
    joiner = "in" if ctx.book.is_group(request.conversation) else "with"
    what = "messages" if count is None else plural(count, "message")
    return f"the {request.which} {what} {joiner} {ctx.book.label(request.conversation)}"


def run_browse(ctx: ToolContext, request: BrowseRequest, budget: int) -> ToolOutcome:
    found = browse(
        ctx.archive, request.conversation, edge=request.edge, limit=request.limit,
        since=request.since, until=request.until,
    )
    title = f"{ctx.book.title(request.conversation)}{ctx.when(request.since, request.until)}"
    if not found.anchored:
        return ToolOutcome(_NO_WAY_IN.format(title=title), f"Could not read {_words(ctx, request)}", ok=False)
    messages = [message for message in found.messages if ctx.allows(message)]
    if not messages:
        return ToolOutcome(f"No messages in {title}.", f"Read {_words(ctx, request)}: nothing there")
    backwards = request.edge == "oldest"
    facts = ctx.book.get(request.conversation)
    everything = facts is not None and len(messages) >= int(facts.get("message_count") or 0)
    whole = everything and not (request.since or request.until)

    def header(count: int) -> str:
        if not found.reached_edge:
            return f"{plural(count, 'message')} near the {'start' if backwards else 'end'} of {title}:"
        note = " (all of it)" if whole and count == len(messages) else ""
        return f"{request.which.capitalize()} {plural(count, 'message')} of {title}{note}:"

    writer, shown = fit_messages(
        messages,
        header=header,
        drop=lambda rows: rows[:-1] if backwards else rows[1:],
        budget=budget,
        book=ctx.book,
        registry=ctx.registry,
        limits=ctx.limits,
        marks={},
        titled=False,
    )
    if len(shown) < len(messages):
        writer.add("[Result cut to fit. Ask again with a smaller limit, or read_around a line.]", force=True)
    if not found.reached_edge:
        side = "start" if backwards else "end"
        writer.add(f"[This conversation is very long: its {side} was not reached.]", force=True)
    return ToolOutcome(writer.text(), f"Read {_words(ctx, request, len(shown))}")
