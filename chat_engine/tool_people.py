"""find_people: people found by the kind of relationship and its story.

The people come from `ports.PeopleSearch`. This tool keeps the scope, masks
the text of each profile and names each conversation by its alias, so the
model can check a person with conversation_stats, conversation_digest or
read_around before it names them.

Rules:
- Scope: the conversations of the scope go to the port, which ranks only
  the people who have one of them. A person none of whose conversations is
  in the scope is left out again here.
  Only the conversations inside the scope are named. A person whose figures
  lie outside the days is left out, and so is a turning point outside them.
  In a limited scope the figures are not shown: they count every
  conversation of the person.
- Masking: a profile was written by a model from whole conversations, so it
  can hold any phone number or address. Every one goes, as in a digest
  (`handle_mask.mask_every_handle`). The description is masked the same way
  before it leaves.
- A failure of the port gives one line, and the question goes on. The port
  is not asked again in that question: a provider that hangs costs the
  question one wait, not one for each call.
- A cancel that comes while the port searches is looked at as soon as the
  port returns: nothing more is read.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .handle_mask import mask_every_handle
from .ports import PERSON_KINDS, PERSON_STATUSES
from .render import ResultWriter
from .scope import parse_day
from .tool_args import ToolInputError, day_arg, int_arg, text_arg
from .tool_context import Prepared, ToolContext, ToolOutcome, plural

logger = logging.getLogger(__name__)
PEOPLE_LIMIT_DEFAULT = 5
PEOPLE_LIMIT_MAX = 10
SEED_PEOPLE = 5  # the people the seeded lookup asks for
SUMMARY_CHARS = 700
TURNING_POINTS = 5
_NOT_AVAILABLE = "The people index was not available: nobody was looked up. Use the other tools."
_NO_QUOTE = "A story is a summary, not a quote. Check a person in their own conversation before you name them."
_HIDDEN = "   Figures: call conversation_stats inside the scope."
_NO_FIGURES = "   Figures: call conversation_stats."
_CUT = "[Result cut to fit: {left} not shown. Ask for fewer people, or narrow the days.]"
_STOPPED = ToolOutcome("", "Stopped before people were looked up", ok=False)


@dataclass(frozen=True)
class PeopleRequest:
    """One search of people. `description` is masked. `what` names the search; `what_one` for one person."""

    description: str
    kinds: Optional[tuple[str, ...]]
    since: Optional[str]
    until: Optional[str]
    limit: int
    what: str
    what_one: str


@dataclass(frozen=True)
class Person:
    """One person of the port, cut to the scope and masked."""

    conversations: tuple[str, ...]
    kind: str
    status: str
    summary: str
    turning_points: tuple[tuple[str, str], ...]
    figures: Mapping[str, Any]


def prepare_find_people(ctx: ToolContext, arguments: Mapping[str, Any]) -> Prepared:
    description = text_arg(arguments, "description")
    if not description:
        raise ToolInputError('description needs a few words on the relationship, such as "a talking stage that faded".')
    kinds = _kinds(arguments.get("kinds"))
    since, until = ctx.days(day_arg(arguments, "since"), day_arg(arguments, "until"))
    limit = int_arg(arguments, "limit", default=PEOPLE_LIMIT_DEFAULT, low=1, high=PEOPLE_LIMIT_MAX)
    masked = masked_text(ctx, description)
    what = f"like “{masked}”"
    request = PeopleRequest(masked, kinds, since, until, limit, what, what)
    return Prepared(f"Finding people {what}", people_lookup(ctx, request))


def seeded_people(ctx: ToolContext, question: str) -> Prepared:
    """The lookup before the first model call: the question is the description."""
    since, until = ctx.days(None, None)
    request = PeopleRequest(
        masked_text(ctx, question), None, since, until, SEED_PEOPLE, "who fit the question", "who fits the question"
    )
    return Prepared("Finding people who fit your question", people_lookup(ctx, request))


def masked_text(ctx: ToolContext, text: str) -> str:
    """Text with no phone number and no address, on one line."""
    ctx.listed()  # the labeler knows every handle of the archive before the text is masked
    return " ".join(mask_every_handle(text, ctx.labeler).split())


def _kinds(value: object) -> Optional[tuple[str, ...]]:
    if value is None or value in ([], ""):
        return None
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(item, str) and item in PERSON_KINDS for item in items):
        raise ToolInputError(f"kinds must be a list of: {', '.join(PERSON_KINDS)}.")
    return tuple(dict.fromkeys(items))


def people_lookup(ctx: ToolContext, request: PeopleRequest) -> Callable[[int], ToolOutcome]:
    """The lookup. The port is asked one time, also when the result is written twice."""
    asked: list[Optional[list[Any]]] = []

    def run(budget: int) -> ToolOutcome:
        if not asked:
            if ctx.memory.cancel.cancelled:
                return _STOPPED
            asked.append(None if ctx.people_failed else _ask_port(ctx, request))
        if ctx.memory.cancel.cancelled:
            return _STOPPED
        rows = asked[0]
        if rows is None:
            return ToolOutcome(_NOT_AVAILABLE, "Could not look for people", ok=False)
        found = [one for one in (_inside(ctx, request, row) for row in rows) if one is not None]
        return _write(ctx, request, found[:request.limit], budget)

    return run


def _ask_port(ctx: ToolContext, request: PeopleRequest) -> Optional[list[Any]]:
    """The rows of the port. None when it fails: then the question does not ask it again."""
    scoped = ctx.conversations(None)
    try:
        found = ctx.memory.people.search(  # type: ignore[union-attr]
            request.description,
            kinds=None if request.kinds is None else list(request.kinds),
            since=request.since,
            until=request.until,
            limit=request.limit,
            conversations=None if scoped is None else list(scoped),
        )
        if not isinstance(found, (list, tuple)):
            raise TypeError("the port gave no list")
    except Exception as error:  # the question goes on with the other tools
        logger.warning("the search of people failed: %s", type(error).__name__)
        ctx.people_failed = True
        return None
    return list(found[:request.limit])


def _day(value: object) -> str:
    try:
        return parse_day(value) or ""
    except (TypeError, ValueError):
        return ""


def _inside(ctx: ToolContext, request: PeopleRequest, row: object) -> Optional[Person]:
    """One row as a person inside the scope and the days. None when it has another form or lies outside."""
    if not isinstance(row, Mapping) or not isinstance(row.get("conversations"), (list, tuple)):
        return None
    listed = [c for c in row["conversations"] if isinstance(c, str)]
    conversations = tuple(c for c in dict.fromkeys(listed) if ctx.scope.allows(c) and ctx.book.get(c) is not None)
    figures = row.get("figures") if isinstance(row.get("figures"), Mapping) else {}
    first, last = _day(figures.get("first_day")), _day(figures.get("last_day"))
    low, high = request.since or "", request.until or ""
    if not conversations or (last and low and last < low) or (first and high and first > high):
        return None
    kind, status = row.get("kind"), row.get("status")
    summary = masked_text(ctx, row["summary"]) if isinstance(row.get("summary"), str) else ""
    return Person(
        conversations=conversations,
        kind=kind if kind in PERSON_KINDS else "unclear",
        status=status if status in PERSON_STATUSES else "unclear",
        summary=summary if len(summary) <= SUMMARY_CHARS else f"{summary[:SUMMARY_CHARS].rstrip()}...",
        turning_points=_turning_points(ctx, request, row.get("turning_points")),
        figures=figures,
    )


def _turning_points(ctx: ToolContext, request: PeopleRequest, value: object) -> tuple[tuple[str, str], ...]:
    """The dated turning points inside the days, masked. At most `TURNING_POINTS`."""
    kept: list[tuple[str, str]] = []
    for item in value if isinstance(value, (list, tuple)) else ():
        if not isinstance(item, Mapping) or not isinstance(item.get("what"), str):
            continue
        day, what = _day(item.get("date")), masked_text(ctx, item["what"])
        inside = (not request.since or day >= request.since) and (not request.until or day <= request.until)
        if day and what and inside:
            kept.append((day, what))
    return tuple(kept[:TURNING_POINTS])


def _write(ctx: ToolContext, request: PeopleRequest, found: Sequence[Person], budget: int) -> ToolOutcome:
    kinds = "" if not request.kinds else f" (kinds: {', '.join(k.replace('_', ' ') for k in request.kinds)})"
    place = " in the scope the user chose" if ctx.scope.conversations else ""
    head = f"People {request.what}{kinds}{place}{ctx.when(request.since, request.until)}"
    if not found:
        text = f"{head}: nobody found. Use the other tools."
        return ToolOutcome(text, f"Looked for people {request.what}: nobody found")
    writer = ResultWriter(budget, ctx.limits.count_tokens)
    writer.add(f"{head}: {len(found)} found, best first. {_NO_QUOTE}", force=True)
    shown = 0
    for index, one in enumerate(found, start=1):
        if not writer.add("\n" + _block(ctx, index, one)):
            break
        shown += 1
    if shown < len(found):
        writer.add(_CUT.format(left=plural(len(found) - shown, "more person", "more people")), force=True)
    what = request.what_one if shown == 1 else request.what
    return ToolOutcome(writer.text(), f"Found {plural(shown, 'person', 'people')} {what}")


def _block(ctx: ToolContext, index: int, one: Person) -> str:
    book = ctx.book
    aliases = ", ".join(book.alias(conversation) for conversation in one.conversations)
    lines = [
        f"{index}. {book.label(one.conversations[0])} · {one.kind.replace('_', ' ')} · {one.status} · "
        f"conversations {aliases}",
        _HIDDEN if ctx.scope.is_limited else _figures(one, book.is_group(one.conversations[0])),
    ]
    if one.summary:
        lines.append(f"   Story: {one.summary}")
    if one.turning_points:
        lines.append("   Turning points: " + "; ".join(f"{day} {what}" for day, what in one.turning_points))
    return "\n".join(lines)


def _number(value: object) -> Optional[float]:
    known = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return float(value) if known else None  # type: ignore[arg-type]


def _figures(one: Person, group: bool) -> str:
    """The span and the counts of the person, as far as the port gave them."""
    figures = one.figures
    first, last = _day(figures.get("first_day")), _day(figures.get("last_day"))
    messages, sent, received = (_number(figures.get(key)) for key in ("messages", "sent", "received"))
    active, gap, share = (_number(figures.get(key)) for key in ("active_days", "longest_gap_days", "wrote_first_share"))
    parts = [f"{first} to {last}"] if first and last else []
    if messages is not None:
        other = "others" if group else "them"
        sides = f" (you {int(sent):,}, {other} {int(received):,})" if sent is not None and received is not None else ""
        parts.append(plural(int(messages), "message") + sides)
    if active is not None:
        parts.append(f"active on {plural(int(active), 'day')}")
    if gap is not None:
        parts.append(f"longest gap {plural(round(gap), 'day')}")
    if share is not None and share <= 1:
        parts.append(f"you wrote first on {round(share * 100)}% of the active days")
    return "   " + " · ".join(parts) if parts else _NO_FIGURES
