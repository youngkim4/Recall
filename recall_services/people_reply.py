"""A reply of the profile model, checked for shape and cleaned.

- A reply that is not an object with `kind`, `status`, `summary` (text) and
  `turning_points` (a list of objects with `date` and `what` as text) is in
  the wrong shape: `parse_reply` gives None. So is a summary with no words left.
- A kind or a status that is not one of the words is `unclear`. A group is
  always `group`, and only a group is one.
- A turning point whose date is no day, or a day outside the span of the
  figures, or whose text is empty, is dropped. The first `TURNING_POINTS`
  that are left are kept, in the order of their days.
- Text loses the line references of the engine (`chat_digest_refs`), and
  every handle is masked by `mask`: the build gives `Planner.mask`
  (`handle_mask.mask_every_handle` with the labeler of chat), so a handle of
  the archive becomes its saved name or its mask, and any other phone number
  or email address a fixed phrase. Text longer than its limit is cut at a
  word, with a mark at its end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from recall_services.people_prompt import (
    GROUP_KIND,
    KINDS,
    STATUSES,
    SUMMARY_CHARS,
    TURNING_POINTS,
    UNCLEAR,
    WHAT_CHARS,
)
from report_engine.chat_digest_refs import without_refs
from report_engine.transcript import squash

CUT_MARK = "…"
DAY_CHARS = 10

Mask = Callable[[str], str]


@dataclass(frozen=True)
class TurningPoint:
    date: str
    what: str


@dataclass(frozen=True)
class ProfileText:
    """What the model said of one profile, checked and cleaned."""

    kind: str
    status: str
    summary: str
    turning_points: tuple[TurningPoint, ...]


def clean_text(text: str, mask: Mask) -> str:
    """Text of a reply as a profile keeps it: one line, no reference of the engine, no handle."""
    return squash(mask(without_refs(text)))


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit - len(CUT_MARK)]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip(" ,;:.") + CUT_MARK


def _day(value: str) -> str | None:
    if len(value) != DAY_CHARS:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _is_text(value: object) -> bool:
    return isinstance(value, str)


def _shaped(reply: object) -> bool:
    if not isinstance(reply, Mapping):
        return False
    if not all(_is_text(reply.get(name)) for name in ("kind", "status", "summary")):
        return False
    points = reply.get("turning_points")
    if not isinstance(points, list):
        return False
    return all(isinstance(point, Mapping) and _is_text(point.get("date")) and _is_text(point.get("what")) for point in points)


def _points(raw: Sequence[Mapping[str, Any]], span: tuple[str, str], mask: Mask) -> tuple[TurningPoint, ...]:
    first, last = span
    kept: list[TurningPoint] = []
    for point in raw:
        day = _day(str(point["date"]).strip())
        what = _cut(clean_text(str(point["what"]), mask), WHAT_CHARS)
        if day is not None and first <= day <= last and what:
            kept.append(TurningPoint(day, what))
    # the first ones that the model gave, in the order of their days
    return tuple(sorted(kept[:TURNING_POINTS], key=lambda point: point.date))


def _kind(said: str, is_group: bool) -> str:
    if is_group:
        return GROUP_KIND
    return said if said in KINDS and said != GROUP_KIND else UNCLEAR


def parse_reply(reply: object, *, span: tuple[str, str], is_group: bool, mask: Mask) -> ProfileText | None:
    """The profile that a reply gives, or None when the reply is in the wrong shape."""
    if not _shaped(reply):
        return None
    assert isinstance(reply, Mapping)
    summary = _cut(clean_text(str(reply["summary"]), mask), SUMMARY_CHARS)
    if not summary:
        return None
    status = str(reply["status"])
    return ProfileText(
        kind=_kind(str(reply["kind"]), is_group),
        status=status if status in STATUSES else UNCLEAR,
        summary=summary,
        turning_points=_points(reply["turning_points"], span, mask),
    )
