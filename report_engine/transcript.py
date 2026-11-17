"""Transcript lines: what the model reads, and how a reference maps back.

A line is `m12 09:41 Mina: text`. References restart at m1 in every period, so
a period's text depends only on its own messages. Speaker names come from the
caller's labeler; a name that is empty or looks like a phone number or an
e-mail address is replaced before it can reach a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from report_engine.dates import DAY_LENGTH, parse_day
from report_engine.errors import SourceError
from report_engine.ports import Labeler

OWN_LABEL = "You"
UNSAVED_LABEL = "Unsaved contact"
MAX_LABEL_CHARS = 40
MIN_PHONE_DIGITS = 7
ELISION = " [...] "
COUNT_REF = "m000"
TRANSCRIPT_OPEN = "<transcript>"
TRANSCRIPT_CLOSE = "</transcript>"
ZERO_WIDTH_SPACE = chr(0x200B)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_STAMP_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2} ([01]\d|2[0-3]):[0-5]\d:[0-5]\d")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_REF = re.compile(r"[\[(\s]*[mM](\d{1,9})[\])\s.,;]*")
_FRAME = re.compile(r"<(/?transcript)", re.IGNORECASE)
# zero-width space, zero-width non-joiner, word joiner, byte-order mark, object replacement
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u2060\ufeff\ufffc"))


@dataclass(frozen=True, slots=True)
class Line:
    """One message with text. `text` is kept as stored, for quote checks."""

    id: str
    day: str
    clock: str
    label: str
    text: str


@dataclass(frozen=True)
class Rendered:
    """A period as the model reads it, and the speaker names in it."""

    text: str
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Row:
    stamp: str
    id: str
    own: bool
    sender: str
    text: str


def squash(text: str) -> str:
    """One line: runs of white space become one space, invisible marks go."""
    return " ".join(text.translate(_INVISIBLE).split())


def looks_like_handle(name: str) -> bool:
    """True for anything that reads as a phone number or an e-mail address."""
    if _EMAIL.search(name):
        return True
    return sum(char.isdigit() for char in name) >= MIN_PHONE_DIGITS


def clean_label(raw: object) -> str:
    flat = squash(str(raw or "")).replace(":", "")
    return squash(flat)[:MAX_LABEL_CHARS].strip()


def resolve_labels(senders: Iterable[str], labeler: Labeler) -> dict[str, str]:
    """Name every sender. Unsaved ones are numbered in order of first appearance,
    so a sender keeps its name when later messages add new people."""
    names: dict[str, str] = {}
    unsaved = 0
    for sender in senders:
        if sender in names:
            continue
        name = clean_label(labeler(sender))
        if not name or name.casefold() == OWN_LABEL.casefold() or looks_like_handle(name):
            unsaved += 1
            name = UNSAVED_LABEL if unsaved == 1 else f"{UNSAVED_LABEL} {unsaved}"
        names[sender] = name
    return names


def _read_row(message: Mapping[str, Any], known_days: set[str]) -> _Row | None:
    try:
        mid = str(message["id"])
        stamp = message["timestamp"]
        own = bool(message["is_from_me"])
    except KeyError as exc:
        raise SourceError(f"a message from the source has no {exc.args[0]!r} field") from exc
    if not isinstance(stamp, str) or not _STAMP_SHAPE.fullmatch(stamp):
        raise SourceError(f"message {mid!r} has a bad timestamp")
    day = stamp[:DAY_LENGTH]
    if day not in known_days:
        try:
            parse_day(day)
        except ValueError as exc:
            raise SourceError(f"message {mid!r} has a bad timestamp") from exc
        known_days.add(day)
    raw = message.get("text")
    text = raw if isinstance(raw, str) else "" if raw is None else str(raw)
    if not squash(text):
        return None
    return _Row(stamp=stamp, id=mid, own=own, sender=str(message.get("sender") or ""), text=text)


def build_lines(messages: Iterable[Mapping[str, Any]], labeler: Labeler) -> tuple[Line, ...]:
    """Check, order and name the messages. Messages with no text are left out."""
    known_days: set[str] = set()
    rows = [row for row in (_read_row(message, known_days) for message in messages) if row]
    if any(later.stamp < earlier.stamp for earlier, later in zip(rows, rows[1:])):
        rows = sorted(rows, key=lambda row: row.stamp)
    names = resolve_labels((row.sender for row in rows if not row.own), labeler)
    return tuple(
        Line(
            id=row.id,
            day=row.stamp[:DAY_LENGTH],
            clock=row.stamp[11:16],
            label=OWN_LABEL if row.own else names[row.sender],
            text=row.text,
        )
        for row in rows
    )


def day_header(day: str) -> str:
    return f"--- {day} {WEEKDAYS[parse_day(day).weekday()]} ---"


def render_text(text: str, max_chars: int) -> str:
    """Message text on one line. A very long message keeps its start and its end.

    Text that imitates the transcript frame gets an invisible break, so a
    message can not end the transcript early. Quote checks ignore the break.
    """
    flat = _FRAME.sub(lambda match: f"<{ZERO_WIDTH_SPACE}{match.group(1)}", squash(text))
    if len(flat) <= max_chars:
        return flat
    head = max_chars * 3 // 4
    tail = max_chars - head
    return flat[:head].rstrip() + ELISION + flat[len(flat) - tail :].lstrip()


def render_line(ref: str, line: Line, max_chars: int) -> str:
    return f"{ref} {line.clock} {line.label}: {render_text(line.text, max_chars)}"


def _render(lines: Sequence[Line], max_chars: int, ref_for: Callable[[int], str]) -> str:
    parts: list[str] = []
    day = ""
    for number, line in enumerate(lines, start=1):
        if line.day != day:
            day = line.day
            parts.append(day_header(day))
        parts.append(render_line(ref_for(number), line, max_chars))
    return "\n".join(parts)


def render_period(lines: Sequence[Line], max_chars: int) -> Rendered:
    """The text of one period, with references m1..mN in order."""
    text = _render(lines, max_chars, lambda number: f"m{number}")
    return Rendered(text=text, labels=tuple(sorted({line.label for line in lines})))


def render_for_count(lines: Sequence[Line], max_chars: int) -> str:
    """Same text with one fixed reference, so a size does not depend on numbering."""
    return _render(lines, max_chars, lambda number: COUNT_REF)


def resolve_ref(ref: object, lines: Sequence[Line]) -> Line | None:
    """Map a reference from the model back to its line. None when it is not one."""
    match = _REF.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        return None
    number = int(match.group(1))
    return lines[number - 1] if 1 <= number <= len(lines) else None
