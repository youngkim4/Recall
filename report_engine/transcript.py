"""Transcript lines: what the model reads, and how a reference maps back.

A line is `m12 09:41 Mina: text`. References restart at m1 in every period, so
a period's text depends only on its own messages. Speaker names come from the
caller's labeler; a name that is empty or looks like a phone number, a short
code or an e-mail address is replaced before it can reach a prompt.

Every piece of outside text that goes into a prompt passes `shield`: it is
put on one line, and anything that imitates a data frame of a prompt is
broken with an invisible character.

A message has a hard cap. A longer one keeps its start and gets a mark at
its end. The cut is made once, when the lines are built, so every size and
every transcript sees the same text. Quote checks read the whole message.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from report_engine.dates import DAY_LENGTH, has_stamp_shape, parse_day
from report_engine.errors import SourceError
from report_engine.ports import Labeler

OWN_LABEL = "You"
UNSAVED_LABEL = "Unsaved contact"
MAX_LABEL_CHARS = 40
MAX_MESSAGE_CHARS = 4_000
MIN_PHONE_DIGITS = 7
ELISION_MARK = "[...]"
SPACED_SCRIPTS_END = 0x0530  # Latin, Greek and Cyrillic end here
COUNT_REF = "m00000"
FRAME_TAGS = ("transcript", "notes", "periods", "moments")
TRANSCRIPT_OPEN = "<transcript>"
TRANSCRIPT_CLOSE = "</transcript>"
ZERO_WIDTH_SPACE = chr(0x200B)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_REF = re.compile(r"[\[(\s]*[mM](\d{1,9})[\])\s.,;]*", re.ASCII)
_FRAME = re.compile(rf"<(/?(?:{'|'.join(FRAME_TAGS)}))", re.IGNORECASE)
_SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")
# control characters that are not white space, and marks that take no room:
# zero-width space, zero-width non-joiner, word joiner, byte-order mark, object replacement
_INVISIBLE = dict.fromkeys(
    (*range(0x00, 0x09), *range(0x0E, 0x1C), 0x7F, 0x200B, 0x200C, 0x2060, 0xFEFF, 0xFFFC)
)
_HAS_INVISIBLE = re.compile("[" + "".join(re.escape(chr(code)) for code in _INVISIBLE) + "]")


@dataclass(frozen=True, slots=True)
class Line:
    """One message with text. `text` is kept as stored, for quote checks. `flat`
    is the text as a transcript shows it: on one line, shielded, and with only
    its start when the message is too long. `cut` says that it was shortened."""

    id: str
    day: str
    clock: str
    label: str
    text: str
    flat: str = ""
    cut: bool = False


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
    flat: str
    cut: bool


def scrub(text: str) -> str:
    """Repair what UTF-8 can not carry: a surrogate with no partner becomes U+FFFD."""
    if not _SURROGATE.search(text):
        return text
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def squash(text: str) -> str:
    """One line: runs of white space become one space, invisible marks go."""
    if _HAS_INVISIBLE.search(text):
        text = text.translate(_INVISIBLE)
    return " ".join(text.split())


def shield(text: str) -> str:
    """Outside text made safe for a prompt: one line, and no tag of a data frame.
    The break is a zero-width space, which quote checks ignore."""
    return _FRAME.sub(lambda match: f"<{ZERO_WIDTH_SPACE}{match.group(1)}", squash(text))


def frame(tag: str, body: str) -> str:
    """A block of data in a prompt. `body` must be made of shielded text."""
    return f"<{tag}>\n{body}\n</{tag}>"


def looks_like_handle(name: str) -> bool:
    """True for anything that reads as a phone number, a short code or an e-mail address."""
    plain = unicodedata.normalize("NFKC", name)
    if "@" in plain:
        return True
    digits = sum(char.isdigit() for char in plain)
    if digits >= MIN_PHONE_DIGITS:
        return True
    return digits > 0 and not any(char.isalpha() for char in plain)


def clean_label(raw: object) -> str:
    """A name on one line, with no colon and no frame tag. Not yet cut to length."""
    return shield(shield(scrub(str(raw or ""))).replace(":", ""))


def resolve_labels(senders: Iterable[str], labeler: Labeler) -> dict[str, str]:
    """Name every sender. Unsaved ones are numbered in order of first appearance,
    so a sender keeps its name when later messages add new people.

    The handle check reads the whole name. Only then is the name cut to length:
    a cut must never turn an address into something that passes."""
    # `taken` maps each label in use to the full name behind it
    names: dict[str, str] = {}
    taken: dict[str, str] = {}
    unsaved = 0
    for sender in senders:
        if sender in names:
            continue
        name = clean_label(_ask(labeler, sender))
        if not name or name.casefold() == OWN_LABEL.casefold() or looks_like_handle(name):
            unsaved += 1
            name = UNSAVED_LABEL if unsaved == 1 else f"{UNSAVED_LABEL} {unsaved}"
        names[sender] = _cut_label(name, taken)
    return names


def _ask(labeler: Labeler, sender: str) -> object:
    """The labeler is foreign code. Its error becomes a SourceError that names
    only the type: the text of the error could hold a handle."""
    try:
        return labeler(sender)
    except Exception as exc:
        raise SourceError(f"the labeler raised {type(exc).__name__}") from exc


def _cut_label(name: str, taken: dict[str, str]) -> str:
    """The name cut to length. Two senders with one full name are one person and
    share the label. When the cut alone makes two names equal, the later one
    gets a number, so two people never read as one."""
    label = name[:MAX_LABEL_CHARS].strip()
    number = 1
    while taken.get(label, name) != name:
        number += 1
        suffix = f" {number}"
        label = name[: MAX_LABEL_CHARS - len(suffix)].strip() + suffix
    taken[label] = name
    return label


def _own_flag(value: object, mid: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise SourceError(f"message {mid!r} has an is_from_me value that is not true or false")


def is_word_char(char: str) -> bool:
    """A letter or a digit of a script that puts spaces between words."""
    return char.isalnum() and ord(char) < SPACED_SCRIPTS_END


def word_cut(text: str, keep: int) -> str:
    """At most the first `keep` characters of a text. The cut does not fall
    inside a word: a quote can not end inside a word, so a text that was cut
    there could not be found in its message."""
    end = min(keep, len(text))
    if is_word_char(text[end : end + 1]):
        while end and is_word_char(text[end - 1]):
            end -= 1
    return text[:end].rstrip()


def cut_start(flat: str, keep: int) -> str:
    """The start of a text that is too long, and the mark that says it was cut."""
    return f"{word_cut(flat, keep)} {ELISION_MARK}".lstrip()


def _read_row(message: Mapping[str, Any], known_days: set[str], max_chars: int) -> _Row | None:
    try:
        mid = scrub(str(message["id"]))
        stamp = message["timestamp"]
        own = _own_flag(message["is_from_me"], mid)
    except KeyError as exc:
        raise SourceError(f"a message from the source has no {exc.args[0]!r} field") from exc
    if not mid.strip():
        raise SourceError("a message from the source has an empty id")
    if not has_stamp_shape(stamp):
        raise SourceError(f"message {mid!r} has a bad timestamp")
    day = stamp[:DAY_LENGTH]
    if day not in known_days:
        try:
            parse_day(day)
        except ValueError as exc:
            raise SourceError(f"message {mid!r} has a bad timestamp") from exc
        known_days.add(day)
    raw = message.get("text")
    text = scrub(raw if isinstance(raw, str) else "" if raw is None else str(raw))
    flat = shield(text)
    if not flat:
        return None
    sender = scrub(str(message.get("sender") or ""))
    cut = len(flat) > max_chars
    shown = cut_start(flat, max_chars) if cut else flat
    return _Row(stamp=stamp, id=mid, own=own, sender=sender, text=text, flat=shown, cut=cut)


def build_lines(
    messages: Iterable[Mapping[str, Any]],
    labeler: Labeler,
    max_message_chars: int = MAX_MESSAGE_CHARS,
) -> tuple[Line, ...]:
    """Check, order and name the messages. Messages with no text are left out.
    A message longer than `max_message_chars` keeps its start."""
    known_days: set[str] = set()
    read = (_read_row(message, known_days, max_message_chars) for message in messages)
    rows = [row for row in read if row]
    if any(later.stamp < earlier.stamp for earlier, later in zip(rows, rows[1:], strict=False)):
        rows = sorted(rows, key=lambda row: row.stamp)
    names = resolve_labels((row.sender for row in rows if not row.own), labeler)
    return tuple(
        Line(
            id=row.id,
            day=row.stamp[:DAY_LENGTH],
            clock=row.stamp[11:16],
            label=OWN_LABEL if row.own else names[row.sender],
            text=row.text,
            flat=row.flat,
            cut=row.cut,
        )
        for row in rows
    )


def day_header(day: str) -> str:
    return f"--- {day} {WEEKDAYS[parse_day(day).weekday()]} ---"


def render_line(ref: str, line: Line) -> str:
    return f"{ref} {line.clock} {line.label}: {line.flat or shield(line.text)}"


def _render(lines: Sequence[Line], ref_for: Callable[[int], str]) -> str:
    parts: list[str] = []
    day = ""
    for number, line in enumerate(lines, start=1):
        if line.day != day:
            day = line.day
            parts.append(day_header(day))
        parts.append(render_line(ref_for(number), line))
    return "\n".join(parts)


def render_period(lines: Sequence[Line]) -> Rendered:
    """The text of one period, with references m1..mN in order."""
    text = _render(lines, lambda number: f"m{number}")
    return Rendered(text=text, labels=tuple(sorted({line.label for line in lines})))


def render_for_count(lines: Sequence[Line]) -> str:
    """Same text with one fixed reference, so a size does not depend on numbering.
    The fixed reference is as long as the longest real one, so a size never runs low."""
    return _render(lines, lambda _number: COUNT_REF)


def resolve_ref(ref: object, lines: Sequence[Line]) -> Line | None:
    """Map a reference from the model back to its line. None when it is not one."""
    match = _REF.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        return None
    number = int(match.group(1))
    return lines[number - 1] if 1 <= number <= len(lines) else None
