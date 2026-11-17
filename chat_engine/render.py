"""Compact text the model reads: message lines, conversation titles, budgets.

Every line carries a reference from the evidence registry. People appear by
label only. A long message is cut: the start stays, and a note says how much
is missing. A whole result never goes past its token budget.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence

from .handles import UNKNOWN_MASK
from .ports import ChatArchive, Conversation, Labeler, Message
from .registry import ConversationAliases, EvidenceRegistry

OWNER = "You"
MIN_MESSAGE_CHARS = 200
_READ_AHEAD = 4  # source characters read for each character that can be shown
_NUMBER_AT_END_RE = re.compile(r"(?:\d|[^\w\n]|_)+$")
_NUMBER_AT_START_RE = re.compile(r"^(?:\d|[^\w\n]|_)+")
_MAX_ADDRESS_CHARS = 320
_ADDRESS_MARKS = "._+-@"
_MIME_RE = re.compile(r"^[a-z0-9.+\-]+/[a-z0-9.+\-]+$")
_MEDIA = {
    "image": {"heic", "jpg", "jpeg", "png", "gif", "webp", "tiff", "bmp"},
    "video": {"mov", "mp4", "m4v", "avi"},
    "audio": {"caf", "m4a", "mp3", "amr", "wav", "aac"},
    "contact card": {"vcf"},
    "pdf": {"pdf"},
}

Redact = Callable[[str], str]


def time_key(message: Message) -> tuple[int, str]:
    """The order of messages in time.

    `sent_at` decides. The local time alone is not enough: one hour comes
    twice in the night when summer time ends.
    """
    return int(message.get("sent_at") or 0), str(message.get("timestamp") or "")


def estimate_tokens(text: str) -> int:
    """A cautious token count without a tokenizer: three bytes per token."""
    return (len(str(text or "").encode("utf-8")) + 2) // 3


@dataclass(frozen=True)
class Limits:
    max_message_chars: int = 4000
    result_tokens: int = 3000
    count_tokens: Callable[[str], int] = estimate_tokens


def _address_at(source: str, edge: int) -> Optional[tuple[int, int]]:
    """The run with an "@" that a cut at `edge` would split, as (start, end). None when no such run."""
    if edge <= 0 or edge >= len(source):
        return None
    left = right = edge
    while left > 0 and edge - left < _MAX_ADDRESS_CHARS and _in_address(source[left - 1]):
        left -= 1
    while right < len(source) and right - edge < _MAX_ADDRESS_CHARS and _in_address(source[right]):
        right += 1
    if edge in (left, right) or "@" not in source[left:right]:
        return None
    return left, right


def _in_address(char: str) -> bool:
    return char.isalnum() or char in _ADDRESS_MARKS


def _piece(source: str, start: int, end: int, redact: Optional[Redact]) -> str:
    """source[start:end] with handles masked and whitespace collapsed.

    A number or an address that the slice cuts in two is dropped at that
    edge: half a handle is never shown, because a mask needs the whole of it.
    """
    low, high = max(0, start), min(len(source), end)
    split = _address_at(source, low)
    low = split[1] if split else low
    split = _address_at(source, high)
    high = split[0] if split else high
    piece = source[low:high] if low < high else ""
    if low > 0:
        piece = _NUMBER_AT_START_RE.sub("", piece)
    if high < len(source):
        piece = _NUMBER_AT_END_RE.sub("", piece)
    return " ".join((redact(piece) if redact else piece).split())


def clip_text(
    text: str, max_chars: int, focus: Optional[int] = None, redact: Optional[Redact] = None
) -> str:
    """The text with whitespace collapsed, cut to `max_chars` with a note.

    The start is kept. When `focus` (a match) lies past the start, the part
    around it is shown too. `redact` runs before any cut. The work is bounded
    for a text of any size.
    """
    cap = max(1, int(max_chars))
    source = str(text or "")
    head = _piece(source, 0, cap * _READ_AHEAD, redact)
    if len(source) <= cap * _READ_AHEAD and len(head) <= cap:
        return head
    window = max(MIN_MESSAGE_CHARS // 2, (cap * 2) // 5)
    if focus is None or focus <= cap or window >= cap:
        shown = head[:cap]
        return f"{shown} [cut: {max(1, len(source) - len(shown)):,} more characters]"
    start = max(0, focus - window // 3)
    near = _piece(source, start, start + window * _READ_AHEAD, redact)[:window]
    first = head[:cap - window]
    skipped = max(1, start - len(first))
    rest = len(source) - start - len(near)
    tail = f" [cut: {rest:,} more characters]" if rest > 0 else ""
    return f"{first} [cut: {skipped:,} characters] {near}{tail}"


def _attachment_kind(name: str) -> str:
    """ "image", "video", ... from a mime type or a file name. Never the name itself."""
    lowered = str(name or "").lower().strip()
    extension = lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    for kind, extensions in _MEDIA.items():
        if extension in extensions:
            return kind
    if _MIME_RE.match(lowered):
        major, _, minor = lowered.partition("/")
        if major in ("image", "video", "audio"):
            return major
        if "vcard" in minor:
            return "contact card"
        if minor == "pdf":
            return "pdf"
    return "file"


def _attachment_note(attachments: Sequence[str]) -> str:
    kinds = Counter(_attachment_kind(name) for name in attachments)
    if not kinds:
        return ""
    parts = [kind if count == 1 else f"{kind} x{count}" for kind, count in kinds.items()]
    word = "attachment" if sum(kinds.values()) == 1 else "attachments"
    return f"[{word}: {', '.join(parts)}]"


def message_body(
    message: Message, limits: Limits, focus: Optional[int] = None, redact: Optional[Redact] = None
) -> str:
    if message.get("unsent"):
        return "[unsent message]"
    parts = []
    text = clip_text(message.get("text") or "", limits.max_message_chars, focus, redact)
    if text:
        parts.append(f"[system] {text}" if message.get("kind") == "system" else text)
    note = _attachment_note(message.get("attachments") or [])
    if note:
        parts.append(note)
    if message.get("edited"):
        parts.append("(edited)")
    return " ".join(parts) or "[empty message]"


class ConversationBook:
    """Facts, label and alias of each conversation, read once."""

    def __init__(self, archive: ChatArchive, labeler: Labeler, aliases: ConversationAliases) -> None:
        self._archive = archive
        self._labeler = labeler
        self._aliases = aliases
        self._facts: dict[str, Optional[Conversation]] = {}
        self._labels: dict[str, str] = {}

    def put(self, conversation: Conversation) -> None:
        self._facts.setdefault(str(conversation["id"]), conversation)

    def get(self, conversation_id: str) -> Optional[Conversation]:
        key = str(conversation_id)
        if key not in self._facts:
            self._facts[key] = self._archive.conversation(key)
        return self._facts[key]

    def label(self, conversation_id: str) -> str:
        key = str(conversation_id)
        if key not in self._labels:
            facts = self.get(key)
            self._labels[key] = self._labeler.conversation(facts) if facts else UNKNOWN_MASK
        return self._labels[key]

    def labels(self) -> list[str]:
        """Every label given so far."""
        return list(self._labels.values())

    def redact(self, text: str) -> str:
        return self._labeler.redact(text)

    def alias(self, conversation_id: str) -> str:
        return self._aliases.alias(str(conversation_id))

    def resolve(self, alias: object) -> Optional[str]:
        return self._aliases.resolve(alias)

    def is_group(self, conversation_id: str) -> bool:
        facts = self.get(conversation_id)
        return bool(facts and facts.get("is_group"))

    def title(self, conversation_id: str) -> str:
        kind = "group" if self.is_group(conversation_id) else "direct"
        return f"{self.alias(conversation_id)} · {self.label(conversation_id)} · {kind}"

    def short(self, conversation_id: str) -> str:
        return f"{self.alias(conversation_id)} ({self.label(conversation_id)})"

    def speaker(self, message: Message) -> str:
        if message.get("is_from_me"):
            return OWNER
        conversation_id = str(message["conversation"])
        sender = str(message.get("sender") or "")
        if self.is_group(conversation_id):
            return self._labeler.sender(sender) if sender else "someone in the group"
        return self.label(conversation_id)


def render_block(
    messages: Sequence[Message],
    *,
    book: ConversationBook,
    registry: EvidenceRegistry,
    limits: Limits,
    marks: Mapping[str, str],
    focus: Optional[Mapping[str, int]] = None,
) -> list[str]:
    """One exchange from one conversation: a title, date lines, message lines."""
    if not messages:
        return []
    lines = [f"== {book.title(messages[0]['conversation'])} =="]
    day = ""
    for message in messages:
        stamp = str(message.get("timestamp") or "")
        if stamp[:10] != day:
            day = stamp[:10]
            lines.append(day)
        speaker = book.speaker(message)  # first: the labeler learns the handle before the text is masked
        ref = registry.add(message)
        body = message_body(message, limits, (focus or {}).get(str(message["id"])), book.redact)
        thread = registry.ref_of(str(message.get("reply_to") or ""))
        note = f" (in the thread that starts at {thread})" if thread and thread != ref else ""
        mark = marks.get(str(message["id"]))
        tail = f"  <<< {mark}" if mark else ""
        lines.append(f"{ref} {stamp[11:16]} {speaker}: {body}{note}{tail}")
    return lines


class ResultWriter:
    """Collects the lines of one tool result under a token budget."""

    def __init__(self, budget_tokens: int, count_tokens: Callable[[str], int]) -> None:
        self._budget = max(0, budget_tokens)
        self._count = count_tokens
        self._parts: list[str] = []
        self.tokens = 0

    def fits(self, text: str) -> bool:
        return self.tokens + self._count(text) <= self._budget

    def add(self, text: str, *, force: bool = False) -> bool:
        if not force and not self.fits(text):
            return False
        self._parts.append(text)
        self.tokens += self._count(text)
        return True

    def text(self) -> str:
        return "\n".join(self._parts)


def _shorter(limits: Limits, cap: int) -> Limits:
    return replace(limits, max_message_chars=cap)


def add_block(
    writer: ResultWriter,
    messages: Sequence[Message],
    *,
    book: ConversationBook,
    registry: EvidenceRegistry,
    limits: Limits,
    marks: Mapping[str, str],
    focus: Optional[Mapping[str, int]] = None,
) -> bool:
    """Add one exchange to a result. When it does not fit, cut its messages shorter and try again.

    False when it does not fit at all. Then nothing is added and nothing is registered.
    """
    cap = limits.max_message_chars
    while True:
        mark = registry.checkpoint()
        lines = render_block(
            messages, book=book, registry=registry, limits=_shorter(limits, cap), marks=marks, focus=focus
        )
        if writer.add("\n" + "\n".join(lines)):
            return True
        registry.rollback(mark)
        if cap <= MIN_MESSAGE_CHARS:
            return False
        cap = max(MIN_MESSAGE_CHARS, cap // 2)


def fit_messages(
    messages: Sequence[Message],
    *,
    header: Callable[[int], str],
    drop: Callable[[list[Message]], list[Message]],
    budget: int,
    book: ConversationBook,
    registry: EvidenceRegistry,
    limits: Limits,
    marks: Mapping[str, str],
    titled: bool = True,
) -> tuple[ResultWriter, list[Message]]:
    """Render messages under a header; drop messages until the text fits the budget.

    Only the messages that are shown stay in the registry. One message is
    always shown: when it alone is too long, it is cut shorter.
    """
    kept = list(messages)
    cap = limits.max_message_chars
    while True:
        writer = ResultWriter(budget, limits.count_tokens)
        writer.add(header(len(kept)), force=True)
        mark = registry.checkpoint()
        lines = render_block(kept, book=book, registry=registry, limits=_shorter(limits, cap), marks=marks)
        block = "\n".join(lines if titled else lines[1:])
        if writer.fits(block) or (len(kept) <= 1 and cap <= MIN_MESSAGE_CHARS):
            writer.add(block, force=True)
            return writer, kept
        registry.rollback(mark)
        if len(kept) > 1:
            fewer = drop(kept)
            kept = fewer if 0 < len(fewer) < len(kept) else kept[:-1]
        else:
            cap = max(MIN_MESSAGE_CHARS, cap // 2)
