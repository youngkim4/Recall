"""Compact text the model reads: message lines, conversation titles, budgets.

Every line carries a reference from the evidence registry. People appear by
label only. Long text is cut with a note that says how much is missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from .labeler import UNKNOWN_MASK
from .ports import ChatArchive, Conversation, Labeler, Message
from .registry import ConversationAliases, EvidenceRegistry

OWNER = "You"
_MEDIA = {
    "image": {"heic", "jpg", "jpeg", "png", "gif", "webp", "tiff", "bmp"},
    "video": {"mov", "mp4", "m4v", "avi"},
    "audio": {"caf", "m4a", "mp3", "amr", "wav", "aac"},
    "contact card": {"vcf"},
    "pdf": {"pdf"},
}


def estimate_tokens(text: str) -> int:
    """A cautious token count without a tokenizer: three bytes per token."""
    return (len(str(text or "").encode("utf-8")) + 2) // 3


@dataclass(frozen=True)
class Limits:
    max_message_chars: int = 4000
    result_tokens: int = 3000
    count_tokens: Callable[[str], int] = estimate_tokens


def clip_text(text: str, max_chars: int, focus: Optional[int] = None) -> str:
    """Whitespace collapsed; cut to `max_chars` around `focus`, with a note."""
    source = str(text or "")
    flat = " ".join(source.split())
    if len(flat) <= max_chars:
        return flat
    start = 0
    if focus is not None and focus > 0:
        centre = len(" ".join(source[:focus].split()))
        start = max(0, min(centre - max_chars // 4, len(flat) - max_chars))
    end = start + max_chars
    before = f"[cut: {start} characters before] " if start else ""
    after = f" [cut: {len(flat) - end} more characters]" if end < len(flat) else ""
    return f"{before}{flat[start:end]}{after}"


def _attachment_kind(name: str) -> str:
    lowered = str(name or "").lower()
    extension = lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    for kind, extensions in _MEDIA.items():
        if extension in extensions or lowered.startswith(f"{kind}/"):
            return kind
    return "file"


def _attachment_note(attachments: Sequence[str]) -> str:
    kinds = Counter(_attachment_kind(name) for name in attachments)
    if not kinds:
        return ""
    parts = [kind if count == 1 else f"{kind} x{count}" for kind, count in kinds.items()]
    word = "attachment" if sum(kinds.values()) == 1 else "attachments"
    return f"[{word}: {', '.join(parts)}]"


def message_body(message: Message, limits: Limits, focus: Optional[int] = None) -> str:
    if message.get("unsent"):
        return "[unsent message]"
    parts = []
    text = clip_text(message.get("text") or "", limits.max_message_chars, focus)
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
        ref = registry.add(message)
        body = message_body(message, limits, (focus or {}).get(str(message["id"])))
        reply = registry.ref_of(str(message.get("reply_to") or ""))
        note = f" (reply to {reply})" if reply else ""
        mark = marks.get(str(message["id"]))
        tail = f"  <<< {mark}" if mark else ""
        lines.append(f"{ref} {stamp[11:16]} {book.speaker(message)}: {body}{note}{tail}")
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
