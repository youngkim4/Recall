"""The scope the user chose. Tools can narrow it. Nothing can widen it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence


def parse_day(value: object) -> Optional[str]:
    """A real calendar day "YYYY-MM-DD", or None when the value is empty.

    Raises ValueError when the value is not a day.
    """
    text = str(value or "").strip()
    if not text:
        return None
    return date.fromisoformat(text[:10]).isoformat()


@dataclass(frozen=True)
class Scope:
    """Conversation ids and days the question is limited to. Empty means all."""

    conversations: tuple[str, ...] = ()
    since: Optional[str] = None
    until: Optional[str] = None

    def __post_init__(self) -> None:
        given = (self.conversations,) if isinstance(self.conversations, str) else self.conversations
        object.__setattr__(self, "conversations", tuple(str(c) for c in given))
        object.__setattr__(self, "since", parse_day(self.since))
        object.__setattr__(self, "until", parse_day(self.until))
        if self.since and self.until and self.since > self.until:
            raise ValueError("scope starts after it ends")

    @property
    def is_limited(self) -> bool:
        return bool(self.conversations or self.since or self.until)

    def allows(self, conversation_id: str) -> bool:
        return not self.conversations or str(conversation_id) in self.conversations

    def allows_time(self, timestamp: str) -> bool:
        day = str(timestamp or "")[:10]
        if not (self.since or self.until):
            return True
        if not day or (self.since and day < self.since):
            return False
        return not (self.until and day > self.until)

    def narrow(self, requested: Optional[Sequence[str]]) -> Optional[tuple[str, ...]]:
        """Conversation ids a lookup may read.

        None means every conversation. An empty tuple means the request lies
        outside the scope and nothing may be read.
        """
        if requested is None:
            return self.conversations or None
        wanted = tuple(dict.fromkeys(str(c) for c in requested))
        if not self.conversations:
            return wanted
        return tuple(c for c in wanted if c in self.conversations)

    def clamp(self, since: Optional[str], until: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """The requested days cut down to the scope. May return since > until."""
        low = max((d for d in (since, self.since) if d), default=None)
        high = min((d for d in (until, self.until) if d), default=None)
        return low, high


EVERYTHING = Scope()
