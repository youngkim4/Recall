"""Find conversations and people by name.

Names match without case or diacritics. Digits match a handle, so "720" finds
an unsaved number, but the handle itself is never shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .labeler import handle_kind, is_mask
from .ports import ChatArchive, Conversation, Labeler
from .render import ConversationBook
from .scope import Scope
from .textnorm import words

_MIN_NAME_LENGTH = 3
_MIN_DIGITS = 3
_MAX_NAMES = 4


def _is_year(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and 1990 <= int(token) <= 2099


@dataclass(frozen=True)
class Entry:
    conversation: Conversation
    label: str
    names: frozenset[str]
    digits: tuple[str, ...]
    via: str  # "phone" or "email" for a direct thread, else ""

    def matches(self, token: str) -> bool:
        if token.isdigit():
            return len(token) >= _MIN_DIGITS and any(token in digits for digits in self.digits)
        if len(token) < _MIN_NAME_LENGTH:
            return token in self.names
        return any(name.startswith(token) for name in self.names)


def _entry(conversation: Conversation, labeler: Labeler, book: ConversationBook) -> Entry:
    conversation_id = str(conversation["id"])
    handles = [conversation_id, *(str(h) for h in conversation.get("members") or [])]
    label = book.label(conversation_id)
    labels = [label]
    if conversation.get("is_group"):
        labels.extend(labeler.sender(handle) for handle in handles[1:])
    names = frozenset(token for text in labels if not is_mask(text) for token in words(text))
    kinds = [handle_kind(handle) for handle in handles]
    phones = tuple(
        dict.fromkeys(re.sub(r"\D", "", h) for h, kind in zip(handles, kinds) if kind == "phone")
    )
    via = "" if conversation.get("is_group") else next((kind for kind in kinds if kind != "other"), "")
    return Entry(conversation=conversation, label=label, names=names, digits=phones, via=via)


class Directory:
    def __init__(self, entries: Sequence[Entry]) -> None:
        self.entries = tuple(
            sorted(entries, key=lambda e: (-int(e.conversation.get("message_count") or 0), e.label))
        )

    @classmethod
    def load(cls, archive: ChatArchive, labeler: Labeler, book: ConversationBook, scope: Scope) -> "Directory":
        if scope.conversations:
            found = [book.get(conversation_id) for conversation_id in scope.conversations]
            conversations = [conversation for conversation in found if conversation]
        else:
            conversations = archive.conversations(limit=None, offset=0, order="count")
            for conversation in conversations:
                book.put(conversation)
        return cls([_entry(conversation, labeler, book) for conversation in conversations])

    def __len__(self) -> int:
        return len(self.entries)

    def top(self, limit: int) -> list[Entry]:
        return list(self.entries[:max(0, limit)])

    def find(self, name: str) -> list[Entry]:
        """Entries where every word of `name` matches a name or handle digits."""
        tokens = words(name)
        if not tokens:
            return []
        return [entry for entry in self.entries if all(entry.matches(token) for token in tokens)]

    def named_in(self, question: str) -> list[str]:
        """Words of the question that are a name or handle digits in this directory."""
        found: list[str] = []
        for token in dict.fromkeys(words(question)):
            if _is_year(token) or len(token) < _MIN_NAME_LENGTH:
                continue
            if token.isdigit():
                hit = any(token in digits for entry in self.entries for digits in entry.digits)
            else:
                hit = any(token in entry.names for entry in self.entries)
            if hit:
                found.append(token)
        return found[:_MAX_NAMES]
