"""Find conversations and people by name.

Names match without case or diacritics. Digits match a handle only in the
forms a person knows: the four digits a label shows, or the whole number. So
nobody can find the digits of an unsaved number by guessing. The handle
itself is never shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from .handles import MIN_PHONE_DIGITS, Phone, handle_kind, is_mask, same_phone
from .ports import ChatArchive, Conversation, CountRow, Labeler
from .render import ConversationBook
from .scope import Scope
from .textnorm import words

_MIN_NAME_LENGTH = 3
_TAIL_DIGITS = 4
_AREA_DIGITS = 3
_NATIONAL_DIGITS = 10
_MAX_NAMES = 4


def _is_year(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and 1990 <= int(token) <= 2099


def _digits_match(token: str, digits: str, loose: bool) -> bool:
    """Whether digits typed by a person point at a handle.

    Always: the last four digits, or the whole number. `loose` (the user's own
    words) also allows an area code.
    """
    if token == digits:
        return True
    if len(token) == _TAIL_DIGITS:
        return digits.endswith(token)
    if len(token) >= MIN_PHONE_DIGITS and len(digits) >= MIN_PHONE_DIGITS:
        return same_phone(Phone(token, False), Phone(digits, False))
    if loose and len(token) == _AREA_DIGITS and len(digits) >= _NATIONAL_DIGITS:
        return digits[-_NATIONAL_DIGITS:-_NATIONAL_DIGITS + _AREA_DIGITS] == token
    return False


@dataclass(frozen=True)
class Entry:
    conversation: Conversation
    label: str
    names: frozenset[str]
    digits: tuple[str, ...]
    via: str  # "phone", "email" or "short code" for a direct thread, else ""
    counted: Optional[CountRow] = None  # counts inside the days of the scope, when it has days
    people: frozenset[str] = frozenset()  # the words of the names of people: not of a group title

    @property
    def total(self) -> int:
        if self.counted is not None:
            return int(self.counted.get("total") or 0)
        return int(self.conversation.get("message_count") or 0)

    def matches(self, token: str, loose_digits: bool = False) -> bool:
        if token.isdigit():
            return any(_digits_match(token, digits, loose_digits) for digits in self.digits)
        if len(token) < _MIN_NAME_LENGTH:
            return token in self.names
        return any(name.startswith(token) for name in self.names)


def _entry(
    conversation: Conversation, labeler: Labeler, book: ConversationBook, counted: Optional[CountRow]
) -> Entry:
    conversation_id = str(conversation["id"])
    handles = [conversation_id, *(str(h) for h in conversation.get("members") or [])]
    label = book.label(conversation_id)
    labels = [label]
    if conversation.get("is_group"):
        labels.extend(labeler.sender(handle) for handle in handles[1:])
    names = frozenset(token for text in labels if not is_mask(text) for token in words(text))
    of_people = labels[1:] if conversation.get("is_group") else labels  # a group label is a title
    people = frozenset(token for text in of_people if not is_mask(text) for token in words(text))
    kinds = [handle_kind(handle) for handle in handles]
    numbers = [re.sub(r"\D", "", h) for h, kind in zip(handles, kinds, strict=True) if kind in ("phone", "short_code")]
    via = "" if conversation.get("is_group") else next((kind for kind in kinds if kind != "other"), "")
    return Entry(
        conversation=conversation,
        label=label,
        names=names,
        digits=tuple(dict.fromkeys(numbers)),
        via=via.replace("_", " "),
        counted=counted,
        people=people,
    )


def _counts_in_days(archive: ChatArchive, scope: Scope) -> Optional[dict[str, CountRow]]:
    """Counts of each conversation inside the days of the scope. None when it has no days."""
    if not (scope.since or scope.until):
        return None
    rows = archive.count(
        conversations=list(scope.conversations) or None,
        since=scope.since,
        until=scope.until,
        from_me=None,
        text=None,
        group_by="conversation",
    )
    return {str(row["key"]): row for row in rows} if isinstance(rows, list) else {}


def _row_of(counted: Optional[dict[str, CountRow]], conversation_id: str) -> Optional[CountRow]:
    """The counts of one conversation inside the days. A conversation with no row has no message there."""
    if counted is None:
        return None
    return counted.get(conversation_id) or {"key": conversation_id, "total": 0, "sent": 0, "received": 0}


class Directory:
    def __init__(self, entries: Sequence[Entry]) -> None:
        self.entries = tuple(sorted(entries, key=lambda entry: (-entry.total, entry.label)))

    @classmethod
    def load(
        cls,
        archive: ChatArchive,
        book: ConversationBook,
        labeler: Labeler,
        scope: Scope,
        listed: Sequence[Conversation],
    ) -> Directory:
        """The conversations of the scope, out of all `listed` ones.

        With days, only those that have messages in the days.
        """
        if scope.conversations:
            by_id = {str(conversation["id"]): conversation for conversation in listed}
            found = [by_id.get(cid) or book.get(cid) for cid in scope.conversations]
            conversations = [conversation for conversation in found if conversation]
        else:
            conversations = list(listed)
        for conversation in conversations:
            book.put(conversation)
        counted = _counts_in_days(archive, scope)
        entries = [
            _entry(conversation, labeler, book, _row_of(counted, str(conversation["id"])))
            for conversation in conversations
        ]
        return cls([entry for entry in entries if counted is None or entry.total > 0])

    def __len__(self) -> int:
        return len(self.entries)

    def top(self, limit: int) -> list[Entry]:
        return list(self.entries[:max(0, limit)])

    def find(self, name: str, loose_digits: bool = False) -> list[Entry]:
        """Entries where every word of `name` matches a name or the digits of a handle."""
        tokens = words(name)
        if not tokens:
            return []
        return [entry for entry in self.entries if all(entry.matches(token, loose_digits) for token in tokens)]

    def name_words(self, question: str) -> list[tuple[str, bool]]:
        """Every word of the question that is a name or points at a handle, with True when it names a person.

        A word that only a group title holds gives False: it can be a common word ("That Is Not A Drill").
        """
        found: list[tuple[str, bool]] = []
        for token in dict.fromkeys(words(question)):
            if _is_year(token) or len(token) < _MIN_NAME_LENGTH:
                continue
            if token.isdigit():
                if any(entry.matches(token, loose_digits=True) for entry in self.entries):
                    found.append((token, True))
            elif any(token in entry.names for entry in self.entries):
                found.append((token, any(token in entry.people for entry in self.entries)))
        return found

    def named_in(self, question: str) -> list[str]:
        """Words of the user's question that are a name or point at a handle in this directory."""
        found: list[str] = []
        for token in dict.fromkeys(words(question)):
            if _is_year(token) or len(token) < _MIN_NAME_LENGTH:
                continue
            if token.isdigit():
                hit = any(entry.matches(token, loose_digits=True) for entry in self.entries)
            else:
                hit = any(token in entry.names for entry in self.entries)
            if hit:
                found.append(token)
        return found[:_MAX_NAMES]
