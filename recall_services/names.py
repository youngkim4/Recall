"""The one place for names.

Names come in as data: the records of the Contacts cache, each
`{name, phones[], emails[]}` (the shape of `saves/contact_names.json`). This
module never reads the Contacts database of the Mac.

The rules of "Names" in CONTRACTS.md:
- A conversation goes by a saved name, the title of a group, a group name made
  from the saved names of its members, or an empty string. The interface labels
  a conversation that has no name.
- Text for the chat model uses a saved name or a mask, never a handle that is
  not saved.
- Text for a report uses a saved name or an empty string. The report engine
  gives each unsaved sender a name of its own.

A handle matches a record as the archive matches people: a phone number by its
digits, the last 10 when there are more, and an email address in lower case.
The first record wins a handle that two names share.
"""

from __future__ import annotations

import json
import threading
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from archive.handles import email_key, handle_match_key, phone_key
from chat_engine.labeler import MaskingLabeler

from group_names import group_name

ReportLabeler = Callable[[str], str]
_RECORD_FIELDS = (("phones", phone_key), ("emails", email_key))
_NO_NAMES: Mapping[str, str] = MappingProxyType({})


def records(cache_path: str | Path) -> list[Any] | None:
    """The records of a Contacts cache file, or None when the file is missing or holds no list of records.

    An import takes these records for the people of the archive. None keeps
    the people that the archive has.
    """
    try:
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    found = payload.get("contacts") if isinstance(payload, dict) else None
    return found if isinstance(found, list) else None


def _name_of(record: Mapping[str, object]) -> str:
    """The name with single spaces, in one spelling of each accented letter."""
    return " ".join(unicodedata.normalize("NFC", str(record.get("name") or "")).split())


def _values(record: Mapping[str, object], name: str) -> list[str]:
    found = record.get(name)
    if isinstance(found, str):
        return [found]  # one value written without a list
    if isinstance(found, (list, tuple)):
        return [value for value in found if isinstance(value, str)]
    return []


def _keys_of(record: Mapping[str, object]) -> list[str]:
    keys = [to_key(value) for name, to_key in _RECORD_FIELDS for value in _values(record, name)]
    return [key for key in keys if key is not None]


def _is_blank(sender: object) -> bool:
    return sender is None or (isinstance(sender, str) and not sender.strip())


@dataclass(frozen=True)
class Names:
    """Saved names, by the match key of each handle that has one."""

    _by_key: Mapping[str, str] = field(default=_NO_NAMES)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]] | None) -> Names:
        """Names from the records of the Contacts cache. An entry that is not a record is left out."""
        if records is None:
            return cls()
        if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
            raise TypeError("contacts must be a list of {name, phones, emails} records")
        by_key: dict[str, str] = {}
        for record in records:
            name = _name_of(record) if isinstance(record, Mapping) else ""
            if not name:
                continue
            for key in _keys_of(record):
                by_key.setdefault(key, name)
        return cls(MappingProxyType(by_key))

    @classmethod
    def load(cls, cache_path: str | Path) -> Names:
        """Names from a Contacts cache file. A file that is missing or can not be read gives no names."""
        return cls.from_records(records(cache_path))

    def __len__(self) -> int:
        return len(self._by_key)

    def saved(self, handle: str) -> str:
        """The saved name of a handle, or "" when nobody saved it."""
        if not isinstance(handle, str) or not handle.strip():
            return ""
        return self._by_key.get(handle_match_key(handle), "")

    def sender(self, handle: str) -> str:
        """The name of the person who wrote a message: the saved name, or ""."""
        return self.saved(handle)

    def group(self, members: Sequence[str]) -> str:
        """A name for a group nobody named: its saved people, the busiest first.

        `members` are handles with the busiest writer first, as the archive
        lists them. Two handles of one saved name are one person.

        Two differences from `group_names.derive_group_names`, which counts
        the writers of an export:
        - A member who never wrote is a person of the group. The archive
          knows every member; the export knew only the writers.
        - A person with two handles stands where the busier handle stands.
          The export rule added the messages of both handles.
        """
        people: dict[str, str | None] = {}
        for handle in members:
            if _is_blank(handle):
                continue
            name = self.saved(handle)
            people.setdefault(name.casefold() if name else f"\x00{handle}", name or None)
        return group_name(list(people.values())) if people else ""

    def conversation(self, conversation: Mapping[str, Any]) -> str:
        """The name a conversation goes by on a screen, or "" when it has none."""
        saved = self.saved(str(conversation.get("id") or ""))
        if saved:
            return saved
        title = " ".join(str(conversation.get("name") or "").split())
        if title:
            return title
        if conversation.get("is_group"):
            return self.group(conversation.get("members") or [])
        return ""

    def conversations(self, listed: Iterable[Mapping[str, Any]]) -> dict[str, str]:
        """The name of each conversation, by conversation id."""
        return {str(conversation["id"]): self.conversation(conversation) for conversation in listed}

    def chat_labeler(self, conversations: Iterable[Mapping[str, Any]]) -> MaskingLabeler:
        """The labeler for chat: saved names as they are, a mask for everyone else.

        `conversations` are the conversations of the archive. Each of their
        handles is given with the name that `saved` gives, so chat and the
        screens name a person by one rule. Each handle that is not saved is
        given too, so `redact` masks it in any text from the first call.
        """
        handles: dict[str, None] = {}
        for conversation in conversations:
            for handle in (conversation.get("id"), *(conversation.get("members") or [])):
                if isinstance(handle, str):
                    handles.setdefault(handle)
        named = {handle: self.saved(handle) for handle in handles}
        return MaskingLabeler(
            {handle: name for handle, name in named.items() if name},
            # known from the start: text of one conversation can hold the handle of another
            known_handles=[handle for handle, name in named.items() if not name],
        )

    def report_labeler(self, conversation: Mapping[str, Any]) -> ReportLabeler:
        """The labeler for a report on one conversation.

        A sender who is not saved gets "": the report engine then gives each
        of them a name of its own. Words such as "an unsaved contact" would
        make one speaker of two people. The empty sender of a one-to-one
        conversation is the contact.
        """
        contact = "" if conversation.get("is_group") else self.saved(str(conversation.get("id") or ""))

        def label(sender: str) -> str:
            return contact if _is_blank(sender) else self.saved(sender)

        return label


_cache_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int] | None, Names]] = {}


def _signature(path: Path) -> tuple[int, int] | None:
    try:
        found = path.stat()
    except OSError:
        return None
    return (found.st_mtime_ns, found.st_size)


def cached(cache_path: str | Path) -> Names:
    """`Names.load`, read again only when the file changed. Names follow the Contacts cache."""
    path = Path(cache_path)
    signature = _signature(path)
    with _cache_lock:
        held = _cache.get(str(path))
        if held is not None and held[0] == signature:
            return held[1]
    loaded = Names.load(path)
    with _cache_lock:
        _cache[str(path)] = (signature, loaded)
    return loaded
