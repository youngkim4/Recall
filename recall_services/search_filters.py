"""The filters of /api/search and /api/stats, read from a query string.

Each filter is optional. All of them are combined with AND.

- `contacts`: conversation ids, comma separated. The older `contact` (one id)
  still works and is added to the list. An id matches whole, never in part.
- `since`, `until`: local days, `YYYY-MM-DD`, both included.
- `from`: `me`, `them` or empty.
- `kind`: `direct`, `group` or empty.
- `attachments`: `any`, `none`, `image`, `video`, `audio`, `other` or empty.

A value that is not allowed raises ValueError with one sentence: the route
answers 400 with it. The sentence never repeats the value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from archive.narrow import ATTACHMENT_CHOICES, KINDS
from recall_services import inputs

SENDERS = ("me", "them")
ORDERS = ("newest", "oldest", "relevance")
_FROM_ME = {"me": True, "them": False}
_FLAGS = {"true": True, "false": False}
# /api/search with no `prefix`: the app of today sends none and searches as the reader types
PREFIX_WHEN_NOT_GIVEN = True
_DAY = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class Filters:
    """What a search or a count is limited to. An empty value is no filter."""

    contacts: tuple[str, ...] = ()
    since: str = ""
    until: str = ""
    sender: str = ""  # `from` in the query string
    kind: str = ""
    attachments: str = ""

    @property
    def is_empty(self) -> bool:
        return self == NO_FILTERS

    def with_contact(self, contact: str) -> Filters:
        """These filters with one more conversation first. "" changes nothing."""
        if not contact:
            return self
        return replace(self, contacts=tuple(dict.fromkeys((contact, *self.contacts))))

    def archive_args(self) -> dict[str, Any]:
        """The keyword arguments of Archive.search and Archive.count."""
        return {
            "conversations": list(self.contacts) if self.contacts else None,
            "since": self.since or None,
            "until": self.until or None,
            "from_me": _FROM_ME.get(self.sender),
            "kind": self.kind or None,
            "attachments": self.attachments or None,
        }

    def payload(self) -> dict[str, Any]:
        """The filters as the screen reads them: every key, "" or [] when not applied."""
        return {
            "contacts": list(self.contacts),
            "since": self.since,
            "until": self.until,
            "from": self.sender,
            "kind": self.kind,
            "attachments": self.attachments,
        }


NO_FILTERS = Filters()


def _value(params: Mapping[str, object], name: str) -> str:
    return inputs.text(params.get(name)).strip()


def _choice(params: Mapping[str, object], name: str, choices: tuple[str, ...]) -> str:
    value = _value(params, name)
    if value and value not in choices:
        raise ValueError(f"{name} must be {', '.join(choices)} or empty.")
    return value


def _day(params: Mapping[str, object], name: str) -> str:
    value = _value(params, name)
    if not value:
        return ""
    try:
        if _DAY.fullmatch(value) is None:
            raise ValueError(value)
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be a day written as YYYY-MM-DD.") from None
    return value


def _contacts(params: Mapping[str, object]) -> tuple[str, ...]:
    listed = [_value(params, "contact"), *inputs.text(params.get("contacts")).split(",")]
    return tuple(dict.fromkeys(item.strip() for item in listed if item.strip()))


def read_filters(params: Mapping[str, object]) -> Filters:
    """The filters of a query string. See the module docstring."""
    return Filters(
        contacts=_contacts(params),
        since=_day(params, "since"),
        until=_day(params, "until"),
        sender=_choice(params, "from", SENDERS),
        kind=_choice(params, "kind", KINDS),
        attachments=_choice(params, "attachments", ATTACHMENT_CHOICES),
    )


def read_order(params: Mapping[str, object]) -> str:
    """`order`: newest (the default), oldest or relevance."""
    value = _value(params, "order") or ORDERS[0]
    if value not in ORDERS:
        raise ValueError(f"order must be {', '.join(ORDERS[:-1])} or {ORDERS[-1]}.")
    return value


def read_prefix(params: Mapping[str, object], *, default: bool = PREFIX_WHEN_NOT_GIVEN) -> bool:
    """`prefix`: true makes the last word a prefix, false makes every word whole.

    No `prefix` (or an empty one, which the app drops) is `default`. For
    /api/search it is true: the search of the app of today, which finds words
    as the reader types them. /api/stats gives false.
    """
    value = _value(params, "prefix").lower()
    if value and value not in _FLAGS:
        raise ValueError("prefix must be true or false.")
    return _FLAGS.get(value, default)
