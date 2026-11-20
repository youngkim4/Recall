"""The people and conversations of the eval world. Every name, number and address is invented.

The owner is Sam Fennimore. Numbers are in the 555-01xx range, which no
real line uses. Addresses are at example.org.

Saved people (in the contacts list): Wren, Juniper, Theo, Marisol, Mom,
Dad and Clem. Not saved: Dash, the recruiter Ada and the landlord Gus. Their
handles must be masked in what the model reads and in each answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from evals.world.script import local

OWNER = "Sam Fennimore"
# the "today" of every question: two days after the last message
NOW = datetime(2025, 12, 22, 10, 0)
FIRST_DAY = "2023-01-01"
LAST_DAY = "2025-12-20"


@dataclass(frozen=True)
class Person:
    key: str
    handle: str
    saved_as: str  # "" when the owner never saved the handle


@dataclass(frozen=True)
class Chat:
    """One conversation. `identifier` is the chat id the archive keeps."""

    key: str
    identifier: str
    members: tuple[str, ...]  # person keys
    title: str = ""

    @property
    def is_group(self) -> bool:
        return len(self.members) > 1


PEOPLE: tuple[Person, ...] = (
    Person("wren", "+14155550101", "Wren Halloway"),
    Person("wrenmail", "wren.halloway@example.org", "Wren Halloway"),
    Person("dash", "+12125550134", ""),
    Person("juni", "+15105550112", "Juniper Okafor"),
    Person("theo", "+15105550147", "Theo Marchetti"),
    Person("mari", "+14155550163", "Marisol Ventnor"),
    Person("ada", "+16465550177", ""),
    Person("gus", "+15105550188", ""),
    Person("mom", "+15035550121", "Mom"),
    Person("dad", "+15035550122", "Dad"),
    Person("clem", "+15035550125", "Clem"),
)
BY_KEY = {person.key: person for person in PEOPLE}

FAMILY_CHAT = "chat410100"
FAMILY_TITLE = "Fennimore Fam"

CHATS: tuple[Chat, ...] = (
    Chat("wren", BY_KEY["wren"].handle, ("wren",)),
    Chat("wrenmail", BY_KEY["wrenmail"].handle, ("wrenmail",)),
    Chat("dash", BY_KEY["dash"].handle, ("dash",)),
    Chat("juni", BY_KEY["juni"].handle, ("juni",)),
    Chat("theo", BY_KEY["theo"].handle, ("theo",)),
    Chat("mari", BY_KEY["mari"].handle, ("mari",)),
    Chat("ada", BY_KEY["ada"].handle, ("ada",)),
    Chat("gus", BY_KEY["gus"].handle, ("gus",)),
    Chat("family", FAMILY_CHAT, ("mom", "dad", "clem"), FAMILY_TITLE),
)
CHAT_BY_KEY = {chat.key: chat for chat in CHATS}

# the Contacts cache, in the shape `import_chat_db` and `Names.from_records` read
CONTACTS: tuple[dict[str, object], ...] = (
    {"name": "Wren Halloway", "phones": ["(415) 555-0101"], "emails": ["Wren.Halloway@example.org"]},
    {"name": "Juniper Okafor", "phones": ["+1 510 555 0112"], "emails": []},
    {"name": "Theo Marchetti", "phones": ["510-555-0147"], "emails": []},
    {"name": "Marisol Ventnor", "phones": ["+14155550163"], "emails": []},
    {"name": "Mom", "phones": ["(503) 555-0121"], "emails": []},
    {"name": "Dad", "phones": ["(503) 555-0122"], "emails": []},
    {"name": "Clem", "phones": ["503 555 0125"], "emails": []},
)


def unsaved_handles() -> tuple[str, ...]:
    """The handles nobody saved. An answer must never hold them."""
    return tuple(person.handle for person in PEOPLE if not person.saved_as)


def conversation_id(chat_key: str) -> str:
    """The id of a conversation in the archive."""
    return CHAT_BY_KEY[chat_key].identifier


def start_of(day: str) -> datetime:
    return local(f"{day} 00:00")
