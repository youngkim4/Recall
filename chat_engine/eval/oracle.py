"""Scripted runs of every question, for the fake model.

Each script acts as a careful model would: it reads only what the engine put
in the request (the question block and the tool results), calls tools, and
writes an answer from what came back. So a run proves that the engine and the
tools carry the facts. It proves nothing about a real model.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from ..fake_llm import Step, call, say
from ..ports import LLMRequest

_REF_RE = re.compile(r"^(m\d+) ")
_ALIAS_RE = re.compile(r"\b(c\d+) ·")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ROW_RE = re.compile(r"^(?P<label>.+): (?P<total>[\d,]+)(?: \(you (?P<sent>[\d,]+), others (?P<others>[\d,]+)\))?$")


# --- reading the request ------------------------------------------------------


def block(request: LLMRequest) -> str:
    """The question block: the last user message."""
    return next(m.content for m in reversed(request.messages) if m.role == "user")


def results(request: LLMRequest) -> list[str]:
    return [m.content for m in request.messages if m.role == "tool"]


def line_with(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line
    raise LookupError(f"no line holds {needle!r}")


def ref(text: str, needle: str) -> str:
    """The reference of the message line that holds `needle`."""
    for line in text.splitlines():
        match = _REF_RE.match(line)
        if match and needle in line:
            return match.group(1)
    raise LookupError(f"no message line holds {needle!r}")


def alias(text: str, needle: str) -> str:
    """The conversation alias on the line that holds `needle`."""
    for line in text.splitlines():
        match = _ALIAS_RE.search(line)
        if match and needle in line:
            return match.group(1)
    raise LookupError(f"no conversation line holds {needle!r}")


def day_of(text: str, needle: str) -> str:
    """The day line above the message line that holds `needle`."""
    day = ""
    for line in text.splitlines():
        if _DAY_RE.match(line):
            day = line
        elif _REF_RE.match(line) and needle in line and day:
            return day
    raise LookupError(f"no dated message line holds {needle!r}")


def number(text: str) -> str:
    """The number a plain count ends with: "...: 1,234." gives "1,234"."""
    return text.strip().rstrip(".").rsplit(": ", 1)[1]


def rows(text: str) -> list[tuple[str, int, int]]:
    """Rows of a count table as (label, total, sent)."""
    found = []
    for line in text.splitlines()[1:]:
        match = _ROW_RE.match(line)
        if match:
            sent = int((match.group("sent") or "0").replace(",", ""))
            found.append((match.group("label"), int(match.group("total").replace(",", "")), sent))
    return found


# --- script builders ----------------------------------------------------------


def from_block(answer: Callable[[str], str]) -> list[Step]:
    """One model call: answer from the seeded lookups."""
    return [lambda request: say(answer(block(request)), chunk=24)]


def recall(needle: str, query: str, answer: Callable[[str, str], str], within: Optional[str] = None) -> list[Step]:
    """Answer from the seed when it holds the message; search for it when it does not."""

    def first(request: LLMRequest) -> list:
        text = block(request)
        if needle in text:
            return say(answer(text, ref(text, needle)), chunk=24)
        arguments: dict = {"query": query}
        if within:
            arguments["conversations"] = [alias(text, within)]
        return [call("search_messages", **arguments)]

    def second(request: LLMRequest) -> list:
        text = results(request)[-1]
        return say(answer(text, ref(text, needle)), chunk=24)

    return [first, second]


def lookup(make_calls: Callable[[str], list], answer: Callable[[list[str]], str]) -> list[Step]:
    """Two model calls: tools chosen from the question block, then the answer."""
    return [
        lambda request: make_calls(block(request)),
        lambda request: say(answer(results(request)), chunk=24),
    ]


# --- the scripts --------------------------------------------------------------


def _quote(text: str, needle: str, lead: str, quote: str) -> str:
    return f'{lead} "{quote}" [{ref(text, needle)}].'


def _busiest(table: str) -> str:
    label, total, _ = max(rows(table), key=lambda row: row[1])
    return f"The busiest month with Maya in 2023 was {label} (September): {total} messages in both threads."


def _volume(table: str) -> str:
    counted = rows(table)
    maya = sum(total for label, total, _ in counted if "Maya Okafor" in label)
    jonas = sum(total for label, total, _ in counted if "Jonas Weber" in label)
    return f"You text Maya more: {maya} messages in her two threads, against {jonas} with Jonas."


def _top(table: str) -> str:
    label, total, _ = rows(table)[0]
    name = label.split(" · ")[1]
    return f"You text {name} the most: {total} messages."


def _sent_received(figures: str) -> str:
    match = re.search(r"messages: ([\d,]+) \(you ([\d,]+), others ([\d,]+)\)", figures)
    if not match:
        raise LookupError("no message figures")
    return f"You sent {match.group(2)} messages and Maya sent {match.group(3)}."


def _first_message(figures: str) -> str:
    return f"Your first messages with Maya are from {line_with(figures, 'first message:').split(': ', 1)[1][:10]}."


def _members(text: str) -> str:
    members = line_with(text, "Lisbon crew · group of").split("group of 3: ", 1)[1].split(" · ", 1)[0]
    return f"The Lisbon crew group has you and three others: {members}."


def _threads(text: str) -> str:
    phone = line_with(text, "Maya Okafor · direct, phone").split(" · ")[3]
    mail = line_with(text, "Maya Okafor · direct, email").split(" · ")[3]
    return (
        f"You have two direct threads with Maya. One is on her phone number: {phone}. "
        f"The other is on her email address: {mail}."
    )


def _dated(text: str, needle: str, lead: str) -> str:
    return f"{lead} {day_of(text, needle)} [{ref(text, needle)}]."


SCRIPTS: dict[str, Callable[[], list[Step]]] = {
    "recall_starter": lambda: recall(
        "its name is Clint Yeastwood", "name",
        lambda text, r: f'Jonas named it Clint Yeastwood. He wrote "its name is Clint Yeastwood" [{r}].',
        within="Jonas Weber · direct",
    ),
    "recall_wifi": lambda: recall(
        "pinecone-42-lantern", "wifi password",
        lambda text, r: f'Maya wrote "the wifi password at the cabin is pinecone-42-lantern" [{r}].',
    ),
    "recall_restaurant": lambda: recall(
        "Taberna do Largo", '"Taberna do Largo"',
        lambda text, r: f'Maya said "we have to eat at Taberna do Largo" [{r}].',
    ),
    "count_sourdough": lambda: lookup(
        lambda text: [call("count_messages", text="sourdough")],
        lambda out: f"{number(out[0])} messages mention sourdough.",
    ),
    "count_priya_2024": lambda: lookup(
        lambda text: [
            call("count_messages", conversations=[alias(text, "Priya Raman · direct")],
                 since="2024-01-01", until="2024-12-31")
        ],
        lambda out: f"You and Priya exchanged {number(out[0])} messages in 2024.",
    ),
    "count_top": lambda: lookup(
        lambda text: [call("count_messages", group_by="conversation")],
        lambda out: _top(out[0]),
    ),
    "count_busiest_month": lambda: lookup(
        lambda text: [
            call(
                "count_messages",
                conversations=[alias(text, "Maya Okafor · direct, phone"), alias(text, "Maya Okafor · direct, email")],
                since="2023-01-01", until="2023-12-31", group_by="month",
            )
        ],
        lambda out: _busiest(out[0]),
    ),
    "count_sent_received": lambda: lookup(
        lambda text: [call("conversation_stats", conversation=alias(text, "Maya Okafor · direct, phone"))],
        lambda out: _sent_received(out[0]),
    ),
    "first_sourdough": lambda: lookup(
        lambda text: [
            call("search_messages", query="sourdough", conversations=[alias(text, "Jonas Weber · direct")],
                 order="oldest", limit=1)
        ],
        lambda out: _dated(out[0], "sourdough starter is alive", "You first talked about sourdough on"),
    ),
    "last_sourdough": lambda: lookup(
        lambda text: [
            call("search_messages", query="sourdough", conversations=[alias(text, "Jonas Weber · direct")],
                 order="newest", limit=1)
        ],
        lambda out: _dated(out[0], "sourdough again this weekend?", "Sourdough last came up on"),
    ),
    "first_message_maya": lambda: lookup(
        lambda text: [call("conversation_stats", conversation=alias(text, "Maya Okafor · direct, phone"))],
        lambda out: _first_message(out[0]),
    ),
    "who_number": lambda: lookup(
        lambda text: [
            call("search_messages", query="", order="oldest", limit=5,
                 conversations=[alias(text, "a number ending in 0199 · direct")])
        ],
        lambda out: _quote(
            out[0], "this is Dana from the climbing gym", "That number is Dana from the climbing gym. She wrote",
            "hey this is Dana from the climbing gym",
        ),
    ),
    "who_email": lambda: recall(
        "this is Rafael from Northwind Outfitters", "tent",
        lambda text, r: (
            "An unsaved email contact wrote to you: Rafael from Northwind Outfitters. "
            f'He wrote "Your tent order has shipped" [{r}].'
        ),
    ),
    "compare_volume": lambda: lookup(
        lambda text: [
            call(
                "count_messages",
                conversations=[
                    alias(text, "Maya Okafor · direct, phone"),
                    alias(text, "Maya Okafor · direct, email"),
                    alias(text, "Jonas Weber · direct"),
                ],
                group_by="conversation",
            )
        ],
        lambda out: _volume(out[0]),
    ),
    "compare_marathon": lambda: lookup(
        lambda text: [
            call("count_messages", "maya", text="marathon",
                 conversations=[alias(text, "Maya Okafor · direct, phone")]),
            call("count_messages", "priya", text="marathon", conversations=[alias(text, "Priya Raman · direct")]),
        ],
        lambda out: (
            f"Maya talks about it more: {number(out[0])} messages in the thread with Maya mention the marathon, "
            f"against {number(out[1])} in the thread with Priya."
        ),
    ),
    "relative_summer": lambda: lookup(
        lambda text: [
            call("search_messages", query="baked", conversations=[alias(text, "Jonas Weber · direct")],
                 since="2026-06-01", until="2026-08-31")
        ],
        lambda out: _quote(
            out[0], "baked a rye loaf", "Last summer Jonas baked a rye loaf. He wrote",
            "baked a rye loaf for the lake trip, crust came out perfect",
        ),
    ),
    "relative_two_years": lambda: lookup(
        lambda text: [
            call("search_messages", query="ferry", conversations=[alias(text, "Priya Raman · direct")],
                 since="2024-01-01", until="2024-12-31")
        ],
        lambda out: _quote(
            out[0], "ferry to Bainbridge", "Two years ago Priya took the ferry to Bainbridge. She wrote",
            "the ferry to Bainbridge was so worth it",
        ),
    ),
    "relative_last_month": lambda: lookup(
        lambda text: [
            call("count_messages", conversations=[alias(text, "Priya Raman · direct")],
                 since="2026-10-01", until="2026-10-31")
        ],
        lambda out: f"You and Priya exchanged {number(out[0])} messages last month, in October 2026.",
    ),
    "follow_when": lambda: from_block(
        lambda text: _dated(text, "its name is Clint Yeastwood", "Jonas wrote that on")
    ),
    "follow_count": lambda: lookup(
        lambda text: [call("count_messages", text="sourdough", conversations=[alias(text, "Jonas Weber · direct")])],
        lambda out: f"{number(out[0])} of them are in your thread with Jonas.",
    ),
    "absent_blood": lambda: from_block(
        lambda text: "The archive does not hold this. I found no message about Maya's blood type."
    ),
    "absent_zanzibar": lambda: from_block(
        lambda text: "I found no message that mentions Zanzibar, from Jonas or from anyone else."
    ),
    "scoped_absent": lambda: from_block(
        lambda text: "In this conversation there is no message about a lease. I searched it and found nothing."
    ),
    "scoped_lease": lambda: from_block(
        lambda text: _dated(text, "the lease got signed btw", "Maya told you the lease was signed on")
    ),
    "portuguese_question": lambda: recall(
        "a cozinha é enorme", "cozinha",
        lambda text, r: f'A Inês disse "O apartamento é ótimo, a cozinha é enorme" [{r}].',
    ),
    "portuguese_content": lambda: lookup(
        lambda text: [
            call("search_messages", query="apartamento cozinha ótimo", match="any",
                 conversations=[alias(text, "Inês Duarte · direct")])
        ],
        lambda out: _quote(
            out[0], "a cozinha é enorme", "Inês liked it. She wrote", "O apartamento é ótimo, a cozinha é enorme"
        ),
    ),
    "long_message": lambda: recall(
        "the spare badge is in the grey cabinet", "badge",
        lambda text, r: f'Tomás wrote "the spare badge is in the grey cabinet, drawer three" [{r}].',
    ),
    "group_flights": lambda: recall(
        "flights to Lisbon are booked", "flights booked",
        lambda text, r: f'Maya booked them. She wrote "flights to Lisbon are booked, I paid for all three" [{r}].',
    ),
    "group_members": lambda: from_block(_members),
    "email_thread": lambda: recall(
        "sending the lease scan", "lease scan",
        lambda text, r: (
            f'By email Maya sent the lease scan: "sending the lease scan from my laptop, the PDF is attached" [{r}].'
        ),
        within="Maya Okafor · direct, email",
    ),
    "threads_with_maya": lambda: from_block(_threads),
}
