"""A synthetic archive with invented people, and the facts the questions test.

Several years, direct threads and groups, one person with a phone thread and
an email thread, a Portuguese conversation, one long message, a topic that
comes back, an unsaved number and an unsaved email address.

The same seed gives the same archive. The facts are counted from the messages
here, with plain loops, not through the archive port: so a wrong count in the
port shows up as a failed question.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping

from ..fake_archive import FakeArchive, make_message
from ..labeler import MaskingLabeler
from ..ports import Message
from . import synthetic_data as data

TODAY = datetime(2026, 11, 16, 9, 30, 0)
DEFAULT_SEED = 7
_WORD_RE = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class SyntheticArchive:
    messages: tuple[Message, ...]
    conversations: tuple[Mapping[str, Any], ...]
    saved_names: Mapping[str, str]
    keys: Mapping[str, str]  # short name -> conversation id
    planted: Mapping[str, str]  # planted key -> message id
    facts: Mapping[str, Any]
    today: datetime

    def archive(self) -> FakeArchive:
        return FakeArchive(self.messages, self.conversations)

    def labeler(self) -> MaskingLabeler:
        return MaskingLabeler(self.saved_names)


def build_synthetic_archive(seed: int = DEFAULT_SEED) -> SyntheticArchive:
    rng = random.Random(seed)
    messages: list[Message] = []
    for chatter in data.CHATTER:
        messages.extend(_chatter(chatter, rng, len(messages)))
    planted = {}
    for item in data.PLANTED:
        message_id = f"p:{item.key}"
        planted[item.key] = message_id
        messages.append(
            make_message(
                message_id,
                item.conversation,
                item.timestamp,
                item.text,
                is_from_me=item.sender is None,
                sender=item.sender or "",
                kind=item.kind,
                attachments=["image/jpeg"] if item.kind == "attachment" else [],
            )
        )
    ordered = tuple(sorted(messages, key=lambda m: (m["timestamp"], m["id"])))
    return SyntheticArchive(
        messages=ordered,
        conversations=tuple(data.CONVERSATIONS),
        saved_names=dict(data.SAVED_NAMES),
        keys=dict(data.KEYS),
        planted=planted,
        facts=_facts(ordered),
        today=TODAY,
    )


def _chatter(chatter: data.Chatter, rng: random.Random, offset: int) -> list[Message]:
    pool = data.POOLS[chatter.pool]
    day = date.fromisoformat(chatter.start)
    end = date.fromisoformat(chatter.end)
    messages: list[Message] = []
    while day <= end:
        if rng.random() < chatter.active:
            moment = datetime(day.year, day.month, day.day, rng.randint(8, 21), rng.randint(0, 59), 0)
            for _ in range(rng.randint(chatter.low, chatter.high)):
                from_me = rng.random() < 0.5
                messages.append(
                    make_message(
                        f"f{offset + len(messages):06d}",
                        chatter.conversation,
                        moment.strftime("%Y-%m-%d %H:%M:%S"),
                        rng.choice(pool),
                        is_from_me=from_me,
                        sender="" if from_me else rng.choice(chatter.senders),
                    )
                )
                moment += timedelta(seconds=rng.randint(20, 240))
        day += timedelta(days=1)
    return messages


# --- ground truth, counted with plain loops ---------------------------------


def _has_word(message: Message, word: str) -> bool:
    return word in _WORD_RE.findall(message["text"].lower())


def _in(messages: tuple[Message, ...], conversation: str) -> list[Message]:
    return [m for m in messages if m["conversation"] == conversation]


def _facts(messages: tuple[Message, ...]) -> dict[str, Any]:
    maya = _in(messages, data.MAYA_PHONE)
    maya_mail = _in(messages, data.MAYA_EMAIL)
    jonas = _in(messages, data.JONAS)
    priya = _in(messages, data.PRIYA)
    per_conversation: dict[str, int] = {}
    for message in messages:
        per_conversation[message["conversation"]] = per_conversation.get(message["conversation"], 0) + 1
    top_id = max(per_conversation, key=lambda cid: per_conversation[cid])
    months: dict[str, int] = {}
    for message in [*maya, *maya_mail]:
        if message["timestamp"].startswith("2023"):
            month = message["timestamp"][:7]
            months[month] = months.get(month, 0) + 1
    busiest = max(months, key=lambda month: months[month])
    sourdough_jonas = [m for m in jonas if _has_word(m, "sourdough")]
    return {
        "messages": len(messages),
        "sourdough_total": sum(1 for m in messages if _has_word(m, "sourdough")),
        "sourdough_jonas": len(sourdough_jonas),
        "sourdough_first_day": sourdough_jonas[0]["timestamp"][:10],
        "sourdough_last_day": sourdough_jonas[-1]["timestamp"][:10],
        "priya_2024": sum(1 for m in priya if m["timestamp"].startswith("2024")),
        "priya_last_month": sum(1 for m in priya if m["timestamp"].startswith("2026-10")),
        "top_conversation": next(key for key, cid in data.KEYS.items() if cid == top_id),
        "top_conversation_total": per_conversation[top_id],
        "maya_phone_total": len(maya),
        "maya_email_total": len(maya_mail),
        "maya_total": len(maya) + len(maya_mail),
        "jonas_total": len(jonas),
        "maya_sent": sum(1 for m in maya if m["is_from_me"]),
        "maya_received": sum(1 for m in maya if not m["is_from_me"]),
        "maya_first_day": maya[0]["timestamp"][:10],
        "maya_busiest_month_2023": busiest,
        "maya_busiest_month_2023_total": months[busiest],
        "marathon_maya": sum(1 for m in maya if _has_word(m, "marathon")),
        "marathon_priya": sum(1 for m in priya if _has_word(m, "marathon")),
    }
