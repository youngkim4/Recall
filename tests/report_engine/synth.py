"""Synthetic conversations for tests. Every name, number and line is invented.

Phone numbers use the 555-01xx block and e-mail uses example.com; both are
reserved for fiction.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Iterator, Sequence

from report_engine.model_info import LongContextRule, ModelInfo
from report_engine.ports import SourceMessage

MINA = "+12025550143"
THEO = "theo@example.com"
RAVI = "+12025550171"
NAMES = {MINA: "Mina", THEO: "Theo", RAVI: "Ravi"}

CONVERSATION = MINA
GROUP = "chat000000000000000042"

WORDS = (
    "harbor", "lantern", "maple", "ticket", "window", "garden", "signal", "paper",
    "river", "kettle", "ladder", "orchard", "pencil", "station", "blanket", "compass",
    "tonight", "maybe", "later", "really", "sorry", "thanks", "okay", "again",
)


def labeler(sender: str) -> str:
    return NAMES.get(sender, "")


def model_info(**overrides: object) -> ModelInfo:
    """A made-up model. No real id, price or limit."""
    values: dict[str, object] = {
        "id": "test-model-1",
        "context_window": 200_000,
        "max_input_tokens": 180_000,
        "max_output_tokens": 20_000,
        "input_usd_per_mtok": 2.0,
        "cached_input_usd_per_mtok": 0.2,
        "output_usd_per_mtok": 10.0,
        "long_context": LongContextRule(
            threshold_tokens=150_000, input_multiplier=2.0, output_multiplier=1.5
        ),
        "tokens_per_minute": None,
        "tokenizer_family": "words",
    }
    values.update(overrides)
    return ModelInfo(**values)  # type: ignore[arg-type]


class WordCounter:
    """One token per whitespace-separated word. Exact and offline."""

    family = "words"

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(text.split()) for text in texts]


def message(
    mid: int | str,
    when: datetime | str,
    text: str,
    *,
    sender: str = MINA,
    is_from_me: bool = False,
) -> SourceMessage:
    stamp = when if isinstance(when, str) else when.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": str(mid),
        "timestamp": stamp,
        "is_from_me": is_from_me,
        "sender": "" if is_from_me else sender,
        "text": text,
    }


def phrase(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def conversation(
    start: str = "2024-06-01",
    days: int = 10,
    per_day: int = 4,
    *,
    seed: int = 7,
    words: int = 6,
    others: Sequence[str] = (MINA,),
    first_id: int = 1,
    skip_days: Sequence[int] = (),
) -> list[SourceMessage]:
    """Evenly spread chatter. Speakers take turns, the owner included."""
    rng = random.Random(seed)
    first = datetime.strptime(start, "%Y-%m-%d").replace(hour=9)
    speakers: list[str | None] = [None, *others]
    out: list[SourceMessage] = []
    mid = first_id
    for day in range(days):
        if day in skip_days:
            continue
        for slot in range(per_day):
            who = speakers[(day + slot) % len(speakers)]
            when = first + timedelta(days=day, minutes=7 * slot)
            out.append(
                message(
                    mid,
                    when,
                    phrase(rng, words),
                    sender=who or MINA,
                    is_from_me=who is None,
                )
            )
            mid += 1
    return out


def bounds(since: str | None, until: str | None) -> tuple[str, str]:
    low = "0000-00-00 00:00:00" if since is None else f"{since[:10]} 00:00:00"
    high = "9999-12-31 23:59:59" if until is None else f"{until[:10]} 23:59:59"
    return low, high


class ListSource:
    """In-memory ReportSource. Dates are whole days, both ends included."""

    def __init__(self, messages: Sequence[SourceMessage], conversation_id: str = CONVERSATION) -> None:
        self._messages = tuple(messages)
        self._conversation = conversation_id
        self.calls: list[tuple[str, str | None, str | None]] = []

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        self.calls.append((conversation, since, until))
        if conversation != self._conversation:
            return
        low, high = bounds(since, until)
        for item in self._messages:
            if low <= item["timestamp"] <= high:
                yield dict(item)  # type: ignore[misc]
