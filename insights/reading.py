"""The model reads the recent messages of one thread and finds plans, promises and news.

`window` picks what the model reads: the last `WINDOW_DAYS` before `now`,
newest kept when the text is longer than `MAX_CHARS`. `read` makes one call
and checks each item before it becomes an insight:

- every ref must be a ref of the window; unknown refs are dropped, and an
  item with no ref left is dropped
- a promise must cite one of my own messages
- an item marked done is dropped
- two items of the same kind with the same refs are one item

`worth_reading` says which threads are worth a call: one where both sides
wrote in the last `ACTIVE_DAYS`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from insights import prompt
from insights.thread import Insight, Note, Thread
from report_engine.ports import LLM, LLMUsage

WINDOW_DAYS = 90
ACTIVE_DAYS = 60
MAX_CHARS = 40_000
OUTPUT_TOKENS = 2_000
MAX_ITEMS = 8


@dataclass(frozen=True)
class Reading:
    """What one call found, and what it cost in tokens."""

    insights: tuple[Insight, ...]
    usage: LLMUsage
    dropped: int  # items that failed a check


def window(thread: Thread, now: datetime) -> tuple[Note, ...]:
    """The notes the model reads, oldest first."""
    recent = thread.between(now - timedelta(days=WINDOW_DAYS), now)
    kept: list[Note] = []
    size = 0
    for note in reversed(recent):
        size += len(note.text) + 24
        if size > MAX_CHARS and kept:
            break
        kept.append(note)
    return tuple(reversed(kept))


def worth_reading(thread: Thread, now: datetime) -> bool:
    recent = thread.between(now - timedelta(days=ACTIVE_DAYS), now)
    return any(note.from_me for note in recent) and any(not note.from_me for note in recent)


def _checked(raw: Mapping[str, Any], thread: Thread, notes: Sequence[Note]) -> Insight | None:
    by_ref = {prompt.ref(place): note for place, note in enumerate(notes)}
    kind = raw.get("kind")
    if kind not in prompt.READ_KINDS or raw.get("done") is True:
        return None
    refs = raw.get("refs")
    named = [item for item in refs if isinstance(item, str)] if isinstance(refs, list) else []
    cited = [by_ref[item] for item in dict.fromkeys(named) if item in by_ref]
    summary = " ".join(str(raw.get("summary") or "").split())
    if not cited or not summary:
        return None
    if kind == "promise" and not any(note.from_me for note in cited):
        return None
    cited.sort(key=lambda note: note.when)
    return Insight(
        kind=kind,
        thread=thread.key,
        name=thread.name,
        summary=summary,
        when=cited[0].when,
        message_ids=tuple(note.id for note in cited),
    )


def parse(parsed: Mapping[str, Any] | None, thread: Thread, notes: Sequence[Note]) -> tuple[tuple[Insight, ...], int]:
    """The checked insights of one reply, and how many items were dropped."""
    raw_items = (parsed or {}).get("items")
    items = [item for item in raw_items if isinstance(item, Mapping)] if isinstance(raw_items, list) else []
    kept: list[Insight] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for raw in items:
        found = _checked(raw, thread, notes)
        if found is None or (found.kind, found.message_ids) in seen:
            continue
        seen.add((found.kind, found.message_ids))
        kept.append(found)
    return tuple(kept[:MAX_ITEMS]), len(items) - len(kept[:MAX_ITEMS])


def read(llm: LLM, thread: Thread, now: datetime, *, reasoning: str | None = None) -> Reading:
    """One call on the window of `thread`. A thread with nothing to read makes no call."""
    notes = window(thread, now)
    if not notes:
        return Reading((), LLMUsage(), 0)
    result = llm.generate(
        [
            {"role": "system", "content": prompt.SYSTEM},
            {"role": "user", "content": prompt.user_text(thread.name, notes, now)},
        ],
        schema=prompt.SCHEMA,
        max_output_tokens=OUTPUT_TOKENS,
        reasoning=reasoning,
    )
    insights, dropped = parse(result.parsed, thread, notes)
    return Reading(insights, result.usage, dropped)
