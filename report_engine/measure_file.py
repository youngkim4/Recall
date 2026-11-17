"""Numbers about the rows of an export file.

Nothing here returns or prints the text of a message, a name or a handle:
only counts, sizes and shares.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from report_engine.frame_source import direction_of
from report_engine.tapbacks import ADDED, CLOSE_MARKS, OPEN_MARKS, REMOVED, is_tapback
from report_engine.transcript import squash

LENGTH_STEPS = (250, 500, 1_000, 2_000, 4_000, 6_000, 10_000, 100_000)
QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
PLACEHOLDER = chr(0xFFFC)
START_CHARS = 8
_GROUP_ID = r"chat\d+"
_TAPBACK_STARTS = tuple(f"{phrase} " for phrase in (*ADDED, *REMOVED, "Reacted"))


@dataclass(frozen=True)
class RowCounts:
    """Counts of rows by what the engine does with them."""

    rows: int
    conversations: int
    group_conversations: int
    no_conversation: int
    no_timestamp: int
    sent: int
    received: int
    no_direction: int
    no_id: int
    repeated_ids: int
    no_text: int
    placeholder_only: int
    tapbacks: int
    tapback_lookalikes: int


@dataclass(frozen=True)
class TapbackCheck:
    """Where the quoted text of the tapback rows is found. A high share that is
    found says that the phrases mark real tapbacks. Each row is counted one
    time, in the first of these that holds:

    - `found_earlier`: it is the whole text of an earlier row of the conversation;
    - `found_as_start`: an earlier row of the conversation starts with it (a
      long text that was quoted in part);
    - `found_elsewhere`: a row after it, or a row of another conversation, holds it;
    - `placeholder`: the quote holds only the attachment placeholder."""

    with_quote: int
    found_earlier: int
    found_as_start: int = 0
    found_elsewhere: int = 0
    placeholder: int = 0

    @property
    def not_found(self) -> int:
        found = self.found_earlier + self.found_as_start + self.found_elsewhere + self.placeholder
        return self.with_quote - found


@dataclass(frozen=True)
class LengthStep:
    """Rows longer than `limit` characters, the share of all characters that
    these rows hold, and the share of all characters that a cap at `limit` keeps."""

    limit: int
    rows_over: int
    share_in_rows_over: float
    share_kept_by_cap: float


@dataclass(frozen=True)
class Lengths:
    characters: int
    longest: int
    quantiles: tuple[tuple[float, int], ...]
    steps: tuple[LengthStep, ...]


def texts_of(frame: pd.DataFrame) -> list[str]:
    """The text column as strings. A missing text is an empty string."""
    return frame["text"].fillna("").astype(str).tolist()


def _quoted(text: str) -> str | None:
    """The text inside the quote marks of a tapback, or None when it has none."""
    stripped = text.strip()
    starts = [stripped.find(mark) for mark in OPEN_MARKS if mark in stripped]
    if not starts or stripped[-1] not in CLOSE_MARKS:
        return None
    return stripped[min(starts) + 1 : -1]


def count_rows(frame: pd.DataFrame, texts: Sequence[str], tapbacks: Sequence[bool]) -> RowCounts:
    chats = frame["chat_id"].astype("string")
    directions = [direction_of(value) for value in frame["is_from_me"].tolist()]
    shown_and_text = [(squash(text), text) for text in texts]
    ids = frame["message_id"]
    return RowCounts(
        rows=len(frame),
        conversations=int(chats.nunique()),
        group_conversations=int(chats[chats.str.fullmatch(_GROUP_ID).fillna(False)].nunique()),
        no_conversation=int(chats.isna().sum()),
        no_timestamp=int(frame["timestamp"].isna().sum()),
        sent=sum(direction is True for direction in directions),
        received=sum(direction is False for direction in directions),
        no_direction=sum(direction is None for direction in directions),
        no_id=int(ids.isna().sum()),
        repeated_ids=int(ids.dropna().duplicated().sum()),
        no_text=sum(not shown and PLACEHOLDER not in text for shown, text in shown_and_text),
        placeholder_only=sum(not shown and PLACEHOLDER in text for shown, text in shown_and_text),
        tapbacks=sum(tapbacks),
        tapback_lookalikes=sum(
            not tapback and text.lstrip().startswith(_TAPBACK_STARTS)
            for tapback, text in zip(tapbacks, texts, strict=True)
        ),
    )


def _without_dots(text: str) -> str:
    """A quoted text with the dots taken off its end: a long text can be quoted
    in part, with dots where it was cut."""
    for dots in (chr(0x2026), "..."):
        if text.endswith(dots):
            return text[: -len(dots)].rstrip()
    return text


class _Seen:
    """The texts of one conversation so far, to look a quoted text up in."""

    def __init__(self) -> None:
        self._whole: set[str] = set()
        self._by_start: dict[str, list[str]] = {}

    def add(self, text: str) -> None:
        self._whole.add(text)
        if len(text) >= START_CHARS:
            self._by_start.setdefault(text[:START_CHARS], []).append(text)

    def has(self, text: str) -> bool:
        return text in self._whole

    def has_start(self, text: str) -> bool:
        start = _without_dots(text)
        if len(start) < START_CHARS:
            return False
        return any(known.startswith(start) for known in self._by_start.get(start[:START_CHARS], ()))


def check_tapbacks(
    frame: pd.DataFrame, texts: Sequence[str], tapbacks: Sequence[bool]
) -> TapbackCheck:
    """Walk each conversation in time order and look for the quoted text of
    every tapback: among the rows before it, as the start of one of them, and
    then anywhere in the file."""
    flat = [squash(text) for text in texts]
    anywhere = {shown for shown, tapback in zip(flat, tapbacks, strict=True) if not tapback}
    order = (
        pd.DataFrame({"chat": frame["chat_id"].astype("string"), "when": frame["timestamp"]})
        .reset_index(drop=True)
        .dropna(subset=["chat"])
        .sort_values(["chat", "when"], kind="stable")
    )
    counts = dict.fromkeys(("quote", "earlier", "start", "elsewhere", "placeholder"), 0)
    current: object = None
    seen = _Seen()
    for position, chat in zip(order.index.tolist(), order["chat"].tolist(), strict=True):
        if chat != current:
            current, seen = chat, _Seen()
        quoted = _quoted(texts[position]) if tapbacks[position] else None
        if quoted is None:
            if not tapbacks[position]:
                seen.add(flat[position])
            continue
        counts["quote"] += 1
        shown = squash(quoted)
        if not shown:
            counts["placeholder"] += 1
        elif seen.has(shown):
            counts["earlier"] += 1
        elif seen.has_start(shown):
            counts["start"] += 1
        elif shown in anywhere:
            counts["elsewhere"] += 1
    return TapbackCheck(
        with_quote=counts["quote"],
        found_earlier=counts["earlier"],
        found_as_start=counts["start"],
        found_elsewhere=counts["elsewhere"],
        placeholder=counts["placeholder"],
    )


def measure_lengths(texts: Sequence[str]) -> Lengths:
    lengths = pd.Series([len(text) for text in texts], dtype="int64")
    total = int(lengths.sum())

    def share(part: int) -> float:
        return part / total if total else 0.0

    steps = tuple(
        LengthStep(
            limit=limit,
            rows_over=int((lengths > limit).sum()),
            share_in_rows_over=share(int(lengths[lengths > limit].sum())),
            share_kept_by_cap=share(int(lengths.clip(upper=limit).sum())),
        )
        for limit in LENGTH_STEPS
    )
    levels = (
        lengths.quantile(list(QUANTILES)) if len(lengths) else pd.Series([0.0] * len(QUANTILES))
    )
    return Lengths(
        characters=total,
        longest=int(lengths.max()) if len(lengths) else 0,
        quantiles=tuple(
            (level, int(value)) for level, value in zip(QUANTILES, levels.tolist(), strict=True)
        ),
        steps=steps,
    )


def tapback_flags(texts: Sequence[str]) -> list[bool]:
    return [is_tapback(text) for text in texts]
