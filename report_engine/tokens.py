"""Token counting.

The engine counts with tiktoken. It uses the encoding named by
`ModelInfo.tokenizer_family` when tiktoken has it, and `o200k_base` when it
does not. A count made with another family than the model's gets a safety
margin. When no vocabulary can load (no file on disk, no network), the engine
falls back to a byte heuristic and says so in the log.

Nothing here loads at import time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from report_engine.model_info import ModelInfo

logger = logging.getLogger(__name__)

ENGINE_ENCODING = "o200k_base"
HEURISTIC_FAMILY = "heuristic"
HEURISTIC_BYTES_PER_TOKEN = 3
DEFAULT_MISMATCH_MARGIN = 1.35


class TokenCounter(Protocol):
    """Counts tokens for one tokenizer family."""

    family: str

    def count_many(self, texts: Sequence[str]) -> list[int]: ...


class HeuristicCounter:
    """One token per three UTF-8 bytes, rounded up. Runs high for plain English."""

    family = HEURISTIC_FAMILY

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [math.ceil(len(text.encode("utf-8")) / HEURISTIC_BYTES_PER_TOKEN) for text in texts]


class TiktokenCounter:
    """Exact counts for one tiktoken encoding."""

    def __init__(self, encoding: Any, family: str) -> None:
        self._encoding = encoding
        self.family = family

    def count_many(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        return [len(ids) for ids in self._encoding.encode_ordinary_batch(list(texts))]


def load_counter(family: str) -> TokenCounter:
    """Best counter for a tokenizer family. Never raises."""
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken is not installed; counting tokens with the byte heuristic")
        return HeuristicCounter()
    name = family if family in set(tiktoken.list_encoding_names()) else ENGINE_ENCODING
    try:
        encoding = tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - any load failure means the same thing here
        logger.warning(
            "tokenizer %s could not load (%s); counting tokens with the byte heuristic",
            name,
            type(exc).__name__,
        )
        return HeuristicCounter()
    return TiktokenCounter(encoding, name)


@dataclass(frozen=True)
class TokenMeter:
    """A counter plus the margin that makes its numbers safe for one model."""

    counter: TokenCounter
    margin: float = 1.0

    def __post_init__(self) -> None:
        if not self.margin >= 1.0:
            raise ValueError("token margin must be 1.0 or more")

    def count_many(self, texts: Sequence[str]) -> list[int]:
        counts = self.counter.count_many(texts)
        if self.margin == 1.0:
            return counts
        return [math.ceil(count * self.margin) for count in counts]

    def count(self, text: str) -> int:
        return self.count_many([text])[0]

    def describe(self) -> str:
        return f"{self.counter.family} x{self.margin:g}"


def meter_for(
    model_info: ModelInfo,
    counter: TokenCounter | None = None,
    mismatch_margin: float = DEFAULT_MISMATCH_MARGIN,
) -> TokenMeter:
    """Meter for one model: no margin when the counter is the model's own tokenizer."""
    chosen = counter if counter is not None else load_counter(model_info.tokenizer_family)
    margin = 1.0 if chosen.family == model_info.tokenizer_family else mismatch_margin
    return TokenMeter(chosen, margin)
