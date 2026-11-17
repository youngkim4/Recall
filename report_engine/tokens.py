"""Token counting.

The engine counts with tiktoken. It uses the encoding named by
`ModelInfo.tokenizer_family` when tiktoken has it, and `o200k_base` when it
does not. A count made with another family than the model's gets a safety
margin. When no vocabulary can load (no file on disk, no network), the engine
falls back to a count by character class and says so in the log.

tiktoken fetches a vocabulary file from the network the first time it is used
on a machine. To stay offline, point TIKTOKEN_CACHE_DIR at a folder that
holds the file, or pass a counter of your own.

Nothing here loads at import time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from report_engine.model_info import ModelInfo

logger = logging.getLogger(__name__)

ENGINE_ENCODING = "o200k_base"
HEURISTIC_FAMILY = "heuristic"
DEFAULT_MISMATCH_MARGIN = 1.35

# how much text one token covers, by class of character
LETTERS_PER_TOKEN = 4.0
DIGITS_PER_TOKEN = 1.5
SPACES_PER_TOKEN = 4.0
WIDE_BYTES_PER_TOKEN = 3.0

_LETTERS = bytes(range(ord("a"), ord("z") + 1)) + bytes(range(ord("A"), ord("Z") + 1))
_DIGITS = bytes(range(ord("0"), ord("9") + 1))
_SPACES = b" \t\n\r\x0b\x0c"
_WIDE = bytes(range(0x80, 0x100))


class TokenCounter(Protocol):
    """Counts tokens for one tokenizer family."""

    family: str

    def count_many(self, texts: Sequence[str]) -> list[int]: ...


def _rough_count(text: str) -> int:
    data = text.encode("utf-8", "surrogatepass")
    total = len(data)
    if total == 0:
        return 0
    letters = total - len(data.translate(None, _LETTERS))
    digits = total - len(data.translate(None, _DIGITS))
    spaces = total - len(data.translate(None, _SPACES))
    wide = total - len(data.translate(None, _WIDE))
    marks = total - letters - digits - spaces - wide
    return math.ceil(
        letters / LETTERS_PER_TOKEN
        + digits / DIGITS_PER_TOKEN
        + spaces / SPACES_PER_TOKEN
        + wide / WIDE_BYTES_PER_TOKEN
        + marks
        + data.count(b"\n")
        + 1
    )


class HeuristicCounter:
    """Counts by class of character when no tokenizer can load.

    One token for 4 letters, for 1.5 digits, for 4 spaces, for 3 bytes of any
    other script, for each punctuation mark and for each line. Measured
    against the engine encoding on transcript lines in English, Spanish,
    Russian, Korean, Japanese and Chinese, with digits, links and emoji, the
    count is 1.0 to 1.5 times the real one. It stands in for that encoding,
    so a model of that family gets no margin."""

    family = HEURISTIC_FAMILY
    approximates = ENGINE_ENCODING

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [_rough_count(text) for text in texts]


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
        counts = list(self.counter.count_many(texts))
        if len(counts) != len(texts):
            raise ValueError(
                f"token counter {self.counter.family} gave {len(counts)} counts for {len(texts)} texts"
            )
        if self.margin == 1.0:
            return counts
        return [math.ceil(round(count * self.margin, 9)) for count in counts]

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
    stands_for = {chosen.family, getattr(chosen, "approximates", chosen.family)}
    margin = 1.0 if model_info.tokenizer_family in stands_for else mismatch_margin
    return TokenMeter(chosen, margin)
