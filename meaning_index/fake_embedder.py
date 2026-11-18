"""A fixed embedder for tests and trial runs. It calls no service.

A vector is a hashed bag of words. Each word is put in lower case, a plural
`s` is dropped, and a small table maps words that mean the same to one word.
Two texts that share words then have near vectors, so a test can rank.
The day of a segment and the labels `Me` and `Them` are not words.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence

FAKE_MODEL = "fake-embedding-1"
FAKE_DIMENSIONS = 64
SAME_MEANING: Mapping[str, str] = {
    "boat": "ferry",
    "ship": "ferry",
    "supper": "dinner",
    "meal": "dinner",
    "cook": "dinner",
    "automobile": "car",
    "pup": "dog",
    "puppy": "dog",
    "mom": "mother",
    "dad": "father",
    "trip": "travel",
    "vacation": "travel",
    "glad": "happy",
    "joyful": "happy",
    "upset": "sad",
    "unhappy": "sad",
}
NOT_WORDS = frozenset({"me", "them", "the", "a", "an", "and", "of", "to", "for", "at", "on", "in", "is", "we", "you"})
MIN_STEM_CHARS = 4

_WORD = re.compile(r"[^\W\d_]+")


def _stem(word: str) -> str:
    plural = len(word) >= MIN_STEM_CHARS and word.endswith("s") and not word.endswith("ss")
    return word[:-1] if plural else word


def _slot(word: str, dimensions: int) -> tuple[int, float]:
    """The place of a word in the vector and its sign. Fixed for all time: a hash, not `hash()`."""
    digest = hashlib.sha256(word.encode("utf-8", "surrogatepass")).digest()
    return int.from_bytes(digest[:4], "big") % dimensions, 1.0 if digest[4] & 1 else -1.0


class FakeEmbedder:
    """Embedder port with fixed vectors. `calls` holds the texts of each call."""

    def __init__(
        self,
        dimensions: int = FAKE_DIMENSIONS,
        model: str = FAKE_MODEL,
        synonyms: Mapping[str, str] | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be 1 or more")
        self.model = model
        self.dimensions = dimensions
        self.calls: list[tuple[str, ...]] = []
        self._same = {**SAME_MEANING, **(synonyms or {})}

    @property
    def texts(self) -> list[str]:
        """Every text that was embedded, in order."""
        return [text for call in self.calls for text in call]

    def _words(self, text: str) -> list[str]:
        found = (_stem(word) for word in _WORD.findall(text.lower()))
        return [self._same.get(word, word) for word in found if word not in NOT_WORDS]

    def _vector(self, text: str) -> list[float]:
        sums = [0.0] * self.dimensions
        for word in self._words(text):
            place, sign = _slot(word, self.dimensions)
            sums[place] += sign
        length = math.sqrt(sum(value * value for value in sums))
        if length == 0.0:
            return [1.0, *([0.0] * (self.dimensions - 1))]
        return [value / length for value in sums]

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [self._vector(text) for text in texts]
