"""A word index over the texts of an export, and the query rules of the archive.

Words are AND-ed, "quoted text" is a phrase, a trailing `*` is a prefix, case
and diacritics do not matter. A term in a script that is written without
spaces (Chinese, Japanese, Korean), or with a sign in it, or with no letter
or digit at all, is matched as plain text.

The index holds the first `max_chars` characters of each text. A match after
that point is not found. The stored text is not cut.
"""

from __future__ import annotations

import re
import unicodedata
from array import array
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .textnorm import find_unit, fold, query_units, token_spans, word_pattern

_PIECE_RE = re.compile(r'"[^"]*"|\S+')
_EMPTY = np.empty(0, dtype=np.uint32)
_LAST_CHARACTER = "\U0010ffff"
_UNSPACED = (
    range(0x2E80, 0xA000),  # CJK radicals, kana, CJK ideographs
    range(0xAC00, 0xD7B0),  # Hangul syllables
    range(0xF900, 0xFB00),  # CJK compatibility ideographs
    range(0x20000, 0x2FA20),  # CJK ideographs, extensions
)


@dataclass(frozen=True)
class Query:
    units: tuple[str, ...]  # words, prefixes* and "phrases", folded
    fragments: tuple[str, ...]  # matched as plain text, without case

    @property
    def empty(self) -> bool:
        return not self.units and not self.fragments


def _is_fragment(term: str) -> bool:
    if not any(char.isalnum() for char in term):
        return True
    return any(
        unicodedata.category(char) == "So" or any(ord(char) in block for block in _UNSPACED) for char in term
    )


def parse_query(query: str) -> Query:
    """Split a query into word units and plain-text fragments. Bad syntax never raises."""
    source = str(query)
    if source.count('"') % 2:
        source = source.replace('"', " ")
    fragments: list[str] = []
    kept: list[str] = []
    for piece in _PIECE_RE.findall(source):
        term = piece.strip('"').strip()
        if term and _is_fragment(term):
            # a star after such a term adds nothing: plain text already matches the start of a longer word
            fragments.append(unicodedata.normalize("NFC", term.rstrip("*")) or term)
        else:
            kept.append(piece)
    units = tuple(dict.fromkeys(fold(unit) for unit in query_units(" ".join(kept))))
    return Query(units, tuple(dict.fromkeys(fragments)))


def fragment_pattern(fragment: str) -> re.Pattern[str]:
    return re.compile(re.escape(fragment), re.IGNORECASE)


class WordIndex:
    """Which texts hold which words. Positions are indexes into the given texts."""

    def __init__(self, texts: Iterable[str], max_chars: int) -> None:
        self.max_chars = max_chars
        postings = _postings(texts, max_chars)
        self._tokens = sorted(postings)
        sizes = np.fromiter((len(postings[token]) for token in self._tokens), dtype=np.int64, count=len(self._tokens))
        self._starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes)))
        joined = b"".join(postings[token].tobytes() for token in self._tokens)
        self._rows = np.frombuffer(joined, dtype=np.uint32) if joined else _EMPTY

    @property
    def words(self) -> int:
        return len(self._tokens)

    @property
    def entries(self) -> int:
        return len(self._rows)

    def frequency(self, word: str) -> int:
        """How many texts hold the word."""
        return len(self._of_word(word))

    def ranked_words(self) -> list[str]:
        """Every word, the one in the most texts first."""
        sizes = np.diff(self._starts)
        return [self._tokens[index] for index in np.argsort(-sizes, kind="stable")]

    def holders(self, unit: str) -> np.ndarray:
        """Positions that can match the unit, ascending. A phrase still needs a check of its word order."""
        if unit.startswith('"'):
            found: np.ndarray | None = None
            for word in unit.strip('"').split():
                rows = self._of_word(word)
                found = rows if found is None else np.intersect1d(found, rows, assume_unique=True)
                if not len(found):
                    return _EMPTY
            return _EMPTY if found is None else found
        if unit.endswith("*"):
            return self._of_prefix(unit[:-1])
        return self._of_word(unit)

    def _of_word(self, word: str) -> np.ndarray:
        index = bisect_left(self._tokens, word)
        if index == len(self._tokens) or self._tokens[index] != word:
            return _EMPTY
        return self._rows[self._starts[index]:self._starts[index + 1]]

    def _of_prefix(self, stem: str) -> np.ndarray:
        if not stem:
            return _EMPTY
        low = bisect_left(self._tokens, stem)
        high = bisect_left(self._tokens, stem + _LAST_CHARACTER)
        if high - low == 1:
            return self._rows[self._starts[low]:self._starts[high]]
        return np.unique(self._rows[self._starts[low]:self._starts[high]])


def _postings(texts: Iterable[str], max_chars: int) -> dict[str, array]:
    if array("I").itemsize != np.dtype(np.uint32).itemsize:
        raise RuntimeError("this platform has no 4-byte array type")
    pattern = word_pattern()
    postings: dict[str, array] = {}
    for position, text in enumerate(texts):
        if not text:
            continue
        source = text if len(text) <= max_chars else text[:max_chars]
        if source.isascii():
            found = set(pattern.findall(source.lower()))
        else:
            found = {fold(word) for word in pattern.findall(source)}
        for word in found:
            rows = postings.get(word)
            if rows is None:
                rows = postings[word] = array("I")
            rows.append(position)
    return postings


def match_ranges(text: str, query: Query, max_chars: int) -> list[list[int]]:
    """Where the query matches in the text, or [] unless every part of it matches."""
    source = text if len(text) <= max_chars else text[:max_chars]
    ranges: list[tuple[int, int]] = []
    if query.units:
        spans = token_spans(source)
        for unit in query.units:
            matched = find_unit(unit, spans)
            if not matched:
                return []
            ranges.extend(matched)
    for fragment in query.fragments:
        matched = [match.span() for match in fragment_pattern(fragment).finditer(source)]
        if not matched:
            return []
        ranges.extend(matched)
    return [list(pair) for pair in sorted(set(ranges))]
