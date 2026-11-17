"""Text folding and query parsing shared by search, names and the test archives.

No stopword list and no word endings of one language. A word is a run of
letters and digits in any script, with the marks that belong to them. Words
are compared without case or diacritics.

Two forms of a word:
- `fold` is for comparisons inside the engine. It takes scripts apart
  (Hangul becomes its letters, kana lose their voicing marks), so a folded
  word must never go to the archive.
- `surface` is the word as it was written, in lower case. This form goes
  to the archive, which does its own matching.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Callable, Optional

_QUOTED_RE = re.compile(r'"([^"]*)"')
_MARK_SCAN_END = 0x1EA00  # the last block that holds combining marks, apart from variation selectors
_NOT_IN_WORDS = (range(0xFE00, 0xFE10), range(0x20D0, 0x2100), range(0x180B, 0x1810))

Span = tuple[int, int, str]


def _is_word_mark(code: int) -> bool:
    """A mark that belongs inside a word: a vowel sign, an accent. Not a variation selector."""
    if any(code in block for block in _NOT_IN_WORDS):
        return False
    return unicodedata.category(chr(code)).startswith("M")


def _ranges(codes: list[int]) -> str:
    """Code points as a regex character class body, runs merged."""
    parts: list[str] = []
    start: Optional[int] = None
    previous = -2
    for code in [*codes, -1]:
        if code != previous + 1:
            if start is not None:
                parts.append(f"\\U{start:08x}-\\U{previous:08x}")
            start = code
        previous = code
    return "".join(parts)


@lru_cache(maxsize=1)
def word_pattern() -> re.Pattern[str]:
    """A letter or digit, then letters, digits and the marks that belong to them."""
    marks = _ranges([code for code in range(0x300, _MARK_SCAN_END) if _is_word_mark(code)])
    return re.compile(rf"[^\W_](?:[^\W_]|[{marks}])*")


@lru_cache(maxsize=1)
def _diacritics() -> dict[int, None]:
    """Combining marks that fold away: accents, not vowel signs."""
    return {
        code: None for code in range(0x300, _MARK_SCAN_END) if unicodedata.combining(chr(code))
    }


def fold(text: str) -> str:
    """Lower-case and strip diacritics, so 'São' equals 'sao'."""
    source = str(text or "")
    if source.isascii():
        return source.lower()
    return unicodedata.normalize("NFKD", source.casefold()).translate(_diacritics())


def surface(word: str) -> str:
    """A word as written: lower case, composed characters, accents kept."""
    return unicodedata.normalize("NFC", str(word or "").lower())


def surface_words(text: str) -> list[str]:
    return [surface(word) for word in word_pattern().findall(str(text or ""))]


def token_spans(text: str) -> list[Span]:
    """Each word as (start, end, folded word), offsets into `text`."""
    source = str(text or "")
    return [(m.start(), m.end(), fold(m.group())) for m in word_pattern().finditer(source)]


def words(text: str) -> list[str]:
    return [fold(word) for word in word_pattern().findall(str(text or ""))]


def query_units(query: str) -> list[str]:
    """Split a search query into folded units: word, prefix* or "quoted phrase".

    Bad syntax never raises: an unbalanced quote makes its words plain words.
    A phrase of one word is that word.
    """
    return _units(query, fold)


def surface_units(query: str) -> list[str]:
    """The units of a query as they were written, to send to the archive.

    Two spellings that fold to the same unit are one unit: the first stays.
    """
    return _units(query, surface)


def _units(query: str, form: Callable[[str], str]) -> list[str]:
    source = str(query or "")
    if source.count('"') % 2:
        source = source.replace('"', " ")
    units: list[str] = []
    cursor = 0
    for match in _QUOTED_RE.finditer(source):
        units.extend(_bare_units(source[cursor:match.start()], form))
        phrase = [form(word) for word in word_pattern().findall(match.group(1))]
        if len(phrase) == 1:
            units.append(phrase[0])
        elif phrase:
            units.append('"' + " ".join(phrase) + '"')
        cursor = match.end()
    units.extend(_bare_units(source[cursor:], form))
    first: dict[str, str] = {}
    for unit in units:
        first.setdefault(fold(unit), unit)
    return list(first.values())


def _bare_units(chunk: str, form: Callable[[str], str]) -> list[str]:
    units: list[str] = []
    for match in word_pattern().finditer(chunk):
        token = form(match.group())
        units.append(f"{token}*" if chunk[match.end():match.end() + 1] == "*" else token)
    return units


def find_unit(unit: str, spans: list[Span]) -> list[tuple[int, int]]:
    """Character ranges where one query unit matches the tokenised text. The unit may have any spelling."""
    unit = fold(unit)
    if unit.startswith('"'):
        return _find_phrase(unit.strip('"').split(), spans)
    if unit.endswith("*"):
        stem = unit[:-1]
        return [(s, e) for s, e, token in spans if stem and token.startswith(stem)]
    return [(s, e) for s, e, token in spans if token == unit]


def _find_phrase(phrase: list[str], spans: list[Span]) -> list[tuple[int, int]]:
    if not phrase:
        return []
    tokens = [token for _, _, token in spans]
    size = len(phrase)
    return [
        (spans[i][0], spans[i + size - 1][1])
        for i in range(len(tokens) - size + 1)
        if tokens[i:i + size] == phrase
    ]


def unit_matches(unit: str, text: str) -> bool:
    units = query_units(unit)
    spans = token_spans(text)
    return bool(units) and all(find_unit(part, spans) for part in units)
