"""Text folding and query parsing shared by search, names and the fake archive.

No stopword list and no word endings of one language: a word is any run of
letters or digits in any script, compared without case or diacritics.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[^\W_]+")
_QUOTED_RE = re.compile(r'"([^"]*)"')

Span = tuple[int, int, str]


def fold(text: str) -> str:
    """Lower-case and strip diacritics, so 'São' equals 'sao'."""
    decomposed = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def token_spans(text: str) -> list[Span]:
    """Each word as (start, end, folded word), offsets into `text`."""
    source = str(text or "")
    return [(m.start(), m.end(), fold(m.group())) for m in _WORD_RE.finditer(source)]


def words(text: str) -> list[str]:
    return [token for _, _, token in token_spans(text)]


def query_units(query: str) -> list[str]:
    """Split a search query into units: word, prefix* or "quoted phrase".

    Bad syntax never raises: an unbalanced quote makes its words plain words.
    """
    source = str(query or "")
    if source.count('"') % 2:
        source = source.replace('"', " ")
    units: list[str] = []
    cursor = 0
    for match in _QUOTED_RE.finditer(source):
        units.extend(_bare_units(source[cursor:match.start()]))
        phrase = " ".join(words(match.group(1)))
        if phrase:
            units.append(f'"{phrase}"')
        cursor = match.end()
    units.extend(_bare_units(source[cursor:]))
    return list(dict.fromkeys(units))


def _bare_units(chunk: str) -> list[str]:
    units: list[str] = []
    for start, end, token in token_spans(chunk):
        is_prefix = chunk[end:end + 1] == "*"
        units.append(f"{token}*" if is_prefix else token)
    return units


def find_unit(unit: str, spans: list[Span]) -> list[tuple[int, int]]:
    """Character ranges where one query unit matches the tokenised text."""
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
