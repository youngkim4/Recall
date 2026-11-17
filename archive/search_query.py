"""Turn text that a user typed into a safe full-text query.

Rules:
- Words are AND-ed.
- "Quoted text" is a phrase.
- A trailing * makes a prefix: `fam*`, `"good mor"*`.
- Case and diacritics do not matter.
- Nothing the user types is query syntax. Every term is passed to FTS5 as a
  quoted string, so AND, OR, NOT, NEAR, brackets and colons are plain words.

The index tokenizer splits on spaces, punctuation, symbols and emoji. It
cannot find a word inside Chinese, Japanese, Korean or Thai text, and it does
not index emoji or punctuation. A term like that is matched as a substring:
case matters only for A-Z, and an accent must be the same as in the text.
A term that mixes letters with a symbol or an emoji (`c++`, `$5`, a word
with an emoji) must match both ways: its words in the index and the whole
term as a substring.

A query is cut to MAX_QUERY_CHARS characters and MAX_TERMS terms. A longer
query would only cost time: no person types one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_QUERY_CHARS = 1000
MAX_TERMS = 32

# curly and fullwidth double quotes count as plain double quotes
_QUOTES = dict.fromkeys((0x201C, 0x201D, 0x201E, 0x201F, 0xFF02), '"')
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# a phrase in quotes with an optional star, or a run of characters that are not space or quote
_TERM = re.compile(r'"([^"]*)"(\*?)|([^\s"]+)')
# Scripts that write words without spaces between them, and letters that the index does not know.
_UNSPACED_RANGES = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x1780, 0x17FF),  # Khmer
    (0x19B0, 0x19C9),  # New Tai Lue vowel signs: letters for Python, separators for the index
    (0x1CF2, 0x1CF3),  # two Vedic signs, the same
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3130, 0x318F),  # Hangul compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK ideographs, extension A
    (0x4E00, 0x9FFF),  # CJK ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF66, 0xFF9F),  # halfwidth Katakana
    (0x20000, 0x3134F),  # CJK ideographs, extensions B to G
)
_UNSPACED = re.compile("[" + "".join(f"{chr(first)}-{chr(last)}" for first, last in _UNSPACED_RANGES) + "]")
_ASCII_LOWER = {code: code + 32 for code in range(ord("A"), ord("Z") + 1)}


@dataclass(frozen=True)
class ParsedQuery:
    """A user query, split into the part the index answers and the substrings."""

    match: str | None  # FTS5 MATCH expression, None when no term can use the index
    substrings: tuple[str, ...]  # needles for instr(lower(text), needle)
    marked: str | None = None  # the part of `match` whose hits are marked from the index

    @property
    def is_empty(self) -> bool:
        return self.match is None and not self.substrings


def ascii_lower(text: str) -> str:
    """Lower-case A-Z only, like SQLite's lower(). The length never changes."""
    return text.translate(_ASCII_LOWER)


def _terms(query: str) -> list[tuple[str, bool]]:
    """(text, is_prefix) for each term of the query."""
    cleaned = _CONTROL.sub(" ", query[:MAX_QUERY_CHARS].translate(_QUOTES))
    if cleaned.count('"') % 2:
        last = cleaned.rindex('"')
        cleaned = cleaned[:last] + " " + cleaned[last + 1 :]  # a quote that is never closed is not a quote
    terms: list[tuple[str, bool]] = []
    for found in _TERM.finditer(cleaned):
        if found.group(3) is None:
            text, prefix = found.group(1).strip(), bool(found.group(2))
        else:
            word = found.group(3)
            text = word.rstrip("*")
            prefix = text != word
        if text:
            terms.append((text, prefix))
    return terms[:MAX_TERMS]


def _uses_index(text: str) -> bool:
    return any(char.isalnum() for char in text) and _UNSPACED.search(text) is None


def _has_symbol(text: str) -> bool:
    """True for a term with a symbol or an emoji. The index drops these, so the term is also a substring."""
    return any(unicodedata.category(char)[0] == "S" for char in text)


def parse_query(query: str) -> ParsedQuery:
    """Parse text typed by a user. Never raises for any string."""
    if not isinstance(query, str):
        raise TypeError("a search query must be text")
    indexed: list[str] = []
    marked: list[str] = []
    substrings: list[str] = []
    for text, prefix in _terms(query.encode("utf-8", "replace").decode("utf-8")):  # no lone surrogate goes to SQLite
        if _uses_index(text):
            quoted = '"' + text.replace('"', '""') + '"'
            indexed.append(f"{quoted} *" if prefix else quoted)
            if _has_symbol(text):
                substrings.append(ascii_lower(text))  # its range is the whole term, not each word of it
            else:
                marked.append(indexed[-1])
        else:
            substrings.append(ascii_lower(text))
    return ParsedQuery(
        match=" ".join(indexed) or None, substrings=tuple(dict.fromkeys(substrings)), marked=" ".join(marked) or None
    )
