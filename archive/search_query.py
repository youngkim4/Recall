"""Turn text that a user typed into a safe full-text query.

Rules:
- Words are AND-ed.
- "Quoted text" is a phrase.
- A trailing * makes a prefix: `fam*`, `"good mor"*`.
- Case and diacritics do not matter.
- Nothing the user types is query syntax. Every term is passed to FTS5 as a
  quoted string, so AND, OR, NOT, NEAR, brackets and colons are plain words.

The index tokenizer (unicode61) splits on spaces and punctuation. It cannot
find a word inside Chinese, Japanese, Korean or Thai text, and it does not
index emoji or punctuation. A term like that is matched as a substring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "＂": '"'})
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# a phrase in quotes with an optional star, or a run of characters that are not space or quote
_TERM = re.compile(r'"([^"]*)"(\*?)|([^\s"]+)')
# scripts that write words without spaces between them
_UNSPACED = re.compile(
    "[฀-໿က-႟ᄀ-ᇿក-៿぀-ヿ㄰-㆏ㇰ-ㇿ"
    "㐀-䶿一-鿿가-힯豈-﫿ｦ-ﾟ\U00020000-\U0003134f]"
)
_ASCII_LOWER = {code: code + 32 for code in range(ord("A"), ord("Z") + 1)}


@dataclass(frozen=True)
class ParsedQuery:
    """A user query, split into the part the index answers and the substrings."""

    match: str | None  # FTS5 MATCH expression, None when no term can use the index
    substrings: tuple[str, ...]  # needles for instr(lower(text), needle)

    @property
    def is_empty(self) -> bool:
        return self.match is None and not self.substrings


def ascii_lower(text: str) -> str:
    """Lower-case A-Z only, like SQLite's lower(). The length never changes."""
    return text.translate(_ASCII_LOWER)


def _terms(query: str) -> list[tuple[str, bool]]:
    """(text, is_prefix) for each term of the query."""
    cleaned = _CONTROL.sub(" ", query.translate(_QUOTES))
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
    return terms


def _uses_index(text: str) -> bool:
    return any(char.isalnum() for char in text) and _UNSPACED.search(text) is None


def parse_query(query: str) -> ParsedQuery:
    """Parse text typed by a user. Never raises for any string."""
    if not isinstance(query, str):
        raise TypeError("a search query must be text")
    indexed: list[str] = []
    substrings: list[str] = []
    for text, prefix in _terms(query):
        if _uses_index(text):
            quoted = '"' + text.replace('"', '""') + '"'
            indexed.append(f"{quoted} *" if prefix else quoted)
        else:
            substrings.append(ascii_lower(text))
    return ParsedQuery(match=" ".join(indexed) or None, substrings=tuple(dict.fromkeys(substrings)))
