"""Check that the answer quotes only what a message says.

Every quoted span of 12 characters or more must be part of one of the sources
after spaces, quote marks and case are normalised. A span that is not found is
returned, so the caller can flag it. Nothing is removed from the answer.

A straight quote mark opens a quote only at the start of a word and closes one
only at the end of a word. So an inch mark or an apostrophe pairs with nothing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

MIN_QUOTE_CHARS = 12
_MAX_QUOTE = 2000
_SPAN = f"{{1,{_MAX_QUOTE}}}"
_PATTERNS = (
    re.compile(rf'(?<!\w)"(?=\S)([^"]{_SPAN})(?<=\S)"(?!\w)'),
    re.compile(rf"[“„]([^“”„]{_SPAN})[”“]"),
    re.compile(rf"«\s*([^«»]{_SPAN}?)\s*»"),
    re.compile(rf"[「『]([^「」『』]{_SPAN})[」』]"),
    re.compile(rf"‘(.{_SPAN}?)’(?!\w)"),
    re.compile(rf"(?<!\w)'(?=\S)((?:[^'\n]|(?<=\w)'(?=\w)){_SPAN}?)'(?!\w)"),
    re.compile(r"^>\s?(.+)$", re.MULTILINE),
)
_QUOTE_MARKS = str.maketrans(
    {
        "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'",
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    }
)
_REFERENCE_RE = re.compile(r"\[\s*m?\d+(?:\s*[,\-–]\s*m?\d+)*\s*\]", re.IGNORECASE)
_ELISION_RE = re.compile(r"\[\s*\.{3,}\s*\]|\.{3,}")
_EDGE = " \t.,;:!?\"'()[]"


def quoted_spans(answer: str) -> list[str]:
    """Text inside quotation marks or on blockquote lines, in reading order."""
    text = str(answer or "")
    covered = bytearray(len(text))
    found: list[tuple[int, str]] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if covered.find(1, start, end) != -1:
                continue  # inside a span that another pattern took
            covered[start:end] = b"\x01" * (end - start)
            span = match.group(1).strip()
            if span:
                found.append((start, span))
    return [span for _, span in sorted(found)]


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", str(text or "")).translate(_QUOTE_MARKS)
    return " ".join(folded.split()).casefold()


def _pieces(span: str) -> list[str]:
    """The parts of a quote between elisions, long enough to check."""
    cleaned = normalise(_REFERENCE_RE.sub(" ", span))
    parts = (part.strip(_EDGE) for part in _ELISION_RE.split(cleaned))
    return [part for part in parts if len(part) >= MIN_QUOTE_CHARS]


def unverified_quotes(answer: str, sources: Iterable[str]) -> list[str]:
    """Quoted spans of the answer that no source holds."""
    haystack = "\n".join(normalise(source) for source in sources)
    held: dict[str, bool] = {}
    for span in quoted_spans(answer):
        if span not in held:
            held[span] = all(piece in haystack for piece in _pieces(span))
    return [span for span, found in held.items() if not found]
