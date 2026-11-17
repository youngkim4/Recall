"""Check that the answer quotes only what a message says.

Every quoted span of 12 characters or more must be part of one source after
spaces, quote marks and case are normalised. A span that is not found is
returned, so the caller can flag it. Nothing is removed from the answer.

A quote with an elision ("the start ... the end") must have all its parts in
one source, in their order: parts of two messages are not one quote.

A straight double quote mark opens a quote only before text and closes one
only after text. So an inch mark pairs with nothing. A straight single quote
mark next to a Latin letter or a digit is an apostrophe.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

MIN_QUOTE_CHARS = 12
_MAX_QUOTE = 2000
_SPAN = f"{{1,{_MAX_QUOTE}}}"
_LATIN = "A-Za-z0-9\u00c0-\u024f"  # the letters an apostrophe stands between
_PATTERNS = (
    re.compile(rf'"(?=\S)([^"]{_SPAN})(?<=\S)"'),
    re.compile(rf"[“„]([^“”„]{_SPAN})[”“]"),
    re.compile(rf"«\s*([^«»]{_SPAN}?)\s*»"),
    re.compile(rf"[「『]([^「」『』]{_SPAN})[」』]"),
    re.compile(rf"‘(.{_SPAN}?)’(?![{_LATIN}])"),
    re.compile(rf"(?<![{_LATIN}])'(?=\S)((?:[^'\n]|(?<=[{_LATIN}])'(?=[{_LATIN}])){_SPAN}?)'(?![{_LATIN}])"),
    re.compile(r"^>\s?(.+)$", re.MULTILINE),
)
_QUOTE_MARKS = str.maketrans(
    {
        "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'",
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    }
)
_REFERENCE_RE = re.compile(r"\[\s*m?\d+(?:\s*[,\-–]\s*m?\d+)*\s*\]", re.IGNORECASE)
_ELISION_RE = re.compile(r"\[\s*\.{3,}\s*\]|\.{3,}|…")
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


def _pieces(span: str, without_references: bool) -> list[str]:
    """The parts of a quote between elisions. [] when the whole quote is too short to check."""
    text = _REFERENCE_RE.sub(" ", span) if without_references else span
    parts = [part.strip(_EDGE) for part in _ELISION_RE.split(normalise(text))]
    parts = [part for part in parts if part]
    return parts if sum(len(part) for part in parts) >= MIN_QUOTE_CHARS else []


def _in_order(pieces: Sequence[str], source: str) -> bool:
    cursor = 0
    for piece in pieces:
        found = source.find(piece, cursor)
        if found < 0:
            return False
        cursor = found + len(piece)
    return True


def _is_held(span: str, sources: Sequence[str], joined: str) -> bool:
    """True when one source holds the quote, or the quote is too short to check.

    A reference inside the quote marks, such as [m3], is not part of the
    quote. But "[3]" can also be text of the message, so both readings count.
    """
    for without_references in (True, False):
        pieces = _pieces(span, without_references)
        if not pieces:
            return True
        if len(pieces) == 1:
            if pieces[0] in joined:
                return True
        elif any(_in_order(pieces, source) for source in sources):
            return True
    return False


def unverified_quotes(answer: str, sources: Iterable[str]) -> list[str]:
    """Quoted spans of the answer that no source holds."""
    normalised = [normalise(source) for source in sources]
    joined = "\n".join(normalised)
    held: dict[str, bool] = {}
    for span in quoted_spans(answer):
        if span not in held:
            held[span] = _is_held(span, normalised, joined)
    return [span for span, found in held.items() if not found]
