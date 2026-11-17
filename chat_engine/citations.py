"""Renumber references while the answer streams.

The model cites messages as [m3]. The reader sees [1], [2], ... in the order
the answer first uses them. The work is done on the stream, so the text shown
while it arrives is the text of the final answer.

A number in brackets without the "m", such as [3], is read as [m3]: the reader
takes every number in brackets for a citation, so none may pass unchecked.
"""

from __future__ import annotations

import re
from typing import Callable

_MAX_MARKER = 64
_MAX_RANGE = 20
_DASH = r"[-–—]"
_ONE = rf"m?\d+(?:\s*{_DASH}\s*m?\d+)?"
_BETWEEN = r"(?:\s*(?:[,;&]|and)\s*|\s+|(?=m))"  # never empty before a digit: one run of digits is one number
_MARKER_RE = re.compile(rf"(?P<space>[ \t]*)\[(?P<body>[^\[\]\n]{{1,{_MAX_MARKER}}})\]")
_BODY_RE = re.compile(rf"^\s*{_ONE}(?:{_BETWEEN}{_ONE})*\s*$", re.IGNORECASE)
_PIECE_RE = re.compile(rf"m?(\d+)(?:\s*{_DASH}\s*m?(\d+))?", re.IGNORECASE)


def _expand(body: str) -> list[str]:
    refs: list[str] = []
    for match in _PIECE_RE.finditer(body):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        numbers = range(start, end + 1) if 0 < end - start <= _MAX_RANGE else (start, end)
        refs.extend(f"m{number}" for number in numbers)
    return list(dict.fromkeys(refs))


class CitationStream:
    """Feed it the raw answer in pieces; it returns the pieces to show."""

    def __init__(self, known: Callable[[str], bool]) -> None:
        self._known = known
        self._pending = ""
        self._started = False
        self._order: list[str] = []
        self._unknown: list[str] = []

    @property
    def order(self) -> tuple[str, ...]:
        """References the answer used, in order of first use."""
        return tuple(self._order)

    @property
    def unknown(self) -> tuple[str, ...]:
        """References the answer used that no shown message has."""
        return tuple(self._unknown)

    def feed(self, text: str) -> str:
        self._pending += text
        ready, self._pending = self._split(self._pending)
        return self._emit(ready)

    def finish(self) -> str:
        ready, self._pending = self._pending, ""
        return self._emit(ready).rstrip()

    @staticmethod
    def _split(buffer: str) -> tuple[str, str]:
        """Hold back a marker that is not closed yet, and trailing whitespace."""
        cut = len(buffer.rstrip())
        opened = buffer.rfind("[")
        if opened != -1 and "]" not in buffer[opened:] and len(buffer) - opened <= _MAX_MARKER:
            cut = len(buffer[:opened].rstrip(" \t"))
            cut = min(cut, len(buffer[:cut].rstrip()))
        return buffer[:cut], buffer[cut:]

    def _emit(self, text: str) -> str:
        out = _MARKER_RE.sub(self._replace, text)
        if not self._started:
            out = out.lstrip()
            self._started = bool(out)
        return out

    def _replace(self, match: re.Match[str]) -> str:
        body = match.group("body")
        if not _BODY_RE.match(body):
            return match.group(0)
        numbers = []
        for ref in _expand(body):
            if self._known(ref):
                numbers.append(self._number(ref))
            elif ref not in self._unknown:
                self._unknown.append(ref)
        if not numbers:
            return ""
        return match.group("space") + "".join(f"[{number}]" for number in numbers)

    def _number(self, ref: str) -> int:
        if ref not in self._order:
            self._order.append(ref)
        return self._order.index(ref) + 1
