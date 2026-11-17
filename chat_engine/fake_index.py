"""A small word index for the in-memory archive, so searches stay fast."""

from __future__ import annotations

from typing import Optional, Sequence

from .textnorm import Span, find_unit, token_spans


class TokenIndex:
    """Which texts hold which words. Positions are indexes into the given texts."""

    def __init__(self, texts: Sequence[str]) -> None:
        self._spans: list[list[Span]] = [token_spans(text) for text in texts]
        self._postings: dict[str, set[int]] = {}
        for position, spans in enumerate(self._spans):
            for _, _, token in spans:
                self._postings.setdefault(token, set()).add(position)

    def candidates(self, units: Sequence[str]) -> list[int]:
        """Positions that hold every word of every unit, in ascending order."""
        found: Optional[set[int]] = None
        for unit in units:
            holders = self._holders(unit)
            found = holders if found is None else found & holders
            if not found:
                return []
        return sorted(found or ())

    def ranges(self, position: int, units: Sequence[str]) -> list[tuple[int, int]]:
        """Match ranges of all units in one text, or [] unless every unit matches."""
        spans = self._spans[position]
        ranges: list[tuple[int, int]] = []
        for unit in units:
            matched = find_unit(unit, spans)
            if not matched:
                return []
            ranges.extend(matched)
        return sorted(set(ranges))

    def _holders(self, unit: str) -> set[int]:
        if unit.startswith('"'):
            holders: Optional[set[int]] = None
            for word in unit.strip('"').split():
                with_word = self._postings.get(word, set())
                holders = with_word if holders is None else holders & with_word
            return set(holders or ())
        if unit.endswith("*"):
            stem = unit[:-1]
            merged: set[int] = set()
            for token, positions in self._postings.items():
                if stem and token.startswith(stem):
                    merged |= positions
            return merged
        return set(self._postings.get(unit, set()))
