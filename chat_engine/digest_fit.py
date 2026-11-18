"""How a conversation digest is cut to its budget. Token counts in, a plan out.

The rule:

1. Every period stays: its title line is never dropped.
2. The summaries come first. When they do not fit together, each summary is
   cut by the same share and no moment is shown.
3. When the summaries fit, the moments take the room that is left. Moments
   are dropped one at a time from the weakest end of the period that still
   has the most, until the rest fits. On a tie the earliest period gives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .render import squeeze

CUT_MARK = "[cut]"


@dataclass(frozen=True)
class Fit:
    """share: the part of each summary that stays, 0 to 1. kept: the moments each period keeps."""

    share: float
    kept: tuple[int, ...]

    def summary_tokens(self, tokens: int) -> int:
        """The tokens a summary of `tokens` tokens may keep."""
        return tokens if self.share >= 1.0 else int(tokens * self.share)

    def dropped(self, moments: Sequence[Sequence[int]]) -> int:
        return sum(len(costs) - kept for costs, kept in zip(moments, self.kept, strict=True))

    def cut(self, moments: Sequence[Sequence[int]]) -> bool:
        return self.share < 1.0 or self.dropped(moments) > 0


def fit_digest(
    *, fixed: int, summaries: Sequence[int], moments: Sequence[Sequence[int]], room: int
) -> Fit:
    """The plan for one digest.

    fixed: tokens of the lines that always stay. summaries: tokens of each
    summary. moments: tokens of each moment of each period, the strongest
    first. room: the tokens the digest may hold.
    """
    left = max(0, room - fixed)
    wanted = sum(summaries)
    if wanted > left:
        return Fit(left / wanted, tuple(0 for _ in moments))
    left -= wanted
    kept = [len(costs) for costs in moments]
    used = sum(sum(costs) for costs in moments)
    while used > left and any(kept):
        index = kept.index(max(kept))
        kept[index] -= 1
        used -= moments[index][kept[index]]
    return Fit(1.0, tuple(kept))


def cut_summary(text: str, tokens: int, count_tokens: Callable[[str], int]) -> str:
    """The start of a summary that fits in `tokens`, with a mark that it was cut."""
    if count_tokens(text) <= tokens:
        return text
    kept = squeeze(text, tokens - count_tokens(f" {CUT_MARK}"), count_tokens)
    return f"{kept} {CUT_MARK}" if kept else CUT_MARK
