"""The recipe of the segments: which switches cut them and write their text.

The default recipe has no switch on. Its segments are the segments of
`segments.py` byte for byte. Each switch can be on alone, so that each one
can be measured alone.

- `context`: one line before the text of each segment. It holds the month and
  the year of the first message, and the words that the caller gives for the
  conversation (`SegmentContext`). The line is embedded. Chat does not show it.
- `overlap`: each segment also holds the last OVERLAP_MESSAGES messages of the
  segment before it, when the cut between the two was not a gap. So a moment
  on a cut is not split in two.
- `adaptive`: the gap that starts a new segment comes from the rhythm of the
  conversation (`adaptive_gap`), not from the fixed GAP_MINUTES.

The meta file of an index names the recipe of its build. A meta file with no
name is the default recipe. Change RECIPE_VERSION with each change to a rule
of a switch: the status then says that the index is behind.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

RECIPE_VERSION = 1
SWITCHES = ("context", "overlap", "adaptive")
OVERLAP_MESSAGES = 2
ADAPTIVE_PERCENTILE = 85
ADAPTIVE_MIN = timedelta(minutes=30)
ADAPTIVE_MAX = timedelta(hours=8)
ADAPTIVE_MIN_GAPS = 8
CONTEXT_LABEL = "Context"
CONTEXT_CHARS = 400
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip

# (conversation id, timestamp of the first message of the segment) -> the words of the context line
SegmentContext = Callable[[str, str], str]


@dataclass(frozen=True)
class Recipe:
    """The switches of a build. The name holds the version and each switch that is on."""

    context: bool = False
    overlap: bool = False
    adaptive: bool = False

    @property
    def name(self) -> str:
        on = [switch for switch in SWITCHES if getattr(self, switch)]
        return "+".join([f"v{RECIPE_VERSION}", *on])


DEFAULT_RECIPE = Recipe()
DEFAULT_NAME = DEFAULT_RECIPE.name


def recipe_from_words(text: str) -> tuple[Recipe, tuple[str, ...]]:
    """The recipe that a comma list of switches names, and each word that names no switch.

    Case and spaces do not count. An empty list is the default recipe.
    """
    words = [word.strip().lower() for word in text.split(",") if word.strip()]
    unknown = tuple(dict.fromkeys(word for word in words if word not in SWITCHES))
    return Recipe(**{switch: switch in words for switch in SWITCHES}), unknown


def adaptive_gap(moments: Sequence[datetime], default: timedelta) -> timedelta:
    """The gap that starts a new segment in one conversation.

    The gaps are the times between consecutive messages with text, oldest
    first, in whole seconds. A time that goes back is a gap of 0. The gap is
    the 85th percentile of them by nearest rank: sorted, the gap at place
    ceil(0.85 * n). It is then clamped to [30 min, 8 h]. It needs no clock and
    no random number, so the same messages always give the same gap.

    Why: most gaps of a conversation fall inside a talk, and a few are the
    pauses between talks. The 85th percentile sits at the top of the gaps of
    a talk, so a pause longer than nearly every gap of a talk starts a new
    segment. Below 30 minutes a short break of a chatty pair would cut one
    talk in two. Above 8 hours a segment of a slow conversation would join
    two days. With fewer than ADAPTIVE_MIN_GAPS gaps the rhythm is not known,
    and the gap is `default`.
    """
    gaps = sorted(max(0, int((later - earlier).total_seconds())) for earlier, later in zip(moments, moments[1:]))
    if len(gaps) < ADAPTIVE_MIN_GAPS:
        return default
    rank = math.ceil(ADAPTIVE_PERCENTILE * len(gaps) / 100)
    return min(max(timedelta(seconds=gaps[rank - 1]), ADAPTIVE_MIN), ADAPTIVE_MAX)


def context_line(first_timestamp: str, words: str) -> str:
    """`Context: June 2024; <words>`, at most CONTEXT_CHARS characters, on one line.

    `first_timestamp` is "YYYY-MM-DD HH:MM:SS" and was checked before.
    """
    head = f"{CONTEXT_LABEL}: {MONTHS[int(first_timestamp[5:7]) - 1]} {first_timestamp[:4]}"
    flat = " ".join(str(words or "").split())
    return (f"{head}; {flat}" if flat else head)[:CONTEXT_CHARS]
