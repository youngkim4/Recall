"""Choose the moments of a report.

Three steps, in this order: remove duplicates, apply a score floor, spread
what is left across the periods. Nothing is added to reach a number: a quiet
period gives no moment.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass

from report_engine.dates import parse_day
from report_engine.models import Moment

DUPLICATE_WINDOW_DAYS = 7
TITLE_RATIO = 0.75
TITLE_WORD_OVERLAP = 0.7


@dataclass(frozen=True)
class _Candidate:
    period: int
    moment: Moment


def _title_key(title: str) -> str:
    kept = "".join(char if char.isalnum() else " " for char in title.casefold())
    return " ".join(kept.split())


def _same_title(first: str, second: str) -> bool:
    if not first or not second:
        return False
    if first == second:
        return True
    words_a, words_b = set(first.split()), set(second.split())
    if len(words_a & words_b) / len(words_a | words_b) >= TITLE_WORD_OVERLAP:
        return True
    return difflib.SequenceMatcher(None, first, second).ratio() >= TITLE_RATIO


def _without_duplicates(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    """Two moments are one event when they cite the same message, or when their
    dates are within a week and their titles read the same. The stronger stays."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.moment.score,
            not item.moment.quote_verified,
            item.moment.date,
            item.period,
        ),
    )
    kept: list[_Candidate] = []
    titles_by_day: dict[int, list[str]] = {}
    cited: set[str] = set()
    for item in ranked:
        moment = item.moment
        if any(message_id in cited for message_id in moment.message_ids):
            continue
        day = parse_day(moment.date).toordinal()
        key = _title_key(moment.title)
        window = range(day - DUPLICATE_WINDOW_DAYS, day + DUPLICATE_WINDOW_DAYS + 1)
        if any(_same_title(key, other) for near in window for other in titles_by_day.get(near, ())):
            continue
        kept.append(item)
        titles_by_day.setdefault(day, []).append(key)
        cited.update(moment.message_ids)
    return kept


def _spread(candidates: Sequence[_Candidate], max_moments: int) -> list[Moment]:
    """Take the best of every period, then the second best of every period, and
    so on, until the cap. Inside one round the higher score goes first."""
    queues: dict[int, list[Moment]] = {}
    for item in candidates:
        queues.setdefault(item.period, []).append(item.moment)
    ordered = [
        sorted(queue, key=lambda moment: (-moment.score, moment.date)) for queue in queues.values()
    ]
    picked: list[Moment] = []
    depth = 0
    while len(picked) < max_moments:
        layer = [queue[depth] for queue in ordered if len(queue) > depth]
        if not layer:
            break
        layer.sort(key=lambda moment: (-moment.score, moment.date))
        picked.extend(layer[: max_moments - len(picked)])
        depth += 1
    return picked


def select_moments(
    per_period: Sequence[Sequence[Moment]], *, score_floor: float, max_moments: int
) -> tuple[Moment, ...]:
    """The report's moments, oldest first. `per_period` is in period order."""
    candidates = [
        _Candidate(period=index, moment=moment)
        for index, moments in enumerate(per_period)
        for moment in moments
    ]
    unique = _without_duplicates(candidates)
    strong = [item for item in unique if item.moment.score >= score_floor]
    picked = _spread(strong, max(0, max_moments))
    return tuple(sorted(picked, key=lambda moment: (moment.date, -moment.score, moment.title)))
