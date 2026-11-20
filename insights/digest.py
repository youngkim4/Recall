"""The weekly digest: the few people most worth reaching out to now, and why.

Each insight has a weight. A fresh insight weighs more than an old one. A
person's score is the sum of their two heaviest reasons of different kinds,
so one person with many items (two pieces of the same news, say) does not
push out someone with one big one. `balance` has
no weight: it never picks a person, but it is shown with a person who is
picked.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from insights.thread import Insight

TOP = 3
REASONS_SHOWN = 3
SCORED_REASONS = 2
WEIGHTS: dict[str, float] = {
    "news": 3.0,
    "birthday": 3.0,
    "unanswered": 2.5,
    "promise": 2.0,
    "plan": 1.5,
    "drift": 1.0,
    "balance": 0.0,
}
STALE_AFTER = timedelta(days=30)
STALE_SHARE = 0.5
SOON_DAYS = 3
SOON_BONUS = 1.0
YEAR_WEEKS = 52


@dataclass(frozen=True)
class Pick:
    """One person in the digest. `reasons` are heaviest first."""

    thread: str
    name: str
    score: float
    reasons: tuple[Insight, ...]


def weight(insight: Insight, now: datetime) -> float:
    base = WEIGHTS.get(insight.kind, 0.0)
    if insight.kind == "drift":
        weeks = insight.figure("active_weeks") or 0.0
        return base + min(1.0, weeks / YEAR_WEEKS)
    if insight.kind == "birthday":
        days = insight.figure("days_until")
        return base + (SOON_BONUS if days is not None and days <= SOON_DAYS else 0.0)
    if insight.kind in ("news", "promise", "plan") and now - insight.when > STALE_AFTER:
        return base * STALE_SHARE
    return base


def picks(insights: Iterable[Insight], now: datetime, top: int = TOP) -> tuple[Pick, ...]:
    """The `top` people by score, highest first. A person with no weighted reason is never picked."""
    by_thread: dict[str, list[Insight]] = defaultdict(list)
    for insight in insights:
        by_thread[insight.thread].append(insight)
    ranked: list[Pick] = []
    for thread, found in by_thread.items():
        ordered = sorted(found, key=lambda item: (-weight(item, now), -item.when.timestamp()))
        heaviest_each_kind = list({item.kind: item for item in reversed(ordered)}.values())
        heaviest_each_kind.sort(key=lambda item: -weight(item, now))
        score = sum(weight(item, now) for item in heaviest_each_kind[:SCORED_REASONS])
        if score <= 0:
            continue
        ranked.append(Pick(thread, ordered[0].name, round(score, 3), tuple(ordered[:REASONS_SHOWN])))
    ranked.sort(key=lambda pick: (-pick.score, -max(item.when.timestamp() for item in pick.reasons), pick.name))
    return tuple(ranked[:top])
