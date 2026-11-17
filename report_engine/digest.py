"""The reply of one map call, checked for shape.

A reply is outside data. This module accepts it only as far as its shape goes:
a summary is required, loose values are coerced, long text is cut. Whether a
date, a quote or a reference is true is the job of `verify`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from report_engine.errors import ReportEngineError

MAX_MOMENTS_PER_PERIOD = 8
MAX_SUMMARY_CHARS = 6_000
MAX_CARRY_CHARS = 1_500
MAX_TITLE_CHARS = 200
MAX_DETAIL_CHARS = 1_000
MAX_QUOTE_CHARS = 600
MAX_REFS = 8


class DigestError(ReportEngineError):
    """A map reply that can not be used."""


@dataclass(frozen=True)
class RawMoment:
    """A moment as the model gave it. Nothing in it is verified yet."""

    date: str
    title: str
    detail: str
    category: str
    score: float
    quote: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class PeriodDigest:
    summary: str
    carry_over: str
    moments: tuple[RawMoment, ...]


def _text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return (value if isinstance(value, str) else str(value))[:limit]


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return 0.0
    else:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _refs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))[:MAX_REFS]


def _moment(item: Mapping[str, Any]) -> RawMoment:
    return RawMoment(
        date=_text(item.get("date"), 32),
        title=_text(item.get("title"), MAX_TITLE_CHARS),
        detail=_text(item.get("detail"), MAX_DETAIL_CHARS),
        category=_text(item.get("category"), 32),
        score=_score(item.get("score")),
        quote=_text(item.get("quote"), MAX_QUOTE_CHARS),
        refs=_refs(item.get("refs")),
    )


def _strongest(moments: list[RawMoment]) -> tuple[RawMoment, ...]:
    """At most the allowed number, the highest scores, in their original order."""
    if len(moments) <= MAX_MOMENTS_PER_PERIOD:
        return tuple(moments)
    ranked = sorted(range(len(moments)), key=lambda index: (-moments[index].score, index))
    keep = sorted(ranked[:MAX_MOMENTS_PER_PERIOD])
    return tuple(moments[index] for index in keep)


def parse_digest(parsed: Any) -> PeriodDigest:
    """Turn a parsed reply into a digest. Raises DigestError when it has no summary."""
    if not isinstance(parsed, Mapping):
        raise DigestError("the reply is not a JSON object")
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise DigestError("the reply has no summary")
    items = parsed.get("moments", [])
    if not isinstance(items, (list, tuple)):
        raise DigestError("the reply's moments are not a list")
    moments = [_moment(item) for item in items if isinstance(item, Mapping)]
    return PeriodDigest(
        summary=summary.strip()[:MAX_SUMMARY_CHARS],
        carry_over=_text(parsed.get("carry_over"), MAX_CARRY_CHARS * 4).strip()[:MAX_CARRY_CHARS],
        moments=_strongest(moments),
    )


def digest_payload(digest: PeriodDigest) -> dict[str, Any]:
    """Plain JSON for the cache. `parse_digest` reads it back unchanged."""
    return {
        "summary": digest.summary,
        "carry_over": digest.carry_over,
        "moments": [
            {
                "date": moment.date,
                "title": moment.title,
                "detail": moment.detail,
                "category": moment.category,
                "score": moment.score,
                "quote": moment.quote,
                "refs": list(moment.refs),
            }
            for moment in digest.moments
        ],
    }
