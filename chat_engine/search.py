"""Search over the archive port.

"all" passes the query to the archive: words are AND-ed there.
"any" ranks messages that hold any of the words. A word counts for more the
fewer messages hold it, so no stopword list of one language is needed.
One long exchange never takes all the slots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from .ports import ChatArchive, Message, SearchHit
from .textnorm import find_unit, query_units, token_spans

MAX_FETCH = 60
_ANY_FETCH = 40
_PAIR_FETCH = 20
_MAX_UNITS = 12
_RANKING_UNITS = 4
_COMMON_SHARE = 0.02
_COMMON_FLOOR = 5
_BURST_WINDOW = timedelta(minutes=10)
_MOMENT_WINDOW = timedelta(minutes=30)
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SearchRequest:
    query: str
    match: str = "all"
    conversations: Optional[tuple[str, ...]] = None
    since: Optional[str] = None
    until: Optional[str] = None
    from_me: Optional[bool] = None
    order: str = "newest"
    limit: int = 8
    boost: tuple[str, ...] = ()


@dataclass(frozen=True)
class Found:
    message: Message
    focus: Optional[int] = None
    score: float = 0.0


@dataclass(frozen=True)
class SearchOutcome:
    hits: tuple[Found, ...]
    total: Optional[int] = None  # exact; None when words were ranked ("any")
    word_counts: tuple[tuple[str, int], ...] = ()
    used: tuple[str, ...] = ()
    common: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()


def run_search(archive: ChatArchive, request: SearchRequest) -> SearchOutcome:
    units = query_units(request.query)[:_MAX_UNITS]
    if request.match == "any" and len(units) > 1:
        return _search_any(archive, request, units)
    return _search_all(archive, request, units)


def dedupe_moments(
    messages: Sequence[Message], per_moment: int = 2, window: timedelta = _MOMENT_WINDOW
) -> list[Message]:
    """Keep at most `per_moment` messages of one exchange. Order is kept."""
    kept: list[Message] = []
    times: dict[str, list[datetime]] = {}
    for message in messages:
        moment = _moment(message)
        near = times.setdefault(str(message["conversation"]), [])
        if moment is not None and sum(1 for other in near if abs(other - moment) <= window) >= per_moment:
            continue
        if moment is not None:
            near.append(moment)
        kept.append(message)
    return kept


def _filters(request: SearchRequest) -> dict[str, Any]:
    return {
        "conversations": None if request.conversations is None else list(request.conversations),
        "since": request.since,
        "until": request.until,
        "from_me": request.from_me,
    }


def _moment(message: Message) -> Optional[datetime]:
    try:
        return datetime.strptime(str(message.get("timestamp") or "")[:19], _TIME_FORMAT)
    except ValueError:
        return None


def _focus(hit: SearchHit) -> Optional[int]:
    ranges = hit.get("ranges") or []
    try:
        start = int(ranges[0][0])
    except (IndexError, TypeError, ValueError):
        return None
    return start if 0 <= start <= len(hit["message"].get("text") or "") else None


def _capped(founds: Sequence[Found], limit: int) -> tuple[Found, ...]:
    keep = {str(m["id"]) for m in dedupe_moments([found.message for found in founds])}
    return tuple(found for found in founds if str(found.message["id"]) in keep)[:limit]


def _count(archive: ChatArchive, request: SearchRequest, text: Optional[str]) -> int:
    result = archive.count(**_filters(request), text=text, group_by=None)
    return int(result) if isinstance(result, int) else 0


def _search_all(archive: ChatArchive, request: SearchRequest, units: list[str]) -> SearchOutcome:
    result = archive.search(
        request.query,
        **_filters(request),
        limit=min(MAX_FETCH, max(request.limit * 3, request.limit)),
        offset=0,
        order=request.order,
    )
    founds = [Found(hit["message"], _focus(hit)) for hit in result["hits"]]
    total = int(result["total"])
    word_counts: tuple[tuple[str, int], ...] = ()
    if total == 0 and len(units) > 1:
        word_counts = tuple((unit, _count(archive, request, unit)) for unit in units)
    return SearchOutcome(hits=_capped(founds, request.limit), total=total, word_counts=word_counts)


def _weigh(archive: ChatArchive, request: SearchRequest, units: list[str]) -> tuple[int, dict[str, int]]:
    population = _count(archive, request, None)
    return population, {unit: _count(archive, request, unit) for unit in units}


def _is_common(count: int, population: int) -> bool:
    return count > _COMMON_FLOOR and count > population * _COMMON_SHARE


def _weight(count: int, population: int) -> float:
    return max(0.1, math.log((population + 1) / (count + 0.5))) ** 2


def _search_any(archive: ChatArchive, request: SearchRequest, units: list[str]) -> SearchOutcome:
    population, counts = _weigh(archive, request, units)
    absent = tuple(unit for unit in units if counts[unit] == 0)
    common = tuple(unit for unit in units if _is_common(counts[unit], population))
    ranked = sorted(
        (unit for unit in units if unit not in absent and unit not in common),
        key=lambda unit: counts[unit],
    )[:_RANKING_UNITS]
    weights = {unit: _weight(counts[unit], population) for unit in ranked}
    outcome = SearchOutcome(
        hits=(), word_counts=tuple(counts.items()), used=tuple(ranked), common=common, absent=absent
    )
    if not ranked:
        return outcome
    candidates = _candidates(archive, request, ranked)
    founds = _score(candidates, weights, request)
    return SearchOutcome(
        hits=_capped(founds, request.limit),
        word_counts=outcome.word_counts,
        used=outcome.used,
        common=common,
        absent=absent,
    )


@dataclass
class _Candidate:
    message: Message
    focus: Optional[int]
    units: set[str]


def _candidates(archive: ChatArchive, request: SearchRequest, ranked: list[str]) -> list[_Candidate]:
    order = request.order if request.order in ("newest", "oldest", "relevance") else "newest"
    allowed = _filters(request)["conversations"]
    queries: list[tuple[str, tuple[str, ...], Optional[list[str]], int]] = [
        (unit, (unit,), allowed, _ANY_FETCH) for unit in ranked
    ]
    if len(ranked) > 1:
        pair = (ranked[0], ranked[1])
        queries.append((" ".join(pair), pair, allowed, _PAIR_FETCH))
    boosted = _boosted(request)
    if boosted:
        queries.extend((unit, (unit,), boosted, _PAIR_FETCH) for unit in ranked[:2])
    found: dict[str, _Candidate] = {}
    for query, matched, conversations, limit in queries:
        result = archive.search(
            query,
            conversations=conversations,
            since=request.since,
            until=request.until,
            from_me=request.from_me,
            limit=limit,
            offset=0,
            order=order,
        )
        for hit in result["hits"]:
            message = hit["message"]
            candidate = found.setdefault(str(message["id"]), _Candidate(message, _focus(hit), set()))
            candidate.units.update(matched)
    return list(found.values())


def _boosted(request: SearchRequest) -> Optional[list[str]]:
    if not request.boost:
        return None
    if request.conversations is None:
        return list(request.boost)
    return [c for c in request.boost if c in request.conversations] or None


def _score(
    candidates: list[_Candidate], weights: dict[str, float], request: SearchRequest
) -> list[Found]:
    for candidate in candidates:
        spans = token_spans(candidate.message.get("text") or "")
        candidate.units.update(unit for unit in weights if find_unit(unit, spans))
    bonus = 0.5 * max(weights.values())
    scored = []
    for candidate in candidates:
        score = sum(weights[unit] for unit in candidate.units if unit in weights)
        score += 0.5 * sum(weights[unit] for unit in _burst_units(candidate, candidates) if unit in weights)
        if str(candidate.message["conversation"]) in request.boost:
            score += bonus
        scored.append(Found(candidate.message, candidate.focus, score))
    newest_first = request.order != "oldest"
    scored.sort(key=lambda found: str(found.message.get("timestamp") or ""), reverse=newest_first)
    scored.sort(key=lambda found: found.score, reverse=True)
    return scored


def _burst_units(candidate: _Candidate, candidates: list[_Candidate]) -> set[str]:
    """Words that neighbours in the same exchange hold and this message does not."""
    moment = _moment(candidate.message)
    if moment is None:
        return set()
    nearby: set[str] = set()
    for other in candidates:
        if other is candidate or other.message["conversation"] != candidate.message["conversation"]:
            continue
        other_moment = _moment(other.message)
        if other_moment is not None and abs(other_moment - moment) <= _BURST_WINDOW:
            nearby |= other.units
    return nearby - candidate.units
