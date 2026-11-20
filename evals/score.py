"""Scores of one answer, and the summary of a run.

Citation recall: the share of the gold messages that the answer cites. An
answer is not asked to cite more than `CITE_TARGET` messages, so the share
is hits / min(gold, CITE_TARGET), at most 1. A question with no gold
messages (an absent one) has no recall. Citation precision is the share of
the cited messages that are gold.

Citations are mapped back to guids through the archive: a citation counts
only when the archive holds a message with its id.

Correctness comes from the judge (`evals.judge`). The summary gives, per kind
and overall: the share correct, the mean judge score, the mean recall and
precision, the money and the tokens.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Protocol

CITE_TARGET = 3


class MessageLookup(Protocol):
    def message(self, message_id: str) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class Metrics:
    """The figures of one question. Each is a number, so a run can be compared with another."""

    correct: bool
    score: float
    citation_recall: float | None
    citation_precision: float | None
    citations: int
    unverified_quotes: int
    invalid_references: int
    leaked_handle: bool
    limited: bool
    truncated: bool
    model_calls: int
    tool_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    chat_usd: float
    judge_usd: float
    wall_seconds: float
    error: str = ""

    @property
    def usd(self) -> float:
        return self.chat_usd + self.judge_usd


def cited_guids(citations: Iterable[Mapping[str, Any]], archive: MessageLookup) -> tuple[str, ...]:
    """The guids of the cited messages that the archive holds, in order, once each."""
    found: list[str] = []
    for citation in citations:
        message = archive.message(str(citation.get("messageId") or ""))
        if message is not None:
            found.append(str(message["id"]))
    return tuple(dict.fromkeys(found))


def citation_recall(gold: Sequence[str], cited: Sequence[str]) -> float | None:
    if not gold:
        return None
    hits = len(set(gold) & set(cited))
    return min(1.0, hits / min(len(gold), CITE_TARGET))


def citation_precision(gold: Sequence[str], cited: Sequence[str]) -> float | None:
    if not gold or not cited:
        return None
    return len(set(gold) & set(cited)) / len(set(cited))


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return fmean(kept) if kept else None


@dataclass(frozen=True)
class Summary:
    """The figures of a group of questions."""

    questions: int
    correct: int
    mean_score: float | None
    mean_recall: float | None
    mean_precision: float | None
    errors: int
    leaked: int
    unverified_quotes: int
    invalid_references: int
    model_calls: int
    tool_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    usd: float
    mean_wall_seconds: float | None

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.questions if self.questions else None


def summarize(rows: Sequence[Metrics]) -> Summary:
    return Summary(
        questions=len(rows),
        correct=sum(row.correct for row in rows),
        mean_score=_mean(row.score for row in rows),
        mean_recall=_mean(row.citation_recall for row in rows),
        mean_precision=_mean(row.citation_precision for row in rows),
        errors=sum(bool(row.error) for row in rows),
        leaked=sum(row.leaked_handle for row in rows),
        unverified_quotes=sum(row.unverified_quotes for row in rows),
        invalid_references=sum(row.invalid_references for row in rows),
        model_calls=sum(row.model_calls for row in rows),
        tool_calls=sum(row.tool_calls for row in rows),
        input_tokens=sum(row.input_tokens for row in rows),
        cached_input_tokens=sum(row.cached_input_tokens for row in rows),
        output_tokens=sum(row.output_tokens for row in rows),
        usd=sum(row.usd for row in rows),
        mean_wall_seconds=_mean(row.wall_seconds for row in rows),
    )


def by_kind(kinds: Sequence[str], rows: Sequence[tuple[str, Metrics]]) -> dict[str, Summary]:
    """A summary for each kind that has a row, in the order of `kinds`, then "all"."""
    grouped = {kind: [metrics for got, metrics in rows if got == kind] for kind in kinds}
    table = {kind: summarize(found) for kind, found in grouped.items() if found}
    return {**table, "all": summarize([metrics for _kind, metrics in rows])}


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def table(summaries: Mapping[str, Summary]) -> str:
    """The summary as a short text table."""
    head = f"{'kind':<13}{'n':>4}{'acc':>7}{'score':>7}{'recall':>8}{'prec':>7}{'calls':>7}{'usd':>9}{'sec':>7}"
    lines = [head, "-" * len(head)]
    for kind, found in summaries.items():
        wall = "-" if found.mean_wall_seconds is None else f"{found.mean_wall_seconds:.1f}"
        lines.append(
            f"{kind:<13}{found.questions:>4}{_share(found.accuracy):>7}{_share(found.mean_score):>7}"
            f"{_share(found.mean_recall):>8}{_share(found.mean_precision):>7}{found.model_calls:>7}"
            f"{found.usd:>9.4f}{wall:>7}"
        )
    return "\n".join(lines)
