"""Scores of one answer, and the summary of a run.

Citation recall: the share of the gold messages that the answer cites. An
answer is not asked to cite more than `CITE_TARGET` messages, so the share
is hits / min(gold, CITE_TARGET), at most 1. A question with no gold
messages (an absent one) has no recall. Citation precision is the share of
the cited messages that are gold. A count question has neither: a count
answer cites no single message, so its gold guids are a sample to read, not
a target (`NO_CITATION_KINDS`).

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
NO_CITATION_KINDS = frozenset({"count"})


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


def citation_scores(kind: str, gold: Sequence[str], cited: Sequence[str]) -> tuple[float | None, float | None]:
    """Recall and precision for a question of `kind`. A kind in NO_CITATION_KINDS has neither."""
    if kind in NO_CITATION_KINDS:
        return None, None
    return citation_recall(gold, cited), citation_precision(gold, cited)


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


@dataclass(frozen=True)
class Spread:
    """The accuracy of one kind over the rounds of a --repeat run."""

    rounds: tuple[float, ...]

    @property
    def mean(self) -> float:
        return fmean(self.rounds)

    @property
    def low(self) -> float:
        return min(self.rounds)

    @property
    def high(self) -> float:
        return max(self.rounds)


def spread(kinds: Sequence[str], rows: Sequence[tuple[str, int, Metrics]]) -> dict[str, Spread]:
    """The accuracy of each round, per kind that has rows and for "all". A row is (kind, round, metrics)."""
    rounds = sorted({number for _kind, number, _metrics in rows})

    def accuracies(wanted: str | None) -> tuple[float, ...]:
        found = []
        for number in rounds:
            kept = [m for kind, got, m in rows if got == number and (wanted is None or kind == wanted)]
            if kept:
                found.append(sum(m.correct for m in kept) / len(kept))
        return tuple(found)

    per_kind = {kind: Spread(accuracies(kind)) for kind in kinds if any(row[0] == kind for row in rows)}
    return {**per_kind, "all": Spread(accuracies(None))} if rows else {}


def spread_table(spreads: Mapping[str, Spread]) -> str:
    """Mean accuracy and its range over the rounds, per kind."""
    head = f"{'kind':<13}{'rounds':>7}{'mean':>7}{'min':>7}{'max':>7}"
    lines = [head, "-" * len(head)]
    for kind, found in spreads.items():
        lines.append(f"{kind:<13}{len(found.rounds):>7}{found.mean:>7.2f}{found.low:>7.2f}{found.high:>7.2f}")
    return "\n".join(lines)
