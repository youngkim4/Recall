"""The shapes of a question of the eval. See `evals.questions` for the kinds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.world import World

BASE_KINDS: tuple[str, ...] = (
    "fact", "first_last", "count", "pattern", "relationship", "multi_hop", "absent", "follow_up",
)
# the kinds that only the hard world asks (`evals.hard_questions`)
HARD_KINDS: tuple[str, ...] = ("paraphrase", "disambiguate", "temporal", "aggregate", "long_story", "needle")
KINDS: tuple[str, ...] = (*BASE_KINDS, *HARD_KINDS)


@dataclass(frozen=True)
class CountQuery:
    """The same count as the archive takes it: `ArchiveAdapter.count(...)`. A test checks the two agree."""

    conversations: tuple[str, ...] | None = None
    from_me: bool | None = None
    since: str | None = None
    until: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class Spec:
    """A question written by hand. `labels` name the messages that support it."""

    id: str
    kind: str
    text: str
    gold: str
    key_facts: tuple[str, ...]
    labels: tuple[str, ...] = ()
    prior: str = ""
    absent_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    text: str
    gold: str
    key_facts: tuple[str, ...]
    gold_guids: tuple[str, ...]
    prior: str = ""  # follow_up: the question asked before this one
    contact: str = ""  # a conversation id; "" asks of the whole archive
    expected_count: int | None = None
    count_query: CountQuery | None = None
    absent_terms: tuple[str, ...] = ()  # absent: words that no message holds


def from_spec(made: World, spec: Spec) -> Question:
    """The question of a hand-written spec, with the guids of its labels in `made`."""
    return Question(
        id=spec.id, kind=spec.kind, text=spec.text, gold=spec.gold, key_facts=spec.key_facts,
        gold_guids=made.guids(spec.labels), prior=spec.prior, absent_terms=spec.absent_terms,
    )  # fmt: skip
