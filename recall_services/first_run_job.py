"""The `first_run` job: build the steps of the first run that are not ready, in order, with the work of their own jobs.

The server gives the work of each step (`build_meaning_index`,
`prepare_memory` with `MEMORY_MIN_MESSAGES`, `build_people_index` in
`ui_server.py`). Each one takes a listener of updates, as its own job gets
them: `progress`, `log` and `spent`. This module:

- skips a step that is `ready` with one log line. A memory that is read
  already gets its marker (`memory_prep.mark_completed`), so the automatic
  reading after a later import works;
- turns the progress of a step into `{"step", "of": 3, "label", "done",
  "total"}`, with the label of the step's own job;
- adds what each step spent into one `spent` (`{"usd", "usage"}`). The index
  of meanings tells no money: its tokens are priced with the embedding facts;
- lets the failure of a step end the job with that step's sentence and code.
  Steps that ended before stay saved, so a new run starts at the first step
  that is not ready.

The result holds counts and money only.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from meaning_index import EmbedderFacts
from recall_services import meaning, memory_prep, people_words
from recall_services import memory_prep_words as memory_words
from recall_services.first_run import SKIPPED, STEP_NAMES

Listener = Callable[[dict[str, Any]], None]
Work = Callable[[Listener], dict[str, Any]]

# the label of each step before its own job tells one
FIRST_LABELS = {
    "meaning": meaning.STEPS["read"][1],
    "memory": memory_words.STEPS["look"][1],
    "people": people_words.STEPS["look"][1],
}
USAGE_COUNTS = ("input_tokens", "cached_input_tokens", "output_tokens", "cost_usd", "calls")


class Spent:
    """What all steps spent, told to the job as one `spent` after each step that tells its own."""

    def __init__(self, told: Listener) -> None:
        self._told = told
        self.usd = 0.0
        self.usage: dict[str, Any] | None = None

    def tell(self) -> None:
        self._told({"spent": {"usd": self.usd, "usage": None if self.usage is None else dict(self.usage)}})

    def add(self, usd: float, usage: Mapping[str, Any] | None = None) -> None:
        self.usd = float(self.usd + float(usd or 0.0))
        if usage:
            before = self.usage or {}
            self.usage = {name: (before.get(name) or 0) + (usage.get(name) or 0) for name in USAGE_COUNTS}
        self.tell()


class StepTeller:
    """The listener of one step: its progress as a step of the first run, its lines as they are, its spend added."""

    def __init__(self, place: int, of: int, told: Listener, spent: Spent) -> None:
        self._place, self._of, self._told, self._spent = place, of, told, spent

    def started(self, name: str) -> None:
        self._told({"progress": self._progress({"label": FIRST_LABELS[name]})})

    def _progress(self, inner: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "step": self._place,
            "of": self._of,
            "label": str(inner.get("label") or ""),
            "done": int(inner.get("done") or 0),
            "total": int(inner.get("total") or 0),
        }

    def tell(self, update: Mapping[str, Any]) -> None:
        if update.get("progress"):
            self._told({"progress": self._progress(update["progress"])})
        if update.get("log"):
            self._told({"log": update["log"]})
        if update.get("spent"):
            told = update["spent"]
            self._spent.add(float(told.get("usd") or 0.0), told.get("usage"))


def meaning_usd(result: Mapping[str, Any], facts: EmbedderFacts | None = None) -> float:
    """What a build of the index of meanings paid: its tokens at the price of the embedding model."""
    return int(result.get("tokens") or 0) * (facts or meaning.default_facts()).usd_per_mtok / 1_000_000


def run(
    steps: Sequence[Mapping[str, Any]],
    work: Mapping[str, Work],
    told: Listener,
    *,
    archive_file: str | os.PathLike[str],
    facts: EmbedderFacts | None = None,
) -> dict[str, Any]:
    """Build the steps that are not ready, in the order of `STEP_NAMES`. Gives the result of each step
    (None when it was skipped), `skipped` and `usd`. A step that fails raises its own failure."""
    spent = Spent(told)
    spent.tell()
    states = {str(step["name"]): str(step["state"]) for step in steps}
    results: dict[str, Any] = {}
    skipped: list[str] = []
    for place, name in enumerate(STEP_NAMES, start=1):
        if states.get(name) == "ready":
            told({"log": SKIPPED[name]})
            if name == "memory":
                memory_prep.mark_completed(archive_file)
            results[name], skipped = None, [*skipped, name]
            continue
        teller = StepTeller(place, len(STEP_NAMES), told, spent)
        teller.started(name)
        results[name] = work[name](teller.tell)
        if name == "meaning":
            spent.add(meaning_usd(results[name], facts))
    return {**results, "skipped": skipped, "usd": spent.usd}
