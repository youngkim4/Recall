"""A build of the people index: look, write the profiles that changed, embed, save.

1. Look (`look`): plan every profile of the archive, and read the store.
   A profile is written when its notes are read and the store holds no row
   with the same hash of its inputs. A profile whose notes are not all read
   is left out; its row from an earlier build stays as it was.
2. Write: one call of the model for each profile to write (`people_calls.py`).
3. Embed the text of each row that has no vector for the embedding model and
   its dimensions, `EMBED_BATCH` texts in one request.
4. Save the whole store at once (`people_store.py`). A row of a profile that
   the archive no longer has, or that has no text now, is removed.

A build that stops (a failure that each profile would have, or a cancel)
embeds nothing more, saves the profiles that were paid for with no vector,
and raises. The next build embeds them and does not pay for them again.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np

from archive import Archive
from meaning_index import EmbedderFacts
from recall_services.meaning_failures import failure_of as embedding_failure
from recall_services.names import Names
from recall_services.people_calls import Written
from recall_services.people_inputs import EMPTY, READ, Planned, Planner
from recall_services.people_store import (
    Held,
    PeopleFormatError,
    PeopleStore,
    PeopleStoreError,
    Profile,
    profile_text,
    text_hash,
)
from recall_services.people_words import Teller, failure
from report_engine import Usage

EMBED_BATCH = 96
EMBED_FAILED = "embed_failed"
Embedded = tuple[tuple[Profile, ...], int, int, str | None]


@dataclass(frozen=True)
class Look:
    """What a build finds before any call."""

    planner: Planner
    planned: tuple[Planned, ...]
    held: Held | None

    @property
    def rows(self) -> dict[str, Profile]:
        return {} if self.held is None else {profile.key: profile for profile in self.held.profiles}

    @property
    def to_write(self) -> tuple[Planned, ...]:
        rows = self.rows
        return tuple(one for one in self.planned if one.notes.state == READ and not _current(one, rows.get(one.subject.key)))

    @property
    def people(self) -> int:
        return sum(one.notes.state != EMPTY for one in self.planned)

    @property
    def not_read(self) -> int:
        return sum(one.notes.state not in (READ, EMPTY) for one in self.planned)


def _current(planned: Planned, row: Profile | None) -> bool:
    return row is not None and row.input_hash == planned.input_hash


def read_store(store: PeopleStore) -> Held | None:
    """What the store holds. Raises PeopleFailure (`store_newer`, `store_error`)."""
    try:
        return store.read()
    except PeopleFormatError as error:
        raise failure("store_newer") from error
    except PeopleStoreError as error:
        raise failure("store_error") from error


def look(
    archive: Archive, names: Names, store: PeopleStore, teller: Teller | None = None, *, model: str, **settings: Any
) -> Look:
    """Plan each profile and read the store. Makes no model call and writes nothing."""
    held = read_store(store)
    planner = Planner(archive, names, model=model, **settings)
    subjects = planner.subjects()
    planned: list[Planned] = []
    for place, subject in enumerate(subjects, start=1):
        planned.append(planner.plan(subject))
        if teller is not None:
            teller.tell(step="look", done=place, total=len(subjects))
    return Look(planner, tuple(planned), held)


def _written(one: Written) -> Profile:
    subject, text, figures = one.planned.subject, one.text, one.planned.figures
    assert text is not None and figures is not None and one.planned.input_hash is not None
    return Profile(
        key=subject.key,
        conversations=subject.conversations,
        label=subject.label,
        is_group=subject.is_group,
        figures=figures,
        kind=text.kind,
        status=text.status,
        summary=text.summary,
        turning_points=text.turning_points,
        input_hash=one.planned.input_hash,
        text_hash="",
    )


def assemble(seen: Look, written: Sequence[Written]) -> tuple[Profile, ...]:
    """The rows of the new store, in the order of the profiles: written now, kept, or kept as they were."""
    rows = seen.rows
    new = {one.planned.subject.key: _written(one) for one in written if one.text is not None}
    kept: list[Profile] = []
    for planned in seen.planned:
        subject, old = planned.subject, rows.get(planned.subject.key)
        if subject.key in new:
            kept.append(new[subject.key])
        elif planned.notes.state == READ and old is not None and _current(planned, old):
            kept.append(replace(old, conversations=subject.conversations, label=subject.label))
        elif planned.notes.state != EMPTY and old is not None:
            kept.append(old)  # not read, failed, or not reached: as it was
    return tuple(kept)


def _vector(values: Sequence[float]) -> bytes:
    found = np.asarray(values, dtype="<f4")
    length = float(np.linalg.norm(found))
    return (found / length if length > 0 else found).astype("<f4").tobytes()


def needs_vector(profile: Profile, facts: EmbedderFacts) -> bool:
    return profile.vector is None or profile.text_hash != text_hash(profile_text(profile), facts.model, facts.dimensions)


def embed(
    profiles: Sequence[Profile], facts: EmbedderFacts, factory: Callable[[str, int], Any], count_tokens: Callable[[str], int]
) -> Embedded:
    """Each row with a vector for these facts. Gives the rows, the count embedded, the tokens and a failure code or None."""
    done = list(profiles)
    waiting = [place for place, profile in enumerate(done) if needs_vector(profile, facts)]
    embedded = tokens = 0
    try:
        embedder = factory(facts.model, facts.dimensions) if waiting else None
        for start in range(0, len(waiting), EMBED_BATCH):
            places = waiting[start : start + EMBED_BATCH]
            texts = [profile_text(done[place]) for place in places]
            vectors = embedder.embed(texts)  # type: ignore[union-attr]
            for place, text, values in zip(places, texts, vectors, strict=True):
                stamp = text_hash(text, facts.model, facts.dimensions)
                done[place] = replace(done[place], text_hash=stamp, vector=_vector(values))
            embedded += len(places)
            tokens += sum(count_tokens(text) for text in texts)
    except Exception as error:  # noqa: BLE001 - a failure of the provider or of the factory: told in a code
        code = embedding_failure(error).code
        return tuple(done), embedded, tokens, EMBED_FAILED if code in ("model_error", "build_failed") else code
    return tuple(done), embedded, tokens, None


def usage_of(written: Sequence[Written]) -> Usage:
    total = Usage()
    for one in written:
        total = total.plus(one.usage)
    return total


def save(store: PeopleStore, held: Held, usage: Usage) -> None:
    try:
        store.write(held)
    except PeopleStoreError as error:
        raise failure("store_error", usage.model_dump(mode="json")) from error


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Progress:
    """The count of the profiles that are written, for the progress of the job, under one lock."""

    def __init__(self, teller: Teller, total: int) -> None:
        self._teller = teller
        self._total = total
        self._done = 0
        self._lock = threading.Lock()

    def heard(self) -> None:
        with self._lock:
            self._done += 1
            self._teller.tell(step="write", done=self._done, total=self._total)
