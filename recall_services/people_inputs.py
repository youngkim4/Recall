"""Who has a profile, what the model reads of each profile, and the hash of those inputs.

A profile is one of these (`subjects_of`):
- a person: the archive's `person_id` with all of its direct threads (phone
  and email), busiest first. Its key is `person:<person_id>`.
- a group conversation, or a direct conversation with no `person_id`. Its key
  is `conversation:<conversation id>`.
The label is the name chat gives the busiest conversation: a saved name, a
group title or name, or a mask (`names.chat_labeler`).

The notes of a profile are the digests of all its conversations, read from
the notes of the reports with the model of the memory (`Planner.plan`). A
read makes no model call and changes no entry of the notes: it is a digest
with no new part allowed (`max_new_periods=0`) over a cache that does not
mark what it reads. The digests are wired as `digests.build` wires them
(`digests._reading`), so the keys of their parts are the keys of the memory.

- `read`: every part of every conversation is in the notes.
- `not_read`: a part is not in the notes. The profile is left out of a build.
- `empty`: no conversation has a message with text. The profile has no notes,
  and it is left out and not counted.

The hash of the inputs is a hash of the prompt version, the model and the
request itself, which holds the label, the figures and the notes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from archive import Archive
from archive import ArchiveError as ArchiveFault
from recall_services import digests, models, people_prompt, reports
from recall_services.names import Names
from recall_services.people_figures import Figures, read_figures
from recall_services.people_words import failure
from report_engine import ConfigError, EmptyConversationError, PeriodCache, ReportEngineError, ReportOptions, StoreError
from report_engine.cache import CacheEntry
from report_engine.chat_digest import DigestTooLarge, PeriodNote, conversation_digest
from report_engine.ports import ChatMessage, LLMResult
from report_engine.tokens import TokenCounter

READ, NOT_READ, EMPTY = "read", "not_read", "empty"


@dataclass(frozen=True)
class Subject:
    key: str
    conversations: tuple[str, ...]
    is_group: bool
    label: str


@dataclass(frozen=True)
class Notes:
    state: str
    periods: tuple[PeriodNote, ...] = ()


@dataclass(frozen=True)
class Planned:
    """One profile as a build finds it. `figures`, `messages` and `input_hash` are set only when it is read."""

    subject: Subject
    notes: Notes
    figures: Figures | None = None
    messages: tuple[ChatMessage, ...] = ()
    input_hash: str | None = None


def subjects_of(archive: Any, names: Names) -> tuple[Subject, ...]:
    """The profiles of the archive, the busiest first. Raises ArchiveError."""
    listed = archive.conversations()
    labeler = names.chat_labeler(listed)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in listed:
        person = record.get("person_id")
        own = person is not None and not record.get("is_group")
        grouped.setdefault(f"person:{person}" if own else f"conversation:{record['id']}", []).append(record)
    return tuple(
        Subject(
            key=key,
            conversations=tuple(str(record["id"]) for record in records),
            is_group=bool(records[0].get("is_group")),
            label=labeler.conversation(records[0]),
        )
        for key, records in grouped.items()
    )


class _Unmarked:
    """The notes, read only: what is read is not marked as used, and nothing is put."""

    def __init__(self, cache: PeriodCache) -> None:
        self._cache = cache

    def get_many(self, keys: Sequence[str], touch: bool = False) -> Mapping[str, CacheEntry]:
        return self._cache.get_many(keys, touch=False)

    def put(self, entry: CacheEntry, **_where: Any) -> None:
        raise StoreError("a reading of the notes for the people index writes nothing")


class _NoModel:
    """The model port of a reading of the notes. A digest with no new part never calls it."""

    def generate(self, *_messages: Any, **_settings: Any) -> LLMResult:
        raise RuntimeError("a reading of the notes makes no model call")


def _notes_file(cache_path: Path) -> _Unmarked | None:
    if not cache_path.exists():
        return None
    try:
        return _Unmarked(PeriodCache(cache_path))
    except StoreError as error:
        raise failure("cache_unreadable") from error


def _together(found: Sequence[Notes]) -> Notes:
    if any(notes.state == NOT_READ for notes in found):
        return Notes(NOT_READ)
    periods = sorted((period for notes in found for period in notes.periods), key=lambda p: (p.first_day, p.last_day))
    return Notes(READ, tuple(periods)) if periods else Notes(EMPTY)


class Planner:
    """Plans the profiles of one archive with the notes of one model. Raises PeopleFailure and ModelChoiceError."""

    def __init__(
        self,
        archive: Archive,
        names: Names,
        *,
        model: str,
        places: reports.ReportPlaces | None = None,
        options: ReportOptions | None = None,
        counter: TokenCounter | None = None,
        **_more: Any,
    ) -> None:
        self._archive = archive
        self._names = names
        self.info = models.model_info(model)
        try:
            self._listed = archive.conversations()
        except ArchiveFault as error:
            raise failure("archive_error") from error
        self._records = {str(record["id"]): record for record in self._listed}
        self.redact = names.chat_labeler(self._listed).redact
        self._options = options
        self._counter = counter
        self._notes = _notes_file((places or reports.default_places()).cache_path)

    def subjects(self) -> tuple[Subject, ...]:
        return subjects_of(_Listed(self._listed), self._names)

    def _conversation(self, conversation: str) -> Notes:
        # with no notes file each part is not read, and a conversation with no text is still empty
        record =self._records.get(conversation) or {"id": conversation, "is_group": False}
        asked = reports._Asked(conversation, None, None, self.info, self._names.report_labeler(record), self.redact)
        try:
            found = conversation_digest(
                llm=_NoModel(),
                cache=self._notes,
                max_new_periods=0,
                **digests._reading(self._archive, asked, self._options, self._counter),
            )
        except (DigestTooLarge, ConfigError):
            return Notes(NOT_READ)
        except EmptyConversationError:
            return Notes(EMPTY)
        except (ReportEngineError, ArchiveFault) as error:
            raise failure("archive_error") from error
        return Notes(READ, found.periods) if found.periods_failed == 0 else Notes(NOT_READ)

    def plan(self, subject: Subject) -> Planned:
        """What the model would read of this profile, and the hash of it."""
        notes = _together([self._conversation(conversation) for conversation in subject.conversations])
        if notes.state != READ:
            return Planned(subject, notes)
        try:
            figures = read_figures(self._archive, subject.conversations)
        except ArchiveFault as error:
            raise failure("archive_error") from error
        person = people_prompt.PromptInput(subject.label, subject.is_group, figures, notes.periods)
        messages = tuple(people_prompt.profile_messages(person, self.redact))
        return Planned(subject, notes, figures, messages, input_hash(self.info.id, messages))


class _Listed:
    """The conversations of the archive as they were read once."""

    def __init__(self, listed: Sequence[Mapping[str, Any]]) -> None:
        self._listed = listed

    def conversations(self, **_more: Any) -> Sequence[Mapping[str, Any]]:
        return self._listed


def input_hash(model: str, messages: Sequence[ChatMessage]) -> str:
    material = json.dumps([people_prompt.PROMPT_VERSION, model, list(messages)], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()
