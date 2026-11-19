"""A look at the profiles of an archive that reads no message: the state, the fingerprint and the size of each.

For each conversation the keys of its parts come from the plans that are
kept (`people_plans.py`). Only a conversation whose fingerprint changed is
laid out again, the way a digest lays it out (`span_plan`). With no notes
file nothing can be read, and nothing is laid out.

Then one read of the notes, with no mark of use, says which parts are
there. A profile is:
- `read` when every part of every conversation is in the notes. Its
  fingerprint is `profile_fingerprint`, and `input_tokens` is an upper bound
  of its request from the lengths of its notes (`input_bound`);
- `empty` when no conversation has a message with text;
- `not_read` otherwise. With no notes file, a profile that was never laid out
  is `not_read`, also when it has no text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from archive import ArchiveError as ArchiveFault
from recall_services.people_inputs import EMPTY, NOT_READ, READ, Planner, Subject
from recall_services.people_plans import (
    NO_TEXT,
    PARTS,
    TOO_LARGE,
    ConversationPlan,
    PlanStore,
    conversation_fingerprint,
    input_bound,
    profile_fingerprint,
)
from recall_services.people_words import Teller, failure
from report_engine import ConfigError, EmptyConversationError, ReportEngineError, ReportOptions, StoreError
from report_engine.chat_digest_span import span_plan
from report_engine.digest import DigestError, PeriodDigest, parse_digest


@dataclass(frozen=True)
class Seen:
    subject: Subject
    state: str
    fingerprint: str | None = None
    input_tokens: int = 0


def _laid_out(planner: Planner, conversation: str, fingerprint: str) -> ConversationPlan:
    """The keys of the parts of one conversation, as a digest of all its days has them."""
    reading = planner.reading(conversation)
    try:
        span = span_plan(
            reading["source"], conversation, None, None, reading["labeler"], reading["model_info"],
            reading["options"] or ReportOptions(), reading["counter"],
        )  # fmt: skip
    except EmptyConversationError:
        return ConversationPlan(fingerprint, NO_TEXT)
    except ConfigError:
        return ConversationPlan(fingerprint, TOO_LARGE)
    except (ReportEngineError, ArchiveFault) as error:
        raise failure("archive_error") from error
    return ConversationPlan(fingerprint, PARTS, tuple(planned.key for planned in span.whole.periods))


def _parts(planner: Planner, keys: Sequence[str]) -> dict[str, PeriodDigest]:
    """The parts that the notes hold and that a digest would use, by key."""
    if planner.notes is None or not keys:
        return {}
    try:
        entries = planner.notes.get_many(keys)
    except StoreError as error:
        raise failure("cache_unreadable") from error
    found: dict[str, PeriodDigest] = {}
    for key, entry in entries.items():
        try:
            found[key] = parse_digest(entry.payload)
        except DigestError:
            continue  # a digest reads that part again
    return found


def _seen(
    planner: Planner, subject: Subject, chosen: Mapping[str, ConversationPlan | None], parts: Mapping[str, PeriodDigest]
) -> Seen:
    found = [chosen[conversation] for conversation in subject.conversations]
    if all(plan is not None and plan.state == NO_TEXT for plan in found):
        return Seen(subject, EMPTY)
    keys = [key for plan in found if plan is not None for key in plan.keys]
    if any(plan is None or plan.state == TOO_LARGE for plan in found) or not all(key in parts for key in keys):
        return Seen(subject, NOT_READ)
    records = [planner.records[conversation] for conversation in subject.conversations]
    fingerprint = profile_fingerprint(planner.info.id, subject.label, subject.is_group, records, keys)
    return Seen(subject, READ, fingerprint, input_bound(subject.label, [parts[key] for key in keys]))


def look_cheaply(planner: Planner, plans: PlanStore, teller: Teller | None = None) -> tuple[Seen, ...]:
    """The state of every profile. It lays out only the conversations that changed, and keeps their plans."""
    subjects = planner.subjects()
    reading = planner.reading_fingerprint()
    kept = plans.read()
    fresh: dict[str, ConversationPlan] = {}
    chosen: dict[str, ConversationPlan | None] = {}
    for place, subject in enumerate(subjects, start=1):
        for conversation in subject.conversations:
            fingerprint = conversation_fingerprint(planner.records[conversation], reading)
            held = kept.get(conversation)
            if held is not None and held.fingerprint == fingerprint:
                chosen[conversation] = held
            elif planner.notes is None:
                chosen[conversation] = None  # nothing can be read: it is not laid out
            else:
                chosen[conversation] = fresh[conversation] = _laid_out(planner, conversation, fingerprint)
        if teller is not None:
            teller.tell(step="look", done=place, total=len(subjects))
    plans.write(fresh)
    parts = _parts(planner, [key for plan in chosen.values() if plan is not None for key in plan.keys])
    return tuple(_seen(planner, subject, chosen, parts) for subject in subjects)
