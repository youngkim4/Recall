"""Decide which handles are one person.

Input is the Contacts cache written by scripts/export_contacts.swift: a list
of records `{name, phones[], emails[]}`. The AddressBook database is never read.

Rules:
- Handles that match the same record are one person.
- Two records with the same name are two people, unless they also share a
  phone number or an email address: then they are two cards of one person.
- A handle that two different names share belongs to the first record.
- A handle with no record is its own person with an empty name.
- A person keeps its id while its handles stay together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from archive.handles import email_key, phone_key


class HandleRow(NamedTuple):
    id: int
    match_key: str
    person_id: int | None


@dataclass(frozen=True)
class ContactGroup:
    """One person in the address book: a name and every key that reaches them."""

    name: str
    keys: frozenset[str]


@dataclass(frozen=True)
class PersonPlan:
    """What the archive must hold for one person after an import."""

    person_id: int | None  # None: make a new person
    name: str | None  # None: keep the stored name
    handle_ids: tuple[int, ...]


def _record_keys(record: Mapping[str, object]) -> frozenset[str]:
    keys: set[str] = set()
    for field, to_key in (("phones", phone_key), ("emails", email_key)):
        values = record.get(field)
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            key = to_key(value) if isinstance(value, str) else None
            if key is not None:
                keys.add(key)
    return frozenset(keys)


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def contact_groups(contacts: Sequence[Mapping[str, object]]) -> tuple[tuple[ContactGroup, ...], int]:
    """Turn contact records into groups. Return (groups, number of records that are not records)."""
    if isinstance(contacts, (str, bytes, Mapping)) or not isinstance(contacts, Sequence):
        raise TypeError("contacts must be a list of {name, phones, emails} records")
    usable: list[tuple[str, frozenset[str]]] = []
    not_records = 0
    for record in contacts:
        if not isinstance(record, Mapping):
            not_records += 1
            continue
        keys = _record_keys(record)
        if keys:
            usable.append((" ".join(str(record.get("name") or "").split()), keys))

    parents = list(range(len(usable)))
    first_with: dict[tuple[str, str], int] = {}
    for index, (name, keys) in enumerate(usable):
        for key in keys:
            earlier = first_with.setdefault((name.casefold(), key), index)
            low, high = sorted((_find(parents, index), _find(parents, earlier)))
            parents[high] = low  # the earliest record names the group

    merged: dict[int, set[str]] = {}
    for index, (_, keys) in enumerate(usable):
        merged.setdefault(_find(parents, index), set()).update(keys)
    groups = tuple(ContactGroup(usable[root][0], frozenset(keys)) for root, keys in sorted(merged.items()))
    return groups, not_records


def _wanted_people(handles: Sequence[HandleRow], groups: Sequence[ContactGroup]) -> list[tuple[str, tuple[int, ...]]]:
    group_of_key: dict[str, int] = {}
    for index, group in enumerate(groups):
        for key in group.keys:
            group_of_key.setdefault(key, index)
    matched: dict[int, list[int]] = {}
    wanted: list[tuple[str, tuple[int, ...]]] = []
    for row in handles:
        index = group_of_key.get(row.match_key)
        if index is None:
            wanted.append(("", (row.id,)))
        else:
            matched.setdefault(index, []).append(row.id)
    wanted.extend((groups[index].name, tuple(sorted(ids))) for index, ids in matched.items())
    return wanted


def _reuse_ids(wanted: list[tuple[str, tuple[int, ...]]], owner: Mapping[int, int | None]) -> tuple[PersonPlan, ...]:
    """Give each wanted person the stored id that already holds most of its handles."""
    claimed: set[int] = set()
    plans: list[PersonPlan] = []
    for name, handle_ids in sorted(wanted, key=lambda item: (-len(item[1]), item[1][0])):
        votes: dict[int, int] = {}
        for handle_id in handle_ids:
            person_id = owner.get(handle_id)
            if person_id is not None:
                votes[person_id] = votes.get(person_id, 0) + 1
        free = [person_id for person_id in sorted(votes, key=lambda pid: (-votes[pid], pid)) if person_id not in claimed]
        chosen = free[0] if free else None
        if chosen is not None:
            claimed.add(chosen)
        plans.append(PersonPlan(person_id=chosen, name=name, handle_ids=handle_ids))
    return tuple(sorted(plans, key=lambda plan: plan.handle_ids[0]))


def _keep_people(handles: Sequence[HandleRow]) -> tuple[PersonPlan, ...]:
    kept: dict[int, list[int]] = {}
    plans: list[PersonPlan] = []
    for row in handles:
        if row.person_id is None:
            plans.append(PersonPlan(person_id=None, name="", handle_ids=(row.id,)))
        else:
            kept.setdefault(row.person_id, []).append(row.id)
    plans.extend(PersonPlan(person_id=pid, name=None, handle_ids=tuple(sorted(ids))) for pid, ids in kept.items())
    return tuple(sorted(plans, key=lambda plan: plan.handle_ids[0]))


def plan_people(handles: Sequence[HandleRow], groups: Sequence[ContactGroup] | None) -> tuple[PersonPlan, ...]:
    """Every handle goes into exactly one plan.

    `groups` is None when the import has no contact list: stored people stay
    as they are and only new handles get a person.
    """
    if groups is None:
        return _keep_people(handles)
    owner = {row.id: row.person_id for row in handles}
    return _reuse_ids(_wanted_people(handles, groups), owner)
