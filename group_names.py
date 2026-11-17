"""Names for group chats that were never given one.

Messages shows an unnamed group as the people in it. The export only knows who
has written in a group, so the name is built from them, busiest first:
"Avery Stone, Blake Reyes and 2 others".
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from conversation import received_mask

NameLookup = Callable[[list[str]], dict[str, str]]

# a group this small is named in full; a larger one names its two busiest people
NAMED_IN_FULL = 3
NAMED_WHEN_LARGER = 2


def join_names(names: list[str]) -> str:
    """["Avery", "Blake", "Casey"] -> "Avery, Blake and Casey"."""
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _group_name(people: list[str | None]) -> str:
    """people holds one entry per person, busiest first: their saved name, or
    None for someone who is not in Contacts."""
    saved = [name for name in people if name]
    if not saved:
        return ""
    everyone_is_saved = len(saved) == len(people)
    shown = saved if everyone_is_saved and len(people) <= NAMED_IN_FULL else saved[:NAMED_WHEN_LARGER]
    rest = len(people) - len(shown)
    if len(people) == 1:
        # a group always holds more people than its one voice
        return f"{shown[0]} and others"
    if rest == 0:
        return join_names(shown)
    return f"{', '.join(shown)} and {rest} other{'' if rest == 1 else 's'}"


def derive_group_names(
    df: pd.DataFrame,
    group_ids: Iterable[str],
    lookup: NameLookup,
) -> dict[str, str]:
    """Names for the given group chats, keyed by chat id. A group in which
    nobody saved in Contacts has written gets no entry."""
    wanted = {str(group_id) for group_id in group_ids}
    if not wanted or df.empty or not {"chat_id", "sender", "is_from_me"} <= set(df.columns):
        return {}

    # a row you sent carries the handle it went to, so only received rows say who is in the group
    rows = df.loc[df["chat_id"].astype("string").isin(wanted), ["chat_id", "sender", "is_from_me"]]
    rows = rows[received_mask(rows["is_from_me"])]
    rows = rows.assign(
        chat_id=rows["chat_id"].astype(str),
        sender=rows["sender"].astype("string").str.strip(),
    )
    rows = rows[rows["sender"].notna() & rows["sender"].ne("")]
    if rows.empty:
        return {}

    saved = lookup(sorted(rows["sender"].unique()))
    name = rows["sender"].map(lambda sender: str(saved.get(sender) or "").strip())
    # one person can write from a phone number and an email address
    person = name.str.lower().where(name.ne(""), rows["sender"])
    people = (
        rows.assign(name=name, person=person)
        .groupby(["chat_id", "person"], as_index=False)
        .agg(messages=("person", "size"), name=("name", "first"))
        .sort_values(["chat_id", "messages", "person"], ascending=[True, False, True])
    )

    named = {
        str(chat_id): _group_name([value or None for value in members["name"]])
        for chat_id, members in people.groupby("chat_id", sort=False)
    }
    return {chat_id: label for chat_id, label in named.items() if label}
