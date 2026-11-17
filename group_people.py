"""The people who wrote in a group chat, busiest first.

One person can write from a phone number and an email address. When both are
saved under one name in Contacts, they are one person here.
"""
from __future__ import annotations

import pandas as pd

# a group chat holds a few dozen people at most; the limit only guards the payload
SPEAKER_LIMIT = 50


def group_speakers(
    senders: pd.Series,
    saved: dict[str, str],
    limit: int = SPEAKER_LIMIT,
) -> tuple[int, list[dict]]:
    """senders holds the handle of every received message. Returns how many
    people wrote, and the busiest of them: the handle each wrote from most,
    their saved name or "", their count, and their share of what was received."""
    handles = senders.astype("string").str.strip()
    handles = handles[handles.notna() & handles.ne("")].astype(str)
    if handles.empty:
        return 0, []

    name = handles.map(lambda handle: str(saved.get(handle) or "").strip())
    person = name.str.lower().where(name.ne(""), handles)
    by_handle = (
        pd.DataFrame({"person": person, "handle": handles, "name": name})
        .groupby(["person", "handle"], as_index=False)
        .agg(count=("handle", "size"), name=("name", "first"))
        # within one person, the handle they wrote from most comes first
        .sort_values(["person", "count", "handle"], ascending=[True, False, True])
    )
    people = (
        by_handle.groupby("person", as_index=False)
        .agg(handle=("handle", "first"), name=("name", "first"), count=("count", "sum"))
        .sort_values(["count", "person"], ascending=[False, True])
    )
    total = int(people["count"].sum())
    listed = [
        {
            "handle": str(entry["handle"]),
            "name": str(entry["name"]),
            "count": int(entry["count"]),
            "share": round(int(entry["count"]) / total, 3),
        }
        for entry in people.head(limit).to_dict("records")
    ]
    return int(len(people)), listed
