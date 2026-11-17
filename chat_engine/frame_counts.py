"""Counts by group and the figures of one conversation, over the rows of an export."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .frame_load import Rows
from .ports import CountRow

GROUPS = ("day", "month", "year", "weekday_hour", "conversation", "sender")
_SECONDS_IN_DAY = 86_400
_THURSDAY = 3  # 1970-01-01, with Monday as 0
_TIME_LABELS: dict[str, tuple[int, Callable[[int], str]]] = {
    "day": (1, lambda key: f"{key // 10000:04d}-{key // 100 % 100:02d}-{key % 100:02d}"),
    "month": (100, lambda key: f"{key // 100:04d}-{key % 100:02d}"),
    "year": (10000, lambda key: f"{key:04d}"),
}


def day_label(day: int) -> str:
    """The number YYYYMMDD as "YYYY-MM-DD"."""
    return _TIME_LABELS["day"][1](day)


def grouped_counts(
    rows: Rows, positions: np.ndarray, group_by: str, sender_codes: np.ndarray, sender_ids: Any
) -> list[CountRow]:
    """One row for each group that has a message. Time keys ascend; the others have the largest total first."""
    label: Callable[[int], str]
    if group_by == "conversation":
        keys, label = rows.codes[positions], lambda key: rows.conversations[key]
    elif group_by == "sender":
        keys, label = sender_codes[positions], lambda key: str(sender_ids[key])
    elif group_by == "weekday_hour":
        seconds = rows.seconds[positions]
        weekday = (seconds // _SECONDS_IN_DAY + _THURSDAY) % 7
        keys, label = weekday * 100 + seconds % _SECONDS_IN_DAY // 3600, lambda key: f"{key // 100}-{key % 100:02d}"
    else:
        divisor, label = _TIME_LABELS[group_by]
        keys = rows.days[positions] // divisor
    found, inverse = np.unique(keys, return_inverse=True)
    totals = np.bincount(inverse, minlength=len(found))
    sent = np.bincount(inverse, weights=rows.from_me[positions], minlength=len(found))
    table: list[CountRow] = [
        {"key": label(int(key)), "total": int(total), "sent": int(mine), "received": int(total) - int(mine)}
        for key, total, mine in zip(found, totals, sent, strict=True)
    ]
    if group_by in ("conversation", "sender"):
        table.sort(key=lambda row: (-row["total"], row["key"]))
    return table


def conversation_stats(rows: Rows, positions: np.ndarray, conversation_id: str) -> dict[str, Any]:
    """The figures of the archive contract. On a tie: the earliest busiest day, the earliest longest gap."""
    days, per_day = np.unique(rows.days[positions], return_counts=True)
    busiest = int(np.argmax(per_day)) if len(days) else -1
    gaps = np.diff(rows.seconds[positions])
    widest = int(np.argmax(gaps)) if len(gaps) and gaps.max() > 0 else -1
    sent = int(rows.from_me[positions].sum())
    total = len(positions)
    return {
        "conversation": conversation_id,
        "total_messages": total,
        "sent_count": sent,
        "received_count": total - sent,
        "first_timestamp": str(rows.timestamps[positions[0]]) if total else None,
        "last_timestamp": str(rows.timestamps[positions[-1]]) if total else None,
        "active_days": len(days),
        "avg_messages_per_day": total / len(days) if len(days) else 0.0,
        "busiest_day": day_label(int(days[busiest])) if busiest >= 0 else None,
        "busiest_day_count": int(per_day[busiest]) if busiest >= 0 else 0,
        "longest_gap_days": float(gaps[widest]) / _SECONDS_IN_DAY if widest >= 0 else 0.0,
        "longest_gap_start": str(rows.timestamps[positions[widest]]) if widest >= 0 else None,
        "longest_gap_end": str(rows.timestamps[positions[widest + 1]]) if widest >= 0 else None,
    }
