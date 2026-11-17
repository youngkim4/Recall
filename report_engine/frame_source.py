"""ReportSource over a pandas DataFrame with today's export columns.

The frame is read, never changed. A conversation matches by its exact
`chat_id`. Rows with no usable timestamp are left out and counted in the log.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterator

import pandas as pd

from report_engine.dates import STAMP_FORMAT
from report_engine.errors import SourceError
from report_engine.ports import SourceMessage
from report_engine.scope import lower_bound, upper_bound

logger = logging.getLogger(__name__)

EXPORT_COLUMNS = ("message_id", "timestamp", "sender", "text", "chat_id", "is_from_me")
_TRUE_WORDS = frozenset({"1", "true", "t", "yes", "y"})


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or (isinstance(value, float) and math.isnan(value))


def _text(value: Any) -> str:
    return "" if _missing(value) else str(value)


def _message_id(value: Any, position: int) -> str:
    if _missing(value):
        return f"row-{position}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _from_me(value: Any) -> bool:
    if _missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_WORDS
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


class FrameSource:
    def __init__(self, frame: pd.DataFrame) -> None:
        absent = [name for name in EXPORT_COLUMNS if name not in frame.columns]
        if absent:
            raise SourceError(f"the messages frame has no column {', '.join(absent)}")
        self._frame = frame

    def _stamps(self, rows: pd.DataFrame) -> pd.Series:
        """Timestamps as fixed-width text; NaN where a row has none."""
        stamps = pd.to_datetime(rows["timestamp"], errors="coerce")
        if getattr(stamps.dt, "tz", None) is not None:
            stamps = stamps.dt.tz_localize(None)
        return stamps.dt.strftime(STAMP_FORMAT)

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        frame = self._frame
        match = (frame["chat_id"].astype("string") == conversation).fillna(False).to_numpy(dtype=bool)
        rows = frame.loc[match, list(EXPORT_COLUMNS)]
        stamps = self._stamps(rows)
        timeless = int(stamps.isna().sum())
        if timeless:
            logger.warning(
                "%d message%s in this conversation had no usable timestamp and %s left out",
                timeless,
                "" if timeless == 1 else "s",
                "was" if timeless == 1 else "were",
            )
        keep = stamps.notna()
        low, high = lower_bound(since), upper_bound(until)
        if low is not None:
            keep &= stamps >= low
        if high is not None:
            keep &= stamps <= high
        order = stamps[keep].sort_values(kind="stable").index
        chosen = rows.loc[order]
        columns = zip(
            chosen["message_id"].tolist(),
            stamps.loc[order].tolist(),
            chosen["is_from_me"].tolist(),
            chosen["sender"].tolist(),
            chosen["text"].tolist(),
        )
        for position, (mid, stamp, own, sender, text) in enumerate(columns):
            yield {
                "id": _message_id(mid, position),
                "timestamp": stamp,
                "is_from_me": _from_me(own),
                "sender": _text(sender),
                "text": _text(text),
            }
