"""ReportSource over a pandas DataFrame with today's export columns.

The frame is read, never changed. Rows are taken by position, so a frame
whose index repeats a label is read correctly. A conversation matches by its
exact `chat_id`. Rows with no usable timestamp are left out and counted in
the log.

Timestamps with no zone are taken as local time, as the export writes them.
Timestamps with a zone are converted to the local time of this machine.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from report_engine.dates import STAMP_FORMAT, lower_bound, upper_bound
from report_engine.errors import SourceError
from report_engine.ports import SourceMessage

logger = logging.getLogger(__name__)

EXPORT_COLUMNS = ("message_id", "timestamp", "sender", "text", "chat_id", "is_from_me")
ISO_FORMAT = "ISO8601"
_TRUE_WORDS = frozenset({"1", "true", "t", "yes", "y"})


def _missing(value: Any) -> bool:
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    )


def _text(value: Any) -> str:
    return "" if _missing(value) else str(value)


def _whole_number_text(value: Any) -> str:
    """5.0 reads as "5": a column of ids turns into floats when one id is missing."""
    if _missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _message_id(value: Any, position: int) -> str:
    """The id as text. A row with no id is named by its place in the frame, which
    does not change with the scope."""
    return _whole_number_text(value) or f"row-{position}"


def _from_me(value: Any) -> bool:
    if _missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_WORDS
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _same_chat(column: pd.Series, conversation: str) -> npt.NDArray[np.bool_]:
    """True at each position whose chat id is exactly `conversation`."""
    if pd.api.types.is_float_dtype(column):
        text = column.map(_whole_number_text).astype("string")
    else:
        text = column.astype("string")
    return (text == conversation).fillna(False).to_numpy(dtype=bool)


def _parse(column: pd.Series) -> pd.Series:
    """Text timestamps as datetimes; NaT where a value does not parse."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            moments = pd.to_datetime(column, errors="coerce", format=ISO_FORMAT)
    except ValueError:
        moments = None
    if moments is None or not pd.api.types.is_datetime64_any_dtype(moments):
        # values with different UTC offsets: read them all as instants
        moments = pd.to_datetime(column, errors="coerce", format=ISO_FORMAT, utc=True)
    return moments


def _to_local(moments: pd.Series) -> pd.Series:
    """Instants as wall-clock time on this machine, with daylight saving applied per value."""
    local = [
        pd.NaT if pd.isna(moment) else datetime.fromtimestamp(moment.timestamp())
        for moment in moments.dt.tz_convert("UTC")
    ]
    return pd.Series(local, index=moments.index, dtype="datetime64[ns]")


def _local_moments(column: pd.Series) -> pd.Series:
    moments = column if pd.api.types.is_datetime64_any_dtype(column) else _parse(column)
    return _to_local(moments) if moments.dt.tz is not None else moments


def _log_timeless(count: int) -> None:
    if count:
        logger.warning(
            "%d message%s in this conversation had no usable timestamp and %s left out",
            count,
            "" if count == 1 else "s",
            "was" if count == 1 else "were",
        )


class FrameSource:
    def __init__(self, frame: pd.DataFrame) -> None:
        absent = [name for name in EXPORT_COLUMNS if name not in frame.columns]
        if absent:
            raise SourceError(f"the messages frame has no column {', '.join(absent)}")
        self._frame = frame

    def messages(
        self, conversation: str, since: str | None, until: str | None
    ) -> Iterator[SourceMessage]:
        positions = np.flatnonzero(_same_chat(self._frame["chat_id"], conversation))
        rows = self._frame.iloc[positions][list(EXPORT_COLUMNS)].reset_index(drop=True)
        try:
            moments = _local_moments(rows["timestamp"])
        except (TypeError, ValueError, AttributeError) as exc:
            raise SourceError(
                f"the timestamp column can not be read: {type(exc).__name__}"
            ) from exc
        stamps = moments.dt.strftime(STAMP_FORMAT)
        _log_timeless(int(moments.isna().sum()))

        keep = moments.notna()
        low, high = lower_bound(since), upper_bound(until)
        if low is not None:
            keep &= stamps >= low
        if high is not None:
            keep &= stamps <= high
        order = moments[keep].sort_values(kind="stable").index
        chosen = rows.loc[order]
        columns = zip(
            [int(position) for position in positions[order]],
            chosen["message_id"].tolist(),
            stamps.loc[order].tolist(),
            chosen["is_from_me"].tolist(),
            chosen["sender"].tolist(),
            chosen["text"].tolist(),
            strict=True,
        )
        for position, mid, stamp, own, sender, text in columns:
            yield {
                "id": _message_id(mid, position),
                "timestamp": stamp,
                "is_from_me": _from_me(own),
                "sender": _text(sender),
                "text": _text(text),
            }
