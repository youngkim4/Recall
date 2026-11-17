"""Read the rows of a message export into arrays.

The export has one row for each message: `message_id`, `timestamp` (local
time, "YYYY-MM-DD HH:MM:SS"), `sender`, `text`, `chat_id`, `is_from_me` and,
when present, `chat_display_name` and `attachment_types` (mime types joined
with "|").

Three kinds of rows are not messages and are left out:
- a reaction phrase: the export writes a tapback as text, such as
  `Loved "the text"`. The real archive keeps a reaction apart from the
  messages, so this reader does the same.
- a row with no id, no conversation or no real time.
- a second row with the id of an earlier row.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np
import pandas as pd

REQUIRED = ("message_id", "timestamp", "sender", "text", "chat_id", "is_from_me")
OPTIONAL = ("chat_display_name", "attachment_types")
PLACEHOLDER = "￼"  # the character Messages puts where an attachment is
_ATTACHMENT_PREFIX = "[Attachment]"
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_TIME_RE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
_REACTION_RE = r'(?:Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) [“"].*[”"]\s*'
_TRUE = ("1", "1.0", "True", "true")
_GROUP_ID_RE = re.compile(r"chat\d+")


@dataclass(frozen=True)
class Rows:
    """The messages of an export, oldest first. Every array has one entry for each message."""

    ids: np.ndarray  # str
    timestamps: np.ndarray  # str, local time
    seconds: np.ndarray  # int64: the local time read as if it were UTC
    days: np.ndarray  # int32: YYYYMMDD
    codes: np.ndarray  # int32: index into `conversations`
    conversations: tuple[str, ...]
    from_me: np.ndarray  # bool
    senders: np.ndarray  # str, "" for own messages
    texts: np.ndarray  # str, "" for a message that is only an attachment
    is_attachment: np.ndarray  # bool
    attachment_types: np.ndarray  # str: mime types joined with "|"
    display_names: np.ndarray  # str: the group title on the row, or ""
    dropped: Mapping[str, int]

    def __len__(self) -> int:
        return len(self.ids)


def is_group_id(conversation_id: str) -> bool:
    """A group id is "chat" followed by digits only."""
    return _GROUP_ID_RE.fullmatch(conversation_id) is not None


def _as_text(series: pd.Series) -> pd.Series:
    """A column as text: no "nan", and 12.0 reads "12"."""
    if pd.api.types.is_float_dtype(series):
        with contextlib.suppress(TypeError, ValueError):  # not whole numbers: the column stays as it is
            series = series.astype("Int64")
    return series.astype(object).where(series.notna(), "").astype(str)


def _table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"the export has no column named: {', '.join(missing)}")
    columns = {name: _as_text(frame[name]) for name in REQUIRED}
    for name in OPTIONAL:
        columns[name] = _as_text(frame[name]) if name in frame.columns else pd.Series("", index=frame.index)
    return pd.DataFrame(columns).reset_index(drop=True)


def _readable(table: pd.DataFrame) -> pd.Series:
    """Rows with an id, a conversation and a real time in the form of the export."""
    moments = pd.to_datetime(table["timestamp"], format=_TIME_FORMAT, errors="coerce")
    well_formed = table["timestamp"].str.fullmatch(_TIME_RE)
    return well_formed & moments.notna() & (table["message_id"] != "") & (table["chat_id"] != "")


def _clean(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    readable = _readable(table)
    kept = table[readable]
    first = ~kept["message_id"].duplicated()
    kept = kept[first]
    reaction = kept["text"].str.fullmatch(_REACTION_RE, flags=re.DOTALL)
    dropped = {
        "bad_rows": int((~readable).sum()),
        "duplicates": int((~first).sum()),
        "reactions": int(reaction.sum()),
    }
    ordered = kept[~reaction].sort_values("timestamp", kind="stable").reset_index(drop=True)
    return ordered, dropped


def _texts(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The text of each message, and which messages are only an attachment."""
    raw = table["text"]
    plain = raw.str.replace(PLACEHOLDER, "", regex=False)
    only_placeholder = raw.str.contains(PLACEHOLDER, regex=False) & (plain.str.strip() == "")
    is_attachment = only_placeholder | raw.str.startswith(_ATTACHMENT_PREFIX)
    return plain.mask(is_attachment, "").to_numpy(dtype=object), is_attachment.to_numpy(dtype=bool)


def load_rows(frame: pd.DataFrame) -> Rows:
    """The messages of an export frame. Raises ValueError when a needed column is missing."""
    table, dropped = _clean(_table(frame))
    moments = pd.to_datetime(table["timestamp"], format=_TIME_FORMAT)
    seconds = ((moments - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    days = (moments.dt.year * 10000 + moments.dt.month * 100 + moments.dt.day).to_numpy(dtype=np.int32)
    codes, conversations = pd.factorize(table["chat_id"])
    from_me = table["is_from_me"].isin(_TRUE).to_numpy(dtype=bool)
    texts, is_attachment = _texts(table)
    return Rows(
        ids=table["message_id"].to_numpy(dtype=object),
        timestamps=table["timestamp"].to_numpy(dtype=object),
        seconds=seconds,
        days=days,
        codes=codes.astype(np.int32),
        conversations=tuple(str(value) for value in conversations),
        from_me=from_me,
        senders=table["sender"].mask(from_me, "").to_numpy(dtype=object),
        texts=texts,
        is_attachment=is_attachment,
        attachment_types=table["attachment_types"].to_numpy(dtype=object),
        display_names=table["chat_display_name"].to_numpy(dtype=object),
        dropped=dropped,
    )


def read_export(path: Union[str, os.PathLike[str]]) -> pd.DataFrame:
    """The export file as text columns. "NA" and "null" stay text."""
    wanted = {*REQUIRED, *OPTIONAL}
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=lambda name: name in wanted,
        encoding="utf-8",
        encoding_errors="replace",
    )
