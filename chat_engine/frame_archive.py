"""The archive port over a message export (`messages.csv`), in memory.

    archive = FrameArchive.from_csv(path)

It answers the calls of `ChatArchive` with the rules of the archive contract,
as closely as an export allows. Use it to run the chat tools on an export
before the real archive exists, and for tests at full volume.

What an export cannot give: `sent_at` is the local time read as UTC,
`reply_to`, `edited` and `unsent` are empty, `person_id` is None, and the
members of a group are the people who wrote in it. See `frame_load` for the
rows that are left out, and `frame_index` for the search rules.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import date
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .fake_archive import NO_SENDER
from .frame_counts import GROUPS, conversation_stats, grouped_counts
from .frame_index import Query, WordIndex, fragment_pattern, match_ranges, parse_query
from .frame_load import Rows, is_group_id, load_rows, read_export
from .ports import ArchiveSummary, Conversation, CountRow, Message, MessagePage, SearchResult
from .textnorm import find_unit, token_spans

DEFAULT_INDEX_CHARS = 10_000  # as `archive.INDEXED_TEXT_CHARS`: search looks at the start of a very long text
_RELEVANCE_POOL = 2_000
_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NO_ROWS = np.empty(0, dtype=np.int64)


def _day_key(value: Optional[str]) -> Optional[int]:
    """A day "YYYY-MM-DD" as the number YYYYMMDD. Any other text raises ValueError."""
    if value is None or value == "":
        return None
    text = str(value)
    if not _DAY_RE.fullmatch(text):
        raise ValueError(f"not a day in the form YYYY-MM-DD: {text[:20]!r}")
    day = date.fromisoformat(text)
    return day.year * 10000 + day.month * 100 + day.day


def _copy_conversation(conversation: Conversation) -> Conversation:
    copied = dict(conversation)
    copied["members"] = list(conversation["members"])
    return copied  # type: ignore[return-value]


class FrameArchive:
    def __init__(self, frame: pd.DataFrame, *, index_chars: int = DEFAULT_INDEX_CHARS) -> None:
        """`index_chars` is how much of each text the search index holds."""
        self._rows: Rows = load_rows(frame)
        self._index_chars = max(1, int(index_chars))
        self._index: Optional[WordIndex] = None
        self._lock = threading.Lock()
        rows = self._rows
        self._code_of = {conversation: code for code, conversation in enumerate(rows.conversations)}
        grouped = np.argsort(rows.codes, kind="stable")
        bounds = np.searchsorted(rows.codes[grouped], np.arange(len(rows.conversations) + 1))
        self._positions_of = [grouped[bounds[code]:bounds[code + 1]] for code in range(len(rows.conversations))]
        self._id_index = pd.Index(rows.ids)
        nameless = (rows.senders == "") & ~rows.from_me  # received, with no handle: not the owner
        self._sender_codes, self._sender_ids = pd.factorize(np.where(nameless, NO_SENDER, rows.senders))
        self._texts = pd.Series(rows.texts, dtype=object)
        described = sorted((self._describe(code) for code in range(len(rows.conversations))), key=lambda c: c["id"])
        self._described = {conversation["id"]: conversation for conversation in described}
        self._by_count = sorted(described, key=lambda c: -c["message_count"])
        self._by_recent = sorted(described, key=lambda c: c["last_timestamp"] or "", reverse=True)

    @classmethod
    def from_csv(
        cls, path: Union[str, os.PathLike[str]], *, index_chars: int = DEFAULT_INDEX_CHARS
    ) -> FrameArchive:
        return cls(read_export(path), index_chars=index_chars)

    @property
    def dropped(self) -> dict[str, int]:
        """Rows of the export that are not messages, by reason."""
        return dict(self._rows.dropped)

    @property
    def index_built(self) -> bool:
        return self._index is not None

    def text_lengths(self) -> np.ndarray:
        """The number of characters of each text, for a report on volume."""
        return self._texts.str.len().to_numpy(dtype=np.int64)

    def longest_messages(self, count: int = 1) -> list[Message]:
        """The messages with the longest text, the longest first."""
        order = np.argsort(-self.text_lengths(), kind="stable")[:max(0, count)]
        return [self._message(int(position)) for position in order]

    @property
    def attachment_only(self) -> int:
        """How many messages are only an attachment."""
        return int(self._rows.is_attachment.sum())

    def build_index(self) -> WordIndex:
        """The search index. It is built at the first search; call this to build it sooner."""
        with self._lock:
            if self._index is None:
                self._index = WordIndex(self._rows.texts, self._index_chars)
            return self._index

    # --- contract ---

    def summary(self) -> ArchiveSummary:
        rows = self._rows
        return {
            "messages": len(rows),
            "conversations": len(rows.conversations),
            "first_timestamp": str(rows.timestamps[0]) if len(rows) else None,
            "last_timestamp": str(rows.timestamps[-1]) if len(rows) else None,
        }

    def conversations(
        self, *, limit: Optional[int] = None, offset: int = 0, order: str = "count"
    ) -> list[Conversation]:
        listed = self._by_recent if order == "recent" else self._by_count
        start = max(0, int(offset))
        end = None if limit is None else start + max(0, int(limit))
        return [_copy_conversation(conversation) for conversation in listed[start:end]]

    def conversation(self, conversation_id: str) -> Optional[Conversation]:
        found = self._described.get(str(conversation_id))
        return _copy_conversation(found) if found else None

    def messages(
        self,
        conversation_id: str,
        *,
        around: Optional[str] = None,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 50,
    ) -> MessagePage:
        cursors = [cursor for cursor in (around, before, after) if cursor is not None]
        if len(cursors) > 1:
            raise ValueError("give at most one of around, before, after")
        code = self._code_of.get(str(conversation_id))
        positions = _NO_ROWS if code is None else self._positions_of[code]
        start, end = self._page(positions, cursors, before, after, max(1, int(limit)))
        return {
            "messages": [self._message(int(position)) for position in positions[start:end]],
            "has_older": 0 < start < end,
            "has_newer": start < end < len(positions),
        }

    def iter_messages(
        self, conversation_id: str, *, since: Optional[str] = None, until: Optional[str] = None
    ) -> Iterator[Message]:
        """The messages of one conversation, oldest first. Not in the port: the engine uses it when it is there."""
        positions = self._filter(None, [str(conversation_id)], _day_key(since), _day_key(until), None)
        return (self._message(int(position)) for position in positions)

    def context(self, message_id: str, *, radius: int = 6) -> list[Message]:
        position = self._position(message_id)
        if position < 0:
            return []
        positions = self._positions_of[int(self._rows.codes[position])]
        index = int(np.searchsorted(positions, position))
        reach = max(0, int(radius))
        return [self._message(int(p)) for p in positions[max(0, index - reach):index + reach + 1]]

    def search(
        self,
        query: str,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "newest",
    ) -> SearchResult:
        if not isinstance(query, str):
            raise TypeError("the query must be text")
        parsed = parse_query(query)
        found = self._matching(parsed, conversations, since, until, from_me)
        ordered = self._ordered(found, parsed, order)
        start = max(0, int(offset))
        page = ordered[start:start + max(0, int(limit))]
        hits = [
            {"message": self._message(int(position)), "ranges": self._ranges(int(position), parsed)}
            for position in page
        ]
        return {"total": len(found), "hits": hits}  # type: ignore[typeddict-item]

    def count(
        self,
        *,
        conversations: Optional[Sequence[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        from_me: Optional[bool] = None,
        text: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> Union[int, list[CountRow]]:
        if group_by is not None and group_by not in GROUPS:
            raise ValueError(f"unknown group_by: {str(group_by)[:20]!r}")
        if text is None:
            positions = self._filter(None, conversations, _day_key(since), _day_key(until), from_me)
        elif isinstance(text, str):
            positions = self._matching(parse_query(text), conversations, since, until, from_me)
        else:
            raise TypeError("text must be text")
        if group_by is None:
            return len(positions)
        return grouped_counts(self._rows, positions, group_by, self._sender_codes, self._sender_ids)

    def stats(
        self, conversation_id: str, *, since: Optional[str] = None, until: Optional[str] = None
    ) -> dict[str, Any]:
        positions = self._filter(None, [str(conversation_id)], _day_key(since), _day_key(until), None)
        return conversation_stats(self._rows, positions, str(conversation_id))

    # --- rows ---

    def _describe(self, code: int) -> Conversation:
        rows = self._rows
        positions = self._positions_of[code]
        conversation_id = rows.conversations[code]
        group = is_group_id(conversation_id)
        sent = int(rows.from_me[positions].sum())
        members = [conversation_id]
        name = ""
        if group:
            members = [str(sender) for sender in pd.unique(rows.senders[positions]) if sender]
            titles = rows.display_names[positions]
            given = titles[titles != ""]
            name = str(given[-1]) if len(given) else ""
        return {
            "id": conversation_id,
            "is_group": group,
            "name": name,
            "message_count": len(positions),
            "sent_count": sent,
            "received_count": len(positions) - sent,
            "first_timestamp": str(rows.timestamps[positions[0]]),
            "last_timestamp": str(rows.timestamps[positions[-1]]),
            "members": members,
            "person_id": None,
        }

    def _message(self, position: int) -> Message:
        rows = self._rows
        types = str(rows.attachment_types[position])
        return {
            "id": str(rows.ids[position]),
            "conversation": rows.conversations[int(rows.codes[position])],
            "timestamp": str(rows.timestamps[position]),
            "sent_at": int(rows.seconds[position]),
            "is_from_me": bool(rows.from_me[position]),
            "sender": str(rows.senders[position]),
            "text": str(rows.texts[position]),
            "kind": "attachment" if rows.is_attachment[position] else "text",
            "attachments": [part for part in types.split("|") if part],
            "reply_to": None,
            "edited": False,
            "unsent": False,
        }

    def _position(self, message_id: object) -> int:
        """The row of a message id, or -1."""
        if not len(self._id_index):
            return -1
        return int(self._id_index.get_indexer([str(message_id)])[0])

    def _page(
        self,
        positions: np.ndarray,
        cursors: list[str],
        before: Optional[str],
        after: Optional[str],
        limit: int,
    ) -> tuple[int, int]:
        """The slice of one page. An unknown anchor gives an empty page."""
        count = len(positions)
        if not cursors:
            return max(0, count - limit), count
        position = self._position(cursors[0])
        index = int(np.searchsorted(positions, position)) if position >= 0 else count
        if index >= count or positions[index] != position:
            return 0, 0
        if before is not None:
            return max(0, index - limit), index
        if after is not None:
            return index + 1, min(count, index + 1 + limit)
        end = min(count, max(0, index - (limit - 1) // 2) + limit)
        return max(0, end - limit), end

    # --- filters and matches ---

    def _filter(
        self,
        positions: Optional[np.ndarray],
        conversations: Optional[Sequence[str]],
        since: Optional[int],
        until: Optional[int],
        from_me: Optional[bool],
    ) -> np.ndarray:
        """Rows that pass the filters, oldest first. `positions=None` starts from every row."""
        rows = self._rows
        if conversations is not None:
            wanted = [conversations] if isinstance(conversations, str) else list(conversations)
            codes = sorted({self._code_of[str(c)] for c in wanted if str(c) in self._code_of})
            if not codes:
                return _NO_ROWS
            if positions is None:
                parts = [self._positions_of[code] for code in codes]
                positions = parts[0] if len(parts) == 1 else np.sort(np.concatenate(parts))
            else:
                positions = positions[np.isin(rows.codes[positions], codes)]
        if positions is None:
            positions = np.arange(len(rows), dtype=np.int64)
        if since is not None:
            positions = positions[rows.days[positions] >= since]
        if until is not None:
            positions = positions[rows.days[positions] <= until]
        if from_me is not None:
            positions = positions[rows.from_me[positions] == bool(from_me)]
        return positions

    def _matching(
        self,
        query: Query,
        conversations: Optional[Sequence[str]],
        since: Optional[str],
        until: Optional[str],
        from_me: Optional[bool],
    ) -> np.ndarray:
        """Rows that match every part of the query and pass the filters, oldest first."""
        days = _day_key(since), _day_key(until)
        if query.empty:
            return _NO_ROWS
        positions = self._filter(self._candidates(query), conversations, *days, from_me)
        phrases = [unit for unit in query.units if unit.startswith('"')]
        if phrases and len(positions):
            texts = self._rows.texts
            keep = [self._has_phrases(str(texts[position]), phrases) for position in positions]
            positions = positions[np.array(keep, dtype=bool)]
        for fragment in query.fragments:
            if not len(positions):
                break
            held = self._texts.iloc[positions].str.contains(fragment_pattern(fragment), regex=True)
            positions = positions[held.to_numpy(dtype=bool)]
        return positions

    def _candidates(self, query: Query) -> Optional[np.ndarray]:
        """Rows that hold every word of the query, or None when the query has no word."""
        if not query.units:
            return None
        index = self.build_index()
        found: Optional[np.ndarray] = None
        for holders in sorted((index.holders(unit) for unit in query.units), key=len):
            found = holders if found is None else np.intersect1d(found, holders, assume_unique=True)
            if not len(found):
                break
        return _NO_ROWS if found is None else found.astype(np.int64)

    def _has_phrases(self, text: str, phrases: list[str]) -> bool:
        spans = token_spans(text if len(text) <= self._index_chars else text[:self._index_chars])
        return all(find_unit(phrase, spans) for phrase in phrases)

    def _ranges(self, position: int, query: Query) -> list[list[int]]:
        return match_ranges(str(self._rows.texts[position]), query, self._index_chars)

    def _ordered(self, found: np.ndarray, query: Query, order: str) -> np.ndarray:
        if order == "oldest":
            return found
        newest = found[::-1]
        if order != "relevance":
            return newest
        pool = newest[:_RELEVANCE_POOL]
        weights = np.array([len(self._ranges(int(position), query)) for position in pool], dtype=np.int64)
        return np.concatenate((pool[np.argsort(-weights, kind="stable")], newest[_RELEVANCE_POOL:]))
