"""The lookups of the volume run, with inputs taken from the archive itself.

The words, the conversations and the days come from the data, so the run is
realistic at any size. None of them is ever printed: a case has a fixed
label, and the report shows the label and numbers only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..frame_archive import FrameArchive
from ..textnorm import token_spans

_MADE_UP_WORD = "zzqxjvzzqxjv"
_MEDIUM_FREQUENCY = 500
_RARE_FREQUENCY = 5


@dataclass(frozen=True)
class Case:
    label: str
    tool: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Picks:
    """Inputs for the cases. Never printed."""

    common: str
    medium: str
    rare: str
    phrase: str
    largest: str  # conversation ids
    group: Optional[str]
    middle: str
    since: str
    until: str
    digits: str
    longest: Optional[str]  # id of the message with the longest text
    longest_word: str


def _plain(word: str) -> bool:
    return word.isascii() and word.isalpha() and 4 <= len(word) <= 12


def _words(archive: FrameArchive) -> tuple[str, str, str]:
    """A word in very many texts, one in some hundreds, one in a few."""
    index = archive.build_index()
    plain = [word for word in index.ranked_words() if _plain(word)]
    if not plain:
        return _MADE_UP_WORD, _MADE_UP_WORD, _MADE_UP_WORD
    medium = next((word for word in plain if index.frequency(word) <= _MEDIUM_FREQUENCY), plain[-1])
    rare = next((word for word in plain if index.frequency(word) <= _RARE_FREQUENCY), plain[-1])
    return plain[0], medium, rare


def _phrase(archive: FrameArchive, conversation_id: str) -> str:
    """Two words that stand side by side in one message of the conversation."""
    for message in archive.messages(conversation_id, limit=200)["messages"]:
        tokens = [token for _, _, token in token_spans(message["text"][:400]) if _plain(token)]
        spans = [token for _, _, token in token_spans(message["text"][:400])]
        for first, second in zip(spans, spans[1:], strict=False):
            if first in tokens and second in tokens:
                return f'"{first} {second}"'
    return f'"{_MADE_UP_WORD} {_MADE_UP_WORD}"'


def _longest(archive: FrameArchive) -> tuple[Optional[str], str]:
    """The message with the longest text, and a plain word from its start."""
    best_id: Optional[str] = None
    best_text = ""
    for message in archive.longest_messages(1):
        best_id, best_text = message["id"], message["text"]
    word = next((token for _, _, token in token_spans(best_text[:2000]) if _plain(token)), _MADE_UP_WORD)
    return best_id, word


def pick(archive: FrameArchive) -> Picks:
    conversations = archive.conversations(limit=None, order="count")
    if not conversations:
        raise ValueError("the archive has no conversation")
    summary = archive.summary()
    first_year = int(str(summary["first_timestamp"])[:4])
    last_year = int(str(summary["last_timestamp"])[:4])
    year = (first_year + last_year) // 2
    groups = [conversation for conversation in conversations if conversation["is_group"]]
    direct = [conversation for conversation in conversations if not conversation["is_group"]]
    common, medium, rare = _words(archive)
    largest = str(conversations[0]["id"])
    longest, longest_word = _longest(archive)
    return Picks(
        common=common,
        medium=medium,
        rare=rare,
        phrase=_phrase(archive, largest),
        largest=largest,
        group=str(groups[0]["id"]) if groups else None,
        middle=str(conversations[len(conversations) // 2]["id"]),
        since=f"{year}-01-01",
        until=f"{year}-12-31",
        digits=re.sub(r"\D", "", str(direct[0]["id"]) if direct else "")[-4:],
        longest=longest,
        longest_word=longest_word,
    )


def search_cases(picks: Picks, alias: Mapping[str, str]) -> list[Case]:
    largest = [alias[picks.largest]]
    several = f"{picks.rare} {picks.medium} {picks.common} {_MADE_UP_WORD} {picks.longest_word}"
    return [
        Case("search word in very many messages", "search_messages", {"query": picks.common}),
        Case("search word in some hundred messages", "search_messages", {"query": picks.medium}),
        Case("search word in a few messages", "search_messages", {"query": picks.rare}),
        Case("search two words", "search_messages", {"query": f"{picks.common} {picks.medium}"}),
        Case("search phrase", "search_messages", {"query": picks.phrase}),
        Case("search prefix", "search_messages", {"query": f"{picks.medium[:3]}*"}),
        Case("search any of five words", "search_messages", {"query": several, "match": "any"}),
        Case("search any, limit 20", "search_messages", {"query": several, "match": "any", "limit": 20}),
        Case("search in the largest conversation", "search_messages",
             {"query": picks.common, "conversations": largest}),
        Case("search in one year", "search_messages",
             {"query": picks.common, "since": picks.since, "until": picks.until}),
        Case("search oldest first", "search_messages", {"query": picks.common, "order": "oldest"}),
        Case("search sent by the owner", "search_messages", {"query": picks.common, "from_me": True}),
        Case("search limit 20", "search_messages", {"query": picks.common, "limit": 20}),
        Case("search with no match", "search_messages", {"query": _MADE_UP_WORD}),
        Case("search word of the longest text", "search_messages", {"query": picks.longest_word, "limit": 20}),
    ]


def browse_cases(picks: Picks, alias: Mapping[str, str]) -> list[Case]:
    largest = [alias[picks.largest]]
    return [
        Case("read the first messages", "search_messages",
             {"query": "", "conversations": largest, "order": "oldest"}),
        Case("read the latest messages", "search_messages", {"query": "", "conversations": largest}),
        Case("read the first messages of one year", "search_messages",
             {"query": "", "conversations": largest, "order": "oldest", "since": picks.since, "until": picks.until}),
        Case("read the latest messages up to a day", "search_messages",
             {"query": "", "conversations": largest, "until": picks.until, "limit": 20}),
    ]


def figure_cases(picks: Picks, alias: Mapping[str, str]) -> list[Case]:
    largest = [alias[picks.largest]]
    cases = [
        Case("count all messages", "count_messages", {}),
        Case("count a word", "count_messages", {"text": picks.common}),
        Case("count by year", "count_messages", {"group_by": "year"}),
        Case("count by month", "count_messages", {"group_by": "month"}),
        Case("count by day, largest conversation", "count_messages", {"group_by": "day", "conversations": largest}),
        Case("count by weekday and hour", "count_messages", {"group_by": "weekday_hour"}),
        Case("count by conversation", "count_messages", {"group_by": "conversation"}),
        Case("count by sender", "count_messages", {"group_by": "sender"}),
        Case("count a word by year", "count_messages", {"text": picks.medium, "group_by": "year"}),
        Case("count in one year, sent by the owner", "count_messages",
             {"since": picks.since, "until": picks.until, "from_me": True}),
        Case("list the largest conversations", "list_conversations", {"limit": 30}),
        Case("figures of the largest conversation", "conversation_stats", {"conversation": largest[0]}),
        Case("figures of one year", "conversation_stats",
             {"conversation": largest[0], "since": picks.since, "until": picks.until}),
        Case("figures of a middle conversation", "conversation_stats", {"conversation": alias[picks.middle]}),
    ]
    if picks.digits:
        cases.append(Case("find a conversation by four digits", "list_conversations", {"name": picks.digits}))
    if picks.group:
        group = [alias[picks.group]]
        cases.append(
            Case("count by sender, largest group", "count_messages", {"group_by": "sender", "conversations": group})
        )
    return cases
