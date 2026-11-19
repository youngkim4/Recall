"""The names in a question: words of a name in the directory that are not common words.

A group can be named "That Is Not A Drill", so "that" and "not" match a name.
A word is taken for a name only when it is rare in the archive: when more than
`COMMON_WORD_SHARE` of the messages hold it, it is a common word.

Cost: the candidates are at most `MAX_COUNTED` words (`Directory.named_in`
gives no more), and each is counted one time. Digits of a handle are not
counted: they are never a common word.
"""

from __future__ import annotations

from .directory import Directory
from .ports import ChatArchive
from .textnorm import fold, surface_words

COMMON_WORD_SHARE = 0.005  # a word in more than 0.5 percent of the messages is a common word, not a name
MAX_COUNTED = 4  # the most words of one question that are counted


def names_in_question(directory: Directory, archive: ChatArchive, question: str, messages: int) -> list[str]:
    """The folded words of the question that name someone. `messages` is the number of messages of the archive."""
    candidates = directory.named_in(question)[:MAX_COUNTED]
    written = _as_written(question)
    limit = COMMON_WORD_SHARE * max(0, messages)
    return [word for word in candidates if word.isdigit() or _count(archive, written.get(word, word)) <= limit]


def _as_written(question: str) -> dict[str, str]:
    """Each folded word of the question, and the first way it was written: that form goes to the archive."""
    first: dict[str, str] = {}
    for word in surface_words(question):
        first.setdefault(fold(word), word)
    return first


def _count(archive: ChatArchive, word: str) -> int:
    """The messages of the whole archive that hold the word."""
    found = archive.count(conversations=None, since=None, until=None, from_me=None, text=word, group_by=None)
    return int(found) if isinstance(found, int) else 0

