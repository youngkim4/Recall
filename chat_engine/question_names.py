"""The names in a question: the words of a name in the directory that are not common words.

A word of the name of a person (a saved contact, a member of a group) is a
name, also when it is a common word ("Will"). A word that only a group title
holds is a name only when it is rare: a group can be named "That Is Not A
Drill". It is common when more messages of the archive hold it than
`common_limit` allows: 0.5 percent of the messages, and at least
`COMMON_WORD_FLOOR`, so a small archive does not make every word common.

Every candidate is looked at, in the order of the question. At most
`MAX_COUNTED` words are counted, one time each, and at most `MAX_NAMES` names
are kept. Digits of a handle are not counted.
"""

from __future__ import annotations

from .directory import Directory
from .ports import ChatArchive
from .textnorm import fold, surface_words

COMMON_WORD_SHARE = 0.005  # a title word in more than 0.5 percent of the messages is a common word
COMMON_WORD_FLOOR = 20  # ... and in more than this many messages
MAX_COUNTED = 8  # the most words of one question that are counted
MAX_NAMES = 4  # the most names of one question that are looked up


def common_limit(messages: int) -> float:
    """The most messages that may hold a word of a group title that is still a name."""
    return max(COMMON_WORD_FLOOR, COMMON_WORD_SHARE * max(0, messages))


def names_in_question(directory: Directory, archive: ChatArchive, question: str, messages: int) -> list[str]:
    """The folded words of the question that name someone. `messages` is the number of messages of the archive."""
    written = _as_written(question)
    limit = common_limit(messages)
    names: list[str] = []
    counted = 0
    for word, of_person in directory.name_words(question):
        if len(names) >= MAX_NAMES:
            break
        if word.isdigit() or of_person:
            names.append(word)
            continue
        if counted >= MAX_COUNTED:
            continue  # not counted, so not known to be rare: not a name
        counted += 1
        if _count(archive, written.get(word, word)) <= limit:
            names.append(word)
    return names


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
