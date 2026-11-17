"""An invented export of any size, to measure volume without a real file.

The shape follows a real export: a few thousand conversations of very
unequal size, eleven years, short messages from a large vocabulary where few
words are very frequent, reaction phrases, attachment rows, and a few texts
that are very long. Every word, name and number is made up. The same seed
gives the same export.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .synthetic import DEFAULT_SEED

_START = datetime(2015, 6, 19).timestamp()
_END = datetime(2026, 11, 16).timestamp()
_SYLLABLES = (
    "ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "du", "fa", "fe", "fi", "fo", "ga", "ge", "go", "ka",
    "ke", "ki", "ko", "la", "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu",
    "pa", "pe", "pi", "po", "ra", "re", "ri", "ro", "ru", "sa", "se", "si", "so", "su", "ta", "te", "ti", "to",
    "tu", "va", "ve", "vi", "vo", "za", "ze", "zo", "sha", "she", "cho", "tra", "pre", "sto", "gli",
)
_VOCABULARY = 120_000
_CONVERSATION_SHARE = 0.0025  # conversations for each message
_GROUP_SHARE = 0.15
_REACTION_SHARE = 0.03
_ATTACHMENT_SHARE = 0.022
_OWN_SHARE = 0.45
_LONG_TEXTS = ((2_900_000, 1), (150_000, 8), (12_000, 66))  # (characters, how many)
_REACTIONS = ("Loved", "Liked", "Laughed at", "Emphasized", "Disliked", "Questioned")


def _vocabulary(rng: np.random.Generator, size: int) -> np.ndarray:
    """Made-up words of two to four syllables. Some repeat: a real vocabulary also has near-doubles."""
    syllables = np.array(_SYLLABLES, dtype=object)
    picks = rng.integers(0, len(syllables), size=(size, 4))
    lengths = rng.integers(2, 5, size=size)
    made = ["".join(syllables[row[:length]]) for row, length in zip(picks, lengths, strict=True)]
    return np.array(made, dtype=object)


def _conversations(rng: np.random.Generator, count: int) -> tuple[list[str], list[list[str]], list[str]]:
    """Ids, the handles that write in each one, and group titles ("" for a direct thread)."""
    ids: list[str] = []
    writers: list[list[str]] = []
    titles: list[str] = []
    for number in range(count):
        if number == 1 or rng.random() < _GROUP_SHARE:  # the second largest is always a group
            ids.append(f"chat{700000 + number}")
            members = int(rng.integers(3, 9))
            writers.append([f"+1555{int(value):07d}" for value in rng.integers(0, 10_000_000, size=members)])
            titles.append(f"group {number}" if rng.random() < 0.5 else "")
        else:
            by_phone = rng.random() < 0.9
            handle = f"+1555{int(rng.integers(0, 10_000_000)):07d}" if by_phone else f"user{number}@example.org"
            ids.append(handle)
            writers.append([handle])
            titles.append("")
    return ids, writers, titles


def _texts(rng: np.random.Generator, rows: int, words: np.ndarray) -> list[str]:
    """One text for each row: mostly a few words, a few very long ones."""
    lengths = np.minimum(rng.geometric(0.12, size=rows), 120)
    ranks = np.minimum(rng.zipf(1.25, size=int(lengths.sum())), len(words)) - 1
    chosen = words[ranks]
    texts: list[str] = []
    cursor = 0
    for length in lengths:
        texts.append(" ".join(chosen[cursor:cursor + length]))
        cursor += int(length)
    position = 0
    for characters, count in _LONG_TEXTS:
        for _ in range(count):
            if position >= rows:
                return texts
            piece = " ".join(words[np.minimum(rng.zipf(1.25, size=2_000), len(words)) - 1])
            texts[position * 97 % rows] = (piece + " ") * (characters // (len(piece) + 1) + 1)
            position += 1
    return texts


def invented_export(rows: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """An export frame with `rows` rows. Made-up text only."""
    rng = np.random.default_rng(seed)
    count = max(2, int(rows * _CONVERSATION_SHARE))
    ids, writers, titles = _conversations(rng, count)
    weights = 1.0 / np.arange(1, count + 1) ** 1.1
    of_row = rng.choice(count, size=rows, p=weights / weights.sum())
    seconds = np.sort(rng.uniform(_START, _END, size=rows)).astype(np.int64)
    stamps = pd.to_datetime(seconds, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    own = rng.random(rows) < _OWN_SHARE
    kind = rng.random(rows)
    texts = _texts(rng, rows, _vocabulary(rng, _VOCABULARY))
    senders, types = [], []
    for row in range(rows):
        conversation = int(of_row[row])
        senders.append("" if own[row] else writers[conversation][row % len(writers[conversation])])
        types.append("")
        if kind[row] < _REACTION_SHARE:
            texts[row] = f"{_REACTIONS[row % len(_REACTIONS)]} “{texts[row][:60]}”"
        elif kind[row] < _REACTION_SHARE + _ATTACHMENT_SHARE:
            texts[row], types[row] = "￼", "image/heic"
    return pd.DataFrame(
        {
            "message_id": [f"v{row:07d}" for row in range(rows)],
            "timestamp": stamps,
            "sender": senders,
            "text": texts,
            "chat_id": [ids[int(conversation)] for conversation in of_row],
            "chat_display_name": [titles[int(conversation)] for conversation in of_row],
            "is_from_me": np.where(own, "1", "0"),
            "service": "iMessage",
            "attachment_types": types,
            "attachment_files": "",
        }
    )
