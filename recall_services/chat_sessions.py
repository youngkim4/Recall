"""What a chat read, kept between its questions, so a follow-up can cite and read on from it.

The app sends the earlier turns of a chat as plain rows (`history`). Rows
hold no evidence: with them alone, a follow-up does not know the messages
that the last answer read, nor that it stopped at the lookup limit.

So the server keeps the `ChatSession` of each answer here, in memory. The
key is the archive and the last question and answer, as the app sends them
back in the next `history`. A follow-up takes the session of the newest pair
of its history that has one, and gives it back when it ends:

- under its own question and answer, when it answered;
- under the pair it was taken by, when it was cancelled or failed (the
  session rolls back by itself, so it is as it was).

A session is taken by one question at a time: a second question that starts
while the first holds it builds its session from the rows. A question asked
again from the same history finds nothing: its session was taken. The store keeps the newest
`KEPT` sessions; a history that matches none gives None, and the caller
builds a session from the rows as before. Nothing is written to disk: the
evidence holds the text of messages.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

from chat_engine.session import ChatSession

KEPT = 32

_LOCK = threading.Lock()
_SESSIONS: OrderedDict[str, ChatSession] = OrderedDict()


def key_of(archive_path: Path | str, question: str, answer: str) -> str:
    material = "\0".join((str(Path(archive_path).resolve()), question.strip(), answer.strip()))
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()


def pairs(history: object) -> list[tuple[str, str]]:
    """Each question with the answer right after it, the newest first."""
    rows = [row for row in history if isinstance(row, Mapping)] if isinstance(history, (list, tuple)) else []
    found: list[tuple[str, str]] = []
    for asked, answered in zip(rows, rows[1:]):
        if asked.get("role") != "user" or answered.get("role") != "assistant":
            continue
        question, answer = str(asked.get("content") or ""), str(answered.get("content") or "")
        if question.strip() and answer.strip():
            found.append((question, answer))
    return found[::-1]


def keep(archive_path: Path | str, question: str, answer: str, session: ChatSession) -> str:
    """Keep the session under this question and answer. Gives the key."""
    key = key_of(archive_path, question, answer)
    with _LOCK:
        _SESSIONS[key] = session
        _SESSIONS.move_to_end(key)
        while len(_SESSIONS) > KEPT:
            _SESSIONS.popitem(last=False)
    return key


def take(archive_path: Path | str, history: object) -> tuple[ChatSession, tuple[str, str]] | None:
    """The kept session of the newest pair of the history that has one, and that pair. It leaves the store.

    The newest pair can have none: a question that failed leaves a question
    with no answer, and a stopped one leaves the part that it wrote. The
    session of the answer before them still holds what the chat read.
    """
    with _LOCK:
        for pair in pairs(history):
            session = _SESSIONS.pop(key_of(archive_path, *pair), None)
            if session is not None:
                return session, pair
    return None


def put_back(archive_path: Path | str, pair: tuple[str, str], session: ChatSession) -> None:
    """Give a taken session back under the pair it was taken by, when its question did not answer."""
    keep(archive_path, *pair, session)


def clear() -> None:
    with _LOCK:
        _SESSIONS.clear()
