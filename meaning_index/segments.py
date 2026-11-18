"""Cut one conversation into segments, and write the text that each segment gives to embed.

A segment is a run of consecutive messages. A new segment starts after a gap
of more than GAP_MINUTES, when the segment holds MAX_MESSAGES, or when its
text reaches MAX_CHARS. The text is the day on the first line, then one line
for each message: `Me: ...` or `Them: ...`. No name and no handle is added.

A change to a rule here changes the ids and the content hashes: the next build
then embeds every segment again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta

from meaning_index.errors import SourceError
from meaning_index.ports import SourceMessage
from meaning_index.shapes import CutSegment, Segment

GAP_MINUTES = 120
MAX_MESSAGES = 16
MAX_CHARS = 1600
LINE_CHARS = 400
OWN_LABEL = "Me"
OTHER_LABEL = "Them"
DAY_LENGTH = 10
STAMP_LENGTH = 19

_GAP = timedelta(minutes=GAP_MINUTES)
_Open = tuple[str, str, str]  # (message id, timestamp, line)


def _digest(parts: Sequence[object]) -> str:
    material = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8", "surrogatepass")).hexdigest()


def segment_id(conversation_id: str, message_ids: Sequence[str]) -> str:
    """Hash of the conversation id and the message ids. The text is not part of it."""
    return _digest([conversation_id, list(message_ids)])


def content_hash(text: str, model: str, dimensions: int) -> str:
    """Hash of the three things that decide a vector: the text, the model and the dimensions."""
    return _digest([text, model, dimensions])


def stored(segment: CutSegment, model: str, dimensions: int) -> Segment:
    """The segment as the index keeps it for one model: the facts and the content hash."""
    return Segment(
        segment_id=segment.segment_id,
        conversation_id=segment.conversation_id,
        first_message_id=segment.first_message_id,
        last_message_id=segment.last_message_id,
        first_timestamp=segment.first_timestamp,
        last_timestamp=segment.last_timestamp,
        message_count=segment.message_count,
        content_hash=content_hash(segment.text, model, dimensions),
    )


def _moment(timestamp: object) -> datetime:
    if isinstance(timestamp, str) and len(timestamp) == STAMP_LENGTH and timestamp[DAY_LENGTH] == " ":
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            pass
    raise SourceError("a message has a timestamp that is not YYYY-MM-DD HH:MM:SS")


def _line(message: SourceMessage) -> str:
    """One line for one message, or "" when the message has no text to read."""
    flat = " ".join(str(message["text"]).split())
    if not flat:
        return ""
    label = OWN_LABEL if message["is_from_me"] else OTHER_LABEL
    return f"{label}: {flat}"[:LINE_CHARS]


def _close(conversation_id: str, held: Sequence[_Open]) -> CutSegment:
    ids = [message_id for message_id, _, _ in held]
    first_stamp, last_stamp = held[0][1], held[-1][1]
    text = "\n".join([first_stamp[:DAY_LENGTH], *(line for _, _, line in held)])
    return CutSegment(
        segment_id=segment_id(conversation_id, ids),
        conversation_id=conversation_id,
        first_message_id=ids[0],
        last_message_id=ids[-1],
        first_timestamp=first_stamp,
        last_timestamp=last_stamp,
        message_count=len(held),
        text=text,
    )


def cut_segments(conversation_id: str, messages: Iterable[SourceMessage]) -> Iterator[CutSegment]:
    """The segments of one conversation, oldest first. `messages` are oldest first.

    A message whose text is empty or only white space is in no segment.
    """
    held: list[_Open] = []
    size = 0
    last: datetime | None = None
    for message in messages:
        line = _line(message)
        if not line:
            continue
        moment = _moment(message["timestamp"])
        full = len(held) >= MAX_MESSAGES or size >= MAX_CHARS
        if held and (full or (last is not None and moment - last > _GAP)):
            yield _close(conversation_id, held)
            held, size = [], 0
        if not held:
            size = DAY_LENGTH
        held = [*held, (str(message["id"]), message["timestamp"], line)]
        size += 1 + len(line)
        last = moment
    if held:
        yield _close(conversation_id, held)
