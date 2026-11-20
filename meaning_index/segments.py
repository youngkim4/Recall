"""Cut one conversation into segments, and write the text that each segment gives to embed.

A segment is a run of consecutive messages. A new segment starts after a gap
of more than GAP_MINUTES, when the segment holds MAX_MESSAGES, or when its
text reaches MAX_CHARS. The text is the day of the earliest message on the
first line, then one line for each message: `Me: ...` or `Them: ...`. No name
and no handle is added.

A change to a rule here changes the ids and the content hashes: the next build
then embeds every segment again.

A recipe (`recipe.py`) can turn on three switches: a context line, an overlap
with the segment before, and a gap from the rhythm of the conversation. With
the default recipe the segments are the ones above, byte for byte. The
context line holds the words that the caller gives: this package adds no name
and no handle of its own. With overlap, the carried messages are part of the
segment: its id, its first message and its count hold them, so a hit reads
back what was embedded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import datetime, timedelta

from meaning_index.errors import SourceError
from meaning_index.ports import SourceMessage
from meaning_index.recipe import DEFAULT_RECIPE, OVERLAP_MESSAGES, Recipe, SegmentContext, adaptive_gap, context_line
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
_Head = Callable[[str], str]  # the timestamp of the first message -> the context line


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


def _close(conversation_id: str, held: Sequence[_Open], head: str | None) -> CutSegment:
    ids = [message_id for message_id, _, _ in held]
    stamps = [stamp for _, stamp, _ in held]
    first_stamp, last_stamp = min(stamps), max(stamps)  # a source can give a time that goes back
    lines = [first_stamp[:DAY_LENGTH], *(line for _, _, line in held)]
    text = "\n".join(lines if head is None else [head, *lines])
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


def _header(recipe: Recipe, conversation_id: str, context: SegmentContext | None) -> _Head | None:
    """The maker of the context line, or None when the recipe has no context."""
    if not recipe.context:
        return None
    return lambda first: context_line(first, context(conversation_id, first) if context is not None else "")


def _rhythm(messages: Sequence[SourceMessage]) -> timedelta:
    """The adaptive gap of the messages that are in a segment."""
    return adaptive_gap([_moment(message["timestamp"]) for message in messages if _line(message)], _GAP)


def cut_segments(
    conversation_id: str,
    messages: Iterable[SourceMessage],
    recipe: Recipe = DEFAULT_RECIPE,
    context: SegmentContext | None = None,
) -> Iterator[CutSegment]:
    """The segments of one conversation, oldest first. `messages` are oldest first.

    A message whose text is empty or only white space is in no segment.
    `context` gives the words of the context line; it is asked one time for
    each segment, and only when the recipe has `context`.

    MAX_MESSAGES counts the messages of the segment itself, not the carried
    ones. MAX_CHARS counts the whole text: the context line and the carried
    lines take room from the messages. So no text is longer than MAX_CHARS +
    LINE_CHARS, with any recipe.
    """
    read: Iterable[SourceMessage] = messages
    gap = _GAP
    if recipe.adaptive:
        listed = list(messages)
        read, gap = listed, _rhythm(listed)
    header = _header(recipe, conversation_id, context)
    held: list[_Open] = []  # the carried messages, then the messages of the segment
    own = size = 0
    head: str | None = None
    last: datetime | None = None
    for message in read:
        line = _line(message)
        if not line:
            continue
        moment = _moment(message["timestamp"])
        opening: _Open = (str(message["id"]), message["timestamp"], line)
        full = own >= MAX_MESSAGES or size >= MAX_CHARS
        gapped = last is not None and moment - last > gap
        if own and (full or gapped):
            yield _close(conversation_id, held, head)
            carried = held[len(held) - own :][-OVERLAP_MESSAGES:] if recipe.overlap and not gapped else []
            held, own = carried, 0
        if not own:
            head = None if header is None else header(min(stamp for _, stamp, _ in (*held, opening)))
            size = DAY_LENGTH + sum(1 + len(text) for _, _, text in held) + (0 if head is None else 1 + len(head))
        held = [*held, opening]
        own += 1
        size += 1 + len(line)
        last = moment
    if own:
        yield _close(conversation_id, held, head)
