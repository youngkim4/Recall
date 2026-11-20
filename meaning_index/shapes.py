"""The data that meaning_index gives and keeps. Every shape is frozen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from meaning_index.recipe import DEFAULT_NAME

IndexState = Literal["none", "ready"]
BuildPhase = Literal["read", "embed", "write"]


@dataclass(frozen=True)
class CutSegment:
    """A run of consecutive messages of one conversation, with the text to embed."""

    segment_id: str
    conversation_id: str
    first_message_id: str
    last_message_id: str
    first_timestamp: str
    last_timestamp: str
    message_count: int
    text: str


@dataclass(frozen=True)
class Segment:
    """A segment as the index keeps it. It holds no text: the archive holds the messages."""

    segment_id: str
    conversation_id: str
    first_message_id: str
    last_message_id: str
    first_timestamp: str
    last_timestamp: str
    message_count: int
    content_hash: str


@dataclass(frozen=True)
class IndexMeta:
    """The content of `meta.json`. A file with no `recipe` was built with the default recipe."""

    format_version: int
    build_id: str
    model: str
    dimensions: int
    built_at: str
    segments: int
    covered_until: str | None
    recipe: str = DEFAULT_NAME


@dataclass(frozen=True)
class IndexStatus:
    """What the app shows about the index. With state "none" every other field is empty."""

    state: IndexState
    segments: int = 0
    model: str | None = None
    dimensions: int | None = None
    built_at: str | None = None
    covered_until: str | None = None


@dataclass(frozen=True)
class BuildEstimate:
    """What a build would do now. `tokens` and `usd` are for the texts that the build would send.

    `upper_bound` is True when the work file of a stopped build could not be
    read. The build can then cost less than the estimate says, and never more.
    `recipe_changed` is True when the index on disk was built with another
    recipe: a build is due, even when it embeds nothing.
    """

    segments: int
    new_segments: int
    tokens: int
    usd: float
    upper_bound: bool = False
    recipe_changed: bool = False


@dataclass(frozen=True)
class BuildResult:
    """What a build did. Every count is a count of segments, and `segments` is `embedded` plus `reused`."""

    segments: int
    embedded: int
    reused: int
    removed: int
    tokens: int


@dataclass(frozen=True)
class BuildProgress:
    """One step of a build. `read` counts conversations, `embed` counts texts, `write` counts segments."""

    phase: BuildPhase
    done: int
    total: int


@dataclass(frozen=True)
class SearchHit:
    """One segment of a search, with the dot product of its vector and the query."""

    segment: Segment
    score: float
