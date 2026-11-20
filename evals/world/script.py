"""A small script language for the scenes of the eval world.

A scene is text, one item on each line:

    @2024-08-02 18:40          set the clock (local time of the world)
    +45                        move the clock 45 minutes; "+3h" moves it 3 hours
    wren: where are you        a message of `wren`; "me" is the owner
    wren: you promised  #tag   a message with a label; questions name it
    me ^love                   a tapback of `me` on the message before it

Before each message the clock moves 1 to 4 minutes, by a fixed rule, so a
scene reads like a real thread. Every id comes from the conversation, the
scene name and the place of the item: the same text gives the same guids.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/Los_Angeles")
# fixed namespace: the guids never change between runs
GUID_SPACE = uuid.UUID("6f1e1d2a-6a43-4c55-9b0e-0e7a1f5e2b10")
TAPBACKS = {"love": 2000, "like": 2001, "dislike": 2002, "laugh": 2003, "emphasize": 2004, "question": 2005}

_CLOCK_RE = re.compile(r"^@(\d{4}-\d{2}-\d{2} \d{2}:\d{2})$")
_MOVE_RE = re.compile(r"^\+(\d+)(h?)$")
_TAPBACK_RE = re.compile(r"^([a-z]+) \^([a-z]+)$")
_LINE_RE = re.compile(r"^([a-z]+): (.+?)(?:\s+#([a-z0-9_]+))?$")


class ScriptError(ValueError):
    """A scene that can not be read. The message names the scene and the line."""


@dataclass(frozen=True)
class Line:
    """One message of the world."""

    conversation: str
    when: datetime
    sender: str  # a person key, or "me"
    text: str
    guid: str
    label: str = ""

    @property
    def from_me(self) -> bool:
        return self.sender == "me"

    @property
    def day(self) -> str:
        return self.when.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Tapback:
    """One reaction on a message of the world."""

    conversation: str
    when: datetime
    sender: str
    target: str  # guid of the message
    kind: str
    guid: str

    @property
    def code(self) -> int:
        return TAPBACKS[self.kind]


Item = Line | Tapback


def guid_for(*parts: object) -> str:
    """A stable guid, in the upper case form Messages writes."""
    return str(uuid.uuid5(GUID_SPACE, "|".join(str(part) for part in parts))).upper()


def local(stamp: str) -> datetime:
    """A "YYYY-MM-DD HH:MM" local time of the world."""
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M").replace(tzinfo=ZONE)


def _gap(place: int) -> timedelta:
    return timedelta(minutes=1 + (place * 37) % 4)


class _Reader:
    def __init__(self, conversation: str, name: str) -> None:
        self.conversation = conversation
        self.name = name
        self.clock: datetime | None = None
        self.fresh = True  # the clock was just set: no gap before the next message
        self.items: list[Item] = []
        self.last: Line | None = None

    def fail(self, place: int, why: str) -> ScriptError:
        return ScriptError(f"scene {self.conversation}/{self.name}, line {place}: {why}")

    def read(self, place: int, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        if clock := _CLOCK_RE.match(text):
            self.clock, self.fresh = local(clock.group(1)), True
            return
        if self.clock is None:
            raise self.fail(place, "no clock is set yet")
        if move := _MOVE_RE.match(text):
            unit = 60 if move.group(2) else 1
            self.clock, self.fresh = self.clock + timedelta(minutes=int(move.group(1)) * unit), True
            return
        if tap := _TAPBACK_RE.match(text):
            self._tapback(place, tap.group(1), tap.group(2))
            return
        line = _LINE_RE.match(text)
        if line is None:
            raise self.fail(place, "not a message, a clock, a move or a tapback")
        self._message(place, line.group(1), line.group(2), line.group(3) or "")

    def _tapback(self, place: int, sender: str, kind: str) -> None:
        if kind not in TAPBACKS:
            raise self.fail(place, f"unknown tapback {kind}")
        if self.last is None or self.clock is None:
            raise self.fail(place, "a tapback needs a message before it")
        self.clock = self.clock + timedelta(minutes=1)
        guid = guid_for(self.conversation, self.name, place, "tapback")
        self.items.append(Tapback(self.conversation, self.clock, sender, self.last.guid, kind, guid))

    def _message(self, place: int, sender: str, text: str, label: str) -> None:
        if self.clock is None:
            raise self.fail(place, "no clock is set yet")
        if not self.fresh:
            self.clock = self.clock + _gap(place)
        self.fresh = False
        guid = guid_for(self.conversation, self.name, place)
        self.last = Line(self.conversation, self.clock, sender, text, guid, label)
        self.items.append(self.last)


def scene(conversation: str, name: str, script: str) -> tuple[Item, ...]:
    """The items of one scene, in order."""
    reader = _Reader(conversation, name)
    for place, raw in enumerate(script.splitlines(), start=1):
        reader.read(place, raw)
    return tuple(reader.items)
