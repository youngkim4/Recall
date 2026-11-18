"""Part F4: a digest inside a question streams the progress of each part while it is read.

The fake source blocks at each part until the reader saw the "start" of that
part. A turn that held its events until the tool returned would leave every
part waiting: the test would see the waits time out.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

import pytest

from chat_engine.cancel import CancelToken
from chat_engine.fake_digests import FakeDigests, period
from chat_engine.fake_llm import call, say
from tests.chat_engine.helpers import alias_of_line, last_user
from tests.chat_engine.memory_bench import maya_story
from tests.chat_engine.test_memory_engine import engine_with
from tests.chat_engine.tiny import MAYA_PHONE, UNSAVED_PHONE

QUESTION = "What is my friendship with Maya like?"
WAIT_SECONDS = 5.0


def _ask_digest(request: Any) -> list:
    alias = alias_of_line(last_user(request), "Maya Okafor · direct, phone")
    return [call("conversation_digest", "d1", conversation=alias)]


def _progress(part: int, state: str, label: str, of: int = 3, conversation: str = "Maya Okafor") -> dict:
    return {
        "type": "progress", "tool": "conversation_digest", "conversation": conversation,
        "part": part, "of": of, "label": label, "state": state,
    }  # fmt: skip


class Gates:
    """One gate for each part. A part waits at its gate; the reader opens it on the start of that part."""

    def __init__(self, parts: int) -> None:
        self.opened = [threading.Event() for _ in range(parts + 1)]
        self.timed_out: list[int] = []

    def wait(self, index: int) -> None:
        if not self.opened[index].wait(WAIT_SECONDS):
            self.timed_out.append(index)

    def seen(self, event: dict) -> None:
        if event.get("type") == "progress" and event["state"] == "start":
            self.opened[event["part"]].set()


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


# --- the events reach the stream while the tool runs --------------------------------


def test_each_part_is_told_while_it_is_read_and_before_the_answer() -> None:
    gates = Gates(3)
    digests = FakeDigests({MAYA_PHONE: maya_story()}, per_part=gates.wait)
    engine, _, _ = engine_with([_ask_digest, say("Close friends.")], digests=digests)
    events: list[dict] = []

    def on_event(event: dict) -> None:
        events.append(event)
        gates.seen(event)

    result = engine.ask(QUESTION, on_event=on_event)
    assert gates.timed_out == [] and result["answer"] == "Close friends."
    told = [event for event in events if event["type"] == "progress"]
    assert told == [
        _progress(1, "start", "2021"), _progress(1, "done", "2021"),
        _progress(2, "start", "2023"), _progress(2, "done", "2023"),
        _progress(3, "start", "2025"), _progress(3, "done", "2025"),
    ]  # fmt: skip
    # the status line of the digest comes first, the answer after the last part, and nothing after done
    first, last = events.index(told[0]), events.index(told[-1])
    assert events[first - 1] == {"type": "status", "text": "Reading your whole conversation with Maya Okafor"}
    assert "delta" not in _types(events[:last]) and events[-1]["type"] == "done"


def test_the_stream_of_the_engine_gives_the_same_events() -> None:
    gates = Gates(3)
    digests = FakeDigests({MAYA_PHONE: maya_story()}, per_part=gates.wait)
    engine, _, _ = engine_with([_ask_digest, say("Close friends.")], digests=digests)
    events = []
    for event in engine.stream(QUESTION):
        events.append(event)
        gates.seen(event)
    assert gates.timed_out == [] and _types(events).count("progress") == 6 and events[-1]["type"] == "done"


def test_parts_in_the_notes_are_told_as_cached() -> None:
    digests = FakeDigests({MAYA_PHONE: maya_story()}, cached=2)
    engine, _, _ = engine_with([_ask_digest, say("Close friends.")], digests=digests)
    events: list[dict] = []
    engine.ask(QUESTION, on_event=events.append)
    assert [event for event in events if event["type"] == "progress"] == [
        _progress(1, "cached", "2021"), _progress(2, "cached", "2023"),
        _progress(3, "start", "2025"), _progress(3, "done", "2025"),
    ]  # fmt: skip


class Reporting:
    """A source that reports what a test gives it, then returns the story."""

    def __init__(self, reports: list[Any], story: Optional[list] = None) -> None:
        self.reports = reports
        self.story = maya_story() if story is None else story

    def estimate(self, conversation_id: str, **_rest: object) -> dict:
        return {"periods": len(self.story), "periods_cached": 0, "usd": 0.01}

    def digest(self, conversation_id: str, *, progress: Any = None, **_rest: object) -> dict:
        for report in self.reports:
            progress(report)
        return {"periods": self.story, "periods_failed": 0}


def _told(reports: list[Any], story: Optional[list] = None) -> list[dict]:
    engine, _, _ = engine_with([_ask_digest, say("Close friends.")], digests=Reporting(reports, story))
    events: list[dict] = []
    engine.ask(QUESTION, on_event=events.append)
    return [event for event in events if event["type"] == "progress"]


def test_a_report_in_another_form_is_dropped_and_a_repeat_is_told_once() -> None:
    good = {"part": 1, "of": 3, "label": "2021", "state": "start"}
    reports = [
        good, dict(good),  # the same change twice
        "start", None, {"part": 1, "of": 3, "label": "2021"}, {**good, "state": "waiting"},
        {**good, "part": 0}, {**good, "part": 4}, {**good, "part": True}, {**good, "of": "3"}, {**good, "label": 7},
        {**good, "state": "failed"},
    ]  # fmt: skip
    assert _told(reports) == [_progress(1, "start", "2021"), _progress(1, "failed", "2021")]


def test_a_handle_in_the_label_of_a_part_is_masked() -> None:
    (event,) = _told([{"part": 1, "of": 3, "label": f"Jan 2025 {UNSAVED_PHONE}", "state": "start"}])
    assert UNSAVED_PHONE not in str(event) and "7205550199" not in str(event)
    assert event["label"].startswith("Jan 2025 ") and event["conversation"] == "Maya Okafor"


# --- a cancel and a reader who leaves -----------------------------------------------


def test_a_cancel_while_the_parts_are_read_stops_the_reading_and_ends_the_question() -> None:
    cancel = CancelToken()
    gates = Gates(3)
    digests = FakeDigests({MAYA_PHONE: maya_story()}, per_part=gates.wait)
    engine, fake, _ = engine_with([_ask_digest, say("never")], digests=digests)
    events: list[dict] = []

    def on_event(event: dict) -> None:
        events.append(event)
        if event.get("type") == "progress" and (event["part"], event["state"]) == (2, "start"):
            cancel.cancel()
        gates.seen(event)

    result = engine.ask(QUESTION, cancel=cancel, on_event=on_event)
    assert result["cancelled"] is True and len(fake.requests) == 1
    told = [(event["part"], event["state"]) for event in events if event["type"] == "progress"]
    assert told == [(1, "start"), (1, "done"), (2, "start"), (2, "done")]  # the part in hand ends; no next part
    assert events[-1]["type"] == "done" and gates.timed_out == []


def test_a_reader_who_leaves_while_the_parts_are_read_stops_the_reading() -> None:
    gates = Gates(3)
    reached = threading.Event()

    def blocked(index: int) -> None:
        reached.set()
        gates.wait(index)

    cancel = CancelToken()
    digests = FakeDigests({MAYA_PHONE: maya_story()}, per_part=blocked)
    engine, _, _ = engine_with([_ask_digest, say("never")], digests=digests)
    events = engine.stream(QUESTION, cancel=cancel)
    for event in events:
        if event["type"] == "progress":
            break
    assert reached.wait(WAIT_SECONDS)
    events.close()  # the reader leaves while part 1 is read
    assert cancel.cancelled
    gates.opened[1].set()
    assert [name for name, _ in digests.calls] == ["estimate", "digest"]


@pytest.mark.parametrize("parts", [1, 4])
def test_a_source_that_tells_nothing_still_gives_the_story(parts: int) -> None:
    story = [period(f"Y{n}", f"202{n}-01-01", f"202{n}-12-31", 10, f"Summary {n}.") for n in range(parts)]
    assert _told([], story) == []
