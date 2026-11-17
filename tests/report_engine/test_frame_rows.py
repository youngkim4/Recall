"""Rows of the export that are not messages a person wrote: tapbacks, bare
attachment placeholders, and rows that do not say who sent them."""

import logging
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from report_engine.errors import SourceError
from report_engine.fake_llm import FakeLLM
from report_engine.frame_source import FrameSource, direction_of
from report_engine.tapbacks import is_tapback

from .support import run
from .synth import CONVERSATION, MINA, labeler

OPEN, CLOSE = chr(0x201C), chr(0x201D)
PLACEHOLDER = chr(0xFFFC)
THUMB = chr(0x1F44D)


def quoted(phrase: str, text: str) -> str:
    return f"{phrase} {OPEN}{text}{CLOSE}"


TAPBACKS = [
    quoted("Loved", "see you at the harbor"),
    quoted("Liked", "ok"),
    quoted("Disliked", "no"),
    quoted("Laughed at", "a line\nwith a break"),
    quoted("Emphasized", "now"),
    quoted("Questioned", "really"),
    quoted("Removed a heart from", "see you"),
    quoted("Removed a like from", "ok"),
    quoted("Removed a dislike from", "no"),
    quoted("Removed a laugh from", "ha"),
    quoted("Removed an exclamation from", "now"),
    quoted("Removed a question mark from", "really"),
    'Liked "straight marks"',
    "Loved an image",
    "Laughed at an image",
    "Liked a movie",
    "Emphasized an audio message",
    "Questioned an attachment",
    "Removed a heart from an image",
    quoted(f"Reacted {THUMB} to", "fine"),
    quoted(f"Removed {THUMB} from", "fine"),
    quoted("Reacted with a sticker to", "fine"),
    quoted("Loved", PLACEHOLDER),
    "  " + quoted("Loved", "padded") + " \n",
]

NOT_TAPBACKS = [
    "Loved it",
    "Liked a lot",
    "Loved an idea",
    f"Loved {OPEN}The Bear{CLOSE} so much",
    f"Loved {OPEN}not closed",
    f"loved {OPEN}lower case{CLOSE}",
    f"I Loved {OPEN}it{CLOSE}",
    f"Reacted badly to {OPEN}the news{CLOSE}",
    f"Removed all doubt from {OPEN}the plan{CLOSE}",
    f"{OPEN}only a quote{CLOSE}",
    "",
    PLACEHOLDER,
]


@pytest.mark.parametrize("text", TAPBACKS)
def test_a_tapback_phrase_is_recognised(text: str) -> None:
    assert is_tapback(text)


@pytest.mark.parametrize("text", NOT_TAPBACKS)
def test_a_typed_message_is_not_a_tapback(text: str) -> None:
    assert not is_tapback(text)


def frame(texts: list[Any], own: list[Any] | None = None) -> pd.DataFrame:
    count = len(texts)
    stamps = pd.date_range("2024-06-01 09:00:00", periods=count, freq="min")
    return pd.DataFrame(
        {
            "message_id": list(range(1, count + 1)),
            "timestamp": stamps,
            "sender": [MINA] * count,
            "text": texts,
            "chat_id": [CONVERSATION] * count,
            "is_from_me": own if own is not None else [0] * count,
        }
    )


def ids(source: FrameSource) -> list[str]:
    return [row["id"] for row in source.messages(CONVERSATION, None, None)]


def test_the_frame_source_leaves_tapback_rows_out_and_says_how_many(
    caplog: pytest.LogCaptureFixture,
) -> None:
    texts = ["see you at the harbor", quoted("Loved", "see you at the harbor"), "Loved it", "ok"]
    with caplog.at_level(logging.INFO, logger="report_engine.frame_source"):
        assert ids(FrameSource(frame(texts))) == ["1", "3", "4"]
    assert "1 tapback row" in caplog.text
    assert "harbor" not in caplog.text


def test_words_in_a_tapback_are_never_quoted_as_the_words_of_the_person_who_tapped() -> None:
    """A tapback row holds the text of the other person's message. In a
    transcript the quote check would find those words in the wrong row."""
    words = "I got the job at the harbor office and I start in June"
    texts = [words, quoted("Loved", words), "so proud of you"]
    llm = FakeLLM()
    report = run([], llm=llm, source=FrameSource(frame(texts, own=[1, 0, 0])), labeler=labeler)
    (call,) = llm.map_calls
    assert "Loved" not in call.transcript
    assert len(call.lines) == 2
    (period,) = report.periods
    assert period.message_count == 2
    quotes = [moment for moment in report.moments if moment.quote == words]
    assert quotes and all(moment.quote_speaker == "You" for moment in quotes)
    assert all(moment.quote_message_id == "1" for moment in quotes)


def test_a_row_with_only_the_attachment_placeholder_is_not_in_the_transcript() -> None:
    texts = ["first words", PLACEHOLDER, f" {PLACEHOLDER} \n", f"look {PLACEHOLDER} at this", None]
    llm = FakeLLM()
    report = run([], llm=llm, source=FrameSource(frame(texts)), labeler=labeler)
    (call,) = llm.map_calls
    assert [line.text for line in call.lines] == ["first words", "look at this"]
    assert PLACEHOLDER not in call.text
    assert report.periods[0].message_count == 2


@pytest.mark.parametrize(
    "value, own",
    [
        (1, True),
        (1.0, True),
        ("1", True),
        ("1.0", True),
        ("true", True),
        (" Yes ", True),
        (True, True),
        (0, False),
        (0.0, False),
        ("0", False),
        ("0.0", False),
        ("false", False),
        ("N", False),
        (False, False),
        (2, False),
    ],
)
def test_the_direction_is_read_as_the_app_reads_it(value: Any, own: bool) -> None:
    (row,) = FrameSource(frame(["hello"], own=[value])).messages(CONVERSATION, None, None)
    assert row["is_from_me"] is own


@pytest.mark.parametrize("unknown", [None, float("nan"), pd.NA, "maybe", ""])
def test_a_row_that_does_not_say_who_sent_it_is_left_out(
    unknown: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A row with no direction is neither sent nor received. Its words must
    not go under a name that is a guess."""
    source = FrameSource(frame(["one", "two", "three"], own=[0, unknown, 1]))
    with caplog.at_level(logging.WARNING, logger="report_engine.frame_source"):
        assert ids(source) == ["1", "3"]
    assert "1 message" in caplog.text and "who sent" in caplog.text


def test_a_conversation_where_no_row_says_who_sent_it_is_an_error() -> None:
    source = FrameSource(frame(["one", "two"], own=[None, None]))
    with pytest.raises(SourceError) as caught:
        list(source.messages(CONVERSATION, None, None))
    assert "who sent" in str(caught.value)
    assert "one" not in str(caught.value)


@pytest.mark.parametrize(
    "value, own",
    [
        (np.int64(1), True),
        (np.float32(1.0), True),
        (np.bool_(True), True),
        (np.int64(0), False),
        (np.bool_(False), False),
        (np.float32("nan"), None),
        (np.float64("nan"), None),
        (np.float32(0.5), None),
        (np.int64(2), None),
    ],
)
def test_a_number_of_another_kind_is_read_as_the_app_reads_it(value: Any, own: bool | None) -> None:
    """The app reads only a plain 0 or 1 as a number. Any other value is read
    as text, and text that is not a known word does not say who sent the row."""
    assert direction_of(value) is own
    column = pd.Series([0, value], dtype=object)
    table = frame(["one", "two"]).assign(is_from_me=column)
    found = list(FrameSource(table).messages(CONVERSATION, None, None))
    assert [row["is_from_me"] for row in found] == ([False] if own is None else [False, own])


def test_a_row_is_counted_for_one_reason_only(caplog: pytest.LogCaptureFixture) -> None:
    texts = ["hi", quoted("Loved", "hi"), "x"]
    with caplog.at_level(logging.INFO, logger="report_engine.frame_source"):
        assert ids(FrameSource(frame(texts, own=[1, None, None]))) == ["1"]
    assert "2 messages in this conversation did not say who sent it" in caplog.text
    assert "tapback" not in caplog.text


def test_the_error_for_no_direction_is_about_the_whole_conversation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Some rows of the conversation say who sent them. A scope that holds only
    rows that do not say it is empty, not an error."""
    table = frame(["one", "two", "three"], own=[1, None, None])
    source = FrameSource(table)
    with caplog.at_level(logging.WARNING, logger="report_engine.frame_source"):
        assert list(source.messages(CONVERSATION, "2024-06-01 09:01:00", None)) == []
    assert "2 messages" in caplog.text
    nobody = FrameSource(frame(["one", "two"], own=[None, None]))
    with pytest.raises(SourceError):
        list(nobody.messages(CONVERSATION, "2024-06-01 09:01:00", None))


@pytest.mark.parametrize("key", ["1", "9", "#", "*"])
def test_a_tapback_with_a_keycap_symbol_is_recognised(key: str) -> None:
    keycap = key + chr(0xFE0F) + chr(0x20E3)
    assert is_tapback(quoted(f"Reacted {keycap} to", "fine"))
    assert is_tapback(quoted(f"Removed {keycap} from", "fine"))
    assert not is_tapback(quoted(f"Reacted {key}5 to", "fine"))


def test_timestamps_with_different_offsets_all_come_through_after_the_apps_own_parse() -> None:
    """The app parses the column before the engine reads it. Two offsets in one
    column give a column of objects, and no row of it may be lost."""
    stamps = ["2024-01-01 10:00:00-05:00", "2024-07-01 10:00:00-04:00", "2024-07-02 09:00:00+00:00"]
    with warnings.catch_warnings():
        # the app's own parse of such a column warns; that is not under test here
        warnings.simplefilter("ignore", FutureWarning)
        parsed = pd.to_datetime(pd.Series(stamps), errors="coerce")
    table = frame(["one", "two", "three"]).assign(timestamp=parsed)
    assert table["timestamp"].dtype == object
    expected = [
        datetime.fromtimestamp(datetime.fromisoformat(stamp).timestamp()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        for stamp in stamps
    ]
    found = list(FrameSource(table).messages(CONVERSATION, None, None))
    assert [row["timestamp"] for row in found] == expected


def test_reading_timestamps_shows_no_warning_of_the_library() -> None:
    """A warning of pandas can quote a cell. The log must never hold a row."""
    table = frame(["one", "two", "three"]).assign(
        timestamp=["2024-01-01 10:00:00 FOO", "2024-01-01 10:00:00-05:00", "2024-01-02T10:00:00"]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(FrameSource(table).messages(CONVERSATION, None, None))
    assert caught == []


def test_a_transcript_with_a_third_of_its_rows_left_out_reads_in_order_with_no_gap() -> None:
    """In a group chat a third of the rows can be tapbacks and bare attachments.
    The lines that stay keep their order, their references have no hole, and
    no request says that rows were left out."""
    said = [f"line number {number} of the talk" for number in range(1, 21)]
    texts: list[Any] = []
    for number, text in enumerate(said, start=1):
        texts.append(text)
        if number % 2 == 0:
            texts.append(quoted("Loved", text))
        if number % 4 == 0:
            texts.append(PLACEHOLDER)
    own = [position % 3 == 0 for position in range(len(texts))]
    llm = FakeLLM()
    report = run([], llm=llm, source=FrameSource(frame(texts, own=own)), labeler=labeler)

    assert len(texts) == 35
    (call,) = llm.map_calls
    assert [line.text for line in call.lines] == said
    assert [line.ref for line in call.lines] == [f"m{number}" for number in range(1, 21)]
    assert [line.clock for line in call.lines] == sorted(line.clock for line in call.lines)
    (reduce_call,) = llm.reduce_calls
    for request in (call.text, reduce_call.text):
        lowered = request.lower()
        assert "loved" not in lowered and "tapback" not in lowered
        assert "left out" not in lowered and "35 messages" not in lowered
        assert PLACEHOLDER not in request
    assert "Messages with text: 20 " in reduce_call.text
    (period,) = report.periods
    assert period.message_count == 20
    assert not any("left out" in warning for warning in report.warnings)
    known = {str(number) for number in range(1, 36) if texts[number - 1] in said}
    assert all(set(moment.message_ids) <= known for moment in report.moments)
