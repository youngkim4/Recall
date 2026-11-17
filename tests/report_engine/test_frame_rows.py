"""Rows of the export that are not messages a person wrote: tapbacks, bare
attachment placeholders, and rows that do not say who sent them."""

import logging
from typing import Any

import pandas as pd
import pytest

from report_engine.errors import SourceError
from report_engine.fake_llm import FakeLLM
from report_engine.frame_source import FrameSource
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
