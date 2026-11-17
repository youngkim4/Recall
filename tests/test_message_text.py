"""Tests for turning stored message text into what the interface shows."""
import pandas as pd
import pytest

from message_text import attachment_label, clip_text, display_text

PLACEHOLDER = "￼"


def test_display_text_drops_the_attachment_placeholder():
    assert display_text(f"look at this {PLACEHOLDER}") == "look at this"
    assert display_text(f"{PLACEHOLDER}{PLACEHOLDER}") == ""
    assert display_text("  plain words  ") == "plain words"


def test_display_text_treats_missing_text_as_empty():
    assert display_text(None) == ""
    assert display_text(float("nan")) == ""
    assert display_text(pd.NA) == ""


@pytest.mark.parametrize(
    ("mime_types", "label"),
    [
        ("image/heic", "Photo"),
        ("image/jpeg", "Photo"),
        ("image/gif", "GIF"),
        ("video/quicktime", "Video"),
        ("audio/x-m4a", "Audio"),
        ("text/vcard", "Contact"),
        ("text/x-vlocation", "Location"),
        ("application/pdf", "File"),
        ("image/heic|image/jpeg|image/png", "3 photos"),
        ("video/quicktime|video/mp4", "2 videos"),
        ("image/heic|video/quicktime", "2 attachments"),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_attachment_label(mime_types, label):
    assert attachment_label(mime_types) == label


def test_clip_text_keeps_short_text_whole():
    assert clip_text("  short enough  ", 40) == "short enough"


def test_clip_text_cuts_at_a_word_and_marks_the_cut():
    text = "one two three four five six seven"

    clipped = clip_text(text, 16)

    assert clipped == "one two three…"
    assert len(clipped) <= 17


def test_clip_text_cuts_a_single_long_word_where_it_must():
    assert clip_text("a" * 30, 10) == "a" * 10 + "…"


def test_clip_text_does_not_drop_a_word_that_ends_at_the_limit():
    assert clip_text("one two three four", 13) == "one two three…"
