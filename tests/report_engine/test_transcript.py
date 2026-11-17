"""Transcript lines: safe names, short references, stable text."""

import pytest

from report_engine.errors import SourceError
from report_engine.transcript import (
    OWN_LABEL,
    UNSAVED_LABEL,
    Line,
    build_lines,
    day_header,
    looks_like_handle,
    render_period,
    resolve_ref,
    squash,
)

from .synth import MINA, RAVI, THEO, labeler, message


def test_own_messages_are_you_and_others_use_the_labeler() -> None:
    lines = build_lines(
        [
            message(1, "2024-06-01 09:00:00", "hi there"),
            message(2, "2024-06-01 09:01:00", "hello", is_from_me=True),
        ],
        labeler,
    )
    assert [(line.id, line.day, line.clock, line.label) for line in lines] == [
        ("1", "2024-06-01", "09:00", "Mina"),
        ("2", "2024-06-01", "09:01", OWN_LABEL),
    ]


def test_the_labeler_is_never_asked_about_own_messages() -> None:
    asked: list[str] = []

    def spy(sender: str) -> str:
        asked.append(sender)
        return "Mina"

    build_lines(
        [
            message(1, "2024-06-01 09:00:00", "a", is_from_me=True),
            message(2, "2024-06-01 09:01:00", "b"),
            message(3, "2024-06-01 09:02:00", "c"),
        ],
        spy,
    )
    assert asked == [MINA]


@pytest.mark.parametrize(
    "bad",
    ["+12025550143", "(202) 555-0143", "202.555.0143", "theo@example.com", "Theo <theo@example.com>", ""],
)
def test_a_label_that_is_a_handle_or_empty_becomes_unsaved(bad: str) -> None:
    lines = build_lines([message(1, "2024-06-01 09:00:00", "hi")], lambda sender: bad)
    assert lines[0].label == UNSAVED_LABEL


def test_looks_like_handle_keeps_ordinary_names() -> None:
    assert not looks_like_handle("Mina")
    assert not looks_like_handle("Mina 2")
    assert not looks_like_handle("Unsaved (…0143)")
    assert looks_like_handle("+1 202 555 0143")
    assert looks_like_handle("a@b.co")


def test_unsaved_senders_keep_their_number_when_more_arrive() -> None:
    first = [message(1, "2024-06-01 09:00:00", "a", sender=MINA)]
    later = [*first, message(2, "2024-06-02 09:00:00", "b", sender=RAVI)]
    nameless = lambda sender: ""  # noqa: E731
    assert [line.label for line in build_lines(first, nameless)] == [UNSAVED_LABEL]
    assert [line.label for line in build_lines(later, nameless)] == [UNSAVED_LABEL, f"{UNSAVED_LABEL} 2"]


def test_a_contact_named_like_the_owner_does_not_pass_as_the_owner() -> None:
    lines = build_lines([message(1, "2024-06-01 09:00:00", "a")], lambda sender: " you ")
    assert lines[0].label == UNSAVED_LABEL


def test_labels_are_cleaned_for_the_line_format() -> None:
    lines = build_lines([message(1, "2024-06-01 09:00:00", "a")], lambda sender: " Dr.\nMina:  Office ")
    assert lines[0].label == "Dr. Mina Office"


def test_messages_without_text_are_left_out() -> None:
    lines = build_lines(
        [
            message(1, "2024-06-01 09:00:00", ""),
            message(2, "2024-06-01 09:01:00", "  \n "),
            message(3, "2024-06-01 09:02:00", "\ufffc"),
            message(4, "2024-06-01 09:03:00", "kept"),
        ],
        labeler,
    )
    assert [line.id for line in lines] == ["4"]


def test_lines_are_sorted_when_the_source_is_not() -> None:
    lines = build_lines(
        [
            message(2, "2024-06-02 09:00:00", "second"),
            message(1, "2024-06-01 09:00:00", "first"),
            message(3, "2024-06-02 09:00:00", "third, same second"),
        ],
        labeler,
    )
    assert [line.id for line in lines] == ["1", "2", "3"]


@pytest.mark.parametrize("stamp", ["2024-06-01", "2024-06-01T09:00:00", "not a time", "2024-13-01 09:00:00"])
def test_a_bad_timestamp_is_a_source_error(stamp: str) -> None:
    with pytest.raises(SourceError) as caught:
        build_lines([message("m-77", stamp, "hi")], labeler)
    assert "m-77" in str(caught.value)
    assert "hi" not in str(caught.value)


def test_build_lines_does_not_change_its_input() -> None:
    source = [message(1, "2024-06-01 09:00:00", "  two   spaces ")]
    snapshot = [dict(item) for item in source]
    build_lines(source, labeler)
    assert source == snapshot


def test_squash_collapses_space_and_drops_invisible_marks() -> None:
    assert squash("  a\n\n b\tc \u200bd\ufeff ") == "a b c d"


def test_day_header_does_not_depend_on_the_locale() -> None:
    assert day_header("2024-06-01") == "--- 2024-06-01 Saturday ---"


def test_render_period_numbers_lines_from_one_and_groups_by_day() -> None:
    lines = build_lines(
        [
            message("a", "2024-06-01 09:00:00", "hi\nthere"),
            message("b", "2024-06-01 09:05:00", "hello", is_from_me=True),
            message("c", "2024-06-02 21:30:00", "night", sender=THEO),
        ],
        labeler,
    )
    rendered = render_period(lines, max_chars=1000)
    assert rendered.text == (
        "--- 2024-06-01 Saturday ---\n"
        "m1 09:00 Mina: hi there\n"
        "m2 09:05 You: hello\n"
        "--- 2024-06-02 Sunday ---\n"
        "m3 21:30 Theo: night"
    )
    assert rendered.labels == ("Mina", "Theo", "You")


def test_render_period_shortens_a_very_long_message_but_keeps_both_ends() -> None:
    text = "start " + "x" * 500 + " end"
    lines = build_lines([message(1, "2024-06-01 09:00:00", text)], labeler)
    rendered = render_period(lines, max_chars=100)
    body = rendered.text.split("Mina: ", 1)[1]
    assert body.startswith("start ")
    assert body.endswith(" end")
    assert "[...]" in body
    assert len(body) <= 100 + len(" [...] ")
    assert lines[0].text == text


@pytest.mark.parametrize("ref, index", [("m1", 0), ("M2", 1), (" [m3] ", 2), ("m3.", 2)])
def test_resolve_ref_maps_back_to_the_line(ref: str, index: int) -> None:
    lines = tuple(Line(id=str(n), day="2024-06-01", clock="09:00", label="Mina", text="t") for n in range(3))
    assert resolve_ref(ref, lines) is lines[index]


@pytest.mark.parametrize("ref", ["m0", "m4", "m", "12", "msg1", "", "m1x", "m-1"])
def test_resolve_ref_rejects_what_is_not_a_line_of_the_period(ref: str) -> None:
    lines = tuple(Line(id=str(n), day="2024-06-01", clock="09:00", label="Mina", text="t") for n in range(3))
    assert resolve_ref(ref, lines) is None
