import random

from chat_engine.citations import CitationStream

KNOWN = {"m1", "m2", "m3", "m4", "m5", "m12"}


def _stream() -> CitationStream:
    return CitationStream(lambda ref: ref in KNOWN)


def _whole(text: str) -> tuple[str, CitationStream]:
    stream = _stream()
    return stream.feed(text) + stream.finish(), stream


def test_references_are_renumbered_in_order_of_first_use() -> None:
    out, stream = _whole("She signed it [m5]. Rent went up [m2], as she said [m5].")
    assert out == "She signed it [1]. Rent went up [2], as she said [1]."
    assert stream.order == ("m5", "m2")
    assert stream.unknown == ()


def test_lists_and_ranges_expand_to_single_markers() -> None:
    out, stream = _whole("Both agree [m1, m3] and later too [m2-m4] [M12].")
    assert out == "Both agree [1][2] and later too [3][2][4] [5]."
    assert stream.order == ("m1", "m3", "m2", "m4", "m12")


def test_unknown_reference_is_dropped_and_reported() -> None:
    out, stream = _whole("It is signed [m99]. Truly [m1][m77].")
    assert out == "It is signed. Truly [1]."
    assert stream.unknown == ("m99", "m77")


def test_other_brackets_are_left_alone() -> None:
    text = "See [sic] and [1st] and [a number ending in 0199] and [maybe m3?]."
    out, stream = _whole(text)
    assert out == text
    assert stream.order == ()


def test_split_across_chunks_gives_the_same_text() -> None:
    text = "First [m3], then  [m1, m2] and a dangling [m4"
    expected, _ = _whole(text)
    rng = random.Random(7)
    for _ in range(50):
        stream = _stream()
        pieces = []
        cursor = 0
        while cursor < len(text):
            step = rng.randint(1, 5)
            pieces.append(stream.feed(text[cursor:cursor + step]))
            cursor += step
        pieces.append(stream.finish())
        assert "".join(pieces) == expected
    assert expected.endswith("[m4")


def test_a_marker_is_never_emitted_half_done() -> None:
    stream = _stream()
    assert stream.feed("signed [m") == "signed"
    assert stream.feed("5") == ""
    assert stream.feed("] ok") == " [1] ok"


def test_outer_whitespace_is_trimmed() -> None:
    out, _ = _whole("  \n The answer [m1].  \n\n")
    assert out == "The answer [1]."


def test_long_bracket_text_is_not_held_forever() -> None:
    stream = _stream()
    text = "[" + "x" * 200
    assert stream.feed(text) == text
