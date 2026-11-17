"""Review findings in the row decoders: attributedBody blobs, tapback codes, tapback targets and group ids.

Every input here is made by hand, or is a fixture that Apple's encoder made from invented text.
No real message is read.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from enum import IntEnum

import pytest

from archive.handles import handle_kind, is_group_identifier, phone_key
from archive.tapbacks import TAPBACK_KINDS, in_tapback_range, split_target, tapback
from archive.typedstream import (
    TypedStreamError,
    attributed_body_text,
    decode_attributed_body,
    parse_text,
    read_integer,
    read_text,
    scan_text,
)
from tests.archive.typedstream_fixtures import blob, expected_text, fixture_names

ARABIC_INDIC_ZERO = 0x0660
FULLWIDTH_ZERO = 0xFF10
SQLITE_MAX_INTEGER = 2**63 - 1

Decoder = Callable[[bytes], str]
STRICT: tuple[Decoder, ...] = (decode_attributed_body, read_text, parse_text, scan_text)
HEADER = b"\x04\x0bstreamtyped\x81\xe8\x03"  # version 4, signature, system version 1000
NEW, NIL = b"\x84", b"\x85"
FIRST_STRING_AGAIN = b"\x92"  # a reference to the first shared string of a stream
TEXT_MARKER = b"\x84\x01+"
RUN_BYTES = 6
DAMAGE_BYTES = (0x00, 0x01, 0x20, 0x2B, 0x40, 0x7F, 0x80, 0x81, 0x82, 0x84, 0x85, 0x92, 0xFF)


def other_digits(text: str, zero: int = ARABIC_INDIC_ZERO) -> str:
    """The same text with every ASCII digit written in another script."""
    return "".join(chr(zero + int(char)) if "0" <= char <= "9" else char for char in text)


def new_string(text: bytes) -> bytes:
    """A shared string that the stream has not held before."""
    return NEW + bytes([len(text)]) + text


def new_class(name: bytes, version: int = 0) -> bytes:
    return NEW + new_string(name) + bytes([version])


def stream(root_classes: list[bytes], text: bytes) -> bytes:
    """A typedstream in the layout of the fixtures: an object with a class chain, then a string."""
    parts = [
        HEADER,
        new_string(b"@"),
        NEW,
        *(new_class(name) for name in root_classes),
        NIL,
        FIRST_STRING_AGAIN,
        NEW,
        new_class(b"NSString", 1),
        NIL,
        new_string(b"+"),
        bytes([len(text)]),
        text,
    ]
    return b"".join(parts)


def damaged_copies(data: bytes, text_start: int) -> Iterator[bytes]:
    """The blob cut at every length, and with one byte changed, removed or added at every place."""
    for size in range(len(data)):
        yield data[:size]
    for index in range(len(data)):
        for value in DAMAGE_BYTES:
            yield data[:index] + bytes([value]) + data[index + 1 :]
    for index in range(text_start + 1):
        yield data[:index] + data[index + 1 :]
        yield data[:index] + NEW + data[index:]


def shown_runs(text: bytes) -> set[str]:
    """Every run of RUN_BYTES bytes of the text, in the ways an error message could show it."""
    runs = [text[start : start + RUN_BYTES] for start in range(len(text) - RUN_BYTES + 1)]
    shown: set[str] = set()
    for run in runs:
        shown.update((run.decode("latin-1"), repr(run)[2:-1], run.hex(), run.hex(" "), run.hex("\\")))
    return shown


def chain_of(error: BaseException) -> Iterator[BaseException]:
    """The error and every error it was raised from or raised while handling."""
    todo, seen = [error], set()
    while todo:
        link = todo.pop()
        if id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        todo.extend(linked for linked in (link.__cause__, link.__context__) if linked is not None)


class TestDeepClassChain:
    DEEP = HEADER + b"\x84\x01\x40\x84" + b"\x84\x84\x00\x00" * 5000

    @pytest.mark.parametrize("decode", STRICT)
    def test_5000_classes_inside_each_other_are_a_typed_stream_error(self, decode: Decoder) -> None:
        with pytest.raises(TypedStreamError):
            decode(self.DEEP)

    def test_the_tolerant_helper_gives_none_for_them(self) -> None:
        assert attributed_body_text(self.DEEP) is None

    def test_a_chain_of_64_classes_is_read(self) -> None:
        names = [b"NSAttributedString", *(b"Level%d" % level for level in range(63))]
        assert parse_text(stream(names, b"deep")) == "deep"
        assert read_text(stream(names, b"deep")) == "deep"

    def test_a_chain_of_65_classes_is_refused(self) -> None:
        names = [b"NSAttributedString", *(b"Level%d" % level for level in range(64))]
        for decode in (parse_text, read_text):
            with pytest.raises(TypedStreamError):
                decode(stream(names, b"deep"))

    def test_a_class_by_reference_brings_only_its_own_superclasses(self) -> None:
        # entries of the table: 0 the object, 1 to 3 its classes, 4 the second object
        root = [b"NSAttributedString", b"NSString", b"NSObject"]
        start = HEADER + new_string(b"@") + NEW + b"".join(new_class(name) for name in root) + NIL
        value = new_string(b"+") + b"\x02ok"
        assert parse_text(start + FIRST_STRING_AGAIN + NEW + b"\x94" + value) == "ok"  # entry 2: NSString, NSObject
        with pytest.raises(TypedStreamError):
            parse_text(start + FIRST_STRING_AGAIN + NEW + b"\x95" + value)  # entry 3: NSObject only

    def test_a_class_that_is_its_own_superclass_is_refused(self) -> None:
        # the object is entry 0 and its class is entry 1: 0x93 points at entry 1 while it is still read
        data = HEADER + new_string(b"@") + NEW + new_class(b"NSAttributedString") + b"\x93" + TEXT_MARKER + b"\x02ok"
        with pytest.raises(TypedStreamError):
            parse_text(data)


class TestDataThatIsNotATypedStream:
    NOT_A_STREAM = b"\x00\x01junk\x84\x01+\x03abc"

    @pytest.mark.parametrize("decode", STRICT)
    def test_text_after_a_marker_is_not_given(self, decode: Decoder) -> None:
        with pytest.raises(TypedStreamError):
            decode(self.NOT_A_STREAM)

    def test_the_tolerant_helper_gives_none(self) -> None:
        assert attributed_body_text(self.NOT_A_STREAM) is None

    @pytest.mark.parametrize(
        "data",
        [
            TEXT_MARKER + b"\x03abc",
            b"\x04\x0bstreamtyped" + TEXT_MARKER + b"\x03abc",  # no system version
            b"\x05\x0bstreamtyped\x81\xe8\x03" + TEXT_MARKER + b"\x03abc",  # another stream version
            b"\x04\x0bstreamtypeX\x81\xe8\x03" + TEXT_MARKER + b"\x03abc",  # another signature
            b"xx" + HEADER + TEXT_MARKER + b"\x03abc",  # the header is not at the start
        ],
    )
    def test_a_marker_with_no_valid_header_before_it_is_refused(self, data: bytes) -> None:
        for decode in (decode_attributed_body, scan_text):
            with pytest.raises(TypedStreamError):
                decode(data)
        assert attributed_body_text(data) is None

    def test_a_typedstream_with_another_layout_is_still_read_by_its_marker(self) -> None:
        data = HEADER + new_string(b"@") + b"another layout" + TEXT_MARKER + b"\x03abc"
        with pytest.raises(TypedStreamError):
            read_text(data)
        assert scan_text(data) == "abc"
        assert decode_attributed_body(data) == "abc"

    def test_a_big_endian_typedstream_is_still_read_by_its_marker(self) -> None:
        data = b"\x04\x0btypedstream\x81\x03\xe8" + b"another layout" + TEXT_MARKER + b"\x81\x01\x2c" + b"a" * 300
        assert scan_text(data) == "a" * 300
        assert decode_attributed_body(data) == "a" * 300


class TestValuesThatAreNotBytes:
    @pytest.mark.parametrize("decode", STRICT)
    @pytest.mark.parametrize("value", ["hello", "", 5, 0, 1.5, None, [1, 2], (b"x",), {"a": 1}, object()])
    def test_strict_functions_raise_a_typed_stream_error(self, decode: Decoder, value: object) -> None:
        with pytest.raises(TypedStreamError):
            decode(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["abc", "", 5, 0, 1.5, [1, 2], object(), None, b"", bytearray(), memoryview(b"")])
    def test_the_tolerant_helper_gives_none(self, value: object) -> None:
        assert attributed_body_text(value) is None  # type: ignore[arg-type]

    def test_the_message_names_the_type_and_not_the_value(self) -> None:
        with pytest.raises(TypedStreamError) as caught:
            decode_attributed_body("the text of a message")  # type: ignore[arg-type]
        assert "str" in str(caught.value)
        assert "message" not in str(caught.value)

    def test_bytes_bytearray_and_memoryview_are_read(self) -> None:
        data, text = blob("accented"), expected_text("accented")
        for value in (data, bytearray(data), memoryview(data), memoryview(bytearray(data))):
            for decode in (*STRICT, attributed_body_text):
                assert decode(value) == text  # type: ignore[arg-type]


class TestErrorsHoldNoBlobBytes:
    # fixtures whose text has no run of six bytes in common with an English sentence
    @pytest.mark.parametrize("name", ["ascii_127", "mutable_ascii_127", "korean", "cjk_japanese"])
    def test_no_error_for_a_damaged_blob_shows_six_bytes_of_the_text(self, name: str) -> None:
        data, text = blob(name), expected_text(name).encode("utf-8")
        shown = shown_runs(text)
        messages: set[str] = set()
        errors = 0
        for copy in damaged_copies(data, data.index(text)):
            for decode in STRICT:
                try:
                    decode(copy)
                except TypedStreamError as error:
                    errors += 1
                    for link in chain_of(error):
                        assert isinstance(link, TypedStreamError), type(link)  # a codec error holds the text
                        messages.update((str(link), repr(link)))
        assert errors > 1000 and len(messages) > 5
        assert sorted(message for message in messages if any(run in message for run in shown)) == []

    def test_a_wrong_type_code_is_named_by_its_offset(self) -> None:
        data = blob("ascii_127")
        marker = data.index(TEXT_MARKER)
        # the type code is now 32 bytes long in place of 1: it runs into the text
        damaged = data[: marker + 1] + b"\x20" + data[marker + 2 :]
        for decode in (parse_text, read_text):
            with pytest.raises(TypedStreamError) as caught:
                decode(damaged)
            assert f"offset {marker}" in str(caught.value)
            assert "00000" not in str(caught.value)

    def test_a_byte_that_starts_no_integer_is_named_by_its_offset(self) -> None:
        with pytest.raises(TypedStreamError) as caught:
            read_integer(b"ab\xc8d", 2)
        assert "offset 2" in str(caught.value)
        assert "c8" not in str(caught.value).lower()

    def test_a_byte_that_is_no_reference_is_named_by_its_offset(self) -> None:
        with pytest.raises(TypedStreamError) as caught:
            parse_text(HEADER + b"\x8a")  # 0x8a stands where the type code of the root object starts
        assert f"offset {len(HEADER)}" in str(caught.value)
        assert "8a" not in str(caught.value).lower()

    def test_text_that_is_not_utf8_leaves_no_codec_error_behind(self) -> None:
        data = bytearray(blob("ascii_127"))
        data[data.index(TEXT_MARKER) + 10] = 0xFF
        for decode in STRICT:
            with pytest.raises(TypedStreamError) as caught:
                decode(bytes(data))
            assert all(isinstance(link, TypedStreamError) for link in chain_of(caught.value))


class TestFixturesStillDecode:
    @pytest.mark.parametrize("name", fixture_names())
    def test_every_entry_point_gives_the_text_of_the_fixture(self, name: str) -> None:
        data, text = blob(name), expected_text(name)
        for decode in (*STRICT, attributed_body_text):
            assert decode(data) == text, decode.__name__
            assert decode(bytearray(data)) == text, decode.__name__  # type: ignore[arg-type]
            assert decode(memoryview(data)) == text, decode.__name__  # type: ignore[arg-type]


class TestTapbackCodes:
    @pytest.mark.parametrize(
        "code",
        [2000.5, 2000.0, 3001.0, "2000", b"2000", [2000], (2000,), float("nan"), float("inf"), True, False, object()],
    )
    def test_a_code_that_is_not_an_int_is_not_a_tapback(self, code: object) -> None:
        assert tapback(code) is None  # type: ignore[arg-type]

    def test_int_codes_still_name_their_kind(self) -> None:
        for index, kind in enumerate(TAPBACK_KINDS):
            assert tapback(2000 + index) == (kind, False)
            assert tapback(3000 + index) == (kind, True)
        assert tapback(None) is None

    def test_a_member_of_an_int_enum_is_an_int_code(self) -> None:
        class Code(IntEnum):
            LOVE = 2000
            UNLIKE = 3001
            NEW_KIND = 2008

        assert tapback(Code.LOVE) == ("love", False)
        assert tapback(Code.UNLIKE) == ("like", True)
        assert tapback(Code.NEW_KIND) is None
        assert in_tapback_range(Code.LOVE) is True and in_tapback_range(Code.NEW_KIND) is True


class TestTapbackRange:
    @pytest.mark.parametrize("code", [2000, 2001, 2007, 2008, 2500, 2999, 3000, 3007, 3008, 3999])
    def test_int_codes_from_2000_to_3999_are_in_the_range(self, code: int) -> None:
        assert in_tapback_range(code) is True

    @pytest.mark.parametrize(
        "code",
        [None, 0, 1, 1000, 1999, 4000, 4001, -2000, 10**30, 2000.0, 2008.0, 2500.5, "2008", b"2008", True, False],
    )
    def test_other_values_are_not_in_the_range(self, code: object) -> None:
        assert in_tapback_range(code) is False

    def test_every_code_that_tapback_knows_is_in_the_range(self) -> None:
        known = [code for code in range(-10, 5000) if tapback(code) is not None]
        assert len(known) == 2 * len(TAPBACK_KINDS)
        assert all(in_tapback_range(code) for code in known)

    def test_a_code_of_the_range_that_tapback_does_not_know(self) -> None:
        assert tapback(2008) is None and in_tapback_range(2008) is True
        assert tapback(3999) is None and in_tapback_range(3999) is True


class TestTapbackTargets:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("p:0/ABC-1", ("ABC-1", 0)),
            ("p:7/ABC-1", ("ABC-1", 7)),
            ("p:0007/ABC-1", ("ABC-1", 7)),
            ("p:9999/ABC-1", ("ABC-1", 9999)),
            ("bp:ABC-1", ("ABC-1", None)),
            ("ABC-1", ("ABC-1", None)),
        ],
    )
    def test_a_part_prefix_has_one_to_four_digits(self, value: str, expected: tuple[str, int | None]) -> None:
        assert split_target(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "p:10000/ABC-1",
            "p:99999999999999999999/X",
            "p:" + "9" * 5000 + "/X",
            "p:/ABC-1",
            "p:-1/ABC-1",
            "p:1.5/ABC-1",
        ],
    )
    def test_text_with_another_prefix_is_the_whole_target(self, value: str) -> None:
        assert split_target(value) == (value, None)

    @pytest.mark.parametrize("zero", [ARABIC_INDIC_ZERO, FULLWIDTH_ZERO])
    def test_digits_of_another_script_are_not_a_part_prefix(self, zero: int) -> None:
        value = other_digits("p:3/ABC-1", zero)
        assert split_target(value) == (value, None)

    @pytest.mark.parametrize(
        "value",
        ["p:0/A", "p:9999/A", "p:10000/A", "p:99999999999999999999/X", "p:" + "9" * 400 + "/X", "bp:A", "A"],
    )
    def test_the_part_always_fits_a_sqlite_integer(self, value: str) -> None:
        found = split_target(value)
        assert found is not None
        guid, part = found
        assert part is None or 0 <= part <= SQLITE_MAX_INTEGER
        conn = sqlite3.connect(":memory:")
        try:
            assert conn.execute("SELECT ?, ?", (guid, part)).fetchone() == (guid, part)
        finally:
            conn.close()


class TestGroupIdentifier:
    @pytest.mark.parametrize("zero", [ARABIC_INDIC_ZERO, FULLWIDTH_ZERO])
    def test_digits_of_another_script_do_not_make_a_group(self, zero: int) -> None:
        assert is_group_identifier(other_digits("chat12", zero)) is False
        assert is_group_identifier("chat12" + chr(zero + 3)) is False
        assert is_group_identifier(chr(zero + 3) + "chat12") is False

    @pytest.mark.parametrize("identifier", ["chat0", "chat12", "chat123456789012345678901234567890"])
    def test_ascii_digits_make_a_group(self, identifier: str) -> None:
        assert is_group_identifier(identifier) is True

    @pytest.mark.parametrize("identifier", ["chat12\n", "chat 12", "Chat12", "chat12 ", " chat12", "chat-12", "chat"])
    def test_anything_else_after_chat_is_one_person(self, identifier: str) -> None:
        assert is_group_identifier(identifier) is False

    def test_phone_rules_still_read_digits_of_another_script(self) -> None:
        number = other_digits("555 010 0199")
        assert handle_kind(number) == "phone"
        assert phone_key(number) == "p:5550100199"
