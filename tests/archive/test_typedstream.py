"""The attributedBody decoder, checked against blobs made by Apple's NSArchiver."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from archive.typedstream import (
    KNOWN_LAYOUTS,
    TypedStreamError,
    attributed_body_text,
    decode_attributed_body,
    parse_text,
    read_integer,
    read_text,
    scan_text,
)
from tests.archive.typedstream_fixtures import (
    FIXTURE_DIR,
    blob,
    counter_text,
    entries,
    expected_text,
    fixture_names,
    manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_BYTE_LENGTHS = {1, 127, 128, 255, 256, 300, 65_535, 70_000}


def _length_marker(name: str) -> bytes:
    data = blob(name)
    start = data.index(b"\x84\x01+") + 3
    return data[start : start + 5]


class TestAppleFixtures:
    @pytest.mark.parametrize("name", fixture_names())
    def test_decodes_every_fixture(self, name: str) -> None:
        assert decode_attributed_body(blob(name)) == expected_text(name)

    @pytest.mark.parametrize("name", fixture_names())
    def test_structured_read_and_marker_scan_agree(self, name: str) -> None:
        assert read_text(blob(name)) == expected_text(name)
        assert scan_text(blob(name)) == expected_text(name)

    @pytest.mark.parametrize("name", fixture_names())
    def test_walking_the_stream_gives_the_same_text_as_the_known_layout(self, name: str) -> None:
        data = blob(name)
        assert sum(1 for layout in KNOWN_LAYOUTS if data.startswith(layout)) == 1
        assert parse_text(data) == read_text(data) == expected_text(name)

    def test_known_layouts_are_the_bytes_apple_writes(self) -> None:
        plain, mutable = KNOWN_LAYOUTS
        assert blob("ascii_1")[: len(plain)] == plain
        assert blob("mutable_short")[: len(mutable)] == mutable
        assert plain.endswith(b"NSString\x01\x94\x84\x01+") and mutable.endswith(b"NSString\x01\x95\x84\x01+")

    def test_another_system_version_is_read_by_walking_the_stream(self) -> None:
        data = blob("ascii_300").replace(b"streamtyped\x81\xe8\x03", b"streamtyped\x81\xe9\x03")
        assert not any(data.startswith(layout) for layout in KNOWN_LAYOUTS)
        assert read_text(data) == counter_text(300)

    @pytest.mark.parametrize("name", fixture_names())
    def test_manifest_describes_the_expected_text(self, name: str) -> None:
        entry = entries()[name]
        encoded = expected_text(name).encode("utf-8")
        assert len(encoded) == entry["utf8_bytes"]
        assert hashlib.sha256(encoded).hexdigest() == entry["sha256"]

    def test_every_required_length_has_a_fixture(self) -> None:
        lengths = {entry["utf8_bytes"] for entry in entries().values()}
        assert lengths >= REQUIRED_BYTE_LENGTHS

    def test_required_kinds_of_text_have_a_fixture(self) -> None:
        assert expected_text("emoji_end").endswith("🎉")
        assert "é" in expected_text("accented")
        assert "寿司" in expected_text("cjk_japanese")
        assert "\n" in expected_text("newline")

    def test_fixtures_come_from_nsarchiver(self) -> None:
        assert manifest()["encoder"] == "NSArchiver.archivedData(withRootObject:)"
        for name in fixture_names():
            assert blob(name).startswith(b"\x04\x0bstreamtyped\x81\xe8\x03")

    @pytest.mark.parametrize(
        ("name", "marker"),
        [
            ("ascii_1", b"\x01"),
            ("ascii_127", b"\x7f"),
            ("ascii_128", b"\x81\x80\x00"),
            ("ascii_255", b"\x81\xff\x00"),
            ("ascii_256", b"\x81\x00\x01"),
            ("ascii_300", b"\x81\x2c\x01"),
            ("ascii_32767", b"\x81\xff\x7f"),
            ("ascii_32768", b"\x82\x00\x80\x00\x00"),
            ("ascii_65535", b"\x82\xff\xff\x00\x00"),
            ("ascii_70000", b"\x82\x70\x11\x01\x00"),
        ],
    )
    def test_apple_writes_the_length_marker_this_way(self, name: str, marker: bytes) -> None:
        assert _length_marker(name).startswith(marker)

    def test_mutable_classes_decode(self) -> None:
        data = blob("mutable_ascii_300")
        assert b"NSMutableAttributedString" in data and b"NSMutableString" in data
        assert decode_attributed_body(data) == counter_text(300)

    def test_multi_byte_text_is_not_cut(self) -> None:
        assert decode_attributed_body(blob("emoji_end_128")).endswith("🎉")
        assert decode_attributed_body(blob("emoji_end_300")).endswith("👩‍👩‍👧‍👦")
        assert decode_attributed_body(blob("cjk_end_256")).endswith("語")

    def test_attachment_placeholder_is_kept_by_the_decoder(self) -> None:
        assert decode_attributed_body(blob("attachment_only")) == chr(0xFFFC)
        assert decode_attributed_body(blob("attachment_with_caption")) == chr(0xFFFC) + "Look at this view"

    def test_empty_string(self) -> None:
        assert decode_attributed_body(blob("empty")) == ""

    @pytest.mark.skipif(
        sys.platform != "darwin" or shutil.which("swift") is None,
        reason="needs the Swift toolchain on macOS",
    )
    def test_swift_script_reproduces_the_committed_fixtures(self) -> None:
        if platform.mac_ver()[0].split(".")[0] != str(manifest()["macos"]).split(".")[0]:
            pytest.skip("fixtures were made on another macOS major version")
        result = subprocess.run(
            ["swift", str(REPO_ROOT / "scripts" / "make_typedstream_fixtures.swift")],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        made = dict(line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line)
        assert sorted(made) == fixture_names()
        for name, hex_text in made.items():
            assert bytes.fromhex(hex_text) == blob(name), f"{name} differs from Apple's encoder output"
        assert sorted(path.stem for path in FIXTURE_DIR.glob("*.hex")) == fixture_names()


class TestIntegerRule:
    @pytest.mark.parametrize(
        ("data", "value", "size"),
        [
            (b"\x00", 0, 1),
            (b"\x05", 5, 1),
            (b"\x7f", 127, 1),
            (b"\x81\x80\x00", 128, 3),
            (b"\x81\x2c\x01", 300, 3),
            (b"\x81\xff\x7f", 32_767, 3),
            (b"\x81\xff\xff", 65_535, 3),
            (b"\x82\x00\x80\x00\x00", 32_768, 5),
            (b"\x82\x70\x11\x01\x00", 70_000, 5),
        ],
    )
    def test_reads_literal_two_byte_and_four_byte_forms(self, data: bytes, value: int, size: int) -> None:
        assert read_integer(data + b"tail", 0) == (value, size)

    def test_reads_at_an_offset(self) -> None:
        assert read_integer(b"xx\x81\x2c\x01", 2) == (300, 5)

    def test_big_endian_streams(self) -> None:
        assert read_integer(b"\x81\x01\x2c", 0, byteorder="big") == (300, 3)

    @pytest.mark.parametrize("head", [0x80, 0x83, 0x84, 0x85, 0x86, 0x92, 0xFF])
    def test_other_tags_are_not_integers(self, head: int) -> None:
        with pytest.raises(TypedStreamError):
            read_integer(bytes([head, 1, 2, 3, 4]), 0)

    @pytest.mark.parametrize("data", [b"", b"\x81", b"\x81\x01", b"\x82\x01\x02\x03"])
    def test_truncated_integers_raise(self, data: bytes) -> None:
        with pytest.raises(TypedStreamError):
            read_integer(data, 0)


class TestBadInput:
    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"\x00",
            b"not a typedstream at all",
            b"\x04\x0bstreamtyped",
            b"\x05\x0bstreamtyped\x81\xe8\x03\x84\x01@",
            b"\x04\x0bstreamtypeX\x81\xe8\x03\x84\x01@",
        ],
    )
    def test_garbage_raises_typed_stream_error(self, data: bytes) -> None:
        with pytest.raises(TypedStreamError):
            decode_attributed_body(data)

    def test_every_cut_of_a_fixture_raises_or_decodes_the_full_text(self) -> None:
        data = blob("two_runs")
        full = expected_text("two_runs")
        text_end = data.index(full.encode()) + len(full.encode())
        for size in range(len(data)):
            cut = data[:size]
            if size >= text_end:
                assert decode_attributed_body(cut) == full
            else:
                with pytest.raises(TypedStreamError):
                    decode_attributed_body(cut)

    def test_length_beyond_the_blob_raises(self) -> None:
        data = bytearray(blob("ascii_300"))
        start = data.index(b"\x84\x01+") + 3
        data[start : start + 3] = b"\x81\xff\x7f"
        with pytest.raises(TypedStreamError):
            decode_attributed_body(bytes(data))

    def test_invalid_utf8_raises(self) -> None:
        data = bytearray(blob("ascii_127"))
        start = data.index(b"\x84\x01+") + 4
        data[start] = 0xFF
        with pytest.raises(TypedStreamError):
            decode_attributed_body(bytes(data))

    def test_another_root_class_is_refused_by_the_structured_read(self) -> None:
        data = blob("ascii_127").replace(b"NSAttributedString", b"NSAttributedStrinX")
        with pytest.raises(TypedStreamError):
            read_text(data)
        assert scan_text(data) == counter_text(127)
        assert decode_attributed_body(data) == counter_text(127)

    def test_accepts_bytearray_and_memoryview(self) -> None:
        data = blob("accented")
        assert decode_attributed_body(bytearray(data)) == expected_text("accented")
        assert decode_attributed_body(memoryview(data)) == expected_text("accented")


class TestTolerantHelper:
    def test_none_and_empty_give_none(self) -> None:
        assert attributed_body_text(None) is None
        assert attributed_body_text(b"") is None

    def test_garbage_gives_none(self) -> None:
        assert attributed_body_text(b"\x00\x01\x02") is None

    def test_good_blob_gives_text(self) -> None:
        assert attributed_body_text(blob("korean")) == expected_text("korean")


class TestLegacyDecoderBug:
    """parse_imessage reads 0x81 as "one length byte follows". Apple writes 0x81 + two bytes."""

    def test_legacy_function_cuts_the_300_byte_fixture(self) -> None:
        from parse_imessage import extract_text_from_attributed_body

        full = counter_text(300)
        legacy = extract_text_from_attributed_body(blob("ascii_300"))
        if legacy == full:
            pytest.skip("the legacy decoder now returns the full text; this test has no bug left to document")
        # 0x81 0x2c 0x01: the legacy code takes 0x2c (44) as the length and starts one byte early
        assert legacy == full[:43]
        assert decode_attributed_body(blob("ascii_300")) == full
