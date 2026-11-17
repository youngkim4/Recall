"""The two decoders of the joined branch read the same text.

The archive has its own decoder of attributedBody (archive/typedstream.py).
The exporter has the decoder of parse_imessage.py, which the decoder fix
corrected. scripts/verify_archive_import.py compares the two on a database.

The first test here holds the numbers that the script prints after the decoder
fix, for the nine fixtures of tests/archive/test_verify_script.py and a
database built the same way. That test expects the same numbers.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from archive.errors import TypedStreamError
from archive.typedstream import decode_attributed_body
from parse_imessage import extract_text_from_attributed_body
from tests.archive import typedstream_fixtures as archive_set
from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, utc
from tests.integration import bridge_data as data

SCRIPT = data.REPO_ROOT / "scripts" / "verify_archive_import.py"
# the rows of tests/archive/test_verify_script.py, from the fixtures of the archive tests
BLOB_ROWS = (
    "ascii_127",
    "korean",
    "part_attribute",
    "tab_and_spaces",
    "ascii_128",
    "ascii_255",
    "emoji_end_130",
    "ascii_256",
    "ascii_300",
)
CHECK = "decoder check (rows whose text is only in the blob)"
APPLE = data.apple_fixtures()  # the fixtures of the decoder fix
# the fixtures that the two decoders read on purpose in two ways: each has a test of its own
APPLE_SPECIAL = ("control_chars", "empty", "not_text")
NO_WAY_OUT = {name: "http://127.0.0.1:9" for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """The database of the archive test: one text column row, nine blob rows, a tapback, an event, an empty row."""
    path = tmp_path / "source" / "chat.db"
    path.parent.mkdir()
    builder = ChatDbBuilder(path)
    try:
        rowan = builder.handle(worlds.ROWAN_PHONE)
        chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
        first = builder.message(chat, when=utc(2024, 1, 1, 12, 0), text="Words in the text column", handle=rowan)
        for index, name in enumerate(BLOB_ROWS):
            builder.message(chat, when=utc(2024, 1, 1, 12, 1 + index), body=archive_set.blob(name), handle=rowan)
        builder.tapback(chat, builder.guid_of(first), 2000, when=utc(2024, 1, 1, 13, 0), handle=rowan)
        builder.message(chat, when=utc(2024, 1, 1, 13, 1), handle=rowan, item_type=2, group_title="A new title")
        builder.message(chat, when=utc(2024, 1, 1, 13, 2), handle=rowan)
    finally:
        builder.close()
    return path


def run_script(database: Path) -> subprocess.CompletedProcess[str]:
    """Run the script as the owner does. It reads the invented database and calls no service."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(database)],
        cwd=data.REPO_ROOT,
        env={**os.environ, **NO_WAY_OUT},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def numbers(output: str) -> dict[str, float]:
    """`section / label` -> the number of that line."""
    section = ""
    found: dict[str, float] = {}
    for line in output.splitlines():
        if not line.startswith("  "):
            section = line.strip()
            continue
        label, value = line.strip().rsplit(": ", 1)
        found[f"{section} / {label}"] = float(value)
    return found


def test_the_script_prints_the_numbers_of_the_joined_branch(source: Path) -> None:
    result = run_script(source)
    assert (result.returncode, result.stderr) == (0, "")
    found = numbers(result.stdout)
    # the import: the same numbers as before the decoder fix
    assert found["import / rows seen"] == 13
    assert found["import / messages"] == 10
    assert found["import / tapbacks"] == 1
    assert found["import / system rows"] == 1
    assert found["import / text from the blob"] == 9
    assert found["import / blobs not readable"] == 0
    assert found["import / rows not stored, empty"] == 1
    assert found[f"{CHECK} / legacy decoder loaded"] == 1
    assert found[f"{CHECK} / rows checked"] == 9
    assert found[f"{CHECK} / read by structure"] == 9
    assert found[f"{CHECK} / read by marker scan"] == 0
    assert found[f"{CHECK} / not readable"] == 0
    # the decoder check: before the decoder fix the two lines after `rows` were 1, 0 / 3, 3 / 2, 2
    assert found[f"{CHECK} / under 128 bytes, rows"] == 4
    assert found[f"{CHECK} / under 128 bytes, legacy length differs"] == 0
    # `tab_and_spaces`: the script trims the text as the old decoder did. The decoder of today does not trim.
    assert found[f"{CHECK} / under 128 bytes, differs after the old cleanup"] == 1
    assert found[f"{CHECK} / 128 to 255 bytes, rows"] == 3
    assert found[f"{CHECK} / 128 to 255 bytes, legacy length differs"] == 0
    assert found[f"{CHECK} / 128 to 255 bytes, differs after the old cleanup"] == 0
    assert found[f"{CHECK} / 256 bytes and more, rows"] == 2
    assert found[f"{CHECK} / 256 bytes and more, legacy length differs"] == 0
    assert found[f"{CHECK} / 256 bytes and more, differs after the old cleanup"] == 0


def test_the_script_names_no_text_of_the_database(source: Path) -> None:
    output = run_script(source).stdout
    for private in ("Words", "title", worlds.ROWAN_PHONE, "Dinner", "Loved"):
        assert private not in output


@pytest.mark.parametrize("name", [name for name in APPLE if name not in APPLE_SPECIAL])
def test_both_decoders_read_each_fixture_of_the_decoder_fix(name: str) -> None:
    fixture = APPLE[name]
    text = decode_attributed_body(fixture.blob)
    assert extract_text_from_attributed_body(fixture.blob) == text
    raw = text.encode("utf-8")
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (fixture.utf8_bytes, fixture.sha256)
    if fixture.text is not None:
        assert text == fixture.text


@pytest.mark.parametrize("name", [name for name in archive_set.fixture_names() if name != "empty"])
def test_both_decoders_read_each_fixture_of_the_archive_tests(name: str) -> None:
    blob = archive_set.blob(name)
    text = decode_attributed_body(blob)
    assert extract_text_from_attributed_body(blob) == text
    assert text == archive_set.expected_text(name)


def test_the_two_sets_of_fixtures_name_every_file_they_hold() -> None:
    assert set(APPLE) == {path.stem for path in data.APPLE_FIXTURE_DIR.glob("*.hex")}
    assert set(archive_set.fixture_names()) == {path.stem for path in archive_set.FIXTURE_DIR.glob("*.hex")}
    assert len(APPLE) == 32 and len(archive_set.fixture_names()) == 39


def test_control_characters_are_kept_by_the_archive_and_dropped_by_the_exporter() -> None:
    """A difference on purpose. The exporter drops C0 controls other than tab and newline.

    parse_imessage.py, `_DROPPED_CONTROLS`: "not part of a message, and NUL breaks the
    CSV". README, "Decoder": the archive's decoder "trims nothing and drops nothing".
    """
    blob = APPLE["control_chars"].blob
    assert decode_attributed_body(blob) == APPLE["control_chars"].text == "a\x00b\x01c\x1fd\te\nf\rg\x7fh"
    assert extract_text_from_attributed_body(blob) == "abcd\te\nfg\x7fh"


@pytest.mark.parametrize("blob", [APPLE["empty"].blob, archive_set.blob("empty")], ids=["decoder fix", "archive tests"])
def test_an_empty_text_is_empty_for_the_archive_and_nothing_for_the_exporter(blob: bytes) -> None:
    """A difference on purpose. Both callers treat the two results as "no text"."""
    assert decode_attributed_body(blob) == ""
    assert extract_text_from_attributed_body(blob) is None


def test_an_archive_with_no_string_is_an_error_for_the_archive_and_nothing_for_the_exporter() -> None:
    """A difference on purpose. README, "Decoder": data that is not such a typedstream raises TypedStreamError."""
    blob = APPLE["not_text"].blob
    with pytest.raises(TypedStreamError):
        decode_attributed_body(blob)
    assert extract_text_from_attributed_body(blob) is None
