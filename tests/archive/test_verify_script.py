"""scripts/verify_archive_import.py prints numbers only. Run here on synthetic data."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.archive import worlds
from tests.archive.builders import ChatDbBuilder, utc
from tests.archive.legacy import REPO_ROOT
from tests.archive.typedstream_fixtures import blob, expected_text

SCRIPT = REPO_ROOT / "scripts" / "verify_archive_import.py"
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
NUMBER_LINE = re.compile(r"^  [a-z0-9 ,()]+: \d+(\.\d+)?$")
HEADER_LINE = re.compile(r"^[a-z ()]+$")


@pytest.fixture
def source(builder: ChatDbBuilder, source_path: Path) -> Path:
    rowan = builder.handle(worlds.ROWAN_PHONE)
    chat = builder.chat(worlds.ROWAN_PHONE, members=[rowan])
    first = builder.message(chat, when=utc(2024, 1, 1, 12, 0), text="Private words in the text column", handle=rowan)
    for index, name in enumerate(BLOB_ROWS):
        builder.message(chat, when=utc(2024, 1, 1, 12, 1 + index), body=blob(name), handle=rowan)
    builder.tapback(chat, builder.guid_of(first), 2000, when=utc(2024, 1, 1, 13, 0), handle=rowan)
    builder.message(chat, when=utc(2024, 1, 1, 13, 1), handle=rowan, item_type=2, group_title="Secret Title")
    builder.message(chat, when=utc(2024, 1, 1, 13, 2), handle=rowan)
    builder.close()
    return source_path


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False
    )


def numbers(output: str) -> dict[str, float]:
    section = ""
    found: dict[str, float] = {}
    for line in output.splitlines():
        if not line.startswith("  "):
            section = line.strip()
            continue
        label, value = line.strip().rsplit(": ", 1)
        found[f"{section} / {label}"] = float(value)
    return found


def test_prints_the_expected_numbers(source: Path) -> None:
    result = run_script("--db", str(source))
    assert result.returncode == 0, result.stderr
    found = numbers(result.stdout)
    assert found["import / rows seen"] == 13
    assert found["import / messages"] == 10
    assert found["import / tapbacks"] == 1
    assert found["import / system rows"] == 1
    assert found["import / text from the blob"] == 9
    assert found["import / blobs not readable"] == 0
    assert found["import / rows not stored, empty"] == 1
    check = "decoder check (rows whose text is only in the blob)"
    assert found[f"{check} / legacy decoder loaded"] == 1
    assert found[f"{check} / rows checked"] == 9
    assert found[f"{check} / read by structure"] == 9
    assert found[f"{check} / read by marker scan"] == 0
    assert found[f"{check} / not readable"] == 0
    # the two decoders give the same length for each row: the old one reads long texts in full now
    assert found[f"{check} / under 128 bytes, rows"] == 4
    assert found[f"{check} / under 128 bytes, legacy length differs"] == 0
    # `tab_and_spaces`: the script trims the text as the old decoder did before. That decoder does not trim now.
    assert found[f"{check} / under 128 bytes, differs after the old cleanup"] == 1
    assert found[f"{check} / 128 to 255 bytes, rows"] == 3
    assert found[f"{check} / 128 to 255 bytes, legacy length differs"] == 0
    assert found[f"{check} / 128 to 255 bytes, differs after the old cleanup"] == 0
    assert found[f"{check} / 256 bytes and more, rows"] == 2
    assert found[f"{check} / 256 bytes and more, legacy length differs"] == 0
    assert found[f"{check} / 256 bytes and more, differs after the old cleanup"] == 0


def test_output_is_numbers_only(source: Path) -> None:
    result = run_script("--db", str(source))
    assert result.returncode == 0
    assert result.stderr == ""
    for line in result.stdout.splitlines():
        assert NUMBER_LINE.match(line) or HEADER_LINE.match(line), line
    private = ["Private", "Secret", "Dinner", worlds.ROWAN_PHONE, "5550100", "0000", "Loved", "invented"]
    private += [expected_text(name)[:12] for name in ("korean", "tab_and_spaces")]
    for needle in private:
        assert needle.strip() not in result.stdout, needle


def test_source_is_not_changed_and_nothing_is_left_behind(source: Path) -> None:
    before = source.read_bytes()
    listed = sorted(path.name for path in source.parent.iterdir())
    assert run_script("--db", str(source)).returncode == 0
    assert source.read_bytes() == before
    assert sorted(path.name for path in source.parent.iterdir()) == listed


def test_missing_database_fails_without_a_traceback(tmp_path: Path) -> None:
    result = run_script("--db", str(tmp_path / "nowhere.db"))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert "SourceError" in result.stderr
