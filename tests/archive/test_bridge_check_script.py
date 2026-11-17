"""scripts/archive_bridge_check.py prints numbers only. Run here on a synthetic export."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from tests.archive.legacy import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "archive_bridge_check.py"
COLUMNS = (
    "message_id",
    "timestamp",
    "sender",
    "text",
    "chat_id",
    "chat_display_name",
    "is_from_me",
    "service",
    "attachment_types",
    "attachment_files",
)
SENDER = "+15550100191"
GROUP_TITLE = "Lantern makers"
PRIVATE_WORDS = ("thermos", "lighthouse", "Saturday")


def write_export(path: Path, count: int) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for index in range(count):
            word = PRIVATE_WORDS[index % len(PRIVATE_WORDS)]
            stamp = f"2024-05-{1 + index % 9:02d} 10:{index % 60:02d}:00"
            chat = SENDER if index % 2 else "chat900900"
            title = "" if index % 2 else GROUP_TITLE
            writer.writerow(
                [
                    index + 1,
                    stamp,
                    SENDER,
                    f"the {word} happy birthday {index}",
                    chat,
                    title,
                    index % 3 == 0,
                    "iMessage",
                    "",
                    "",
                ]
            )
    return path


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False
    )


def test_the_output_holds_numbers_and_fixed_labels_only(tmp_path: Path) -> None:
    export = write_export(tmp_path / "messages.csv", 60)

    done = run_script(str(export))

    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    for private in (*PRIVATE_WORDS, SENDER, GROUP_TITLE, "chat900900", str(tmp_path)):
        assert private not in done.stdout
    lines = done.stdout.splitlines()
    assert "  rows_seen: 60" in lines
    assert "  messages: 60" in lines
    assert any(line.startswith("  search two words: ") and line.endswith("(rows or hits: 60)") for line in lines)
    assert any(line.startswith("  messages(around=...): ") for line in lines)
    assert "  bridge rows: 60" in lines


def test_a_failure_prints_the_type_of_the_error_only(tmp_path: Path) -> None:
    missing = tmp_path / "private-folder-name" / "messages.csv"

    done = run_script(str(missing))

    assert done.returncode == 1
    assert done.stdout == ""
    assert done.stderr.strip() == "failed: SourceError"


def test_an_export_with_no_usable_row_ends_cleanly(tmp_path: Path) -> None:
    export = tmp_path / "messages.csv"
    with export.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(COLUMNS)

    done = run_script(str(export))

    assert done.returncode == 0, done.stderr
    assert "  rows_seen: 0" in done.stdout.splitlines()
