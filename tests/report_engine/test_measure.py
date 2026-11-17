"""The measurement tool: its numbers are right, and it prints nothing but numbers."""

import os
import re
import socket
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from report_engine import measure
from report_engine.measure_file import (
    check_tapbacks,
    count_rows,
    measure_lengths,
    tapback_flags,
    texts_of,
)

from .synth import GROUP, MINA, RAVI, THEO, WORDS, conversation

OPEN, CLOSE = chr(0x201C), chr(0x201D)
PLACEHOLDER = chr(0xFFFC)
SECRET = "pineapple"


def rows_of(messages: list[Any], chat: str) -> list[dict[str, Any]]:
    return [
        {
            "message_id": int(item["id"]),
            "timestamp": item["timestamp"],
            "sender": item["sender"] or None,
            "text": item["text"],
            "chat_id": chat,
            "is_from_me": int(item["is_from_me"]),
            "service": "iMessage",
        }
        for item in messages
    ]


def odd_row(number: int, text: Any, **fields: Any) -> dict[str, Any]:
    row = {
        "message_id": 9_000 + number,
        "timestamp": f"2024-03-05 10:{number:02d}:00",
        "sender": MINA,
        "text": text,
        "chat_id": MINA,
        "is_from_me": 0,
        "service": "iMessage",
    }
    return {**row, **fields}


def export(folder: Path) -> Path:
    first = conversation("2024-01-01", days=90, per_day=5, words=9)
    group = conversation("2024-02-01", days=4, per_day=3, words=5, first_id=5_000)
    known = first[10]["text"]
    odd = [
        odd_row(1, f"Loved {OPEN}{known}{CLOSE}"),
        odd_row(2, f"Liked {OPEN}{SECRET} that no row holds{CLOSE}"),
        odd_row(3, f"Loved the {SECRET} so much"),
        odd_row(4, PLACEHOLDER),
        odd_row(5, None),
        odd_row(6, f"{SECRET} with no direction", is_from_me=None),
        odd_row(7, f"{SECRET} with no time", timestamp="not a time"),
        odd_row(8, f"{SECRET} " * 1_500),
    ]
    path = folder / "messages.csv"
    pd.DataFrame([*rows_of(first, MINA), *rows_of(group, GROUP), *odd]).to_csv(path, index=False)
    return path


def run_tool(path: Path, *flags: str) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = measure.main(
        [str(path), "--heuristic", "--budgets", "300,600", "--cuts", "5", *flags], lines.append
    )
    return code, lines


def numbers(line: str) -> list[int]:
    return [int(found.replace(",", "")) for found in re.findall(r"\d[\d,]*", line)]


def test_the_row_counts_are_right(tmp_path: Path) -> None:
    frame = measure.load_frame(str(export(tmp_path)))
    texts = texts_of(frame)
    flags = tapback_flags(texts)
    rows = count_rows(frame, texts, flags)
    assert rows.rows == 450 + 12 + 8
    assert (rows.conversations, rows.group_conversations, rows.no_conversation) == (2, 1, 0)
    assert rows.no_timestamp == 1
    assert rows.no_direction == 1
    assert rows.sent + rows.received == rows.rows - 1
    assert (rows.no_id, rows.repeated_ids) == (0, 0)
    assert rows.no_text == 1
    assert rows.placeholder_only == 1
    assert rows.tapbacks == 2
    assert rows.tapback_lookalikes == 1
    check = check_tapbacks(frame, texts, flags)
    assert (check.with_quote, check.found_earlier) == (2, 1)


def test_the_length_table_says_what_a_cap_keeps() -> None:
    lengths = measure_lengths(["a" * 100, "b" * 300, "c" * 5_000, ""])
    assert lengths.characters == 5_400
    assert lengths.longest == 5_000
    by_limit = {step.limit: step for step in lengths.steps}
    assert by_limit[250].rows_over == 2
    assert by_limit[250].share_in_rows_over == pytest.approx(5_300 / 5_400)
    assert by_limit[250].share_kept_by_cap == pytest.approx(600 / 5_400)
    assert by_limit[4_000].rows_over == 1
    assert by_limit[4_000].share_kept_by_cap == pytest.approx(4_400 / 5_400)
    assert by_limit[10_000].rows_over == 0
    assert by_limit[10_000].share_kept_by_cap == 1.0
    assert measure_lengths([]).characters == 0


def test_the_tool_reports_the_file_and_the_largest_conversations(tmp_path: Path) -> None:
    code, lines = run_tool(export(tmp_path), "--top", "2", "--estimates", "1")
    assert code == 0
    text = "\n".join(lines)
    assert "rows 470; conversations 2 (groups 1); rows with no conversation 0" in text
    assert "no timestamp 1; no direction 1; no text 1" in text
    assert "only the attachment placeholder 1; tapbacks 2 (0.43%)" in text
    assert "rows that start like one and are kept 1; with quoted text 2, of which" in text
    assert "tokens counted with heuristic" in text
    first = next(line for line in lines if line.startswith("conversation 1:"))
    # 450 messages, the kept look-alike and the long row; the long row is cut
    assert numbers(first)[1:4] == [458, 452, 1]
    assert next(line for line in lines if line.startswith("conversation 2:"))
    budget_lines = [line for line in lines if line.lstrip().startswith("budget")]
    assert len(budget_lines) == 4
    assert all("over the budget 0; cut points not stable 0" in line for line in budget_lines)
    assert sum(line.startswith("estimate ") for line in lines) == 1
    totals = next(line for line in lines if line.startswith("all conversations:"))
    # two conversations; 452 and 12 lines; one line cut
    assert numbers(totals)[:3] == [2, 464, 1]
    assert "not read, by type of error: none" in text
    at_budget = [line for line in lines if line.lstrip().startswith("at budget")]
    assert [numbers(line)[0] for line in at_budget] == [300, 600]
    # every conversation is in exactly one of the five groups of a budget line
    for line in at_budget:
        found = numbers(line)
        assert found[3] + found[6] + found[9] + found[12] + found[14] == 2
        assert found[1] >= 2


def test_a_smaller_budget_gives_more_periods(tmp_path: Path) -> None:
    _, lines = run_tool(export(tmp_path), "--top", "1", "--estimates", "0")
    small, large = (numbers(line) for line in lines if line.lstrip().startswith("budget"))
    assert small[0] == 300 and large[0] == 600
    assert small[1] > large[1] > 1


def test_the_output_holds_no_word_of_a_message_no_name_and_no_handle(tmp_path: Path) -> None:
    code, lines = run_tool(export(tmp_path), "--top", "2", "--estimates", "2")
    assert code == 0
    text = "\n".join(lines).lower()
    for private in (*WORDS, SECRET, MINA, THEO, RAVI, GROUP, "mina", PLACEHOLDER):
        assert private.lower() not in text, private


def test_an_error_is_named_by_its_type_and_its_text_is_not_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"row 7 holds: {SECRET}")

    monkeypatch.setattr(measure, "count_rows", broken)
    code, lines = run_tool(export(tmp_path))
    assert code == 1
    assert lines[-1].startswith("stopped: RuntimeError at ")
    assert SECRET not in "\n".join(lines)


def test_a_file_that_is_not_an_export_stops_the_tool(tmp_path: Path) -> None:
    code, lines = run_tool(tmp_path / "missing.csv")
    assert code == 1
    assert lines == [next(line for line in lines if line.startswith("stopped: FileNotFoundError"))]
    other = tmp_path / "other.csv"
    pd.DataFrame({"text": [SECRET]}).to_csv(other, index=False)
    code, lines = run_tool(other)
    assert code == 1
    assert lines[-1].startswith("stopped: ValueError at ")


def test_a_conversation_that_the_engine_can_not_read_is_counted_by_error_type(
    tmp_path: Path,
) -> None:
    rows = [odd_row(n, f"{SECRET} {n}", is_from_me=None, chat_id=THEO) for n in range(1, 4)]
    path = tmp_path / "messages.csv"
    pd.DataFrame([*rows_of(conversation("2024-01-01", days=3), MINA), *rows]).to_csv(
        path, index=False
    )
    code, lines = run_tool(path)
    assert code == 0
    assert "not read, by type of error: SourceError 1" in lines
    assert any(line.startswith("all conversations: read 1;") for line in lines)


def test_a_conversation_with_no_words_is_counted_and_not_shown(tmp_path: Path) -> None:
    rows = [odd_row(n, PLACEHOLDER, chat_id=THEO) for n in range(1, 4)]
    path = tmp_path / "messages.csv"
    pd.DataFrame([*rows_of(conversation("2024-01-01", days=3), MINA), *rows]).to_csv(
        path, index=False
    )
    code, lines = run_tool(path)
    assert code == 0
    assert "not read, by type of error: no message with text 1" in lines
    assert sum(line.startswith("conversation ") for line in lines) == 1


def test_the_tool_runs_as_a_module_and_prints_numbers(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "report_engine.measure",
        str(export(tmp_path)),
        "--heuristic",
        "--budgets",
        "600",
        "--top",
        "1",
        "--estimates",
        "0",
    ]
    done = subprocess.run(  # noqa: S603 - our own module, with a synthetic file
        command,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(measure.__file__).resolve().parents[1],
        env={**os.environ, "OPENAI_API_KEY": "dummy-not-a-key"},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-400:]
    assert "rows 470; conversations 2" in done.stdout
    assert SECRET not in done.stdout and SECRET not in done.stderr


def test_a_row_with_no_conversation_does_not_stop_the_tool(tmp_path: Path) -> None:
    rows = [
        *rows_of(conversation("2024-01-01", days=3), MINA),
        odd_row(1, f"{SECRET} one", chat_id=None),
        odd_row(2, f"Loved {OPEN}{SECRET}{CLOSE}", chat_id=None),
    ]
    path = tmp_path / "messages.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    code, lines = run_tool(path)
    assert code == 0
    assert any("rows with no conversation 2" in line for line in lines)
    assert any(line.startswith("all conversations: read 1;") for line in lines)


def test_no_warning_of_a_library_reaches_the_output(tmp_path: Path) -> None:
    """A library can quote a cell of the file in a warning. The tool is run on
    private data and all of its output is passed on, so it shows no warning."""
    rows = rows_of(conversation("2024-01-01", days=3), MINA)
    rows.append(odd_row(1, "hello", timestamp=f"2024-01-01 10:00:00 {SECRET[:3].upper()}"))
    rows.append(odd_row(2, "hello", timestamp="2024-01-01T10:00:00-05:00"))
    path = tmp_path / "messages.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code, lines = run_tool(path)
    assert code == 0
    assert caught == []
    done = subprocess.run(  # noqa: S603 - our own module, with a synthetic file
        [sys.executable, "-m", "report_engine.measure", str(path), "--heuristic", "--top", "0"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(measure.__file__).resolve().parents[1],
        env={**os.environ, "OPENAI_API_KEY": "dummy-not-a-key"},
        check=False,
    )
    assert done.returncode == 0
    assert done.stderr == ""
    assert SECRET[:3].upper() not in done.stdout


def test_an_error_in_one_conversation_is_counted_and_the_others_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = measure.read_sized

    def fails_for_the_group(frame: Any, chat: str, meter: Any) -> Any:
        if chat == GROUP:
            raise RuntimeError(f"row 3 holds: {SECRET}")
        return real(frame, chat, meter)

    monkeypatch.setattr(measure, "read_sized", fails_for_the_group)
    code, lines = run_tool(export(tmp_path), "--top", "2")
    assert code == 0
    assert "not read, by type of error: RuntimeError 1" in lines
    assert any(line.startswith("all conversations: read 1;") for line in lines)
    assert SECRET not in "\n".join(lines)


def test_the_tool_says_when_it_counts_without_the_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(measure, "load_counter", lambda family: measure.HeuristicCounter())
    lines: list[str] = []
    code = measure.main([str(export(tmp_path)), "--top", "0", "--budgets", "600"], lines.append)
    assert code == 0
    assert any("the tokenizer did not load" in line for line in lines)
    _, quiet = run_tool(export(tmp_path), "--top", "0")
    assert not any("the tokenizer did not load" in line for line in quiet)


def test_the_tool_can_shut_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))
    for name in measure.LOOKUPS:
        monkeypatch.setattr(socket, name, getattr(socket, name))
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    measure.block_network()
    with pytest.raises(OSError), socket.socket() as probe:
        probe.connect(("127.0.0.1", 9))
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9))
    with pytest.raises(OSError):
        socket.getaddrinfo("localhost", 80)
    with pytest.raises(OSError):
        socket.gethostbyname("localhost")
