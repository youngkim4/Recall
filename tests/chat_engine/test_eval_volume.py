"""The volume run, on an invented export. The real export is never read by a test."""

import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from chat_engine.eval import volume
from chat_engine.eval.synthetic import build_synthetic_archive
from chat_engine.fake_archive import make_message
from tests.chat_engine.frames import frame_of

SYNTHETIC = build_synthetic_archive()
_ALLOWED_LINE = re.compile(r"^[A-Za-z0-9 .,:;()\-]*$")


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    long = "opening remarks about the harbour " + ("lorem ipsum dolor " * 9_000)
    maya = SYNTHETIC.keys["maya_phone"]
    extra = [
        {"message_id": "r1", "timestamp": "2024-03-01 10:00:00", "sender": maya, "chat_id": maya, "is_from_me": "0",
         "text": "Loved “a made-up message”"},
        {"message_id": "r2", "timestamp": "2024-03-01 10:01:00", "sender": maya, "chat_id": maya, "is_from_me": "0",
         "text": "Liked “a text that was cut"},
    ]
    messages = [*SYNTHETIC.messages, make_message("big", maya, "2024-03-02 09:00:00", long, sender=maya)]
    path = tmp_path_factory.mktemp("volume") / "export.csv"
    frame_of(messages, SYNTHETIC.conversations, extra).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def report(export: Path) -> list[str]:
    lines: list[str] = []
    assert volume.main([str(export), "--repeat", "1"], out=lines.append) == 0
    return lines


def test_the_report_holds_numbers_and_fixed_labels_only(report: list[str]) -> None:
    text = "\n".join(report)
    for line in report:
        assert _ALLOWED_LINE.match(line), line
    secrets = [*SYNTHETIC.saved_names, *SYNTHETIC.saved_names.values(), *SYNTHETIC.keys.values()]
    secrets += ["Clint", "Yeastwood", "pinecone", "sourdough", "harbour", "lorem", "Lisbon", "made-up", "example.org"]
    for secret in secrets:
        for part in str(secret).split():
            assert part.lower() not in text.lower(), part
    assert "@" not in text and "+1" not in text


def test_the_report_counts_the_file_and_what_it_leaves_out(report: list[str]) -> None:
    text = "\n".join(report)
    assert f"rows in the file: {len(SYNTHETIC.messages) + 3:,}" in text
    assert f"messages: {len(SYNTHETIC.messages) + 2:,}; conversations: 11" in text
    assert "reaction phrases 1, unreadable rows 0, second rows of one id 0" in text
    assert "do not end as one (kept as messages): 1" in text
    assert "of 100,000 or more: 1; longest: 162,034 characters" in text


def test_every_lookup_runs_and_stays_inside_its_budget(report: list[str]) -> None:
    text = "\n".join(report)
    assert "failed: 0; over the budget: 0" in report[-2]
    assert "OVER THE BUDGET" not in text and "failed:" not in text.replace("failed: 0", "")
    for label in (
        "search word in very many messages", "search any of five words", "read the first messages",
        "page walk to the first messages", "count by sender", "figures of the largest conversation",
        "read around the longest text", "find a conversation by four digits",
    ):
        assert any(line.strip().startswith(label) for line in report), label
    questions = [line for line in report if "ms in all" in line]
    assert len(questions) == 3 and any("1 citations, 1 flagged quotes" in line for line in questions)


def test_a_file_that_is_not_an_export_is_reported_by_the_type_of_the_error_only(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("name,secret\nMaya,+14155550123\n", encoding="utf-8")
    lines: list[str] = []
    assert volume.main([str(path)], out=lines.append) == 1
    assert lines[-1] == "could not read the export: ValueError"
    assert "Maya" not in "\n".join(lines) and "4155550123" not in "\n".join(lines)


def test_the_run_blocks_the_network_first(export: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    @contextmanager
    def blocked() -> Iterator[None]:
        order.append("block")
        yield
        order.append("open")

    monkeypatch.setattr(volume, "network_blocked", blocked)
    def load(_path: str, _invented: int, _out: object) -> None:
        order.append("load")
        raise ValueError

    monkeypatch.setattr(volume, "load", load)
    assert volume.main([str(export)], out=lambda line: None) == 1
    assert order == ["block", "load", "open"]


WAYS_OUT = ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection"), (socket, "getaddrinfo"))


def ways_out_now() -> list[Any]:
    """What each way out is at this moment. None: the class takes it from its base."""
    return [vars(owner).get(name) for owner, name in WAYS_OUT]


def test_inside_the_block_every_way_out_is_stopped() -> None:
    with volume.network_blocked():
        with pytest.raises(RuntimeError):
            socket.create_connection(("example.com", 443), timeout=1)
        with pytest.raises(RuntimeError):
            socket.socket().connect(("127.0.0.1", 9))
        with pytest.raises(RuntimeError):
            socket.socket().connect_ex(("127.0.0.1", 9))
        with pytest.raises(RuntimeError):
            socket.getaddrinfo("example.com", 443)


def test_after_the_block_the_process_is_as_it_was() -> None:
    before = ways_out_now()
    with volume.network_blocked():
        assert all(now is not was for now, was in zip(ways_out_now(), before))
    assert ways_out_now() == before


def test_an_error_inside_the_block_does_not_keep_the_network_shut() -> None:
    before = ways_out_now()
    with pytest.raises(ValueError), volume.network_blocked():
        raise ValueError
    assert ways_out_now() == before


def test_a_way_out_that_the_class_takes_from_its_base_is_given_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(socket.socket, "connect")
    with volume.network_blocked():
        assert "connect" in vars(socket.socket)
    assert "connect" not in vars(socket.socket)


def test_a_run_leaves_the_process_as_it_found_it() -> None:
    before = ways_out_now()
    assert volume.main(["--invented", "600", "--repeat", "1"], out=lambda line: None) == 0
    assert ways_out_now() == before


def test_an_invented_export_has_the_shape_of_a_real_one() -> None:
    from chat_engine.eval.volume_data import invented_export
    from chat_engine.frame_archive import FrameArchive
    from tests.chat_engine.frames import COLUMNS

    frame = invented_export(4_000)
    assert list(frame.columns) == COLUMNS and len(frame) == 4_000
    assert frame.equals(invented_export(4_000))
    archive = FrameArchive(frame)
    dropped = archive.dropped
    assert 60 < dropped["reactions"] < 200 and dropped["bad_rows"] == 0 and dropped["duplicates"] == 0
    assert archive.attachment_only > 40
    lengths = archive.text_lengths()
    assert int((lengths >= 100_000).sum()) >= 8 and int(lengths.max()) > 2_000_000
    listed = archive.conversations()
    assert len(listed) == 10 and any(c["is_group"] for c in listed)
    assert listed[0]["message_count"] > 5 * listed[-1]["message_count"]


def test_the_run_takes_an_invented_export_and_needs_one_source() -> None:
    lines: list[str] = []
    assert volume.main(["--invented", "3000", "--repeat", "1"], out=lines.append) == 0
    assert lines[0] == "export: invented, made-up text only" and "rows in the file: 3,000" in lines[1]
    assert "failed: 0; over the budget: 0" in lines[-2]
    with pytest.raises(SystemExit):
        volume.main([], out=lines.append)
    with pytest.raises(SystemExit):
        volume.main(["some.csv", "--invented", "10"], out=lines.append)
