"""Memory use does not grow with the size of the database.

The 200,000-message figures come from scripts/archive_scale_check.py. These
tests keep the same property under watch at a size that runs in seconds.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from archive import Archive, connection, import_chat_db, ledger
from archive import importer as importer_module
from tests.archive.inspect_archive import fts_is_consistent, scalar, totals_match_messages
from tests.archive.synthetic import SyntheticCounts, build_synthetic_chat_db

MIB = 1024 * 1024


def make(folder: Path, messages: int) -> tuple[Path, Path, SyntheticCounts]:
    source = folder / f"source-{messages}" / "chat.db"
    source.parent.mkdir(parents=True)
    counts = build_synthetic_chat_db(source, messages)
    return source, folder / f"archive-{messages}" / "archive.sqlite3", counts


def python_peak_while_importing(source: Path, archive_path: Path) -> int:
    tracemalloc.start()
    try:
        import_chat_db(source, archive_path)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_synthetic_import_finds_every_row(tmp_path: Path) -> None:
    source, archive_path, counts = make(tmp_path, 6000)
    result = import_chat_db(source, archive_path)
    assert result.rows_seen == counts.source_rows
    assert (result.messages_seen, result.reactions_seen, result.system_seen) == (
        counts.messages, counts.reactions, counts.system_rows,
    )  # fmt: skip
    assert (result.attachments_added, result.blob_texts) == (counts.attachments, counts.blob_texts)
    assert (result.conversations_added, result.handles_added) == (counts.conversations, counts.handles)
    assert dict(result.skipped) == {} and result.decode_failures == 0
    assert scalar(archive_path, "SELECT SUM(message_count) FROM conversations") == counts.messages
    assert totals_match_messages(archive_path) and fts_is_consistent(archive_path)
    again = import_chat_db(source, archive_path, full=True)
    assert (again.messages_added, again.messages_updated, again.reactions_added, again.attachments_added) == (0, 0, 0, 0)


def test_python_memory_of_an_import_does_not_grow_with_the_row_count(tmp_path: Path) -> None:
    small_source, small_archive, _ = make(tmp_path, 3000)
    large_source, large_archive, _ = make(tmp_path, 12000)
    small = python_peak_while_importing(small_source, small_archive)
    large = python_peak_while_importing(large_source, large_archive)
    assert large < small * 1.25 + MIB, f"peak grew from {small / MIB:.1f} MiB to {large / MIB:.1f} MiB for 4x the rows"


def test_rows_are_written_in_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, archive_path, counts = make(tmp_path, 2500)
    monkeypatch.setattr(importer_module, "BATCH_ROWS", 400)
    sizes: list[int] = []
    real_write = ledger.write_messages

    def recording_write(conn: Any, run_id: int, rows: Any) -> Any:
        sizes.append(len(rows))
        return real_write(conn, run_id, rows)

    monkeypatch.setattr(ledger, "write_messages", recording_write)
    import_chat_db(source, archive_path)
    assert max(sizes) <= 400
    assert len(sizes) >= counts.source_rows // 400
    assert sum(sizes) == counts.messages + counts.system_rows


def test_streaming_a_conversation_keeps_python_memory_flat(tmp_path: Path) -> None:
    source, archive_path, _ = make(tmp_path, 12000)
    import_chat_db(source, archive_path)
    with Archive.open(archive_path) as archive:
        biggest = archive.conversations(limit=1)[0]
        assert biggest["message_count"] > 2000
        list(archive.iter_messages(biggest["id"], until="2017-01-02"))  # warm up: connection and statement cache
        tracemalloc.start()
        try:
            streamed = sum(1 for _ in archive.iter_messages(biggest["id"]))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert streamed == biggest["message_count"]
    assert peak < 3 * MIB, f"streaming {streamed} messages held {peak / MIB:.1f} MiB"


def test_messages_are_inserted_many_rows_per_statement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One statement for each row makes FTS5 write one index segment for each message."""
    source, archive_path, counts = make(tmp_path, 2500)
    # the trace also fires for each trigger and repeats the text of the statement: a set keeps one of each
    statements: set[str] = set()
    real_open = connection.open_writer

    def traced_open(path: Path) -> Any:
        conn = real_open(path)
        conn.set_trace_callback(lambda sql: statements.add(sql) if sql.startswith("INSERT INTO messages(") else None)
        return conn

    monkeypatch.setattr(connection, "open_writer", traced_open)
    import_chat_db(source, archive_path)
    stored = counts.messages + counts.system_rows
    batches = stored // importer_module.BATCH_ROWS + 1
    assert 0 < len(statements) <= batches * (importer_module.BATCH_ROWS // ledger.ROWS_PER_INSERT + 1)
    assert fts_is_consistent(archive_path)
