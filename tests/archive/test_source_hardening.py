"""The copy of the source database: one state of the source, nothing new in its folder, a private work folder.

Every database here is synthetic and lives in tmp_path.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import stat
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from archive import source
from archive.errors import SourceError
from tests.archive.builders import ChatDbBuilder, utc

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)
LIVE_WAL_FILES = ["chat.db", "chat.db-shm", "chat.db-wal"]
COPY_NAME = "source-test.sqlite3"


def names(folder: Path) -> list[str]:
    return sorted(entry.name for entry in folder.iterdir())


def add_messages(builder: ChatDbBuilder, count: int, label: str) -> list[str]:
    """Add one chat with `count` text messages and commit. Returns the guids."""
    handle = builder.handle(f"{label}@example.com")
    chat = builder.chat(f"{label}@example.com", members=[handle])
    rowids = [
        builder.message(chat, when=utc(2024, 3, 1, 12, index // 60, index % 60), text=f"{label} {index}", handle=handle)
        for index in range(count)
    ]
    builder.commit()
    return [builder.guid_of(rowid) for rowid in rowids]


def read_guids(db: Path, work: Path, on_step: source.CopyStep | None = None) -> list[str]:
    """The guids in a snapshot of `db`, sorted."""
    with source.open_snapshot(db, work, on_step) as reader:
        return sorted(str(row.guid) for batch in reader.messages_after(0, 500) for row in batch)


def rows_in_main_file_alone(db: Path, scratch: Path) -> int:
    scratch.mkdir()
    alone = scratch / "alone.db"
    shutil.copy(db, alone)
    conn = sqlite3.connect(str(alone))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM message").fetchone()[0])
    finally:
        conn.close()


def new_work_folder(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    return work


def watch_signature(monkeypatch: pytest.MonkeyPatch, changed: Callable[[int], bool]) -> list[int]:
    """Count the calls of the signature helper. A call that `changed` picks reports a later write of the main file."""
    real = source._signature
    calls: list[int] = []

    def fake(path: Path) -> tuple[source.FileState, ...]:
        calls.append(len(calls) + 1)
        found = real(path)
        main = found[0]
        if main is None or not changed(calls[-1]):
            return found
        return ((main[0], main[1], main[2] + calls[-1]), *found[1:])

    monkeypatch.setattr(source, "_signature", fake)
    return calls


def always(call: int) -> bool:
    return True


def never(call: int) -> bool:
    return False


def count_locked_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    real = source._backup
    seen: list[Path] = []

    def backup(path: Path, target: Path, on_step: source.CopyStep | None) -> None:
        seen.append(path)
        real(path, target, on_step)

    monkeypatch.setattr(source, "_backup", backup)
    return seen


@dataclass(frozen=True)
class Leftover:
    """chat.db with a `-wal` and no `-shm`: what a writer leaves when it is killed."""

    path: Path
    in_main: tuple[str, ...]
    in_wal: tuple[str, ...]

    @property
    def guids(self) -> list[str]:
        return sorted(self.in_main + self.in_wal)

    @property
    def folder(self) -> Path:
        return self.path.parent


@pytest.fixture
def leftover(tmp_path: Path) -> Leftover:
    folder = tmp_path / "source"
    folder.mkdir()
    writer = ChatDbBuilder(tmp_path / "writer.db", wal=True)
    writer.raw.execute("PRAGMA wal_autocheckpoint=0")
    in_main = add_messages(writer, 6, "main")
    writer.raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    in_wal = add_messages(writer, 4, "wal")
    shutil.copy(writer.path, folder / "chat.db")
    shutil.copy(f"{writer.path}-wal", folder / "chat.db-wal")
    writer.close()
    assert names(folder) == ["chat.db", "chat.db-wal"]
    assert rows_in_main_file_alone(folder / "chat.db", tmp_path / "alone") == len(in_main)
    return Leftover(folder / "chat.db", tuple(in_main), tuple(in_wal))


@pytest.fixture
def live(tmp_path: Path) -> Iterator[ChatDbBuilder]:
    """A WAL database that a writer keeps open. Its `-wal` and `-shm` exist."""
    folder = tmp_path / "source"
    folder.mkdir()
    builder = ChatDbBuilder(folder / "chat.db", wal=True)
    builder.raw.execute("PRAGMA wal_autocheckpoint=0")
    try:
        yield builder
    finally:
        builder.close()


def quiet_wal_source(tmp_path: Path) -> Path:
    """A WAL database that nothing has open: no side file."""
    folder = tmp_path / "source"
    folder.mkdir()
    builder = ChatDbBuilder(folder / "chat.db", wal=True)
    add_messages(builder, 3, "quiet")
    builder.close()
    assert names(folder) == ["chat.db"] and (folder / "chat.db").read_bytes()[18:20] == b"\x02\x02"
    return folder / "chat.db"


def rollback_source(tmp_path: Path, count: int = 3) -> tuple[Path, list[str]]:
    """A database with a rollback journal that nothing has open."""
    folder = tmp_path / "source"
    folder.mkdir()
    builder = ChatDbBuilder(folder / "chat.db")
    guids = add_messages(builder, count, "plain")
    builder.close()
    assert names(folder) == ["chat.db"] and (folder / "chat.db").read_bytes()[18:20] == b"\x01\x01"
    return folder / "chat.db", guids


class TestOneStateOfTheSource:
    def test_symlinked_database_is_read_with_the_rows_of_its_wal(self, leftover: Leftover, tmp_path: Path) -> None:
        links = tmp_path / "links"
        links.mkdir()
        link = links / "chat.db"
        link.symlink_to(leftover.path)
        before = names(leftover.folder), names(links)
        assert read_guids(link, tmp_path / "work") == leftover.guids
        assert (names(leftover.folder), names(links)) == before == (["chat.db", "chat.db-wal"], ["chat.db"])

    def test_leftover_wal_is_read_and_the_source_folder_stays_as_it_is(
        self, leftover: Leftover, tmp_path: Path
    ) -> None:
        files = {path.name: path.read_bytes() for path in leftover.folder.iterdir()}
        assert read_guids(leftover.path, tmp_path / "work") == leftover.guids
        assert {path.name: path.read_bytes() for path in leftover.folder.iterdir()} == files
        assert not (tmp_path / "work").exists()

    def test_live_wal_database_is_read_with_the_rows_of_its_wal(self, live: ChatDbBuilder, tmp_path: Path) -> None:
        guids = add_messages(live, 5, "live")
        assert names(live.path.parent) == LIVE_WAL_FILES
        assert read_guids(live.path, tmp_path / "work") == sorted(guids)
        assert names(live.path.parent) == LIVE_WAL_FILES

    def test_a_write_during_the_copy_makes_a_second_pass(self, live: ChatDbBuilder, tmp_path: Path) -> None:
        guids = add_messages(live, 5, "early")
        steps: list[tuple[int, int]] = []

        def write_after_the_first_step(done: int, total: int) -> None:
            steps.append((done, total))
            if len(steps) == 1:
                guids.extend(add_messages(live, 3, "late"))

        assert read_guids(live.path, tmp_path / "work", write_after_the_first_step) == sorted(guids)
        totals = sorted({total for _, total in steps})
        assert len(totals) == 2  # the second pass copies a longer WAL
        assert steps[-1] == (totals[-1], totals[-1])
        assert names(live.path.parent) == LIVE_WAL_FILES

    def test_unfinished_transaction_of_a_rollback_database_is_not_in_the_copy(self, tmp_path: Path) -> None:
        folder = tmp_path / "source"
        folder.mkdir()
        writer = ChatDbBuilder(folder / "chat.db")
        try:
            committed = add_messages(writer, 300, "kept")
            clean = writer.path.read_bytes()
            writer.raw.execute("PRAGMA cache_size=5")
            writer.raw.execute("UPDATE message SET text = text || ?", ("x" * 400,))  # not committed
            in_flight = writer.path.read_bytes()
            assert in_flight != clean and (folder / "chat.db-journal").stat().st_size > 0
            with source.open_snapshot(writer.path, tmp_path / "work") as reader:
                rows = [row for batch in reader.messages_after(0, 500) for row in batch]
            assert sorted(str(row.guid) for row in rows) == sorted(committed)
            assert all(len(row.text or "") < 400 for row in rows)
            assert names(folder) == ["chat.db", "chat.db-journal"] and writer.path.read_bytes() == in_flight
        finally:
            writer.raw.rollback()
            writer.close()
        assert not (tmp_path / "work").exists()


class TestSourceThatChanges:
    def test_a_change_during_the_first_pass_is_copied_again(
        self, leftover: Leftover, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = watch_signature(monkeypatch, lambda call: call == 2)
        locked = count_locked_reads(monkeypatch)
        assert read_guids(leftover.path, tmp_path / "work") == leftover.guids
        assert calls == [1, 2, 3, 4] and locked == []
        assert names(leftover.folder) == ["chat.db", "chat.db-wal"]

    @pytest.mark.parametrize("state", ["quiet wal", "wal, no shm", "shm, no wal"])
    def test_no_stable_pass_and_no_safe_locked_read_is_an_error(
        self, state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
    ) -> None:
        if state == "quiet wal":
            db = quiet_wal_source(tmp_path)
        elif state == "wal, no shm":
            db = request.getfixturevalue("leftover").path
        else:
            builder = request.getfixturevalue("live")
            add_messages(builder, 3, "live")
            db = tmp_path / "copied" / "chat.db"
            db.parent.mkdir()
            shutil.copy(builder.path, db)
            shutil.copy(f"{builder.path}-shm", f"{db}-shm")
        before = names(db.parent)
        work = new_work_folder(tmp_path)
        calls = watch_signature(monkeypatch, always)
        locked = count_locked_reads(monkeypatch)
        with pytest.raises(SourceError, match="keeps changing"):
            source.copy_source(db, work / COPY_NAME)
        assert len(calls) == 6 and locked == []  # three passes, then no locked read
        assert names(db.parent) == before
        assert names(work) == []

    def test_no_stable_pass_of_a_snapshot_removes_the_work_folder(
        self, leftover: Leftover, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        watch_signature(monkeypatch, always)
        with pytest.raises(SourceError, match="keeps changing"):
            read_guids(leftover.path, tmp_path / "work")
        assert not (tmp_path / "work").exists()
        assert names(leftover.folder) == ["chat.db", "chat.db-wal"]

    def test_no_stable_pass_of_a_live_wal_database_is_read_with_locks(
        self, live: ChatDbBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        guids = add_messages(live, 5, "live")
        calls = watch_signature(monkeypatch, always)
        locked = count_locked_reads(monkeypatch)
        assert names(live.path.parent) == LIVE_WAL_FILES
        assert read_guids(live.path, tmp_path / "work") == sorted(guids)
        assert len(calls) == 6 and locked == [live.path.resolve()]
        assert names(live.path.parent) == LIVE_WAL_FILES
        assert not (tmp_path / "work").exists()

    def test_no_stable_pass_of_a_rollback_database_is_read_with_locks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, guids = rollback_source(tmp_path)
        watch_signature(monkeypatch, always)
        locked = count_locked_reads(monkeypatch)
        assert read_guids(db, tmp_path / "work") == sorted(guids)
        assert locked == [db.resolve()]
        assert names(db.parent) == ["chat.db"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
    def test_a_side_file_that_cannot_be_read_is_an_error_after_three_passes(
        self, leftover: Leftover, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wal = leftover.folder / "chat.db-wal"
        work = new_work_folder(tmp_path)
        calls = watch_signature(monkeypatch, never)
        wal.chmod(0)
        try:
            with pytest.raises(SourceError, match="cannot copy") as caught:
                source.copy_source(leftover.path, work / COPY_NAME)
        finally:
            wal.chmod(0o600)
        assert isinstance(caught.value.__cause__, PermissionError)
        assert len(calls) == 3
        assert names(work) == [] and names(leftover.folder) == ["chat.db", "chat.db-wal"]

    def test_a_side_file_that_is_gone_at_the_copy_counts_as_a_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = quiet_wal_source(tmp_path)
        real = source._signature
        monkeypatch.setattr(source, "_signature", lambda path: (real(path)[0], (1, 10, 1), None))  # a `-wal` was there
        work = new_work_folder(tmp_path)
        with pytest.raises(SourceError, match="keeps changing") as caught:
            source.copy_source(db, work / COPY_NAME)
        assert isinstance(caught.value.__cause__, FileNotFoundError)
        assert names(work) == [] and names(db.parent) == ["chat.db"]

    def test_a_linked_wal_is_not_followed(self, leftover: Leftover, tmp_path: Path) -> None:
        real_wal = tmp_path / "elsewhere-wal"
        (leftover.folder / "chat.db-wal").rename(real_wal)
        (leftover.folder / "chat.db-wal").symlink_to(real_wal)
        work = new_work_folder(tmp_path)
        with pytest.raises(SourceError):
            source.copy_source(leftover.path, work / COPY_NAME)
        assert names(work) == [] and names(leftover.folder) == ["chat.db", "chat.db-wal"]


class TestHeader:
    @pytest.mark.parametrize("size", [0, 16, 99])
    def test_a_file_with_less_than_a_full_header_is_not_a_database(self, size: int, tmp_path: Path) -> None:
        db, _ = rollback_source(tmp_path)
        cut = tmp_path / "cut.db"
        cut.write_bytes(db.read_bytes()[:size])
        work = new_work_folder(tmp_path)
        with pytest.raises(SourceError, match="is not a SQLite database") as caught:
            source.copy_source(cut, work / COPY_NAME)
        assert type(caught.value) is SourceError
        assert names(work) == []

    def test_a_missing_file_and_a_dead_link_are_source_errors(self, tmp_path: Path) -> None:
        dead = tmp_path / "dead.db"
        dead.symlink_to(tmp_path / "nowhere.db")
        work = new_work_folder(tmp_path)
        for path in (tmp_path / "nowhere.db", dead):
            with pytest.raises(SourceError, match="no Messages database"):
                source.copy_source(path, work / COPY_NAME)
        assert names(work) == []


class TestProgress:
    def test_steps_count_bytes_up_to_the_total(
        self, leftover: Leftover, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(source, "COPY_CHUNK_BYTES", 1024)
        steps: list[tuple[int, int]] = []
        target = new_work_folder(tmp_path) / COPY_NAME
        source.copy_source(leftover.path, target, lambda done, total: steps.append((done, total)))
        size = sum(path.stat().st_size for path in leftover.folder.iterdir())
        assert len(steps) > 4
        assert [done for done, _ in steps] == sorted(done for done, _ in steps)
        assert {total for _, total in steps} == {size}
        assert steps[0][0] > 0 and steps[-1] == (size, size)
        assert target.read_bytes() == leftover.path.read_bytes()

    @pytest.mark.parametrize("error", [ValueError("ui went away"), BrokenPipeError("ui went away")])
    def test_an_error_from_on_step_stops_the_copy_and_leaves_nothing(
        self, error: Exception, leftover: Leftover, tmp_path: Path
    ) -> None:
        calls: list[int] = []

        def on_step(done: int, total: int) -> None:
            calls.append(done)
            raise error

        work = new_work_folder(tmp_path)
        with pytest.raises(type(error)) as caught:
            source.copy_source(leftover.path, work / COPY_NAME, on_step)
        assert caught.value is error and len(calls) == 1
        assert names(work) == []
        assert names(leftover.folder) == ["chat.db", "chat.db-wal"]

    def test_an_error_from_on_step_during_a_locked_read_leaves_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, _ = rollback_source(tmp_path)
        watch_signature(monkeypatch, always)
        locked = count_locked_reads(monkeypatch)

        def on_step(done: int, total: int) -> None:
            if locked:
                raise ValueError("stop")

        work = new_work_folder(tmp_path)
        with pytest.raises(ValueError, match="stop"):
            source.copy_source(db, work / COPY_NAME, on_step)
        assert names(work) == [] and names(db.parent) == ["chat.db"]

    def test_an_existing_target_is_refused_and_kept(self, leftover: Leftover, tmp_path: Path) -> None:
        target = new_work_folder(tmp_path) / COPY_NAME
        target.write_bytes(b"not a copy of this run")
        with pytest.raises(FileExistsError):
            source.copy_source(leftover.path, target)
        assert target.read_bytes() == b"not a copy of this run"

    def test_the_copy_holds_one_chunk_in_memory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        chunk = 256 * 1024
        monkeypatch.setattr(source, "COPY_CHUNK_BYTES", chunk)
        folder = tmp_path / "source"
        folder.mkdir()
        builder = ChatDbBuilder(folder / "chat.db")
        chat = builder.chat("long@example.com")
        for index in range(400):
            builder.message(chat, when=utc(2024, 3, 1, 12, index // 60, index % 60), text="long text " * 400)
        builder.close()
        assert builder.path.stat().st_size > 4 * chunk
        target = new_work_folder(tmp_path) / COPY_NAME
        tracemalloc.start()
        try:
            source.copy_source(builder.path, target)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 1.5 * chunk, f"the copy held {peak / chunk:.2f} chunks"
        assert target.read_bytes() == builder.path.read_bytes()


def loose_guid_source(tmp_path: Path, guids: list[object]) -> Path:
    """A Messages database whose guid column keeps any value, also NULL."""
    folder = tmp_path / "source"
    folder.mkdir()
    builder = ChatDbBuilder(folder / "chat.db", skip_tables={"message"})
    builder.raw.execute(
        "CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid, text TEXT, handle_id INTEGER,"
        " is_from_me INTEGER, date INTEGER)"
    )
    builder.raw.executemany("INSERT INTO message (guid, text) VALUES (?, 'x')", [(guid,) for guid in guids])
    builder.close()
    return folder / "chat.db"


class TestGuids:
    def test_chunks_hold_every_guid(self, tmp_path: Path) -> None:
        db, guids = rollback_source(tmp_path, count=25)
        with source.open_snapshot(db, tmp_path / "work") as reader:
            chunks = list(reader.guids(chunk=10))
            one_chunk = list(reader.guids())
        assert [len(chunk) for chunk in chunks] == [10, 10, 5]
        assert sorted(guid for chunk in chunks for guid in chunk) == sorted(guids)
        assert [sorted(chunk) for chunk in one_chunk] == [sorted(guids)]

    def test_values_are_the_guids_that_the_importer_stores(self, tmp_path: Path) -> None:
        odd: list[object] = [None, "", "   ", " PADDED\t", 12345, b"BLOB-GUID", 1.5, "plain", "\n", None]
        db = loose_guid_source(tmp_path, odd)
        with source.open_snapshot(db, tmp_path / "work") as reader:
            chunks = list(reader.guids(chunk=3))
            rows = [row for batch in reader.messages_after(0, 100) for row in batch]
        found = [guid for chunk in chunks for guid in chunk]
        assert all(0 < len(chunk) <= 3 for chunk in chunks)
        assert sorted(found) == ["1.5", "12345", "BLOB-GUID", "PADDED", "plain"]
        assert sorted(found) == sorted({(row.guid or "").strip() for row in rows} - {""})  # what classify keeps
        assert all(type(guid) is str for guid in found)

    def test_a_table_with_no_guid_gives_no_chunk(self, tmp_path: Path) -> None:
        db = loose_guid_source(tmp_path, [None, ""])
        with source.open_snapshot(db, tmp_path / "work") as reader:
            assert list(reader.guids()) == []

    @pytest.mark.parametrize("chunk", [0, -1])
    def test_a_chunk_below_one_is_refused(self, chunk: int, tmp_path: Path) -> None:
        db, _ = rollback_source(tmp_path)
        with source.open_snapshot(db, tmp_path / "work") as reader, pytest.raises(ValueError, match="chunk"):
            reader.guids(chunk)


TOO_LARGE = [1e30, -1e30, float("inf"), float("-inf"), "123456789012345678901234567890"]
INTEGER_COLUMNS = {"item_type": "item_type", "associated_message_type": "associated_type", "date": "date"}


class TestOddNumbers:
    @pytest.mark.parametrize(
        "value",
        [*TOO_LARGE, float("nan"), INT64_MAX + 1, INT64_MIN - 1, "9" * 5000, "soon", b"\xff", "", [1]],
    )
    def test_a_value_that_is_no_64_bit_integer_becomes_none(self, value: object) -> None:
        assert source._coerce(value, int) is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(INT64_MAX, INT64_MAX), (INT64_MIN, INT64_MIN), (float(2**62), 2**62), (7.9, 7), ("42", 42), (b"42", 42)],
    )
    def test_a_usable_number_is_kept(self, value: object, expected: int) -> None:
        assert source._coerce(value, int) == expected

    @pytest.mark.parametrize("column", list(INTEGER_COLUMNS))
    @pytest.mark.parametrize("value", TOO_LARGE)
    def test_a_number_outside_64_bits_in_a_row_is_read_as_none(
        self, value: object, column: str, tmp_path: Path
    ) -> None:
        folder = tmp_path / "source"
        folder.mkdir()
        builder = ChatDbBuilder(folder / "chat.db")
        (usual,) = add_messages(builder, 1, "usual")
        builder.raw.execute(f"INSERT INTO message (guid, text, {column}) VALUES ('ODD', 'odd row', ?)", (value,))
        builder.close()
        with source.open_snapshot(folder / "chat.db", tmp_path / "work") as reader:
            rows = {row.guid: row for batch in reader.messages_after(0, 10) for row in batch}
        field = INTEGER_COLUMNS[column]
        assert getattr(rows["ODD"], field) is None and rows["ODD"].text == "odd row"
        assert getattr(rows[usual], field) is not None
        check = sqlite3.connect(":memory:")
        try:
            for row in rows.values():  # every value must bind to SQLite: that is what the importer does next
                check.execute(f"SELECT {', '.join('?' for _ in row)}", tuple(row)).fetchone()
        finally:
            check.close()

    def test_member_rows_that_are_not_integers_are_left_out(self, tmp_path: Path) -> None:
        folder = tmp_path / "source"
        folder.mkdir()
        builder = ChatDbBuilder(folder / "chat.db")
        handle = builder.handle("member@example.com")
        chat = builder.chat("member@example.com", members=[handle])
        odd = [("abc", handle), (chat, float("inf")), (None, handle), (b"\x01", handle), (2.5, 1e30)]
        builder.raw.executemany("INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (?, ?)", odd)
        builder.close()
        with source.open_snapshot(folder / "chat.db", tmp_path / "work") as reader:
            assert reader.members() == [(chat, handle)]


class TestWorkFolder:
    @pytest.mark.parametrize("in_its_place", ["link to a folder", "dead link", "file"])
    def test_something_else_in_place_of_the_folder_is_refused(self, in_its_place: str, tmp_path: Path) -> None:
        db, _ = rollback_source(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "source-other.sqlite3").write_bytes(b"not ours")
        work = tmp_path / "work"
        if in_its_place == "link to a folder":
            work.symlink_to(elsewhere, target_is_directory=True)
        elif in_its_place == "dead link":
            work.symlink_to(tmp_path / "nowhere")
        else:
            work.write_bytes(b"a file")
        with pytest.raises(SourceError, match="work folder"), source.open_snapshot(db, work):
            pytest.fail("the snapshot must not open")
        assert work.is_symlink() == (in_its_place != "file") and os.path.lexists(work)
        assert names(elsewhere) == ["source-other.sqlite3"]
        assert names(db.parent) == ["chat.db"]

    def test_a_folder_of_another_user_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db, _ = rollback_source(tmp_path)
        work = new_work_folder(tmp_path)
        (work / "source-other.sqlite3").write_bytes(b"not ours")
        me = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: me + 1)
        with pytest.raises(SourceError, match="work folder"), source.open_snapshot(db, work):
            pytest.fail("the snapshot must not open")
        assert names(work) == ["source-other.sqlite3"]

    def test_a_folder_with_a_loose_mode_is_made_private(self, tmp_path: Path) -> None:
        db, guids = rollback_source(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        work.chmod(0o755)
        (work / "notes.txt").write_text("not a source copy", encoding="utf-8")
        with source.open_snapshot(db, work) as reader:
            modes = {entry.name: stat.S_IMODE(entry.lstat().st_mode) for entry in work.iterdir()}
            assert stat.S_IMODE(work.lstat().st_mode) == 0o700
            assert reader.count_after(0) == len(guids)
        copies = [name for name in modes if name.startswith("source-")]
        assert copies and all(modes[name] & 0o077 == 0 for name in copies)
        assert names(work) == ["notes.txt"] and stat.S_IMODE(work.lstat().st_mode) == 0o700

    def test_clear_work_dir_removes_only_copy_files(self, tmp_path: Path) -> None:
        work = new_work_folder(tmp_path)
        outside = tmp_path / "outside.txt"
        outside.write_text("keep", encoding="utf-8")
        copy = "source-123-456.sqlite3"
        for name in (copy, f"{copy}-wal", f"{copy}-shm", f"{copy}-journal", "notes.txt"):
            (work / name).write_bytes(b"x")
        (work / "source-folder").mkdir()
        (work / "source-link").symlink_to(outside)
        source.clear_work_dir(work)
        assert names(work) == ["notes.txt", "source-folder", "source-link"]
        assert outside.read_text(encoding="utf-8") == "keep"

    def test_clear_work_dir_removes_a_folder_that_holds_only_copies(self, tmp_path: Path) -> None:
        work = new_work_folder(tmp_path)
        for name in ("source-1-2.sqlite3", "source-1-2.sqlite3-wal", "source-1-2.sqlite3-shm"):
            (work / name).write_bytes(b"left by a run that was killed")
        source.clear_work_dir(work)
        assert not work.exists()
        source.clear_work_dir(work)  # a folder that is not there is not an error

    def test_clear_work_dir_does_not_follow_a_link(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "source-other.sqlite3").write_bytes(b"not ours")
        work = tmp_path / "work"
        work.symlink_to(elsewhere, target_is_directory=True)
        source.clear_work_dir(work)
        assert work.is_symlink() and names(elsewhere) == ["source-other.sqlite3"]
