"""The work file: the vectors that a build paid for, kept until a build ends well."""

from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pytest

from meaning_index.errors import StoreError
from meaning_index.workfile import WorkFile
from tests.meaning_index.support import unit

A, B, C = "a" * 64, "b" * 64, "c" * 64


def test_a_work_file_that_is_not_there_holds_nothing_and_a_read_makes_no_file(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    assert work.hashes() == frozenset()
    assert work.get_many([A]) == {}
    assert not (tmp_path / "work.sqlite3").exists()


def test_vectors_are_kept_by_content_hash(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    rows = unit(3, 8)
    work.put([A, B], rows[:2])
    work.put([C], rows[2:])
    again = WorkFile(tmp_path / "work.sqlite3", 8)
    assert again.hashes() == {A, B, C}
    found = again.get_many([C, A, "d" * 64])
    assert set(found) == {A, C}
    assert np.array_equal(found[A], rows[0])
    assert np.array_equal(found[C], rows[2])
    assert found[A].dtype == np.float16


def test_the_file_is_for_its_owner_only(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    work.put([A], unit(1, 8))
    assert stat.S_IMODE((tmp_path / "work.sqlite3").stat().st_mode) == 0o600


def test_a_vector_of_another_size_is_not_given(tmp_path: Path) -> None:
    WorkFile(tmp_path / "work.sqlite3", 8).put([A], unit(1, 8))
    other = WorkFile(tmp_path / "work.sqlite3", 16)
    assert other.hashes() == frozenset()
    assert other.get_many([A]) == {}


def test_many_hashes_are_read_in_parts(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 4)
    hashes = [f"{n:064x}" for n in range(1200)]
    rows = unit(1200, 4)
    work.put(hashes, rows)
    found = work.get_many(hashes)
    assert len(found) == 1200
    assert np.array_equal(found[hashes[1199]], rows[1199])


def test_clear_removes_the_file(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    work.put([A], unit(1, 8))
    work.clear()
    assert not (tmp_path / "work.sqlite3").exists()
    work.clear()
    assert work.hashes() == frozenset()


def test_rows_that_do_not_fit_the_hashes_are_refused(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    with pytest.raises(StoreError):
        work.put([A, B], unit(1, 8))
    with pytest.raises(StoreError):
        work.put([A], unit(1, 4))


def test_a_file_that_is_no_database_is_read_as_empty_and_a_repair_removes_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "work.sqlite3"
    path.write_bytes(b"this is no database at all, only invented bytes" * 40)
    work = WorkFile(path, 8)
    assert work.hashes() == frozenset()
    assert caplog.records
    with pytest.raises(StoreError):
        work.put([A], unit(1, 8))
    work.repair()
    assert not path.exists()
    work.put([A], unit(1, 8))
    assert work.hashes() == {A}


def test_a_repair_keeps_a_file_that_can_be_read(tmp_path: Path) -> None:
    work = WorkFile(tmp_path / "work.sqlite3", 8)
    work.put([A], unit(1, 8))
    work.repair()
    assert work.hashes() == {A}


def test_a_file_that_can_not_be_removed_is_a_warning_and_no_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from meaning_index import workfile

    def refuse(path: Path) -> None:
        raise PermissionError("not yours")

    work = WorkFile(tmp_path / "work.sqlite3", 8)
    work.put([A], unit(1, 8))
    monkeypatch.setattr(workfile, "remove", refuse)
    work.clear()
    assert any("could not be removed" in record.getMessage() for record in caplog.records)
    assert work.hashes() == {A}
