"""places.py: which archive stands for the archive file that the app names."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import recall_paths
from archive import default_archive_path
from recall_services import archive_state, places
from tests.recall_services import state_data as data


@pytest.fixture(autouse=True)
def _no_states_left() -> Iterator[None]:
    yield
    places.forget()


def default_export() -> Path:
    return Path(recall_paths.DEFAULT_MESSAGES_CSV)


@pytest.mark.parametrize("nothing", [None, "", "   "])
def test_nothing_named_is_the_archive_of_the_app(nothing: str | None) -> None:
    assert places.archive_path_for(nothing) == default_archive_path()
    assert places.export_for(nothing) == default_export()
    assert places.state_for(nothing) is archive_state.shared()


def test_the_default_export_and_the_default_archive_name_the_same_archive() -> None:
    """A browser that kept the name of the export of an earlier build still reads the archive of the app."""
    for named in (default_export(), default_archive_path(), str(default_export())):
        assert places.archive_path_for(named) == default_archive_path()
        assert places.export_for(named) == default_export()
        assert places.state_for(named) is archive_state.shared()


def test_a_name_that_ends_in_sqlite3_is_the_archive_itself_and_has_no_export(tmp_path: Path) -> None:
    named = tmp_path / "elsewhere" / "kept.sqlite3"
    assert places.archive_path_for(named) == named.resolve()
    assert places.export_for(named) is None
    assert places.state_for(named).path == named.resolve()


def test_any_other_name_is_an_export_and_its_archive_stands_beside_it(tmp_path: Path) -> None:
    export = tmp_path / "old" / "messages.csv"
    assert places.archive_path_for(export) == export.resolve().with_name("messages.sqlite3")
    assert places.export_for(export) == export.resolve()


def test_one_archive_has_one_state_however_it_is_named(tmp_path: Path) -> None:
    export = tmp_path / "messages.csv"
    by_export = places.state_for(export)
    assert places.state_for(str(export)) is by_export
    assert places.state_for(tmp_path / "messages.sqlite3") is by_export
    assert places.state_for(tmp_path / "sub" / ".." / "messages.csv") is by_export
    assert places.state_for(tmp_path / "other.csv") is not by_export


def test_no_folder_is_listed_and_no_file_is_made_to_find_a_place(tmp_path: Path) -> None:
    (tmp_path / "looks-like-an-archive.sqlite3").write_bytes(b"")
    before = sorted(path.name for path in tmp_path.iterdir())
    assert places.archive_path_for(tmp_path / "messages.csv").name == "messages.sqlite3"
    places.state_for(tmp_path / "messages.csv")
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_only_a_few_archives_beside_the_default_keep_an_open_handle(tmp_path: Path) -> None:
    first = places.state_for(tmp_path / "0.sqlite3")
    database = data.write_database(tmp_path / "chat.db", ["one"])
    first.pull_from_messages(database, None)
    first.archive()
    for number in range(1, places.KEPT_STATES + 1):
        places.state_for(tmp_path / f"{number}.sqlite3")
    # the oldest state was let go and closed: the place gets a new state, on the same file
    again = places.state_for(tmp_path / "0.sqlite3")
    assert again is not first and first._archive is None
    assert data.texts_of(again.archive()) == ["one"]
    assert archive_state.shared() is places.state_for(None)


def test_a_name_that_is_no_path_is_not_an_error(tmp_path: Path) -> None:
    odd = "~no-such-user-here/messages.csv"
    assert places.archive_path_for(odd).name == "messages.sqlite3"
    assert places.export_for(odd) is not None


def test_forget_closes_every_state_but_the_default_one(tmp_path: Path) -> None:
    held = places.state_for(tmp_path / "kept.sqlite3")
    places.forget()
    assert places.state_for(tmp_path / "kept.sqlite3") is not held
