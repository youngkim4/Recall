"""people_store.py: the file of the people index, written whole, for its owner only."""

from __future__ import annotations

import hashlib
import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest

import recall_paths
from recall_services import people_store
from recall_services.people_figures import Figures
from recall_services.people_reply import TurningPoint
from recall_services.people_store import Held, PeopleStore, PeopleStoreError, Profile, StoreMeta

FIGURES = Figures("2023-01-02", "2023-02-20", 8, 4, 4, 4, 39.45, "2023-01", 0.5)
META = StoreMeta(model="gpt-6-luna", embed_model="fake-embedding-1", dimensions=4, built_at="2026-11-18T10:00:00+00:00")


def profile(key: str = "person:1", vector: bytes | None = None, **more: object) -> Profile:
    fields = {
        "key": key,
        "conversations": ("+12025550143", "mina@example.com"),
        "label": "Mina",
        "is_group": False,
        "figures": FIGURES,
        "kind": "friend",
        "status": "ongoing",
        "summary": "You and Mina kept in touch.",
        "turning_points": (TurningPoint("2023-02-14", "A valentine."),),
        "input_hash": "a" * 64,
        "text_hash": "b" * 64,
        "vector": vector,
        **more,
    }
    return Profile(**fields)  # type: ignore[arg-type]


def vector(*values: float) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def test_the_folder_of_an_archive_is_named_by_the_hash_of_its_path(tmp_path: Path) -> None:
    archive_file = tmp_path / "a" / ".." / "archive.sqlite3"
    name = hashlib.sha256(str(archive_file.resolve()).encode("utf-8")).hexdigest()[:16]
    assert people_store.people_folder(archive_file) == Path(recall_paths.SAVES_DIR) / "people" / name


def test_no_file_holds_nothing(tmp_path: Path) -> None:
    store = PeopleStore(tmp_path / "people")
    assert store.read() is None and store.signature() is None
    assert not (tmp_path / "people").exists()


def test_what_is_written_is_read_back(tmp_path: Path) -> None:
    store = PeopleStore(tmp_path / "people")
    held = Held(META, (profile(vector=vector(1, 0, 0, 0)), profile("conversation:chat1", is_group=True)))
    store.write(held)
    assert store.read() == held
    assert store.signature() is not None


def test_the_file_and_its_folder_are_for_their_owner_only(tmp_path: Path) -> None:
    store = PeopleStore(tmp_path / "people")
    store.write(Held(META, (profile(),)))
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.folder.stat().st_mode) == 0o700
    assert sorted(path.name for path in store.folder.iterdir()) == ["people.sqlite3"]


def test_a_write_that_fails_leaves_the_file_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = PeopleStore(tmp_path / "people")
    before = Held(META, (profile(),))
    store.write(before)

    def broken(*_args: object) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(people_store.os, "replace", broken)
    with pytest.raises(PeopleStoreError):
        store.write(Held(META, (profile("person:2"),)))
    assert store.read() == before
    assert sorted(path.name for path in store.folder.iterdir()) == ["people.sqlite3"]


def test_a_file_of_a_newer_format_is_not_read(tmp_path: Path) -> None:
    store = PeopleStore(tmp_path / "people")
    store.write(Held(META, (profile(),)))
    conn = sqlite3.connect(store.path)
    with conn:
        conn.execute("UPDATE meta SET format_version = 99")
    conn.close()
    with pytest.raises(people_store.PeopleFormatError):
        store.read()


def test_a_file_that_is_not_a_store_is_an_error(tmp_path: Path) -> None:
    store = PeopleStore(tmp_path / "people")
    store.folder.mkdir()
    store.path.write_text("not a database", encoding="utf-8")
    with pytest.raises(PeopleStoreError):
        store.read()


def test_the_profile_text_names_kind_status_summary_and_turning_points() -> None:
    assert people_store.profile_text(profile()) == (
        "Kind: friend. Status: ongoing.\nYou and Mina kept in touch.\n2023-02-14: A valentine."
    )


def test_the_text_hash_holds_the_model_and_the_dimensions() -> None:
    one = people_store.text_hash("text", "model-a", 4)
    assert one != people_store.text_hash("text", "model-a", 8) != people_store.text_hash("text", "model-b", 4)
