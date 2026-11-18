"""Faults in the swap, in the put-back and in the repair: the store gives one whole build at every point.

The two builds have one shape. `side` fails when the store pairs the segments
of one with the vectors of the other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meaning_index import store_files
from meaning_index.errors import IndexBusyError, StoreError
from tests.meaning_index.store_kit import FILES, Faults, content, make, put, side, torn_between, two_builds


def leftovers(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.name.endswith((".old", ".new")))


def no_sync(path: Path) -> None:
    raise OSError("the folder can not be synced")


@pytest.mark.parametrize("first", [1, 2, 3, 4])
@pytest.mark.parametrize("second", [None, 1, 2, 3])
def test_a_fault_in_the_swap_and_a_second_fault_in_the_put_back_keep_the_old_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, first: int, second: int | None
) -> None:
    """`first` is the step of the swap that fails: one of the three renames, or the sync after them (4).

    `second` is the step of the put-back that fails too.
    """
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    before = content(folder)
    renames = min(first, 4) if first < 4 else 3
    faults = Faults(({first} if first < 4 else set()) | ({renames + second} if second else set()))
    monkeypatch.setattr(store_files.os, "replace", faults)
    if first == 4:
        monkeypatch.setattr(store_files, "sync_folder", no_sync)
    with pytest.raises(StoreError), store.building():
        store.replace("model-a", 8, b[0], [b[1]])
    monkeypatch.undo()
    # every file of the new build is in place and the put-back could not start: the new build stays whole
    expected = "b" if (first, second) == (4, 1) else "a"
    # the build is given at once, also when the put-back did not end
    assert side(store, a, b) == expected
    with store.building():
        pass
    assert side(store, a, b) == expected
    assert leftovers(folder) == []
    if expected == "a":
        assert content(folder) == before


@pytest.mark.parametrize("at", [1, 2, 3])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_in_the_swap_puts_the_old_files_back_and_goes_on_up(
    folder: Path, monkeypatch: pytest.MonkeyPatch, at: int, error: type
) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    before = content(folder)
    monkeypatch.setattr(store_files.os, "replace", Faults({at}, error()))
    with pytest.raises(error), store.building():
        store.replace("model-a", 8, b[0], [b[1]])
    monkeypatch.undo()
    assert content(folder) == before
    assert leftovers(folder) == []
    assert side(store, a, b) == "a"


@pytest.mark.parametrize("at", [1, 2, 3])
def test_an_interrupt_in_a_first_swap_leaves_no_file(folder: Path, monkeypatch: pytest.MonkeyPatch, at: int) -> None:
    a, b = two_builds()
    store = make(folder)
    monkeypatch.setattr(store_files.os, "replace", Faults({at}, KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt), store.building():
        store.replace("model-a", 8, a[0], [a[1]])
    monkeypatch.undo()
    assert [path.name for path in folder.iterdir()] == ["build.lock"]
    assert side(store, a, b) == "none"


@pytest.mark.parametrize("replaced", [1, 2])
def test_a_folder_that_only_needs_the_put_back_gives_the_old_build(folder: Path, replaced: int) -> None:
    a, b = two_builds()
    store = torn_between(folder, a, b, replaced)
    listed = sorted(path.name for path in folder.iterdir())
    assert side(store, a, b) == "a"
    assert store.status().state == "ready"
    # a reader changes nothing
    assert sorted(path.name for path in folder.iterdir()) == listed


def test_a_swap_that_ended_gives_the_new_build_while_its_backups_are_there(folder: Path) -> None:
    a, b = two_builds()
    store = torn_between(folder, a, b, 3)
    assert side(store, a, b) == "b"
    with store.building():
        pass
    assert side(store, a, b) == "b"
    assert leftovers(folder) == []


def test_files_of_two_builds_with_no_backup_are_no_index(folder: Path) -> None:
    a, b = two_builds()
    store = torn_between(folder, a, b, 2)
    for name in FILES:
        (folder / f"{name}.old").unlink(missing_ok=True)
    assert side(store, a, b) == "none"


@pytest.mark.parametrize(("replaced", "at"), [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_a_fault_at_each_step_of_the_repair_keeps_the_old_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, replaced: int, at: int, error: type
) -> None:
    a, b = two_builds()
    store = torn_between(folder, a, b, replaced)
    monkeypatch.setattr(store_files.os, "replace", Faults({at}, error("stopped")))
    with pytest.raises(StoreError if error is OSError else error), store.building():
        pass
    monkeypatch.undo()
    assert side(store, a, b) == "a"
    with store.building():
        assert side(store, a, b) == "a"
        assert leftovers(folder) == []
    assert side(store, a, b) == "a"


def test_the_put_back_goes_in_the_reverse_order_of_the_swap(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)
    real = Faults(set())
    order: list[str] = []

    def replace(source: object, target: object) -> None:
        order.append(Path(str(target)).name)
        real(source, target)

    monkeypatch.setattr(store_files.os, "replace", replace)
    monkeypatch.setattr(store_files, "sync_folder", no_sync)  # the three files are in place, then the swap fails
    with pytest.raises(StoreError), store.building():
        store.replace("model-a", 8, b[0], [b[1]])
    assert order == [*FILES, *reversed(FILES)]
    monkeypatch.undo()
    assert side(store, a, b) == "a"


def test_a_nested_build_on_one_store_is_refused_and_the_outer_one_keeps_the_folder(folder: Path) -> None:
    a, b = two_builds()
    store = make(folder)
    with store.building():
        with pytest.raises(IndexBusyError), store.building():
            pass
        with pytest.raises(IndexBusyError), make(folder).building():
            pass
        store.replace("model-a", 8, a[0], [a[1]])
    assert side(store, a, b) == "a"
    with make(folder).building():
        pass


def test_a_cleanup_that_fails_does_not_hide_the_failure_of_the_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    a, b = two_builds()
    store = make(folder)
    put(store, a)

    def stream():  # type: ignore[no-untyped-def]
        yield b[1][:2]
        raise RuntimeError("the rows stopped")

    def refuse(path: Path) -> None:
        raise PermissionError("not yours")

    with pytest.raises(RuntimeError, match="the rows stopped"), store.building():
        monkeypatch.setattr(store_files, "remove", refuse)
        store.replace("model-a", 8, b[0], stream())
    monkeypatch.undo()
    assert any("could not be removed" in record.getMessage() for record in caplog.records)
    assert side(store, a, b) == "a"


def test_a_lock_that_can_not_be_closed_does_not_hide_the_failure_of_the_build(
    folder: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from meaning_index import store as store_module

    store = make(folder)
    held: list[int] = []
    real_lock, real_close = store_module.fcntl.flock, store_module.os.close

    def lock(descriptor: int, how: int) -> None:
        held.append(descriptor)
        real_lock(descriptor, how)

    def close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor in held:
            raise OSError("closed too late")

    monkeypatch.setattr(store_module.fcntl, "flock", lock)
    monkeypatch.setattr(store_module.os, "close", close)
    with pytest.raises(RuntimeError, match="the build failed"), store.building():
        raise RuntimeError("the build failed")
    monkeypatch.undo()
    assert any("could not be closed" in record.getMessage() for record in caplog.records)
    with store.building():
        pass


def test_a_build_id_of_another_form_is_not_written(folder: Path) -> None:
    folder.mkdir(parents=True)
    with pytest.raises(StoreError):
        store_files.write_vectors(folder / "vectors.npy.new", "not an id", 0, 8, [])


def test_a_segments_file_that_names_no_build_is_no_index(folder: Path) -> None:
    import sqlite3

    a, b = two_builds()
    store = make(folder)
    put(store, a)
    conn = sqlite3.connect(folder / "segments.sqlite3")
    conn.execute("DELETE FROM build")
    conn.commit()
    conn.close()
    assert side(store, a, b) == "none"


@pytest.mark.parametrize("replaced", [1, 2])
def test_the_repair_goes_in_the_reverse_order_of_the_swap(
    folder: Path, monkeypatch: pytest.MonkeyPatch, replaced: int
) -> None:
    a, b = two_builds()
    store = torn_between(folder, a, b, replaced)
    real = Faults(set())
    order: list[str] = []

    def replace(source: object, target: object) -> None:
        order.append(Path(str(target)).name)
        real(source, target)

    monkeypatch.setattr(store_files.os, "replace", replace)
    with store.building():
        pass
    assert order == list(reversed(FILES[:replaced]))
    assert side(store, a, b) == "a"
