"""An import changes all three files of the archive, or none of them.

A failure at any step (a write, the keeping of an earlier file, a move) leaves
the three files as they were, and nothing else beside them. While the new files
move in, each of the three places holds a whole file. An earlier file that
could not be put back is never lost: the next import puts it back first.
"""
import errno
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

import ui_server
from tests.archive_kit import left_behind, messages_csv, run, texts_in, unchanged
from tests.invented_db import messages_db

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_saved_names", "messages_home")

NAMES = ["messages.csv", "messages_attachments.csv", "messages_reactions.csv"]
NO_SPACE = OSError(errno.ENOSPC, "No space left on device")
NO_LINKS = OSError(errno.ENOTSUP, "Operation not supported")


def import_from(copy, archive_file):
    return run("setup_import", {"dbPath": str(copy), "messagesPath": str(archive_file), "includeContacts": False})


def fail_on_call(monkeypatch, owner, name, failing, error):
    """Make the calls of `owner.name` whose place is in `failing` raise `error`. Gives the list of calls made."""
    real = getattr(owner, name)
    calls = []

    def sometimes(*args, **kwargs):
        calls.append(args)
        if len(calls) in failing:
            raise error
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, sometimes)
    return calls


# ---- a failure at any step ----


@pytest.mark.parametrize("failing_write", [1, 2, 3])
def test_a_write_that_fails_leaves_the_three_files_as_they_were(archive, tmp_path, monkeypatch, failing_write):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    fail_on_call(monkeypatch, pd.DataFrame, "to_csv", {failing_write}, NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "No space left on device" in job["error"] and "not changed" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


@pytest.mark.parametrize("failing_move", [1, 2, 3])
def test_a_move_that_fails_leaves_the_three_files_as_they_were(archive, tmp_path, monkeypatch, failing_move):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    fail_on_call(monkeypatch, os, "replace", {failing_move}, NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "No space left on device" in job["error"] and "not changed" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


@pytest.mark.parametrize("failing_move", [1, 2, 3])
def test_a_first_import_that_fails_leaves_no_file(tmp_path, monkeypatch, failing_move):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    folder = tmp_path / "new"
    fail_on_call(monkeypatch, os, "replace", {failing_move}, NO_SPACE)

    job = import_from(copy, folder / "messages.csv")

    assert job["status"] == "failed"
    assert list(folder.iterdir()) == []
    # there was no archive, so nothing is said about one
    assert "No space left on device" in job["error"] and "archive was not changed" not in job["error"]


def test_a_volume_with_no_hard_links_keeps_the_earlier_files_as_copies(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    fail_on_call(monkeypatch, os, "link", {1, 2, 3}, NO_LINKS)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"one"}
    assert left_behind(archive) == []


@pytest.mark.parametrize("failing_move", [1, 2, 3])
def test_a_move_that_fails_on_a_volume_with_no_hard_links_puts_the_copies_back(archive, tmp_path, monkeypatch, failing_move):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    fail_on_call(monkeypatch, os, "link", {1, 2, 3}, NO_LINKS)
    fail_on_call(monkeypatch, os, "replace", {failing_move}, NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_an_earlier_file_that_cannot_be_kept_stops_the_import_before_a_move(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    fail_on_call(monkeypatch, os, "link", {1, 2, 3}, NO_LINKS)
    fail_on_call(monkeypatch, shutil, "copy2", {2}, NO_SPACE)
    moves = fail_on_call(monkeypatch, os, "replace", set(), NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert moves == []
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_files_that_cannot_be_put_back_stay_beside_the_archive_and_the_error_says_so(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    before = dict(archive)
    # the second move fails, and so does the move that puts the first file back
    fail_on_call(monkeypatch, os, "replace", {2, 3}, NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    folder = messages_csv(archive).parent
    kept = folder / "messages_attachments.before-update.csv"
    assert job["status"] == "failed"
    assert ".before-update" in job["error"] and "not changed" not in job["error"]
    assert kept.read_text(encoding="utf-8") == before[folder / "messages_attachments.csv"]
    # the two files that were not moved are as they were
    for name in ("messages.csv", "messages_reactions.csv"):
        assert (folder / name).read_text(encoding="utf-8") == before[folder / name]


def test_a_file_that_cannot_be_removed_afterwards_does_not_hide_why_the_import_stopped(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    fail_on_call(monkeypatch, pd.DataFrame, "to_csv", {2}, NO_SPACE)
    real_unlink = Path.unlink

    def refuse_to_remove(path, *args, **kwargs):
        if ".tmp-export" in path.name:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_to_remove)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "No space left on device" in job["error"]
    assert unchanged(archive)


def test_an_import_that_finished_is_not_failed_by_what_it_cannot_remove(archive, tmp_path):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    real_unlink = Path.unlink
    strays = ("messages_attachments.before-update.csv", "messages_reactions.before-update.csv")

    def refuse_to_remove(path, *args, **kwargs):
        if path.name in strays and path.exists():
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "unlink", refuse_to_remove)
        job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"one"}
    assert left_behind(archive) == sorted(strays)

    # they are of an import that finished: the next import removes them, and does not put them back
    after = import_from(tmp_path / "gone.db", messages_csv(archive))

    assert after["status"] == "failed"
    assert texts_in(messages_csv(archive)) == {"one"}
    assert left_behind(archive) == []


# ---- what a request reads while the files move ----


def test_each_file_of_the_archive_is_whole_at_every_step_of_the_moves(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    folder = messages_csv(archive).parent
    old = {path.name: text for path, text in archive.items()}
    real_replace = os.replace
    seen = []

    def look_then_move(source, target):
        seen.append({name: (folder / name).read_text(encoding="utf-8") for name in NAMES})
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", look_then_move)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    new = {name: (folder / name).read_text(encoding="utf-8") for name in NAMES}
    attachments, reactions, messages = "messages_attachments.csv", "messages_reactions.csv", "messages.csv"
    # the messages move last: a new file of messages is never beside earlier attachments or reactions
    assert seen == [
        old,
        {**old, attachments: new[attachments]},
        {**old, attachments: new[attachments], reactions: new[reactions]},
    ]
    assert new[messages] != old[messages]


def test_what_an_import_that_was_cut_off_left_is_removed_by_the_next(archive, tmp_path):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    folder = messages_csv(archive).parent
    for name in ("messages.tmp-export.csv", "messages_reactions.tmp-export.csv", "messages_attachments.before-update.csv"):
        (folder / name).write_text("left by an import that was cut off", encoding="utf-8")

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    assert left_behind(archive) == []



# ---- what an import that did not finish left ----


def leave_in_parts(archive, copy):
    """An import whose second move fails, and whose move that puts the first file back fails too."""
    with pytest.MonkeyPatch.context() as patch:
        fail_on_call(patch, os, "replace", {2, 3}, NO_SPACE)
        job = import_from(copy, messages_csv(archive))
    assert job["status"] == "failed" and ".before-update" in job["error"], job["error"]
    assert not unchanged(archive)
    return job


def test_an_import_that_could_not_put_a_file_back_says_that_the_next_one_will(archive, tmp_path):
    job = leave_in_parts(archive, messages_db(tmp_path / "copy.db", ["one"]))

    assert "messages_attachments.before-update.csv" in job["error"]
    assert "The next import puts them back." in job["error"]


def test_the_next_import_first_puts_back_what_the_one_before_it_could_not(archive, tmp_path):
    leave_in_parts(archive, messages_db(tmp_path / "copy.db", ["one"]))

    # this one cannot even start: there is no file to read
    job = import_from(tmp_path / "gone.db", messages_csv(archive))

    assert job["status"] == "failed" and "There is no file" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_the_next_import_that_succeeds_leaves_only_the_new_archive(archive, tmp_path):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    leave_in_parts(archive, copy)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"one"}
    assert left_behind(archive) == []


def test_the_next_import_that_fails_part_way_leaves_the_archive_as_it_first_was(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    leave_in_parts(archive, copy)
    fail_on_call(monkeypatch, pd.DataFrame, "to_csv", {2}, NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_an_archive_that_still_cannot_be_put_back_is_not_imported_over(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    leave_in_parts(archive, copy)
    folder = messages_csv(archive).parent
    kept = folder / "messages_attachments.before-update.csv"
    fail_on_call(monkeypatch, os, "replace", set(range(1, 20)), NO_SPACE)
    monkeypatch.setattr(ui_server, "extract_messages", lambda _path: pytest.fail("the database was read"))

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "messages_attachments.before-update.csv" in job["error"] and "not changed" not in job["error"]
    # the only copy of the earlier attachments is still there
    assert kept.read_text(encoding="utf-8") == archive[folder / "messages_attachments.csv"]


def test_a_file_that_cannot_be_put_back_twice_is_still_put_back_the_third_time(archive, tmp_path):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    leave_in_parts(archive, copy)

    # the next import cannot put the attachments back either; everything else would work
    with pytest.MonkeyPatch.context() as patch:
        fail_on_call(patch, os, "replace", {1}, NO_SPACE)
        second = import_from(copy, messages_csv(archive))
    third = import_from(tmp_path / "gone.db", messages_csv(archive))

    assert second["status"] == "failed" and "messages_attachments.before-update.csv" in second["error"]
    assert third["status"] == "failed" and "There is no file" in third["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_an_import_that_was_cut_off_during_its_moves_is_undone_by_the_next(archive, tmp_path):
    # as the process left it: the attachments are new, and all three earlier files are kept
    folder = messages_csv(archive).parent
    for path in archive:
        os.link(path, path.with_name(f"{path.stem}.before-update{path.suffix}"))
    (folder / "messages_attachments.csv").unlink()
    (folder / "messages_attachments.csv").write_text("message_id,filename\n9,new.jpg\n", encoding="utf-8")

    job = import_from(tmp_path / "gone.db", messages_csv(archive))

    assert job["status"] == "failed"
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_what_an_import_that_finished_left_is_removed_and_not_put_back(archive, tmp_path):
    # as the process left it: all three files are new, and two earlier files were not yet removed
    folder = messages_csv(archive).parent
    for name in ("messages_attachments", "messages_reactions"):
        (folder / f"{name}.before-update.csv").write_text("from the archive before this one", encoding="utf-8")

    job = import_from(tmp_path / "gone.db", messages_csv(archive))

    assert job["status"] == "failed"
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_an_import_whose_last_step_fails_leaves_the_three_files_as_they_were(archive, tmp_path, monkeypatch):
    # the three files have moved in; giving up the earlier messages file is what finishes the import
    copy = messages_db(tmp_path / "copy.db", ["one"])
    real_unlink = Path.unlink
    refused = []

    def refuse_once(path, *args, **kwargs):
        if path.name == "messages.before-update.csv" and path.exists() and not refused:
            refused.append(path)
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_once)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed" and refused
    assert "Permission denied" in job["error"] and "not changed" in job["error"]
    assert unchanged(archive)
    assert left_behind(archive) == []


def test_a_new_file_that_cannot_be_removed_is_named_and_no_earlier_file_is_claimed(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    folder = messages_csv(archive).parent
    (folder / "messages_attachments.csv").unlink()
    # the second move fails; the new attachments file has no earlier file, and it cannot be removed
    fail_on_call(monkeypatch, os, "replace", {2}, NO_SPACE)
    real_unlink = Path.unlink

    def keep_the_new_file(path, *args, **kwargs):
        if path.name == "messages_attachments.csv":
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", keep_the_new_file)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "A new file could not be removed" in job["error"] and "messages_attachments.csv" in job["error"]
    assert "messages_attachments.before-update.csv" not in job["error"]
    for name in ("messages.csv", "messages_reactions.csv"):
        assert (folder / name).read_text(encoding="utf-8") == archive[folder / name]


def test_one_file_that_cannot_be_put_back_does_not_stop_the_others(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    folder = messages_csv(archive).parent
    # the third move fails; of the two files to put back, the first one tried (the reactions) fails
    fail_on_call(monkeypatch, os, "replace", {3, 4}, NO_SPACE)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "failed"
    assert "messages_reactions.before-update.csv" in job["error"]
    assert "messages_attachments.before-update.csv" not in job["error"]
    for name in ("messages.csv", "messages_attachments.csv"):
        assert (folder / name).read_text(encoding="utf-8") == archive[folder / name]
    assert (folder / "messages_reactions.before-update.csv").read_text(encoding="utf-8") == archive[folder / "messages_reactions.csv"]
    # the earlier messages file stays too: it says that the import did not finish
    assert (folder / "messages.before-update.csv").exists()
