"""An import changes all three files of the archive, or none of them.

A failure at any step (a write, the keeping of an earlier file, a move) leaves
the three files as they were, and nothing else beside them. While the new files
move in, each of the three places holds a whole file.
"""
import errno
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

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


def test_an_import_that_succeeds_is_not_failed_by_what_it_cannot_remove(archive, tmp_path, monkeypatch):
    copy = messages_db(tmp_path / "copy.db", ["one"])
    real_unlink = Path.unlink

    def refuse_to_remove(path, *args, **kwargs):
        if ".before-update" in path.name and path.exists():
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_to_remove)

    job = import_from(copy, messages_csv(archive))

    assert job["status"] == "completed", job["error"]
    assert texts_in(messages_csv(archive)) == {"one"}


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
