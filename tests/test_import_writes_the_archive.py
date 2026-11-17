"""What an import writes, and what it tells.

After an import the archive holds what the database holds: the messages, what
hangs on them and the tapbacks. The figures of the server follow the new
archive at once, and no preview of the earlier archive is kept. A database
that lacks more than a fifth of the archive is not imported, unless the
request says so. The names of Contacts are read before the pull.
"""
import json
from datetime import datetime, timedelta

import pytest

import ui_server
from archive import Archive
from tests.archive.builders import ChatDbBuilder
from tests.archive_kit import (
    KEPT,
    PERSON,
    beside,
    bridged,
    ended,
    held,
    import_from,
    no_names,  # noqa: F401
    own_archives,  # noqa: F401
    picture,
    pulled,
    put_in_messages,
    run,
    said,
    unchanged,
    write_database,
)
from tests.invented_db import apple_archive
from tests.loopback import post
from tests.recall_services.state_data import START, ZONE

pytestmark = pytest.mark.usefixtures("fresh_bundle_cache", "no_names", "own_archives", "messages_home")

FIRST = ["one", "two", "three", "four", "five"]


@pytest.fixture
def filled(tmp_path):
    """An archive that a pull filled with the messages of FIRST."""
    return pulled(tmp_path / "saves" / "archive.sqlite3", FIRST)


def write_thread(path, texts, photo=False, tapback=False):
    """A database with one conversation. A photo can hang on its first message, and a tapback can be on it."""
    path.unlink(missing_ok=True)
    builder = ChatDbBuilder(path)
    person = builder.handle(PERSON)
    thread = builder.chat(PERSON, members=[person])
    rows = [
        builder.message(thread, when=START + timedelta(minutes=place), text=text, handle=person, guid=f"guid-{text}")
        for place, text in enumerate(texts)
    ]
    if photo:
        builder.attachment(rows[0])
    if tapback:
        builder.tapback(thread, f"guid-{texts[0]}", 2000, when=START + timedelta(hours=1), handle=person)
    builder.close()
    return path


def what_hangs(archive_file):
    """How many files and how many tapbacks the messages of the archive have."""
    with Archive.open(archive_file) as archive:
        messages = archive.messages(PERSON, limit=1_000)["messages"]
        tapbacks = archive.reactions([message["id"] for message in messages])
    return sum(len(message["attachments"]) for message in messages), sum(len(found) for found in tapbacks.values())


# ---- what the archive holds after an import ----


@pytest.mark.parametrize("photo, tapback", [(False, False), (True, False), (False, True), (True, True)])
def test_after_an_update_the_photos_and_the_tapbacks_are_those_of_the_database(tmp_path, photo, tapback):
    target = tmp_path / "saves" / "archive.sqlite3"
    first = import_from(write_thread(tmp_path / "chat.db", ["one", "two"], photo=True, tapback=True), target)
    assert first["status"] == "completed" and what_hangs(target) == (1, 1)

    job = import_from(write_thread(tmp_path / "chat.db", ["one", "two"], photo=photo, tapback=tapback), target)

    assert job["status"] == "completed", job["error"]
    assert job["result"]["messages"] == 2
    assert held(target) == {"guid-one": "one", "guid-two": "two"}
    # the photo and the tapback that the earlier archive held are gone when Messages no longer has them
    assert what_hangs(target) == (int(photo), int(tapback))


def test_a_first_import_makes_the_folders_and_leaves_the_archive_and_its_note(tmp_path):
    target = tmp_path / "new" / "folder" / "archive.sqlite3"

    job = import_from(write_database(tmp_path / "copy.db", ["one"]), target)

    assert job["status"] == "completed", job["error"]
    assert held(target) == {"guid-one": "one"}
    assert beside(target) == []
    assert json.loads((target.parent / "archive.sqlite3.source.json").read_text(encoding="utf-8")) == {"source": "messages"}


def test_an_update_leaves_nothing_else_beside_the_archive(filled, tmp_path):
    job = import_from(write_database(tmp_path / "chat.db", [*FIRST, "six"]), filled)

    assert job["status"] == "completed", job["error"]
    assert beside(filled) == []
    # no path of a database is kept
    assert json.loads((filled.parent / "archive.sqlite3.source.json").read_text(encoding="utf-8")) == {"source": "messages"}


def test_a_long_message_that_is_only_in_its_body_is_whole_in_the_archive(tmp_path):
    long_text = bytes.fromhex(
        "303132333435363738396162636465666768696a6b6c6d6e6f707172737475767778797a"
        "4142434445464748494a4b4c4d4e4f505152535455565758595a"
    ).decode("ascii")
    builder = ChatDbBuilder(tmp_path / "copy.db")
    person = builder.handle(PERSON)
    thread = builder.chat(PERSON, members=[person])
    builder.message(thread, when=START, text="short", handle=person, guid="guid-short")
    builder.message(thread, when=START, body=apple_archive("ascii_300"), handle=person, guid="guid-long")
    builder.message(thread, when=START, body=apple_archive("spaces_both_ends"), handle=person, guid="guid-spaces")
    builder.close()
    target = tmp_path / "new" / "archive.sqlite3"

    job = import_from(tmp_path / "copy.db", target)

    written = held(target)
    assert job["status"] == "completed", job["error"]
    assert len(written) == 3
    assert len(written["guid-long"]) == 300 and written["guid-long"].startswith(long_text)
    # the text is kept as Messages has it: its ends are not trimmed
    assert written["guid-spaces"] != written["guid-spaces"].strip()


# ---- what a finished import tells ----


def across_the_years(path):
    """A database with one message in 2019 and one in 2024."""
    builder = ChatDbBuilder(path)
    person = builder.handle(PERSON)
    thread = builder.chat(PERSON, members=[person])
    builder.message(thread, when=datetime(2019, 3, 4, 9, 0, tzinfo=ZONE), text="early", handle=person, guid="guid-early")
    builder.message(thread, when=datetime(2024, 7, 1, 9, 0, tzinfo=ZONE), text="late", handle=person, guid="guid-late")
    builder.close()
    return path


def test_a_first_import_tells_the_counts_and_the_years(tmp_path):
    job = import_from(across_the_years(tmp_path / "copy.db"), tmp_path / "new" / "archive.sqlite3")

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 2, "conversations": 1, "firstYear": 2019, "lastYear": 2024, "contactsSkipped": False}
    assert job["error"] is None and job["code"] is None


def test_an_update_tells_the_counts_and_the_years_of_the_archive_as_it_is_now(filled, tmp_path):
    job = import_from(write_database(tmp_path / "chat.db", [*FIRST, "six"]), filled)

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 6, "conversations": 1, "firstYear": 2024, "lastYear": 2024, "contactsSkipped": False}


def test_the_export_job_tells_the_counts_and_the_archive_file(filled, tmp_path):
    put_in_messages(write_database(tmp_path / "chat.db", [*FIRST, "six"]))

    job = run("export", {"messagesPath": str(filled)})

    assert job["status"] == "completed", job["error"]
    assert job["result"] == {"messages": 6, "conversations": 1, "messagesPath": str(filled)}


def test_an_import_says_in_its_lines_which_database_it_reads(filled, tmp_path):
    copy = write_database(tmp_path / "copy.db", ["one"])

    first = import_from(copy, tmp_path / "new" / "archive.sqlite3")
    update = import_from(write_database(tmp_path / "chat.db", FIRST), filled)

    assert f"Reading the copy at {copy.resolve()}" in said(first)
    assert "Reading Messages" in said(update) and "copy" not in said(update)


# ---- the names of Contacts ----


def names_step(monkeypatch, cache, done):
    """The step that reads Contacts: it writes the cache with one name, and notes that it ran."""

    def read_contacts():
        done.append("names")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"contacts": [{"name": "Avery Stone", "phones": [PERSON], "emails": []}]}), encoding="utf-8")
        return {"count": 1}

    monkeypatch.setattr(ui_server, "export_contact_names", read_contacts)


def name_in(archive_file):
    with Archive.open(archive_file) as archive:
        return archive.person(archive.conversation(PERSON)["person_id"])["name"]


def test_the_names_are_read_before_the_pull(tmp_path, monkeypatch, no_names):  # noqa: F811
    done = []
    names_step(monkeypatch, no_names, done)
    real = ui_server.pull_into_archive
    monkeypatch.setattr(ui_server, "pull_into_archive", lambda action, payload: (done.append("pull"), real(action, payload))[1])
    target = tmp_path / "new" / "archive.sqlite3"

    job = import_from(write_database(tmp_path / "copy.db", ["one"]), target, contacts=True)

    assert job["status"] == "completed", job["error"]
    assert done == ["names", "pull"]
    # the pull took the people of the cache that the step before it wrote
    assert name_in(target) == "Avery Stone"
    assert job["result"]["contactsSkipped"] is False


def test_an_import_that_is_told_to_leave_contacts_does_not_read_them(tmp_path, monkeypatch, no_names):  # noqa: F811
    done = []
    names_step(monkeypatch, no_names, done)
    target = tmp_path / "new" / "archive.sqlite3"

    job = import_from(write_database(tmp_path / "copy.db", ["one"]), target, contacts=False)

    assert job["status"] == "completed", job["error"]
    assert done == []
    assert name_in(target) == ""
    assert job["result"]["contactsSkipped"] is False


def test_an_update_takes_the_names_that_contacts_has_now(filled, tmp_path, monkeypatch, no_names):  # noqa: F811
    assert name_in(filled) == ""
    names_step(monkeypatch, no_names, [])

    job = import_from(write_database(tmp_path / "chat.db", FIRST), filled, contacts=True)

    assert job["status"] == "completed", job["error"]
    assert name_in(filled) == "Avery Stone"


# ---- the figures and the previews follow the archive ----


def texts_of_the_bundle(archive_file):
    frame, _catalog, _names = ui_server.cached_messages_bundle(archive_file)
    return sorted(frame["text"])


def test_the_figures_follow_the_new_archive_at_once(filled, tmp_path):
    assert texts_of_the_bundle(filled) == sorted(FIRST)
    assert ui_server.archive_summary(filled)["messages"] == 5

    job = import_from(write_database(tmp_path / "chat.db", [*FIRST, "six"]), filled)

    assert job["status"] == "completed", job["error"]
    # the same server, with no start in between
    assert texts_of_the_bundle(filled) == sorted([*FIRST, "six"])
    assert ui_server.archive_summary(filled)["messages"] == 6
    assert len(ui_server.archive_patterns(filled)["daily"]) >= 1


def test_the_figures_of_a_first_import_are_there_at_once(tmp_path):
    target = tmp_path / "new" / "archive.sqlite3"
    assert ui_server.bundle_or_none(target) is None

    job = import_from(write_database(tmp_path / "copy.db", ["one", "two"]), target)

    assert job["status"] == "completed", job["error"]
    assert texts_of_the_bundle(target) == ["one", "two"]


STORES = [
    (ui_server.PREVIEW_CACHE_NAMESPACE, "PREVIEW_CACHE", "PREVIEW_CACHE_LOCK"),
    (ui_server.ANALYSIS_CACHE_NAMESPACE, "ANALYSIS_CACHE", "ANALYSIS_CACHE_LOCK"),
]


def keep_one_in_each_store():
    for namespace, held_name, lock_name in STORES:
        ui_server.set_cache_entry(namespace, "of-the-earlier-archive", {"stats": {}})
        with getattr(ui_server, lock_name):
            getattr(ui_server, held_name)["of-the-earlier-archive"] = {"stats": {}}


def kept_in_the_stores():
    stored = [ui_server.get_cache_entry(namespace, "of-the-earlier-archive") is not None for namespace, _held, _lock in STORES]
    in_memory = ["of-the-earlier-archive" in getattr(ui_server, held_name) for _namespace, held_name, _lock in STORES]
    return stored + in_memory


@pytest.fixture
def stores_cleared():
    yield
    ui_server.forget_previews()


@pytest.mark.usefixtures("stores_cleared")
@pytest.mark.parametrize("action", ["setup_import", "export"])
def test_after_an_import_no_preview_of_the_earlier_archive_is_kept(filled, tmp_path, action):
    put_in_messages(write_database(tmp_path / "chat.db", [*FIRST, "six"]))
    keep_one_in_each_store()
    assert kept_in_the_stores() == [True] * 4

    job = run(action, {"messagesPath": str(filled), "includeContacts": False})

    assert job["status"] == "completed", job["error"]
    assert kept_in_the_stores() == [False] * 4


@pytest.mark.usefixtures("stores_cleared")
def test_an_import_that_fails_keeps_the_previews_of_the_archive_that_stays(filled, tmp_path):
    keep_one_in_each_store()

    job = import_from(tmp_path / "gone.db", filled)

    assert job["status"] == "failed"
    assert kept_in_the_stores() == [True] * 4


# ---- a database that lacks much of the archive ----


def limit_sentence(gone, stored):
    return f"This database lacks {gone} of the {stored} messages in the archive. That is more than 20%. {KEPT}"


def test_a_database_that_lacks_more_than_a_fifth_of_the_archive_is_not_imported(filled, tmp_path):
    before = picture(filled)

    job = import_from(write_database(tmp_path / "chat.db", FIRST[:3]), filled)

    assert job["status"] == "failed"
    assert job["code"] == "removal_limit"
    assert job["error"] == limit_sentence(2, 5)
    assert job["result"] is None
    assert unchanged(filled, before)


def test_a_database_that_lacks_a_fifth_of_the_archive_is_imported(filled, tmp_path):
    job = import_from(write_database(tmp_path / "chat.db", FIRST[:4]), filled)

    assert job["status"] == "completed", job["error"]
    assert job["code"] is None
    assert sorted(held(filled).values()) == sorted(FIRST[:4])


def test_new_messages_do_not_make_up_for_those_that_are_gone(filled, tmp_path):
    before = picture(filled)

    job = import_from(write_database(tmp_path / "chat.db", ["one", "six", "seven", "eight", "nine", "ten"]), filled)

    assert (job["status"], job["code"], job["error"]) == ("failed", "removal_limit", limit_sentence(4, 5))
    assert unchanged(filled, before)


@pytest.mark.parametrize("action", ["setup_import", "export"])
def test_the_same_request_with_allow_removal_runs(filled, tmp_path, action):
    put_in_messages(write_database(tmp_path / "chat.db", ["one", "six"]))
    asked = {"messagesPath": str(filled), "includeContacts": False}

    refused = run(action, asked)
    job = run(action, {**asked, "allowRemoval": True})

    assert (refused["status"], refused["code"]) == ("failed", "removal_limit")
    assert job["status"] == "completed", job["error"]
    assert job["code"] is None
    assert held(filled) == {"guid-one": "one", "guid-six": "six"}


@pytest.mark.parametrize("given", ["true", "True", 1, 1.0, "yes", [True], {"allow": True}, False, None, 0])
def test_only_true_lifts_the_limit(filled, tmp_path, given):
    put_in_messages(write_database(tmp_path / "chat.db", ["one"]))
    before = picture(filled)

    job = run("setup_import", {"messagesPath": str(filled), "includeContacts": False, "allowRemoval": given})

    assert (job["status"], job["code"], job["error"]) == ("failed", "removal_limit", limit_sentence(4, 5))
    assert unchanged(filled, before)


@pytest.mark.usefixtures("no_jobs")
def test_the_route_tells_the_limit_with_its_code_and_runs_the_same_request_with_allow_removal(server, filled, tmp_path):
    put_in_messages(write_database(tmp_path / "chat.db", ["one"]))
    asked = {"action": "setup_import", "messagesPath": str(filled), "includeContacts": False}

    status, answer = post(server, "/api/jobs", asked)
    refused = ended(answer["job"]["id"])
    again_status, again = post(server, "/api/jobs", {**asked, "allowRemoval": True})
    done = ended(again["job"]["id"])

    assert (status, again_status) == (202, 202)
    assert (refused["status"], refused["code"], refused["error"]) == ("failed", "removal_limit", limit_sentence(4, 5))
    assert (done["status"], done["code"]) == ("completed", None)
    assert held(filled) == {"guid-one": "one"}


def test_the_rows_that_the_bridge_loaded_are_replaced_by_the_first_pull_and_not_counted_by_the_limit(tmp_path):
    export = bridged(tmp_path / "messages.csv", ["a", "b", "c", "d", "e"])
    assert sorted(held(export)) == [f"legacy:{row}" for row in range(1, 6)]

    job = import_from(write_database(tmp_path / "chat.db", ["a"]), export)

    assert job["status"] == "completed", job["error"]
    assert job["code"] is None
    assert held(export) == {"guid-a": "a"}


def test_after_the_first_pull_over_bridged_rows_the_limit_counts_again(tmp_path):
    export = bridged(tmp_path / "messages.csv", ["a", "b", "c", "d", "e"])
    import_from(write_database(tmp_path / "chat.db", ["a", "b", "c"]), export)

    job = import_from(write_database(tmp_path / "chat.db", ["a"]), export)

    assert (job["status"], job["code"], job["error"]) == ("failed", "removal_limit", limit_sentence(2, 3))
    assert sorted(held(export).values()) == ["a", "b", "c"]
