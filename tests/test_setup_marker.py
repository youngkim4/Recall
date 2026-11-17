"""The record of how far setup got.

The client can record three things, each true or false. Nothing else is taken
from it: no path, and no value that only looks like true or false. Two records
made at the same moment are both kept.
"""
import json
import threading

import pytest

import ui_server
from tests.loopback import fetch, post


@pytest.mark.parametrize("said, recorded", [
    ({"completed": True, "skipped": False, "firstWordsShown": True}, {"completed": True, "skipped": False, "firstWordsShown": True}),
    # text is not true or false, whatever it says
    ({"completed": "false", "skipped": "true", "firstWordsShown": ""}, {}),
    ({"completed": 1, "skipped": 0, "firstWordsShown": None}, {}),
    ({"completed": True, "pickedDbPath": "/somewhere/chat.db", "other": True}, {"completed": True}),
    ({}, {}),
])
def test_setup_takes_only_true_and_false(said, recorded):
    assert ui_server.setup_marker_updates(said) == recorded


@pytest.mark.parametrize("body", [[], ["completed"], "completed", 7, None, True])
def test_a_body_that_is_not_a_record_records_nothing(body):
    assert ui_server.setup_marker_updates(body) == {}


def test_a_record_written_by_an_earlier_build_is_read_as_true_and_false_only(marker):
    marker.parent.mkdir()
    kept = {"completed": "false", "skipped": 1, "firstWordsShown": True, "pickedDbPath": "/somewhere/chat.db", "other": True}
    marker.write_text(json.dumps(kept), encoding="utf-8")

    assert ui_server.read_setup_marker() == {"firstWordsShown": True}


def test_text_that_says_false_does_not_finish_setup(server, marker):
    marker.parent.mkdir()
    marker.write_text(json.dumps({"completed": "false"}), encoding="utf-8")

    _, _, body = fetch(server, "/api/defaults")

    assert json.loads(body)["setupCompleted"] is False


def test_two_records_made_at_once_are_both_kept(marker, monkeypatch):
    read = ui_server.read_setup_marker
    first_has_read, first_may_write = threading.Event(), threading.Event()

    def read_and_wait_the_first_time():
        found = read()
        if not first_has_read.is_set():
            first_has_read.set()
            first_may_write.wait(timeout=5)
        return found

    monkeypatch.setattr(ui_server, "read_setup_marker", read_and_wait_the_first_time)
    first = threading.Thread(target=ui_server.write_setup_marker, args=({"completed": True},))
    second = threading.Thread(target=ui_server.write_setup_marker, args=({"firstWordsShown": True},))
    first.start()
    assert first_has_read.wait(timeout=5)
    second.start()
    # the second record waits for the first: it must not read the file in between
    second.join(timeout=0.3)
    first_may_write.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert json.loads(marker.read_text(encoding="utf-8")) == {"completed": True, "firstWordsShown": True}


def test_the_route_records_what_setup_says_and_answers_with_the_record(server, marker):
    status, answer = post(server, "/api/setup/complete", {"completed": True})
    assert (status, answer) == (200, {"setup": {"completed": True}})

    status, answer = post(server, "/api/setup/complete", {"firstWordsShown": True, "skipped": "false"})

    assert (status, answer) == (200, {"setup": {"completed": True, "firstWordsShown": True}})
    assert json.loads(marker.read_text(encoding="utf-8")) == {"completed": True, "firstWordsShown": True}


def test_the_route_keeps_no_path(server, marker):
    status, answer = post(server, "/api/setup/complete", {"completed": True, "pickedDbPath": "/somewhere/chat.db"})

    assert (status, answer) == (200, {"setup": {"completed": True}})
    assert "pickedDbPath" not in marker.read_text(encoding="utf-8")


def test_the_route_answers_a_body_that_is_not_a_record_and_writes_nothing(server, marker):
    status, answer = post(server, "/api/setup/complete", ["completed"])

    assert (status, answer) == (200, {"setup": {}})
    assert not marker.exists()
