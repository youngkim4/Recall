"""The chat routes on an invented archive: /api/ask, /api/ask/stream and /api/ask/cancel.

The model is a script. No request leaves this machine.
"""
import json
import re
import socket
import threading
import time

import pytest

import recall_paths
import ui_server
from chat_engine.fake_llm import FakeLLM, say
from recall_services import chat as chat_service
from recall_services import keys, places
from tests.loopback import _ask, post
from tests.recall_services import world

FLAT_TIRE = "The wheelbarrow has a flat tire"
QUESTION = "Who said the wheelbarrow has a flat tire?"
NO_KEY = "No API key is set. Add OPENAI_API_KEY to the file .env in Recall's data folder, then ask again."


@pytest.fixture(scope="module")
def orchard(tmp_path_factory):
    return world.build_orchard(tmp_path_factory.mktemp("orchard"))


@pytest.fixture
def archive_file(orchard, tmp_path, monkeypatch):
    """The archive that every request of a test names, with the saved names of its world."""
    cache = tmp_path / "contact_names.json"
    cache.write_text(json.dumps({"contacts": world.RECORDS}), encoding="utf-8")
    monkeypatch.setattr(ui_server, "CONTACTS_CACHE_PATH", cache)
    yield str(orchard.archive_path)
    places.forget()
    with ui_server.ASKING_LOCK:
        ui_server.ASKING.clear()


def scripted(monkeypatch, *steps):
    """The model port of the server answers from a script."""
    llm = FakeLLM(list(steps))
    monkeypatch.setattr(chat_service, "default_llm", lambda _config: llm)
    return llm


def shown_ref(request, words):
    """The reference of the line with these words, in what the model was shown."""
    for message in request.messages:
        for line in message.content.splitlines():
            found = re.match(r"(m\d+) ", line)
            if found and words in line:
                return found.group(1)
    raise AssertionError(f"no line that the model was shown holds {words!r}")


def cite(words):
    return lambda request: say(f"Someone in the group wrote it [{shown_ref(request, words)}].", chunk=12)


def sent_to(llm):
    return "\n".join(message.content for request in llm.requests for message in request.messages)


def stream(server, body):
    """Ask /api/ask/stream and give the events that it sent."""
    status, headers, raw = _ask(server, "POST", "/api/ask/stream", json.dumps(body).encode("utf-8"))
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    frames = [frame for frame in raw.decode("utf-8").split("\n\n") if frame.strip()]
    assert all(frame.startswith("data: ") for frame in frames)
    return [json.loads(frame[len("data: "):]) for frame in frames]


def test_a_question_streams_its_answer_and_ends_with_the_whole_answer(server, archive_file, monkeypatch):
    scripted(monkeypatch, cite(FLAT_TIRE))
    events = stream(server, {"messagesPath": archive_file, "question": QUESTION})
    kinds = [event["type"] for event in events]
    assert kinds[-1] == "done" and kinds.count("done") == 1 and set(kinds) == {"status", "delta", "done"}
    payload = events[-1]["payload"]
    assert payload["answer"] == "".join(event["text"] for event in events if event["type"] == "delta")
    assert (payload["mode"], payload["terms"], payload["cancelled"], payload["unverifiedQuotes"]) == ("ai", [], False, [])
    cited = payload["citations"]
    assert [citation["text"] for citation in cited] == [FLAT_TIRE]
    assert cited[0]["chatId"] == world.GARDEN and cited[0]["displayName"] == world.GARDEN_TITLE


def test_the_answer_without_a_stream_is_the_same_payload(server, archive_file, monkeypatch):
    scripted(monkeypatch, cite(FLAT_TIRE))
    status, payload = post(server, "/api/ask", {"messagesPath": archive_file, "question": QUESTION})
    assert status == 200
    assert payload["answer"] == "Someone in the group wrote it [1]." and payload["question"] == QUESTION


def test_no_handle_that_is_not_saved_goes_to_the_model(server, archive_file, monkeypatch):
    llm = scripted(monkeypatch, cite(FLAT_TIRE))
    stream(server, {"messagesPath": archive_file, "question": QUESTION})
    told = sent_to(llm)
    assert FLAT_TIRE in told
    assert world.UNSAVED not in told and world.UNSAVED_EMAIL not in told


def test_the_scope_of_a_question_is_its_conversation(server, archive_file, monkeypatch):
    scripted(monkeypatch, say("Nothing in this conversation says so."))
    events = stream(server, {"messagesPath": archive_file, "question": QUESTION, "contact": world.INES_PHONE})
    payload = events[-1]["payload"]
    assert payload["contact"] == world.INES_PHONE and payload["citations"] == []


def test_an_earlier_answer_goes_to_a_question_of_the_same_scope_only(server, archive_file, monkeypatch):
    """A "user" row of the history says in `contact` which conversation its question was asked of."""
    earlier = "Tomas has the wheelbarrow behind the shed."
    asked_here = [{"role": "user", "content": "Who has it?", "contact": world.GARDEN}, {"role": "assistant", "content": earlier}]
    asked_of_all = [{"role": "user", "content": "Who has it?"}, {"role": "assistant", "content": earlier}]
    body = {"messagesPath": archive_file, "question": "And is it broken?", "contact": world.GARDEN}

    same_scope = scripted(monkeypatch, say("Yes."))
    stream(server, {**body, "history": asked_here})
    assert earlier in sent_to(same_scope)

    other_scope = scripted(monkeypatch, say("Yes."))
    stream(server, {**body, "history": asked_of_all})
    assert earlier not in sent_to(other_scope)


def test_with_no_key_the_answer_is_one_plain_sentence_and_no_keyword_list(server, archive_file, tmp_path, monkeypatch):
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", tmp_path / "no-key-file")
    events = stream(server, {"messagesPath": archive_file, "question": QUESTION})
    assert events[-1] == {"type": "error", "code": "no_key", "error": NO_KEY}
    assert not any(event["type"] in ("delta", "done") for event in events)
    status, said = post(server, "/api/ask", {"messagesPath": archive_file, "question": QUESTION})
    assert (status, said) == (502, {"error": NO_KEY, "code": "no_key"})


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"question": "   "}, "empty_question"),
        ({"question": "x" * 8_001}, "question_too_long"),
        ({"question": QUESTION, "model": "gpt-5.5"}, "model_not_offered"),
        ({"question": QUESTION, "model": "no-such-model"}, "unknown_model"),
    ],
)
def test_a_question_that_cannot_be_asked_is_refused_with_a_code(server, archive_file, monkeypatch, body, code):
    llm = scripted(monkeypatch)
    status, said = post(server, "/api/ask", {"messagesPath": archive_file, **body})
    assert status == 400 and said["code"] == code and said["error"].endswith(".")
    assert stream(server, {"messagesPath": archive_file, **body})[-1]["code"] == code
    assert llm.requests == []


def test_an_answer_with_no_ai_says_that_no_model_was_used(server, archive_file, monkeypatch):
    llm = scripted(monkeypatch)
    status, payload = post(server, "/api/ask", {"messagesPath": archive_file, "question": "wheelbarrow", "ai": False})
    assert status == 200 and payload["mode"] == "local" and payload["answer"].startswith("No AI model was used.")
    assert llm.requests == []


def test_a_question_is_stopped_by_its_id(server, archive_file, monkeypatch):
    started = threading.Event()

    def write_until_stopped(request):
        started.set()
        request.cancel.wait(10)
        return say("This is the start of an answer")

    scripted(monkeypatch, write_until_stopped)
    events = []
    asking = threading.Thread(
        target=lambda: events.extend(stream(server, {"messagesPath": archive_file, "question": QUESTION, "requestId": "q-1"}))
    )
    asking.start()
    assert started.wait(5)
    assert post(server, "/api/ask/cancel", {"requestId": "q-1"}) == (200, {"cancelled": True})
    asking.join(10)
    assert not asking.is_alive()
    assert events[-1]["type"] == "done" and events[-1]["payload"]["cancelled"] is True
    # the id is forgotten with its question
    assert post(server, "/api/ask/cancel", {"requestId": "q-1"}) == (200, {"cancelled": False})
    assert ui_server.ASKING == {}


@pytest.mark.parametrize("body", [{}, {"requestId": "nobody-asked-this"}, {"requestId": 7}, {"requestId": ""}])
def test_stopping_a_question_that_nobody_asked_stops_nothing(server, body):
    assert post(server, "/api/ask/cancel", body) == (200, {"cancelled": False})


def test_a_reader_who_leaves_ends_the_model_call(server, archive_file, monkeypatch):
    """The write to a reader who is gone fails. The question is then stopped, and the model call is closed."""
    llm = scripted(monkeypatch, say("word " * 200_000, chunk=5))
    body = json.dumps({"messagesPath": archive_file, "question": QUESTION}).encode("utf-8")
    reader = socket.create_connection(("127.0.0.1", server.server_port), timeout=5)
    try:
        reader.sendall(
            b"POST /api/ask/stream HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        # stay until the words of the answer arrive, then go in the middle of them
        seen = b""
        while b'"delta"' not in seen:
            part = reader.recv(4096)
            assert part, "the stream ended before a word of the answer came"
            seen += part
    finally:
        reader.close()
    limit = time.monotonic() + 10
    while llm.closed_early == 0 and time.monotonic() < limit:
        time.sleep(0.02)
    assert (llm.closed_early, llm.finished, len(llm.requests)) == (1, 0, 1)


def wait_for_the_import():
    limit = time.monotonic() + 10
    while ui_server.unfinished_import() and time.monotonic() < limit:
        time.sleep(0.02)
    assert ui_server.unfinished_import() is None


def test_a_question_waits_while_the_archive_is_loaded(server, tmp_path, monkeypatch, no_jobs):
    """An export of an earlier build with no archive beside it: the load starts, and the question is told to wait."""
    llm = scripted(monkeypatch, say("The ferry leaves at nine."))
    export = tmp_path / "messages.csv"
    export.write_text(
        "message_id,timestamp,sender,text,chat_id,chat_display_name,is_from_me,service,attachment_types,attachment_files\n"
        "1,2024-01-01 09:00:00,+15550180001,the ferry leaves at nine,+15550180001,,0,iMessage,,\n",
        encoding="utf-8",
    )
    try:
        status, said = post(server, "/api/ask", {"messagesPath": str(export), "question": QUESTION})
        assert status == 503 and said["code"] == "archive_building" and said["job"]["action"] == "bridge"
        assert llm.requests == []
        wait_for_the_import()
        # the archive is there now: the same question is answered
        events = stream(server, {"messagesPath": str(export), "question": "When does the ferry leave?"})
        assert events[-1]["type"] == "done" and events[-1]["payload"]["answer"] == "The ferry leaves at nine."
    finally:
        wait_for_the_import()
        places.forget()


@pytest.mark.parametrize("path", ["/api/ask", "/api/ask/stream", "/api/ask/cancel"])
def test_a_body_that_is_no_record_is_refused(server, path):
    status, _, raw = _ask(server, "POST", path, b"[1, 2]")
    assert status == 400 and json.loads(raw) == {"error": "The request must be a JSON object."}
