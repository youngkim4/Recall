"""GET, POST and DELETE /api/key: a person gives the key in the app, and no other program can.

The key is an invented one, and the data folder is a temporary one. No test
reads or writes a real `.env`.
"""
import http.client
import io
import json
import logging
import stat

import pytest

import ui_server
from recall_services import key_store, keys
from tests.archive_kit import data_folder  # noqa: F401

pytestmark = pytest.mark.usefixtures("data_folder", "no_key_in_the_environment")

INVENTED = "sk-invented-" + "Q7x" * 10
PLACE = keys.KEY_PLACE
NONE_SAVED = {"present": False, "source": None, "place": PLACE}
SAVED = {"present": True, "source": "file", "place": PLACE}


@pytest.fixture
def no_key_in_the_environment(monkeypatch):
    monkeypatch.delenv(keys.KEY_NAME, raising=False)


def ask(server, method, path="/api/key", payload=None, *, origin=None, host=None, raw=None):
    """One request with the headers a browser or another program would send. Gives the status and the answer as text."""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        body = raw if raw is not None else (None if payload is None else json.dumps(payload).encode("utf-8"))
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if origin is not None:
            headers["Origin"] = origin
        if host is not None:
            headers["Host"] = host
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def own_origin(server):
    return f"http://127.0.0.1:{server.server_port}"


# --- the three routes -----------------------------------------------------------------------


def test_get_says_there_is_no_key(server):
    assert ask(server, "GET") == (200, json.dumps(NONE_SAVED))


def test_post_saves_the_key_and_get_tells_where_it_is_with_no_part_of_it(server, data_folder):  # noqa: F811
    status, answer = ask(server, "POST", payload={"key": INVENTED})
    assert (status, json.loads(answer)) == (200, SAVED)
    assert (data_folder / ".env").read_text(encoding="utf-8") == f"OPENAI_API_KEY={INVENTED}\n"
    assert stat.S_IMODE((data_folder / ".env").stat().st_mode) == 0o600
    status, answer = ask(server, "GET")
    assert (status, json.loads(answer)) == (200, SAVED)
    assert INVENTED[-4:] not in answer and keys.api_key() == INVENTED


def test_delete_removes_the_key(server, data_folder):  # noqa: F811
    (data_folder / ".env").write_text(f"KEEP=1\nOPENAI_API_KEY={INVENTED}\n", encoding="utf-8")
    status, answer = ask(server, "DELETE")
    assert (status, json.loads(answer)) == (200, NONE_SAVED)
    assert (data_folder / ".env").read_text(encoding="utf-8") == "KEEP=1\n"


def test_a_value_that_is_no_key_gets_400_in_a_sentence_without_it(server, data_folder):  # noqa: F811
    status, answer = ask(server, "POST", payload={"key": "sk-short"})
    assert status == 400 and "sk-short" not in answer and json.loads(answer)["error"]
    assert not (data_folder / ".env").exists()


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b'{"key": 7}'])
def test_a_body_that_holds_no_key_gets_400(server, raw):
    status, answer = ask(server, "POST", raw=raw)
    assert status == 400 and json.loads(answer)["error"]


def test_with_a_key_in_the_environment_post_gets_409(server, monkeypatch, data_folder):  # noqa: F811
    monkeypatch.setenv(keys.KEY_NAME, "sk-from-the-shell-" + "x" * 20)
    status, answer = ask(server, "POST", payload={"key": INVENTED})
    assert (status, json.loads(answer)) == (409, {"error": key_store.ENVIRONMENT_WINS})
    assert not (data_folder / ".env").exists()
    assert json.loads(ask(server, "GET")[1]) == {"present": True, "source": "environment", "place": PLACE}


def test_a_method_the_route_does_not_have_is_404_for_another_path(server):
    assert ask(server, "DELETE", "/api/jobs")[0] == 404


# --- who may ask -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_a_page_of_another_local_port_is_refused(server, method, data_folder):  # noqa: F811
    status, answer = ask(server, method, payload={"key": INVENTED}, origin="http://127.0.0.1:5173")
    assert status == 403 and json.loads(answer)["error"]
    assert not (data_folder / ".env").exists()


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_a_page_of_another_host_is_refused(server, method):
    status, _answer = ask(server, method, payload={"key": INVENTED}, host="attacker.example:80")
    assert status == 403


def test_the_page_of_this_server_and_a_request_with_no_origin_pass(server):
    assert ask(server, "POST", payload={"key": INVENTED}, origin=own_origin(server))[0] == 200
    assert ask(server, "GET", origin=own_origin(server))[0] == 200
    assert ask(server, "DELETE")[0] == 200


def test_other_routes_keep_the_check_they_had(server):
    """A page of another local port (the Vite dev server) may still read the other routes."""
    status, _answer = ask(server, "GET", "/api/jobs", origin="http://127.0.0.1:5173")
    assert status == 200


# --- the key never leaves -----------------------------------------------------------------------


def test_the_key_is_in_no_log_and_no_answer(server, data_folder, monkeypatch, capfd, caplog):  # noqa: F811
    logged = io.StringIO()
    monkeypatch.setattr(ui_server.sys, "__stderr__", logged)
    caplog.set_level(logging.DEBUG)
    answers = [
        ask(server, "POST", payload={"key": INVENTED}),
        ask(server, "POST", payload={"key": INVENTED + " with a space"}),
        ask(server, "POST", payload={"key": INVENTED}, origin="http://127.0.0.1:5173"),
        ask(server, "GET"),
        ask(server, "GET", "/api/first-run"),
        ask(server, "GET", "/api/jobs"),
        ask(server, "DELETE"),
    ]
    seen = "\n".join([*(text for _status, text in answers), logged.getvalue(), caplog.text, *capfd.readouterr()])
    assert INVENTED not in seen and INVENTED[3:] not in seen and INVENTED[-8:] not in seen


def test_a_key_that_can_not_be_written_is_told_with_no_trace_of_it(server, data_folder, monkeypatch, capfd):  # noqa: F811
    logged = io.StringIO()
    monkeypatch.setattr(ui_server.sys, "__stderr__", logged)
    (data_folder / ".env").mkdir()
    status, answer = ask(server, "POST", payload={"key": INVENTED})
    assert (status, json.loads(answer)) == (500, {"error": key_store.NOT_WRITTEN})
    seen = "\n".join([answer, logged.getvalue(), *capfd.readouterr()])
    assert INVENTED not in seen and INVENTED[-8:] not in seen
