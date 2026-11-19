"""Every POST and DELETE under /api/ comes from the page of this server, or names no origin.

A page of another local port can send a simple POST (`text/plain`, no
preflight). It must not start a paid job or ask a question.
"""
import http.client
import json
import threading

import pytest

import ui_server
from tests.archive_kit import data_folder, listed, no_names  # noqa: F401

pytestmark = pytest.mark.usefixtures("data_folder", "no_names", "no_jobs", "held_jobs")

OTHER_PORT = "http://localhost:9999"


@pytest.fixture
def held_jobs(monkeypatch):
    """Jobs that are made and never run: a request that passes makes a queued job, and nothing is paid."""
    let_go = threading.Event()
    monkeypatch.setattr(ui_server, "run_job", lambda job_id, action, payload: let_go.wait(timeout=5))
    yield
    let_go.set()
    with ui_server.JOBS_LOCK:
        ui_server.JOBS.clear()


def send(server, method, path, payload, origin=None):
    """A simple request, as a page sends it with no preflight: the body is text/plain."""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "text/plain"}
        if origin is not None:
            headers["Origin"] = origin
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


@pytest.mark.parametrize("action", ["first_run", "memory", "people", "meaning"])
def test_a_page_of_another_local_port_starts_no_job(server, action):
    status, answer = send(server, "POST", "/api/jobs", {"action": action}, origin=OTHER_PORT)
    assert status == 403 and json.loads(answer)["error"]
    assert listed() == []


def test_a_page_of_another_local_port_asks_no_question(server, monkeypatch):
    asked = []
    monkeypatch.setattr(ui_server, "answer_question", lambda payload: asked.append(payload) or (200, {}))
    for path in ("/api/ask", "/api/ask/stream"):
        status, _answer = send(server, "POST", path, {"question": "anything"}, origin=OTHER_PORT)
        assert status == 403
    assert asked == []


@pytest.mark.parametrize("path", ["/api/setup/complete", "/api/contact-names", "/api/cache/preview", "/api/ask/cancel"])
def test_every_other_write_refuses_another_local_port(server, path):
    assert send(server, "POST", path, {}, origin=OTHER_PORT)[0] == 403


def test_a_delete_refuses_another_local_port(server):
    assert send(server, "DELETE", "/api/jobs", {}, origin=OTHER_PORT)[0] == 403


@pytest.mark.parametrize("origin", [None, "own"])
def test_the_page_of_this_server_and_a_request_with_no_origin_start_a_job(server, origin):
    named = f"http://127.0.0.1:{server.server_port}" if origin == "own" else None
    status, answer = send(server, "POST", "/api/jobs", {"action": "memory"}, origin=named)
    assert (status, json.loads(answer)["job"]["action"]) == (202, "memory")
    assert listed() == [("memory", "queued")]


def test_reading_routes_keep_the_check_they_had(server):
    """A GET changes nothing: the Vite dev server on another local port may still read."""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", "/api/jobs", headers={"Origin": OTHER_PORT})
        assert connection.getresponse().status == 200
    finally:
        connection.close()
