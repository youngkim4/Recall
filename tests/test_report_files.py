"""Tests for how a saved report file is handed to the browser."""
import http.client
import threading
from urllib.parse import quote

import pytest

import ui_server
from ui_server import report_file_headers, served_report_file


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A data folder of its own, with nothing of the real one in reach."""
    root = tmp_path / "data"
    (root / "out").mkdir(parents=True)
    monkeypatch.setattr(ui_server, "DATA_DIR", root)
    monkeypatch.setattr(ui_server, "ROOT", root)
    return root


def write(path, text="x"):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", [
    "analysis_+15550000001.md",
    "analysis_+15550000001.html",
    "events_timeline_+15550000001.csv",
    "analysis_averyexamplecom.MD",
])
def test_a_report_file_is_served(data_dir, name):
    path = write(data_dir / "out" / name)

    assert served_report_file(str(path)) == path.resolve()


@pytest.mark.parametrize("name", [
    ".env",
    "messages.csv",
    "messages_attachments.csv",
    "contacts_cache.json",
    "analysis_.md.py",
    "notes.md",
    "analysis_x.svg",
    "analysis_x.xhtml",
])
def test_a_file_that_is_not_a_report_is_not_served(data_dir, name):
    path = write(data_dir / "out" / name)

    assert served_report_file(str(path)) is None


def test_a_report_outside_the_data_folder_is_not_served(data_dir, tmp_path):
    outside = write(tmp_path / "analysis_elsewhere.md")

    assert served_report_file(str(outside)) is None
    assert served_report_file(str(data_dir / "out" / ".." / ".." / "analysis_elsewhere.md")) is None


def test_a_link_out_of_the_data_folder_is_followed_before_it_is_judged(data_dir, tmp_path):
    secret = write(tmp_path / "secret.txt", "not a report")
    link = data_dir / "out" / "analysis_link.md"
    link.symlink_to(secret)

    assert served_report_file(str(link)) is None


def test_a_link_to_another_file_inside_the_data_folder_is_judged_by_that_file(data_dir):
    env = write(data_dir / ".env", "KEY=value")
    link = data_dir / "out" / "analysis_link.md"
    link.symlink_to(env)

    assert served_report_file(str(link)) is None


@pytest.mark.parametrize("raw", ["", "analysis_missing.md", "bad\x00name.md"])
def test_a_path_that_names_no_file_is_not_served(data_dir, raw):
    assert served_report_file(raw) is None


def test_a_folder_is_not_served(data_dir):
    folder = data_dir / "out" / "analysis_folder.md"
    folder.mkdir()

    assert served_report_file(str(folder)) is None


def test_every_report_file_is_served_as_what_it_says_it_is(data_dir):
    for name in ("analysis_x.md", "analysis_x.html", "events_timeline_x.csv"):
        headers = report_file_headers(data_dir / "out" / name)

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "charset=utf-8" in headers["Content-Type"]


def test_an_html_report_opens_in_an_origin_of_its_own(data_dir):
    headers = report_file_headers(data_dir / "out" / "analysis_x.html")

    assert headers["Content-Type"].startswith("text/html")
    # A report saved by an older build may hold a handler in its text. Nothing
    # in the page runs, and nothing is fetched, so nothing can be sent out.
    policy = [part.strip() for part in headers["Content-Security-Policy"].split(";")]
    assert policy[0] == "sandbox"
    assert "default-src 'none'" in policy
    assert not any("allow-scripts" in part or "allow-same-origin" in part for part in policy)
    assert not any(part.startswith(("script-src", "connect-src")) for part in policy)


def test_a_report_that_is_text_is_served_as_text(data_dir):
    markdown = report_file_headers(data_dir / "out" / "analysis_x.md")
    timeline = report_file_headers(data_dir / "out" / "events_timeline_x.csv")

    # with the type fixed and sniffing off, neither can be read as a page
    assert markdown["Content-Type"] == "text/markdown; charset=utf-8"
    assert timeline["Content-Type"] == "text/csv; charset=utf-8"


@pytest.fixture
def server(data_dir):
    try:
        httpd = ui_server.LoopbackHTTPServer(("127.0.0.1", 0), ui_server.RecallHandler)
    except OSError as exc:
        pytest.skip(f"no loopback socket here: {exc}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def fetch(server, path):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_the_server_sends_an_html_report_sandboxed(data_dir, server):
    report = write(data_dir / "out" / "analysis_x.html", "<html><body onload='alert(1)'>hi</body></html>")

    status, headers, body = fetch(server, f"/api/report?path={quote(str(report), safe='')}")

    assert status == 200
    assert body == report.read_bytes()
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Content-Security-Policy"].startswith("sandbox; default-src 'none'")
    assert "allow-scripts" not in headers["Content-Security-Policy"]


def test_the_server_sends_a_markdown_report_as_text(data_dir, server):
    report = write(data_dir / "out" / "analysis_x.md", "# Conversation Analysis - x\n\n<script>alert(1)</script>\n")

    status, headers, body = fetch(server, f"/api/report?path={quote(str(report), safe='')}")

    assert status == 200
    assert body == report.read_bytes()
    assert headers["Content-Type"] == "text/markdown; charset=utf-8"
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_the_server_refuses_a_file_that_is_not_a_report(data_dir, server):
    env = write(data_dir / ".env", "KEY=value")

    status, _headers, body = fetch(server, f"/api/report?path={quote(str(env), safe='')}")

    assert status == 404
    assert b"KEY" not in body
