"""Ask the test server (the `server` fixture) for a path."""
import http.client
import json


def _ask(server, method: str, path: str, body: bytes | None = None) -> tuple[int, dict, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def fetch(server, path: str) -> tuple[int, dict, bytes]:
    return _ask(server, "GET", path)


def post(server, path: str, payload) -> tuple[int, dict]:
    """Send `payload` as JSON. Gives the status and the answer, read as JSON."""
    status, _, body = _ask(server, "POST", path, json.dumps(payload).encode("utf-8"))
    return status, json.loads(body)


def post_bytes(server, path: str, body: bytes) -> tuple[int, dict]:
    """Send `body` as it is, for a request that is not well made. Gives the status and the answer, read as JSON."""
    status, _, answer = _ask(server, "POST", path, body)
    return status, json.loads(answer)

