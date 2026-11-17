"""Ask the test server (the `server` fixture) for a path."""
import http.client


def fetch(server, path: str) -> tuple[int, dict, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()
