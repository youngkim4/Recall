"""The SDK client of the report engine takes nothing from the environment that can move a transcript."""

from __future__ import annotations

import pytest

from report_engine.openai_llm import OFFICIAL_BASE_URL, build_client

DUMMY = "sk-test-dummy-not-a-real-key"


def test_the_client_takes_no_proxy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy variable of the server's environment would see where every report goes."""
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    client = build_client(DUMMY)
    assert client._client.trust_env is False


def test_the_client_keeps_its_address_and_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid/v1")
    client = build_client(DUMMY, timeout_s=45.0)
    assert str(client.base_url).rstrip("/") == OFFICIAL_BASE_URL
    assert client.timeout == 45.0 and client.max_retries == 0
