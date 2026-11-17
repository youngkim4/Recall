"""The SDK client that the chat port makes for itself takes nothing from the environment that can move a question."""

from __future__ import annotations

import pytest

from chat_engine.openai_llm import OFFICIAL_BASE_URL, OpenAIResponsesLLM

DUMMY = "sk-test-dummy-not-a-real-key"


def test_the_port_goes_to_the_address_of_the_provider_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OPENAI_BASE_URL in the environment can not send message text to another address."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid/v1")
    client = OpenAIResponsesLLM(api_key=DUMMY)._ready_client()
    assert str(client.base_url).rstrip("/") == OFFICIAL_BASE_URL


def test_the_port_takes_no_proxy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = OpenAIResponsesLLM(api_key=DUMMY)._ready_client()
    assert client._client.trust_env is False
    assert client.max_retries == 0
