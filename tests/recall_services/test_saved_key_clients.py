"""Every model client reads the key that was saved in the app, at the next call, with no new start.

A client is made and asked for its key. No request is sent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import recall_paths
from chat_engine.errors import NoApiKeyError
from chat_engine.openai_llm import OpenAIResponsesLLM
from recall_services import key_store, keys
from report_engine.openai_llm import build_client

INVENTED = "sk-invented-" + "Q7x" * 10


@pytest.fixture
def key_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", path)
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    return path


def test_the_chat_client_reads_a_key_saved_after_it_was_made(key_file: Path) -> None:
    llm = OpenAIResponsesLLM()
    with pytest.raises(NoApiKeyError):
        llm._ready_client()
    key_store.save(INVENTED)
    assert llm._ready_client().api_key == INVENTED


def test_the_chat_client_still_takes_the_key_it_is_given(key_file: Path) -> None:
    key_store.save(INVENTED)
    assert OpenAIResponsesLLM(api_key="sk-given-" + "a" * 20)._secret() == "sk-given-" + "a" * 20


def test_the_chat_client_reads_the_environment_first(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_store.save(INVENTED)
    monkeypatch.setenv(keys.KEY_NAME, "sk-from-the-shell-" + "x" * 20)
    assert OpenAIResponsesLLM()._secret() == "sk-from-the-shell-" + "x" * 20


def test_the_report_client_reads_the_saved_key_when_none_is_given(key_file: Path) -> None:
    key_store.save(INVENTED)
    assert build_client().api_key == INVENTED
    assert build_client("sk-given-" + "a" * 20).api_key == "sk-given-" + "a" * 20


def test_the_report_client_reads_the_environment_first(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_store.save(INVENTED)
    monkeypatch.setenv(keys.KEY_NAME, "sk-from-the-shell-" + "x" * 20)
    assert build_client().api_key == "sk-from-the-shell-" + "x" * 20
