"""The server starts with no network: no vocabulary is fetched at import, and the app ships its own."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"


def no_vocabulary(name):
    raise ConnectionError("no network")


@pytest.fixture
def offline_conversation(monkeypatch):
    """`conversation` imported again while every vocabulary fails to load."""
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", no_vocabulary)
    import conversation

    module = importlib.reload(conversation)
    yield module
    monkeypatch.undo()
    importlib.reload(conversation)


def test_importing_conversation_loads_no_vocabulary(monkeypatch):
    import tiktoken

    asked = []
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: asked.append(name) or no_vocabulary(name))
    import conversation

    importlib.reload(conversation)
    assert asked == []
    monkeypatch.undo()
    importlib.reload(conversation)


def test_with_no_vocabulary_tokens_are_counted_roughly(offline_conversation):
    assert offline_conversation.estimate_tokens("") == 0
    assert offline_conversation.estimate_tokens("abcdef") == 2
    assert offline_conversation.estimate_tokens("abcdefg") == 3


def test_with_no_vocabulary_a_text_is_cut_from_its_end(offline_conversation):
    text = "first part. " * 50 + "the end"
    cut = offline_conversation.truncate_to_tokens(text, 3)
    assert cut == text[-9:]
    assert offline_conversation.truncate_to_tokens("short", 10) == "short"
    assert offline_conversation.truncate_to_tokens("text", 0) == ""


def test_a_cut_never_splits_a_letter(offline_conversation):
    cut = offline_conversation.truncate_to_tokens("ab" + "é" * 10, 1)
    cut.encode("utf-8")
    assert set(cut) <= {"é"}


def test_a_message_that_holds_a_special_token_is_counted():
    import conversation

    try:
        count = conversation.estimate_tokens("see <|endoftext|> here")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"a special token in a message raised {type(exc).__name__}")
    assert count > 0
    assert "<|endoftext|>" in conversation.truncate_to_tokens("x <|endoftext|>", 50)


def load_entry(monkeypatch, frozen_root: Path | None):
    """`packaging/server_entry.py` imported as the frozen app imports it, with no `ui_server` behind it."""
    monkeypatch.setitem(sys.modules, "ui_server", type(sys)("ui_server"))
    if frozen_root is not None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(frozen_root), raising=False)
    else:
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.syspath_prepend(str(PACKAGING))
    sys.modules.pop("server_entry", None)
    return importlib.import_module("server_entry")


def test_the_frozen_app_reads_its_own_vocabulary(monkeypatch, tmp_path):
    (tmp_path / "tiktoken_cache").mkdir()
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    entry = load_entry(monkeypatch, tmp_path)
    entry.use_bundled_vocabulary()
    import os

    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(tmp_path / "tiktoken_cache")


def test_a_folder_that_was_set_is_kept(monkeypatch, tmp_path):
    (tmp_path / "tiktoken_cache").mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/somewhere/else")
    entry = load_entry(monkeypatch, tmp_path)
    entry.use_bundled_vocabulary()
    import os

    assert os.environ["TIKTOKEN_CACHE_DIR"] == "/somewhere/else"


def test_outside_the_app_nothing_is_set(monkeypatch, tmp_path):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    entry = load_entry(monkeypatch, None)
    entry.use_bundled_vocabulary()
    import os

    assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_the_spec_ships_the_vocabulary_folder():
    spec = (PACKAGING / "recall_server.spec").read_text(encoding="utf-8")
    assert '"tiktoken_cache"' in spec
    assert "o200k_base" in spec and "cl100k_base" in spec
