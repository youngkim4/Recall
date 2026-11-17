"""Load legacy modules of the repo for comparison, without their side effects."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_legacy_conversation(monkeypatch: pytest.MonkeyPatch) -> Any:
    """conversation.py under a private name, with a stub tokenizer.

    The real module loads a tiktoken vocabulary at import, which can need the
    network. The functions compared here do not use the tokenizer.
    """
    stub = types.ModuleType("tiktoken")
    stub.get_encoding = lambda name: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", stub)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)  # never read a real .env
    spec = importlib.util.spec_from_file_location("legacy_conversation_for_tests", REPO_ROOT / "conversation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
