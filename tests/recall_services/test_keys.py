"""keys.py: where the key of the model provider comes from."""

from __future__ import annotations

from pathlib import Path

import pytest

import recall_paths
from recall_services import keys


@pytest.fixture
def key_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", path)
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    return path


def test_no_key_anywhere_is_no_key(key_file: Path) -> None:
    assert keys.api_key() == ""


def test_the_key_of_the_environment_is_used(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(keys.KEY_NAME, "  from-the-environment  ")
    key_file.write_text("OPENAI_API_KEY=from-the-file\n", encoding="utf-8")
    assert keys.api_key() == "from-the-environment"


def test_the_key_file_of_the_data_folder_is_read(key_file: Path) -> None:
    """The Mac app starts the server with no shell environment: the key is in the data folder."""
    key_file.write_text('# a note\nOTHER=1\nOPENAI_API_KEY="from-the-file"\n', encoding="utf-8")
    assert keys.api_key() == "from-the-file"


def test_a_key_that_was_added_is_found_without_a_new_start(key_file: Path) -> None:
    assert keys.api_key() == ""
    key_file.write_text("OPENAI_API_KEY=added-later\n", encoding="utf-8")
    assert keys.api_key() == "added-later"


def test_reading_the_key_file_changes_no_environment(key_file: Path) -> None:
    import os

    key_file.write_text("OPENAI_API_KEY=from-the-file\nOPENAI_BASE_URL=https://elsewhere.invalid\n", encoding="utf-8")
    keys.api_key()
    assert keys.KEY_NAME not in os.environ and "OPENAI_BASE_URL" not in os.environ


@pytest.mark.parametrize("content", [b"\xff\xfe\x00not text", b"OPENAI_API_KEY\n", b"OPENAI_API_KEY=\n"])
def test_a_key_file_with_no_key_in_it_is_no_key(key_file: Path, content: bytes) -> None:
    key_file.write_bytes(content)
    assert keys.api_key() == ""


def test_a_folder_in_the_place_of_the_key_file_is_no_key(key_file: Path) -> None:
    key_file.mkdir()
    assert keys.api_key() == ""


def test_no_file_above_the_data_folder_is_looked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout inside another checkout must not find the key file of the outer one."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=of-the-outer-folder\n", encoding="utf-8")
    inner = tmp_path / "inner"
    inner.mkdir()
    monkeypatch.setattr(recall_paths, "DOTENV_PATH", inner / ".env")
    monkeypatch.delenv(keys.KEY_NAME, raising=False)
    monkeypatch.chdir(inner)
    assert keys.api_key() == ""


def test_nothing_of_the_environment_is_filled_into_the_key(key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the key is taken from the file, as it is written."""
    monkeypatch.setenv("SOMETHING_ELSE", "a-value-of-the-environment")
    key_file.write_text("OPENAI_API_KEY=${SOMETHING_ELSE}\n", encoding="utf-8")
    assert keys.api_key() == "${SOMETHING_ELSE}"
