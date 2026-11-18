"""The default token counter. No test lets tiktoken look for its file: a stand-in takes its place."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from meaning_index import tokens


class Encoding:
    def encode_ordinary(self, text: str) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture(autouse=True)
def fresh_counter() -> Iterator[None]:
    tokens.forget_encoding()
    yield
    tokens.forget_encoding()


def test_the_default_counts_with_the_encoding_of_the_embedding_models(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def get_encoding(name: str) -> Encoding:
        asked.append(name)
        return Encoding()

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    assert tokens.default_count_tokens("one two three") == 3
    assert tokens.default_count_tokens("one") == 1
    assert asked == ["cl100k_base"]  # loaded one time


def test_when_the_encoding_can_not_load_the_count_is_rough_and_the_log_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def get_encoding(name: str) -> Encoding:
        raise OSError("no vocabulary file here")

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    with caplog.at_level(logging.WARNING, logger="meaning_index"):
        assert tokens.default_count_tokens("abcdef") == tokens.rough_count("abcdef")
        tokens.default_count_tokens("abcdef")
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "OSError" in warnings[0].getMessage()
    assert "abcdef" not in caplog.text


def test_the_rough_count_never_counts_below_one_token_for_three_bytes() -> None:
    assert tokens.rough_count("") == 0
    assert tokens.rough_count("a") == 1
    assert tokens.rough_count("abcdef") == 2
    assert tokens.rough_count("abcdefg") == 3
    assert tokens.rough_count("ééé") == 2  # six bytes
