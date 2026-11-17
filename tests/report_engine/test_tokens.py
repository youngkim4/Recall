"""Token counting: offline by default in tests, with a margin on a family mismatch."""

import sys
import types
from collections.abc import Sequence

import pytest

from report_engine import tokens
from report_engine.tokens import (
    ENGINE_ENCODING,
    HeuristicCounter,
    TiktokenCounter,
    TokenMeter,
    load_counter,
    meter_for,
)

from .synth import WordCounter, model_info


class StubEncoding:
    """Stands in for a tiktoken encoding: one token per character."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode_ordinary_batch(self, texts: list[str]) -> list[list[int]]:
        self.batches.append(texts)
        return [list(range(len(text))) for text in texts]


def install_tiktoken(
    monkeypatch: pytest.MonkeyPatch, names: Sequence[str], fail: bool = False
) -> list[str]:
    asked: list[str] = []

    def get_encoding(name: str) -> StubEncoding:
        asked.append(name)
        if fail:
            raise OSError("vocabulary file is not on disk")
        return StubEncoding()

    fake = types.ModuleType("tiktoken")
    fake.list_encoding_names = lambda: list(names)  # type: ignore[attr-defined]
    fake.get_encoding = get_encoding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    return asked


def test_heuristic_counts_by_class_of_character_and_by_line() -> None:
    counter = HeuristicCounter()
    # 4 letters, 1.5 digits, 4 spaces or 3 other bytes for one token; one for each mark and line
    assert counter.count_many(["", "abcd", "abcd abcd", "12 34", "a,b", "one\ntwo"]) == [
        0,
        2,
        4,
        4,
        3,
        4,
    ]


def test_heuristic_counts_other_scripts_by_bytes() -> None:
    assert HeuristicCounter().count_many(["가나다", "héllo"]) == [4, 3]


def test_tiktoken_counter_counts_in_one_batch() -> None:
    encoding = StubEncoding()
    counter = TiktokenCounter(encoding, "some_encoding")
    assert counter.count_many(["ab", "", "abcd"]) == [2, 0, 4]
    assert encoding.batches == [["ab", "", "abcd"]]
    assert counter.family == "some_encoding"


def test_load_counter_uses_the_models_encoding_when_tiktoken_has_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = install_tiktoken(monkeypatch, ["enc_a", ENGINE_ENCODING])
    counter = load_counter("enc_a")
    assert counter.family == "enc_a"
    assert asked == ["enc_a"]


def test_load_counter_falls_back_to_the_engine_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = install_tiktoken(monkeypatch, ["enc_a", ENGINE_ENCODING])
    counter = load_counter("a-family-tiktoken-does-not-know")
    assert counter.family == ENGINE_ENCODING
    assert asked == [ENGINE_ENCODING]


def test_load_counter_uses_the_heuristic_when_the_vocabulary_can_not_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    install_tiktoken(monkeypatch, [ENGINE_ENCODING], fail=True)
    with caplog.at_level("WARNING", logger=tokens.__name__):
        counter = load_counter(ENGINE_ENCODING)
    assert isinstance(counter, HeuristicCounter)
    assert "heuristic" in caplog.text


def test_load_counter_uses_the_heuristic_without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    assert isinstance(load_counter(ENGINE_ENCODING), HeuristicCounter)


def test_meter_has_no_margin_when_families_match() -> None:
    meter = meter_for(model_info(tokenizer_family="words"), WordCounter())
    assert meter.margin == 1.0
    assert meter.count("one two three") == 3


def test_meter_adds_the_margin_when_families_differ() -> None:
    meter = meter_for(model_info(tokenizer_family="other"), WordCounter(), mismatch_margin=1.5)
    assert meter.margin == 1.5
    assert meter.count_many(["one two three", "", "one"]) == [5, 0, 2]


def test_meter_rejects_a_margin_below_one() -> None:
    with pytest.raises(ValueError):
        TokenMeter(WordCounter(), 0.9)


def test_meter_names_the_tokenizer_it_counts_with() -> None:
    meter = meter_for(model_info(tokenizer_family="other"), WordCounter(), mismatch_margin=1.35)
    assert meter.describe() == "words x1.35"


def test_meter_loads_a_counter_when_none_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    install_tiktoken(monkeypatch, ["enc_a"])
    meter = meter_for(model_info(tokenizer_family="enc_a"))
    assert meter.counter.family == "enc_a"
    assert meter.margin == 1.0
