"""The facts of the embedding models: one file, read as it is written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meaning_index.errors import ModelFactsError
from meaning_index.model_facts import FACTS_PATH, EmbedderFacts, embedder_facts, load_facts


def test_the_file_holds_the_facts_of_the_contract() -> None:
    raw = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    assert raw["facts_read_on"] == "2026-11-17"
    assert raw["default"] == {"model": "text-embedding-3-large", "dimensions": 1024}
    models = {entry["id"]: entry for entry in raw["models"]}
    small, large = models["text-embedding-3-small"], models["text-embedding-3-large"]
    assert (small["native_dimensions"], small["usd_per_mtok"]) == (1536, 0.02)
    assert (large["native_dimensions"], large["usd_per_mtok"]) == (3072, 0.13)
    for entry in (small, large):
        assert entry["max_input_tokens"] == 8192
        assert entry["can_shorten"] is True
        assert entry["provider"] == "openai"


def test_the_request_limits_are_marked_as_not_read() -> None:
    raw = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    for entry in raw["models"]:
        assert entry["max_inputs_per_request"] is None
        assert entry["max_tokens_per_request"] is None
        assert entry["tokens_per_minute"] is None
    assert any("not read" in note for note in raw["notes"])


def test_the_default_is_the_large_model_at_1024_dimensions() -> None:
    facts = embedder_facts()
    assert facts == EmbedderFacts(
        model="text-embedding-3-large",
        dimensions=1024,
        usd_per_mtok=0.13,
        max_input_tokens=8192,
        tokenizer_family="cl100k_base",
    )


def test_a_model_named_without_dimensions_has_its_own_dimensions() -> None:
    assert embedder_facts("text-embedding-3-small").dimensions == 1536
    assert embedder_facts("text-embedding-3-large").dimensions == 1024
    assert embedder_facts("text-embedding-3-large", 3072).dimensions == 3072
    assert embedder_facts("text-embedding-3-small", 256).usd_per_mtok == 0.02


def test_a_model_that_is_not_in_the_file_is_refused() -> None:
    with pytest.raises(ModelFactsError):
        embedder_facts("text-embedding-ada-002")


@pytest.mark.parametrize("dimensions", [0, -1, 3073, True, 10.5])
def test_dimensions_that_the_model_can_not_give_are_refused(dimensions: object) -> None:
    with pytest.raises(ModelFactsError):
        embedder_facts("text-embedding-3-large", dimensions)  # type: ignore[arg-type]


def write(path: Path, **changes: object) -> Path:
    raw = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**raw, **changes}), encoding="utf-8")
    return path


def test_a_file_that_can_not_be_used_is_a_typed_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelFactsError):
        load_facts(broken)
    with pytest.raises(ModelFactsError):
        load_facts(tmp_path / "missing.json")
    with pytest.raises(ModelFactsError):
        load_facts(write(tmp_path / "no-models.json", models=[]))
    with pytest.raises(ModelFactsError):
        load_facts(write(tmp_path / "bad-default.json", default={"model": "nothing", "dimensions": 8}))


def test_a_model_that_can_not_be_shortened_gives_only_its_own_dimensions(tmp_path: Path) -> None:
    raw = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    fixed = [{**entry, "can_shorten": False} for entry in raw["models"]]
    path = write(tmp_path / "fixed.json", models=fixed, default={"model": "text-embedding-3-small", "dimensions": 1536})
    assert embedder_facts(path=path).dimensions == 1536
    with pytest.raises(ModelFactsError):
        embedder_facts("text-embedding-3-small", 512, path=path)


def test_a_price_that_was_not_read_is_refused_and_no_number_is_made_up(tmp_path: Path) -> None:
    raw = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    unpriced = [{**entry, "usd_per_mtok": None} for entry in raw["models"]]
    with pytest.raises(ModelFactsError):
        embedder_facts(path=write(tmp_path / "unpriced.json", models=unpriced))
