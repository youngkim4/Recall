"""The model registry: one file of facts, and the functions that read it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recall_services import models
from report_engine import ModelInfo
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd
from report_engine.tokens import meter_for

SIX_SERIES = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra")
OLDER = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
# input, cached input, output: USD for a million tokens, as the brief of 2026-11-16 gives them
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}
LEVELS = {"gpt-6-luna": "low", "gpt-6.1-terra": "medium", "gpt-6.1-sol": "high", "gpt-6-astra": "top"}
FACT_KEYS = {
    "id",
    "provider",
    "label",
    "level",
    "offered",
    "context_window",
    "max_input_tokens",
    "max_output_tokens",
    "input_usd_per_mtok",
    "cached_input_usd_per_mtok",
    "output_usd_per_mtok",
    "long_context",
    "tokens_per_minute",
    "tokenizer_family",
    "tokenizer_family_assumed",
    "reasoning_levels",
    "shutdown_date",
    "facts_read_on",
}


class _Counter:
    """Stands in for the tiktoken encoding of the engine: one token for each word."""

    family = "o200k_base"

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]


def test_the_file_holds_every_fact_for_every_model() -> None:
    raw = json.loads(Path(models.FACTS_PATH).read_text(encoding="utf-8"))
    assert [entry["id"] for entry in raw["models"]] == [*SIX_SERIES, *OLDER]
    for entry in raw["models"]:
        assert set(entry) == FACT_KEYS, entry["id"]
        assert entry["provider"] == "openai"
        assert entry["facts_read_on"] == "2026-11-16"
        assert entry["shutdown_date"] is None


@pytest.mark.parametrize("model_id", [*SIX_SERIES, *OLDER])
def test_prices_are_entered_as_given(model_id: str) -> None:
    facts = models.facts(model_id)
    assert (facts.input_usd_per_mtok, facts.cached_input_usd_per_mtok, facts.output_usd_per_mtok) == PRICES[model_id]


@pytest.mark.parametrize("model_id", SIX_SERIES)
def test_the_six_series_shares_its_limits_and_its_long_context_rule(model_id: str) -> None:
    facts = models.facts(model_id)
    assert facts.context_window == 1_050_000
    assert facts.max_output_tokens == 128_000
    assert facts.level == LEVELS[model_id]
    assert facts.long_context is not None
    assert facts.long_context.threshold_tokens == 272_000
    assert facts.long_context.input_multiplier == 2.0
    assert facts.long_context.output_multiplier == 1.5
    # the family of the 6 series is not confirmed
    assert facts.tokenizer_family == "o200k_base"
    assert facts.tokenizer_family_assumed is True


def test_the_facts_that_only_one_model_has() -> None:
    sol = models.facts("gpt-6.1-sol")
    assert sol.max_input_tokens == 922_000
    assert sol.tokens_per_minute == 1_000_000
    assert sol.reasoning_levels == ("low", "medium", "high", "xhigh", "max")
    for other in ("gpt-6-luna", "gpt-6.1-terra", "gpt-6-astra"):
        facts = models.facts(other)
        # not read from the page: the file says so with null, and no number is made up
        assert facts.max_input_tokens is None
        assert facts.tokens_per_minute is None
        assert facts.reasoning_levels is None


def test_the_offered_list_and_the_default() -> None:
    assert models.offered_ids() == ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol"]
    assert models.default_id() == "gpt-6.1-sol"
    assert models.default_id() in models.offered_ids()
    assert models.offered() == [
        {"id": "gpt-6-luna", "label": "GPT-6 Luna", "level": "low"},
        {"id": "gpt-6.1-terra", "label": "GPT-6.1 Terra", "level": "medium"},
        {"id": "gpt-6.1-sol", "label": "GPT-6.1 Sol", "level": "high"},
    ]


def test_models_that_are_still_sold_are_known_and_not_offered() -> None:
    for model_id in ("gpt-6-astra", *OLDER):
        assert models.facts(model_id).offered is False
        assert model_id not in models.offered_ids()


def test_the_defaults_payload_has_the_keys_the_app_reads() -> None:
    payload = models.defaults_payload()
    # Defaults in app/src/types.ts at 89ab504: defaultModel: string, models: string[]
    assert payload["defaultModel"] == "gpt-6.1-sol"
    assert payload["models"] == models.offered_ids()
    assert payload["modelChoices"] == models.offered()
    assert set(payload) == {"defaultModel", "models", "modelChoices"}
    json.dumps(payload)


def test_model_info_is_the_shape_of_the_report_engine() -> None:
    info = models.model_info("gpt-6.1-sol")
    assert isinstance(info, ModelInfo)
    assert info.id == "gpt-6.1-sol"
    assert (info.context_window, info.max_input_tokens, info.max_output_tokens) == (1_050_000, 922_000, 128_000)
    assert info.tokens_per_minute == 1_000_000
    assert info.long_context is not None and info.long_context.threshold_tokens == 272_000


def test_a_largest_input_that_was_not_read_is_the_window_less_the_largest_output() -> None:
    info = models.model_info("gpt-6-luna")
    assert info.max_input_tokens == 1_050_000 - 128_000
    assert info.tokens_per_minute is None


def test_an_assumed_tokenizer_family_keeps_the_safety_margin() -> None:
    info = models.model_info("gpt-6.1-sol")
    # the engine counts with o200k_base and, because the family is not the model's for certain, adds its margin
    assert info.tokenizer_family != "o200k_base"
    assert "o200k_base" in info.tokenizer_family
    meter = meter_for(info, _Counter())
    assert meter.margin == 1.35
    assert meter.count("one two three four") == 6  # 4 words, times 1.35, rounded up


def test_the_long_context_rule_doubles_input_and_adds_half_to_output() -> None:
    info = models.model_info("gpt-6.1-sol")
    below = call_cost_usd(info, LLMUsage(input_tokens=272_000, cached_input_tokens=0, output_tokens=1_000_000))
    above = call_cost_usd(info, LLMUsage(input_tokens=272_001, cached_input_tokens=0, output_tokens=1_000_000))
    assert below == pytest.approx(272_000 * 2.00 / 1e6 + 10.00)
    assert above == pytest.approx(272_001 * 2.00 * 2 / 1e6 + 10.00 * 1.5)


@pytest.mark.parametrize("model_id", ["", "gpt-7", "gpt-6.1-sol-2026-01-01", "GPT-6.1-SOL", " gpt-6.1-sol", None, 7])
def test_an_unknown_id_is_a_typed_error_and_is_never_priced_as_another_model(model_id: object) -> None:
    for read in (models.facts, models.model_info, models.chat_config):
        with pytest.raises(models.UnknownModelIdError) as caught:
            read(model_id)  # type: ignore[arg-type]
        assert caught.value.model_id == str(model_id if model_id is not None else "")
        assert caught.value.code == "unknown_model"


def test_the_error_names_the_id_and_the_choices() -> None:
    with pytest.raises(models.UnknownModelIdError) as caught:
        models.model_info("gpt-7")
    assert str(caught.value) == (
        "Recall does not know the model gpt-7. Choose one of: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol."
    )


@pytest.mark.parametrize("model_id", ["gpt-6-astra", *OLDER])
def test_a_model_that_is_not_offered_can_not_be_run(model_id: str) -> None:
    for read in (models.model_info, models.chat_config):
        with pytest.raises(models.ModelNotOfferedError) as caught:
            read(model_id)
        assert caught.value.model_id == model_id
        assert caught.value.code == "model_not_offered"
        assert str(caught.value) == (
            f"The model {model_id} is not offered. Choose one of: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol."
        )
    # its facts stay readable: the prices of a model that is still sold
    assert models.facts(model_id).id == model_id


def test_no_model_named_is_said_in_plain_words() -> None:
    with pytest.raises(models.UnknownModelIdError) as caught:
        models.chat_config("")
    assert str(caught.value) == "No model was named. Choose one of: gpt-6-luna, gpt-6.1-terra, gpt-6.1-sol."


def test_the_older_models_have_no_largest_output_on_file() -> None:
    # they are kept for their prices; their limits were not read on 2026-11-16
    for model_id in OLDER:
        assert models.facts(model_id).max_output_tokens is None


def test_both_choice_errors_are_one_kind() -> None:
    assert issubclass(models.UnknownModelIdError, models.ModelChoiceError)
    assert issubclass(models.ModelNotOfferedError, models.ModelChoiceError)
    assert issubclass(models.IncompleteModelFactsError, models.ModelFactsError)


def test_chat_config_for_a_model_that_reasons() -> None:
    config = models.chat_config("gpt-6.1-sol")
    assert config.model == "gpt-6.1-sol"
    assert config.provider == "openai"
    assert config.reasoning_effort == "low"
    assert 1 <= config.max_output_tokens <= 128_000


def test_chat_config_sends_no_effort_when_the_levels_were_not_read() -> None:
    config = models.chat_config("gpt-6-luna")
    assert config.reasoning_effort is None
    assert config.max_output_tokens == models.chat_config("gpt-6.1-sol").max_output_tokens


def test_every_offered_model_can_be_run() -> None:
    for model_id in models.offered_ids():
        assert models.model_info(model_id).id == model_id
        assert models.chat_config(model_id).model == model_id


def test_a_file_that_offers_a_model_with_missing_limits_is_refused(tmp_path: Path) -> None:
    raw = json.loads(Path(models.FACTS_PATH).read_text(encoding="utf-8"))
    broken = {**raw, "models": [{**entry, "offered": True} for entry in raw["models"]]}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(models.ModelFactsError, match="gpt-5.5"):
        models.load_registry(path)


def test_a_file_whose_default_is_not_offered_is_refused(tmp_path: Path) -> None:
    raw = json.loads(Path(models.FACTS_PATH).read_text(encoding="utf-8"))
    path = tmp_path / "models.json"
    path.write_text(json.dumps({**raw, "default": "gpt-6-astra"}), encoding="utf-8")
    with pytest.raises(models.ModelFactsError, match="default"):
        models.load_registry(path)


@pytest.mark.parametrize(
    "change",
    [
        {"level": "ultra"},
        {"input_usd_per_mtok": -1},
        {"context_window": 0},
        {"provider": ""},
        {"reasoning_levels": "low"},
        {"surprise": 1},
    ],
)
def test_a_file_with_a_bad_value_is_refused(tmp_path: Path, change: dict[str, object]) -> None:
    raw = json.loads(Path(models.FACTS_PATH).read_text(encoding="utf-8"))
    first, *rest = raw["models"]
    path = tmp_path / "models.json"
    path.write_text(json.dumps({**raw, "models": [{**first, **change}, *rest]}), encoding="utf-8")
    with pytest.raises(models.ModelFactsError):
        models.load_registry(path)


def test_two_models_with_one_id_are_refused(tmp_path: Path) -> None:
    raw = json.loads(Path(models.FACTS_PATH).read_text(encoding="utf-8"))
    path = tmp_path / "models.json"
    path.write_text(json.dumps({**raw, "models": [*raw["models"], raw["models"][0]]}), encoding="utf-8")
    with pytest.raises(models.ModelFactsError, match="two times"):
        models.load_registry(path)


def test_a_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(models.ModelFactsError):
        models.load_registry(path)
    with pytest.raises(models.ModelFactsError):
        models.load_registry(tmp_path / "missing.json")
