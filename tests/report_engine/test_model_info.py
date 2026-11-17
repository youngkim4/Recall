"""ModelInfo carries every model fact. Pricing reads nothing else."""

import pytest
from pydantic import ValidationError

from report_engine.model_info import LongContextRule, ModelInfo
from report_engine.ports import LLMUsage
from report_engine.pricing import call_cost_usd

from .synth import model_info


def test_cost_splits_cached_and_uncached_input() -> None:
    info = model_info()
    usage = LLMUsage(input_tokens=100_000, cached_input_tokens=40_000, output_tokens=5_000)
    # 60k at 2.0 + 40k at 0.2 + 5k at 10.0, all per million
    assert call_cost_usd(info, usage) == pytest.approx(0.12 + 0.008 + 0.05)


def test_cost_is_zero_for_no_tokens() -> None:
    assert call_cost_usd(model_info(), LLMUsage()) == 0.0


def test_long_context_rule_applies_only_above_threshold() -> None:
    info = model_info()
    at_limit = LLMUsage(input_tokens=150_000, output_tokens=1_000)
    over = LLMUsage(input_tokens=150_001, output_tokens=1_000)
    assert call_cost_usd(info, at_limit) == pytest.approx(0.30 + 0.01)
    assert call_cost_usd(info, over) == pytest.approx(150_001 * 2.0 * 2.0 / 1e6 + 0.015)


def test_long_context_rule_scales_cached_input_too() -> None:
    info = model_info()
    usage = LLMUsage(input_tokens=200_000, cached_input_tokens=200_000, output_tokens=0)
    assert call_cost_usd(info, usage) == pytest.approx(200_000 * 0.2 * 2.0 / 1e6)


def test_cached_multiplier_can_differ_from_input_multiplier() -> None:
    rule = LongContextRule(
        threshold_tokens=10,
        input_multiplier=2.0,
        output_multiplier=1.0,
        cached_input_multiplier=1.0,
    )
    info = model_info(long_context=rule)
    usage = LLMUsage(input_tokens=1_000_000, cached_input_tokens=1_000_000)
    assert call_cost_usd(info, usage) == pytest.approx(0.2)


def test_no_long_context_rule_means_flat_prices() -> None:
    info = model_info(long_context=None)
    usage = LLMUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert call_cost_usd(info, usage) == pytest.approx(12.0)


def test_cached_tokens_never_exceed_input_in_the_bill() -> None:
    usage = LLMUsage(input_tokens=100, cached_input_tokens=900, output_tokens=0)
    assert call_cost_usd(model_info(), usage) == pytest.approx(100 * 0.2 / 1e6)


@pytest.mark.parametrize(
    "field, value",
    [
        ("id", ""),
        ("context_window", 0),
        ("max_input_tokens", -1),
        ("max_output_tokens", 0),
        ("input_usd_per_mtok", -0.01),
        ("output_usd_per_mtok", float("nan")),
        ("tokens_per_minute", 0),
        ("tokenizer_family", ""),
    ],
)
def test_model_info_rejects_bad_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        model_info(**{field: value})


def test_model_info_rejects_input_larger_than_the_window() -> None:
    with pytest.raises(ValidationError):
        model_info(context_window=1_000, max_input_tokens=2_000)


def test_model_info_is_immutable() -> None:
    info = model_info()
    with pytest.raises(ValidationError):
        info.id = "other"  # type: ignore[misc]


def test_model_info_accepts_a_plain_mapping() -> None:
    info = ModelInfo.model_validate(
        {
            "id": "from-config",
            "context_window": 100,
            "max_input_tokens": 80,
            "max_output_tokens": 20,
            "input_usd_per_mtok": 1,
            "cached_input_usd_per_mtok": 0.1,
            "output_usd_per_mtok": 4,
            "long_context": {
                "threshold_tokens": 50,
                "input_multiplier": 2,
                "output_multiplier": 1.5,
            },
            "tokens_per_minute": 1000,
            "tokenizer_family": "words",
        }
    )
    assert info.long_context is not None
    assert info.long_context.threshold_tokens == 50
