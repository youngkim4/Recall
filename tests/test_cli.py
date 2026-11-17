import sys
sys.path.insert(0, ".")

import pytest

from cli import parse_date, MODEL_PRICING, get_model_pricing
from analysis import DEFAULT_MODEL


class TestParseDate:
    def test_valid_date(self):
        result = parse_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_date_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_date("not-a-date")

    def test_wrong_format_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_date("01/15/2024")


class TestModelPricing:
    def test_default_model_present(self):
        assert DEFAULT_MODEL in MODEL_PRICING

    def test_gpt5_mini_present(self):
        assert "gpt-5-mini" in MODEL_PRICING

    def test_pricing_tuple(self):
        for model, pricing in MODEL_PRICING.items():
            assert len(pricing) == 2
            assert pricing[0] >= 0  # input rate
            assert pricing[1] >= 0  # output rate

    def test_gpt5_mini_rates(self):
        inp, out = MODEL_PRICING["gpt-5-mini"]
        assert inp == 0.25
        assert out == 2.00

    def test_a_model_with_no_price_on_file_is_not_priced_as_another(self):
        with pytest.raises(ValueError, match="No price is on file for the model gpt-unknown"):
            get_model_pricing("gpt-unknown")
