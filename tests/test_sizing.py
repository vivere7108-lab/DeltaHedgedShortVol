from datetime import date

import pytest

from deltahedger.chain import OptionQuote
from deltahedger.config import SizingConfig
from deltahedger.pricing import black76
from deltahedger.sizing import (
    FixedMarginModel, RegTMarginModel, SpanScanMarginModel,
    build_margin_model, size_short_puts,
)

F = 5000.0
T = 6.4 / 24 / 365


def quote(strike, iv=0.156):
    greeks = black76(F, strike, T, iv, 0.04, "P")
    return OptionQuote(strike, "P", date(2025, 6, 10), greeks.price, iv, greeks, T)


@pytest.fixture
def span():
    return SpanScanMarginModel()


class TestSpanScan:
    def test_scan_range_comes_from_the_future_margin(self, span, es):
        assert span.price_scan_range(es) == pytest.approx(2455.0 / 50.0)

    def test_margin_is_the_right_order_of_magnitude(self, span, es):
        """A short 20-delta 0DTE put should cost less than an outright future."""
        margin = span.short_option_margin(quote(4980.0), F, es)
        assert 500 < margin < es.future_initial_margin

    def test_margin_rises_toward_the_money(self, span, es):
        """Strictly increasing while the strike is inside the scan range."""
        margins = [span.short_option_margin(quote(k), F, es)
                   for k in (4950.0, 4970.0, 4990.0, 5000.0)]
        assert all(a < b for a, b in zip(margins, margins[1:]))

    def test_margin_never_falls_as_the_strike_rises(self, span, es):
        margins = [span.short_option_margin(quote(k), F, es)
                   for k in range(4700, 5001, 25)]
        assert all(a <= b for a, b in zip(margins, margins[1:]))

    def test_strikes_beyond_the_scan_range_pay_only_the_minimum(self, span, es):
        """SPAN charges the floor for a short the scenarios cannot reach."""
        far = F - 3 * span.price_scan_range(es)
        assert span.short_option_margin(quote(far), F, es) == pytest.approx(
            span.short_option_minimum
        )

    def test_the_short_option_minimum_is_a_floor(self, es):
        model = SpanScanMarginModel(short_option_minimum=750.0)
        assert model.short_option_margin(quote(4000.0), F, es) == pytest.approx(750.0)

    def test_higher_vol_costs_more_margin(self, span, es):
        cheap = span.short_option_margin(quote(4950.0, iv=0.12), F, es)
        rich = span.short_option_margin(quote(4950.0, iv=0.40), F, es)
        assert rich > cheap

    def test_regt_wildly_overstates_futures_margin(self, span, es):
        """Documents why reg_t is not the default."""
        assert RegTMarginModel().short_option_margin(quote(4980.0), F, es) > 10 * (
            span.short_option_margin(quote(4980.0), F, es)
        )


class TestBuyingPower:
    def test_contracts_scale_with_the_allocation(self, es, span):
        cfg = SizingConfig()
        small = size_short_puts(250_000, quote(4980.0), F, cfg, es, span)
        cfg_big = SizingConfig(buying_power_pct=0.30)
        big = size_short_puts(250_000, quote(4980.0), F, cfg_big, es, span)
        assert big.contracts > small.contracts

    def test_the_default_allocation_is_fifteen_percent(self):
        assert SizingConfig().buying_power_pct == 0.15

    def test_budget_is_equity_times_the_allocation(self, es, span):
        result = size_short_puts(200_000, quote(4980.0), F, SizingConfig(), es, span)
        assert result.budget == pytest.approx(30_000.0)

    def test_a_reserve_is_held_back_for_the_hedge(self, es, span):
        cfg = SizingConfig(hedge_margin_reserve_pct=0.25)
        result = size_short_puts(200_000, quote(4980.0), F, cfg, es, span)
        assert result.option_budget == pytest.approx(30_000.0 * 0.75)

    def test_margin_never_exceeds_the_option_budget(self, es, span):
        for equity in (50_000, 137_500, 400_000, 2_000_000):
            result = size_short_puts(equity, quote(4980.0), F, SizingConfig(), es, span)
            assert result.total_margin <= result.option_budget

    def test_too_little_buying_power_declines_with_a_reason(self, es, span):
        result = size_short_puts(10_000, quote(4980.0), F, SizingConfig(), es, span)
        assert not result.ok
        assert "buying power supports" in result.reason

    def test_the_hard_cap_applies(self, es, span):
        cfg = SizingConfig(max_short_contracts=3)
        result = size_short_puts(5_000_000, quote(4980.0), F, cfg, es, span)
        assert result.contracts == 3
        assert "capped" in result.reason

    def test_zero_equity_trades_nothing(self, es, span):
        assert not size_short_puts(0.0, quote(4980.0), F, SizingConfig(), es, span).ok


class TestModelSelection:
    @pytest.mark.parametrize("name,expected", [
        ("span_scan", SpanScanMarginModel),
        ("reg_t", RegTMarginModel),
        ("fixed", FixedMarginModel),
    ])
    def test_build_margin_model(self, name, expected, es):
        model = build_margin_model(SizingConfig(margin_model=name), es)
        assert isinstance(model, expected)

    def test_fixed_model_returns_its_constant(self, es):
        cfg = SizingConfig(margin_model="fixed", fixed_margin_per_contract=1234.0)
        model = build_margin_model(cfg, es)
        assert model.short_option_margin(quote(4980.0), F, es) == 1234.0
