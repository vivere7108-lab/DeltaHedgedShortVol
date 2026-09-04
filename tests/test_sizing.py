from datetime import date

import pytest

from deltahedger.chain import select_atm_straddle
from deltahedger.config import SizingConfig, VolConfig
from deltahedger.sizing import (
    FixedMarginModel, RegTMarginModel, SpanScanMarginModel,
    build_margin_model, size_straddles, straddle_debit,
)
from deltahedger.volsurface import VolSurface

F = 5000.0
T = 6.4 / 24 / 365
EXPIRY = date(2025, 6, 10)

LONG, SHORT = 1, -1


def straddle(es, future=F, iv=0.156, t=T):
    return select_atm_straddle(future, EXPIRY, t, iv, es, VolSurface(VolConfig()), 0.04)


@pytest.fixture
def span():
    return SpanScanMarginModel()


class TestDirectionDecidesTheRequirement:
    """A short straddle is margined; a long one is paid for. Conflating the
    two would misstate the risk in both directions."""

    def test_a_long_straddle_costs_its_debit(self, span, es):
        quote = straddle(es)
        assert span.straddle_requirement(quote, F, es, LONG) == pytest.approx(
            quote.price * es.option.multiplier
        )

    def test_the_debit_is_also_the_maximum_loss(self, span, es):
        quote = straddle(es)
        assert straddle_debit(quote, es) == span.straddle_requirement(quote, F, es, LONG)

    def test_a_short_straddle_costs_scenario_margin_not_the_credit(self, span, es):
        quote = straddle(es)
        margin = span.straddle_requirement(quote, F, es, SHORT)
        assert margin > quote.price * es.option.multiplier

    def test_the_two_directions_differ(self, span, es):
        quote = straddle(es)
        assert span.straddle_requirement(quote, F, es, LONG) != span.straddle_requirement(
            quote, F, es, SHORT
        )


class TestSpanScan:
    def test_scan_range_comes_from_the_future_margin(self, span, es):
        assert span.price_scan_range(es) == pytest.approx(2455.0 / 50.0)

    def test_margin_is_the_right_order_of_magnitude(self, span, es):
        """A short 0DTE ATM straddle is riskier than one leg but should still
        be cheaper than carrying two outright futures."""
        margin = span.straddle_requirement(straddle(es), F, es, SHORT)
        assert 1_000 < margin < 2 * es.future_initial_margin

    def test_the_legs_are_netted_within_each_scenario(self, span, es):
        """Only one leg can finish in the money, so charging each its own
        worst case would overstate the requirement and undersize the book."""
        quote = straddle(es)
        combined = span.straddle_requirement(quote, F, es, SHORT)
        legs = sum(
            _single_leg_worst_case(span, leg, quote.time_to_expiry, F, es)
            for leg in quote.legs()
        )
        assert combined < legs

    def test_a_richer_premium_lowers_the_scan_margin(self, span, es):
        """Counter-intuitive but correct, and worth pinning down.

        SPAN charges the worst loss *relative to the entry value*. A short
        straddle sold at 40 vol has already collected most of what a 49-point
        scan move is worth, so the incremental loss is smaller than for the
        same straddle sold at 12 vol. Margin falls as premium rises, and a
        model that did the opposite would size richest-vol days smallest --
        exactly backwards.
        """
        cheap = span.straddle_requirement(straddle(es, iv=0.12), F, es, SHORT)
        rich = span.straddle_requirement(straddle(es, iv=0.40), F, es, SHORT)
        assert rich < cheap

    def test_scanning_volatility_harder_costs_more_margin(self, es):
        """The vol scan itself still binds; it is the entry premium, not the
        scan, that moves the wrong way above."""
        quote = straddle(es)
        gentle = SpanScanMarginModel(vol_scan_pct=0.05)
        harsh = SpanScanMarginModel(vol_scan_pct=0.90)
        assert harsh.straddle_requirement(
            quote, F, es, SHORT
        ) > gentle.straddle_requirement(quote, F, es, SHORT)

    def test_a_wider_price_scan_costs_more_margin(self, es):
        quote = straddle(es)
        narrow = SpanScanMarginModel(scan_multiplier=0.5)
        wide = SpanScanMarginModel(scan_multiplier=2.0)
        assert wide.straddle_requirement(
            quote, F, es, SHORT
        ) > narrow.straddle_requirement(quote, F, es, SHORT)

    def test_the_short_option_minimum_floors_both_legs(self, es):
        model = SpanScanMarginModel(short_option_minimum=750.0)
        # A straddle with essentially no scenario risk still pays 2 x floor.
        quote = straddle(es, iv=0.02, t=1.0 / 24 / 365)
        assert model.straddle_requirement(quote, F, es, SHORT) >= 1500.0

    def test_regt_wildly_overstates_futures_margin(self, span, es):
        """Documents why reg_t is not the default."""
        quote = straddle(es)
        assert RegTMarginModel().straddle_requirement(quote, F, es, SHORT) > 5 * (
            span.straddle_requirement(quote, F, es, SHORT)
        )

    def test_regt_still_charges_only_the_debit_for_a_long(self, es):
        quote = straddle(es)
        assert RegTMarginModel().straddle_requirement(
            quote, F, es, LONG
        ) == pytest.approx(straddle_debit(quote, es))


def _single_leg_worst_case(model, leg, t, future, es):
    """The worst-case loss on one leg alone, for the netting comparison."""
    from deltahedger.pricing import black76
    from deltahedger.sizing import SPAN_SCENARIOS

    scan = model.price_scan_range(es)
    worst = 0.0
    for price_frac, vol_frac, weight in SPAN_SCENARIOS:
        value = black76(
            max(future + price_frac * scan, 1e-9), leg.strike, t,
            max(leg.iv * (1.0 + vol_frac * model.vol_scan_pct), 1e-6),
            model.risk_free_rate, leg.right,
        ).price
        worst = max(worst, (value - leg.price) * es.option.multiplier * weight)
    return max(worst, model.short_option_minimum)


class TestBuyingPower:
    def test_contracts_scale_with_the_allocation(self, es, span):
        quote = straddle(es)
        small = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span)
        big = size_straddles(
            250_000, quote, F, SHORT, SizingConfig(buying_power_pct=0.30), es, span
        )
        assert big.contracts > small.contracts

    def test_the_default_allocation_is_fifteen_percent(self):
        assert SizingConfig().buying_power_pct == 0.15

    def test_budget_is_equity_times_the_allocation(self, es, span):
        result = size_straddles(200_000, straddle(es), F, SHORT, SizingConfig(), es, span)
        assert result.budget == pytest.approx(30_000.0)

    def test_a_reserve_is_held_back_for_the_hedge(self, es, span):
        cfg = SizingConfig(hedge_margin_reserve_pct=0.25)
        result = size_straddles(200_000, straddle(es), F, SHORT, cfg, es, span)
        assert result.option_budget == pytest.approx(30_000.0 * 0.75)

    def test_the_reserve_applies_to_the_long_side_too(self, es, span):
        """The hedge leg needs margin whichever way the straddle is facing."""
        cfg = SizingConfig(hedge_margin_reserve_pct=0.25)
        result = size_straddles(200_000, straddle(es), F, LONG, cfg, es, span)
        assert result.option_budget == pytest.approx(30_000.0 * 0.75)

    @pytest.mark.parametrize("direction", [LONG, SHORT])
    def test_the_requirement_never_exceeds_the_budget(self, es, span, direction):
        for equity in (50_000, 137_500, 400_000, 2_000_000):
            result = size_straddles(
                equity, straddle(es), F, direction, SizingConfig(), es, span
            )
            assert result.total_margin <= result.option_budget

    def test_too_little_buying_power_declines_with_a_reason(self, es, span):
        result = size_straddles(2_000, straddle(es), F, SHORT, SizingConfig(), es, span)
        assert not result.ok
        assert "buying power supports" in result.reason

    def test_the_reason_names_the_right_kind_of_requirement(self, es, span):
        long_result = size_straddles(2_000, straddle(es), F, LONG, SizingConfig(), es, span)
        short_result = size_straddles(2_000, straddle(es), F, SHORT, SizingConfig(), es, span)
        assert "debit" in long_result.reason
        assert "margin" in short_result.reason

    def test_the_hard_cap_applies(self, es, span):
        cfg = SizingConfig(max_straddles=3)
        result = size_straddles(5_000_000, straddle(es), F, SHORT, cfg, es, span)
        assert result.contracts == 3
        assert "capped" in result.reason

    def test_zero_equity_trades_nothing(self, es, span):
        assert not size_straddles(0.0, straddle(es), F, SHORT, SizingConfig(), es, span).ok

    def test_no_direction_sizes_nothing(self, es, span):
        result = size_straddles(250_000, straddle(es), F, 0, SizingConfig(), es, span)
        assert not result.ok
        assert "no direction" in result.reason


class TestSizeMultiplier:
    """The nowcast sizing haircut: applied to the option budget before
    contracts are counted, so it composes with max/min_straddles rather
    than bypassing them."""

    def test_the_default_multiplier_is_full_size(self, es, span):
        quote = straddle(es)
        full = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span)
        explicit = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span, 1.0)
        assert full.contracts == explicit.contracts

    def test_a_haircut_reduces_the_option_budget(self, es, span):
        quote = straddle(es)
        result = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span, 0.4)
        full = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span, 1.0)
        assert result.option_budget == pytest.approx(full.option_budget * 0.4)

    def test_a_haircut_can_reduce_the_contract_count(self, es, span):
        quote = straddle(es)
        full = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span, 1.0)
        haircut = size_straddles(250_000, quote, F, SHORT, SizingConfig(), es, span, 0.4)
        assert haircut.contracts < full.contracts

    def test_the_multiplier_is_reported_on_the_result(self, es, span):
        result = size_straddles(250_000, straddle(es), F, SHORT, SizingConfig(), es, span, 0.4)
        assert result.size_multiplier == pytest.approx(0.4)

    def test_a_zero_multiplier_declines_with_a_reason(self, es, span):
        result = size_straddles(250_000, straddle(es), F, SHORT, SizingConfig(), es, span, 0.0)
        assert not result.ok
        assert "sized" in result.reason

    def test_the_hard_cap_still_applies_under_a_haircut(self, es, span):
        """A haircut only ever shrinks the count -- it cannot be used to
        sneak past max_straddles from the other direction."""
        cfg = SizingConfig(max_straddles=3)
        result = size_straddles(5_000_000, straddle(es), F, SHORT, cfg, es, span, 1.0)
        assert result.contracts == 3


class TestModelSelection:
    @pytest.mark.parametrize("name,expected", [
        ("span_scan", SpanScanMarginModel),
        ("reg_t", RegTMarginModel),
        ("fixed", FixedMarginModel),
    ])
    def test_build_margin_model(self, name, expected, es):
        model = build_margin_model(SizingConfig(margin_model=name), es)
        assert isinstance(model, expected)

    def test_fixed_model_charges_its_constant_per_short_leg(self, es):
        cfg = SizingConfig(margin_model="fixed", fixed_margin_per_contract=1234.0)
        model = build_margin_model(cfg, es)
        assert model.straddle_requirement(straddle(es), F, es, SHORT) == 2468.0

    def test_fixed_model_still_charges_the_debit_for_a_long(self, es):
        cfg = SizingConfig(margin_model="fixed", fixed_margin_per_contract=1234.0)
        model = build_margin_model(cfg, es)
        quote = straddle(es)
        assert model.straddle_requirement(quote, F, es, LONG) == pytest.approx(
            straddle_debit(quote, es)
        )
