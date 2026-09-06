from datetime import date

import pytest

from deltahedger.chain import select_atm_straddle
from deltahedger.config import SizingConfig, VolConfig
from deltahedger.sizing import (
    BIND_CAPITAL, BIND_GAMMA, BIND_MAX, BIND_RISK,
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


#: What one straddle loses at its branch stop, as a fraction of the entry
#: premium: 50% of the debit long, (2.5 - 1) x the credit short.
LONG_STOP, SHORT_STOP = 0.50, 1.50


def sized(es, span, equity=250_000, direction=SHORT, cfg=None, **kwargs):
    stop = LONG_STOP if direction > 0 else SHORT_STOP
    return size_straddles(
        equity, straddle(es), F, direction, cfg or SizingConfig(), es, span,
        stop_fraction=kwargs.pop("stop_fraction", stop), **kwargs,
    )


class TestTheThreeConstraints:
    """The count is the smallest any constraint allows, and which one bound
    is recorded. Each guards a different failure, so each is tested as the
    one that binds."""

    def test_the_risk_budget_binds_at_the_shipped_defaults(self, es, span):
        """The point of the whole rule: at the margin limit the branch stops
        were 5-8x the daily loss limit, so the daily limit was the only stop
        that ever fired. The risk budget is what makes them reachable."""
        for direction in (LONG, SHORT):
            result = sized(es, span, direction=direction)
            assert result.binding == BIND_RISK
            assert result.limits[BIND_RISK] < result.limits[BIND_CAPITAL]

    def test_a_full_stop_out_costs_about_the_risk_budget(self, es, span):
        for direction, stop in ((LONG, LONG_STOP), (SHORT, SHORT_STOP)):
            result = sized(es, span, direction=direction)
            loss = result.contracts * result.risk_per_straddle
            assert loss <= 0.05 * 250_000 + result.risk_per_straddle
            assert result.risk_per_straddle == pytest.approx(
                stop * straddle(es).price * es.option.multiplier
            )

    def test_a_wider_stop_earns_a_smaller_position(self, es, span):
        """Risk-based sizing's defining property: the further away the stop,
        the fewer contracts the same budget buys."""
        near = sized(es, span, direction=SHORT, stop_fraction=0.5)
        far = sized(es, span, direction=SHORT, stop_fraction=3.0)
        assert near.contracts > far.contracts

    def test_the_gamma_ceiling_binds_when_the_risk_budget_is_loose(self, es, span):
        cfg = SizingConfig(risk_budget_pct=0.50)
        result = sized(es, span, cfg=cfg)
        assert result.binding == BIND_GAMMA
        gamma = result.contracts * result.gamma_per_straddle
        assert gamma <= cfg.gamma_ceiling_units_per_100k * 2.5 + result.gamma_per_straddle

    def test_the_gamma_ceiling_holds_the_bet_steady_across_the_day(self, es, span):
        """A cheap late straddle risks less per contract and carries more
        gamma, so a risk budget alone lets the bet grow into the afternoon.
        The ceiling is what stops it."""
        morning, afternoon = straddle(es, t=T), straddle(es, t=1.5 / 24 / 365)
        cfg = SizingConfig()
        sizes = [
            size_straddles(250_000, q, F, LONG, cfg, es, span, stop_fraction=LONG_STOP)
            for q in (morning, afternoon)
        ]
        gammas = [r.contracts * r.gamma_per_straddle for r in sizes]
        assert max(gammas) / min(gammas) < 2.0
        without = SizingConfig(gamma_ceiling_units_per_100k=None)
        loose = [
            size_straddles(250_000, q, F, LONG, without, es, span, stop_fraction=LONG_STOP)
            for q in (morning, afternoon)
        ]
        unbounded = [r.contracts * r.gamma_per_straddle for r in loose]
        assert max(unbounded) / min(unbounded) > 3.0

    def test_the_capital_cap_binds_once_the_others_are_off(self, es, span):
        cfg = SizingConfig(risk_budget_pct=None, gamma_ceiling_units_per_100k=None)
        result = sized(es, span, cfg=cfg)
        assert result.binding == BIND_CAPITAL
        assert result.total_margin <= result.budget

    def test_with_the_others_off_the_count_scales_with_the_allocation(self, es, span):
        off = dict(risk_budget_pct=None, gamma_ceiling_units_per_100k=None)
        small = sized(es, span, cfg=SizingConfig(buying_power_pct=0.30, **off))
        big = sized(es, span, cfg=SizingConfig(buying_power_pct=0.80, **off))
        assert big.contracts > small.contracts

    def test_the_hard_cap_applies(self, es, span):
        cfg = SizingConfig(max_straddles=3)
        result = sized(es, span, equity=5_000_000, cfg=cfg)
        assert result.contracts == 3
        assert result.binding == BIND_MAX

    def test_every_constraint_reports_what_it_would_have_allowed(self, es, span):
        result = sized(es, span)
        assert set(result.limits) == {BIND_RISK, BIND_GAMMA, BIND_CAPITAL, BIND_MAX}
        assert result.contracts == min(result.limits.values())
        assert f"risk {result.limits[BIND_RISK]}" in result.describe_limits()

    def test_a_disabled_stop_falls_back_to_the_requirement(self, es, span):
        """With no branch stop the loss being bounded is the requirement
        itself -- the debit is a long straddle's maximum loss, and the SPAN
        scan is the short's one-day adverse move."""
        for direction in (LONG, SHORT):
            result = sized(es, span, direction=direction, stop_fraction=None)
            assert result.risk_per_straddle == pytest.approx(result.margin_per_contract)


class TestBuyingPower:
    def test_the_default_allocation_is_the_margin_limit_less_a_fifth(self):
        """Everything up to the margin limit, with a 20% buffer left untouched."""
        assert SizingConfig().buying_power_pct == 0.80

    def test_a_fifth_of_equity_is_never_committed(self, es, span):
        for direction in (LONG, SHORT):
            result = sized(es, span, direction=direction)
            assert result.budget <= 0.80 * 250_000 + 1e-6
            assert result.total_margin <= 0.80 * 250_000

    def test_budget_is_equity_times_the_allocation(self, es, span):
        assert sized(es, span, equity=200_000).budget == pytest.approx(160_000.0)

    def test_the_hedge_leg_has_room_without_a_reserve(self, es, span):
        """The reserve is gone: with the risk budget binding, the straddle
        takes about a tenth of equity, so the unused capital cap covers the
        hedge even for a fully in-the-money book at the buffer."""
        result = sized(es, span, direction=LONG)
        worst_case_hedge = (
            result.contracts * 100.0 / es.hedge_quantum * es.hedge_initial_margin
        )
        assert result.total_margin + worst_case_hedge < result.budget

    def test_the_default_cap_does_not_bind_at_an_ordinary_account_size(self, es, span):
        """max_straddles is a backstop against a sizing bug rather than a
        rule, so at a quarter-million account it must not set the size."""
        result = sized(es, span)
        assert result.ok and result.binding != BIND_MAX
        assert result.contracts < SizingConfig().max_straddles

    @pytest.mark.parametrize("direction", [LONG, SHORT])
    def test_the_requirement_never_exceeds_the_budget(self, es, span, direction):
        for equity in (50_000, 137_500, 400_000, 2_000_000):
            result = sized(es, span, equity=equity, direction=direction)
            assert result.total_margin <= result.budget

    def test_too_little_capital_declines_with_a_reason(self, es, span):
        result = sized(es, span, equity=2_000)
        assert not result.ok
        assert "minimum is" in result.reason and "at risk" in result.reason

    def test_the_reason_names_the_right_kind_of_requirement(self, es, span):
        assert "debit" in sized(es, span, equity=500, direction=LONG).reason
        assert "margin" in sized(es, span, equity=500, direction=SHORT).reason

    def test_zero_equity_trades_nothing(self, es, span):
        assert not sized(es, span, equity=0.0).ok

    def test_no_direction_sizes_nothing(self, es, span):
        result = sized(es, span, direction=0)
        assert not result.ok
        assert "no direction" in result.reason


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
