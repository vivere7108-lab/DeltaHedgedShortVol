"""Tests for the delta band.

The properties that matter most are the ones that keep the hedger from
misbehaving in production: it must never oscillate, never trade away from
the target, and never fire inside the band.

The band algebra is target-agnostic, so most of these pin an explicit
``target=20, band=3`` rather than reading whatever the shipped default
happens to be -- that way they test the hedger, and a change of default
cannot quietly stop them from exercising an off-zero target.
``TestNeutralTarget`` covers the default the straddle strategy actually
runs.
"""

import pytest

from deltahedger.config import HedgeConfig
from deltahedger.hedger import DeltaHedger

#: The band these tests are written against, not the shipped default.
OFFSET_BAND = dict(target=20.0, band=3.0)


@pytest.fixture
def hedger(es):
    return DeltaHedger(HedgeConfig(**OFFSET_BAND), es)


class TestBand:
    @pytest.mark.parametrize("net", [17.0, 18.0, 20.0, 22.5, 23.0])
    def test_does_nothing_inside_the_band(self, hedger, net):
        decision = hedger.decide(net)
        assert not decision.should_hedge
        assert decision.contracts == 0

    def test_sells_when_delta_is_too_long(self, hedger):
        decision = hedger.decide(119.3)
        assert decision.should_hedge
        assert decision.contracts == -10
        assert decision.net_delta_after == pytest.approx(19.3)

    def test_buys_when_delta_is_too_short(self, hedger):
        decision = hedger.decide(-50.0)
        assert decision.should_hedge
        assert decision.contracts == 7
        assert decision.net_delta_after == pytest.approx(20.0)

    def test_lands_as_close_to_target_as_a_whole_contract_allows(self, hedger, es):
        for net in range(-200, 201, 7):
            decision = hedger.decide(float(net))
            if decision.should_hedge:
                assert abs(decision.residual_delta) <= es.hedge_quantum / 2 + 1e-9


class TestNoChurn:
    """The band is narrower than one MES contract, so these are the cases
    where a naive hedger would trade back and forth forever."""

    def test_does_not_trade_when_no_contract_improves(self, hedger):
        # 25 is outside [17, 23], but selling one MES lands on 15, which is
        # exactly as far from 20. Trading would be pure cost.
        decision = hedger.decide(25.0)
        assert not decision.should_hedge
        assert "closer" in decision.reason

    @pytest.mark.parametrize("net", [23.1, 24.0, 25.0, 16.9, 16.0, 15.0])
    def test_settles_in_one_step_from_any_breach(self, hedger, net):
        """Hedging must reach a fixed point immediately, never oscillate."""
        first = hedger.decide(net)
        if not first.should_hedge:
            return
        second = hedger.decide(first.net_delta_after)
        assert not second.should_hedge, (
            f"hedger oscillates: {net} -> {first.net_delta_after} -> "
            f"{second.net_delta_after}"
        )

    @pytest.mark.parametrize("start", [-500.0, -73.0, 0.0, 44.0, 250.0, 1000.0])
    def test_converges_from_far_away(self, hedger, start):
        net, steps = start, 0
        while True:
            decision = hedger.decide(net)
            if not decision.should_hedge:
                break
            net = decision.net_delta_after
            steps += 1
            assert steps < 5, f"did not converge from {start}"

    def test_never_moves_away_from_target(self, hedger):
        for net in [n / 2 for n in range(-400, 401)]:
            decision = hedger.decide(float(net))
            if decision.should_hedge:
                assert abs(decision.net_delta_after - 20.0) < abs(net - 20.0)


class TestLimits:
    def test_respects_the_maximum_order_size(self, es):
        hedger = DeltaHedger(HedgeConfig(**OFFSET_BAND, max_hedge_contracts=3), es)
        decision = hedger.decide(1000.0)
        assert decision.contracts == -3

    def test_respects_the_minimum_order_size(self, es):
        hedger = DeltaHedger(HedgeConfig(**OFFSET_BAND, min_hedge_contracts=5), es)
        decision = hedger.decide(-30.0)  # wants +5 contracts
        assert decision.should_hedge and decision.contracts == 5
        assert not hedger.decide(-10.0).should_hedge  # wants +3, below the floor

    def test_respects_the_cooldown(self, es):
        hedger = DeltaHedger(
            HedgeConfig(**OFFSET_BAND, min_seconds_between_hedges=60), es
        )
        assert not hedger.decide(200.0, seconds_since_last_hedge=10).should_hedge
        assert hedger.decide(200.0, seconds_since_last_hedge=90).should_hedge

    def test_a_zero_band_still_does_not_churn(self, es):
        hedger = DeltaHedger(HedgeConfig(target=20.0, band=0.0), es)
        decision = hedger.decide(21.0)
        assert not decision.should_hedge


class TestFlatten:
    @pytest.mark.parametrize("net,expected", [(100.0, -10), (-35.0, 4), (0.0, 0)])
    def test_flatten_returns_the_offsetting_quantity(self, hedger, net, expected):
        assert hedger.flatten(net) == expected


def test_a_custom_target_is_honoured(es):
    hedger = DeltaHedger(HedgeConfig(target=-40.0, band=5.0), es)
    decision = hedger.decide(0.0)
    assert decision.contracts == -4
    assert decision.net_delta_after == pytest.approx(-40.0)


class TestNeutralTarget:
    """The default the straddle strategy runs: hold the book flat.

    The band is a fixed heuristic of +/-10 delta units -- one whole MES
    contract -- which is the smallest width that can both bind and be landed
    inside. Anything under half a contract is inert (see ``hedger.py``).
    """

    @pytest.fixture
    def neutral(self, es):
        return DeltaHedger(HedgeConfig(), es)

    def test_the_shipped_default_is_delta_neutral(self):
        cfg = HedgeConfig()
        assert cfg.target == 0.0
        assert cfg.lower == -10.0 and cfg.upper == 10.0

    def test_the_default_band_is_wide_enough_to_bind(self, es):
        """A band under half a hedge contract fires on exactly the same bars
        as one at half a contract, which makes the parameter inert."""
        assert HedgeConfig().band >= es.hedge_quantum / 2.0

    @pytest.mark.parametrize("net", [-10.0, -4.0, 0.0, 7.5, 10.0])
    def test_does_nothing_inside_the_band(self, neutral, net):
        assert not neutral.decide(net).should_hedge

    def test_a_long_book_is_sold_back_to_flat(self, neutral):
        decision = neutral.decide(64.0)
        assert decision.should_hedge and decision.contracts == -6
        assert decision.net_delta_after == pytest.approx(4.0)

    def test_a_short_book_is_bought_back_to_flat(self, neutral):
        decision = neutral.decide(-64.0)
        assert decision.should_hedge and decision.contracts == 6
        assert decision.net_delta_after == pytest.approx(-4.0)

    def test_it_is_symmetric_about_zero(self, neutral):
        """Neither regime may be hedged more eagerly than the other."""
        for net in (12.0, 27.0, 48.0, 133.0):
            up = neutral.decide(net)
            down = neutral.decide(-net)
            assert up.contracts == -down.contracts

    @pytest.mark.parametrize("start", [-500.0, -73.0, 11.0, 44.0, 250.0])
    def test_converges_without_oscillating(self, neutral, start):
        net, steps = start, 0
        while True:
            decision = neutral.decide(net)
            if not decision.should_hedge:
                break
            net = decision.net_delta_after
            steps += 1
            assert steps < 5, f"did not converge from {start}"

    def test_the_residual_never_exceeds_half_a_contract(self, neutral, es):
        for net in [n / 2 for n in range(-400, 401)]:
            decision = neutral.decide(float(net))
            if decision.should_hedge:
                assert abs(decision.residual_delta) <= es.hedge_quantum / 2 + 1e-9


class TestOvernightBand:
    """The band widens outside the regular session (see ``config.py`` and
    the hedger's module docstring): a position is now carried overnight, so
    a breach that would bind during RTH may not bind outside it."""

    @pytest.fixture
    def neutral(self, es):
        return DeltaHedger(
            HedgeConfig(target=0.0, band=10.0, overnight_band_multiplier=2.5), es
        )

    def test_a_breach_that_binds_intraday_can_be_inert_overnight(self, neutral):
        """15 units breaches the 10-wide intraday band but sits inside the
        25-wide overnight one."""
        assert neutral.decide(15.0, in_session=True).should_hedge
        assert not neutral.decide(15.0, in_session=False).should_hedge

    def test_a_large_enough_breach_still_hedges_overnight(self, neutral):
        """The band widens; it is never switched off. A move past the wider
        bound must still be hedged, because an overnight gap is exactly
        when an unhedged straddle does the most damage."""
        decision = neutral.decide(40.0, in_session=False)
        assert decision.should_hedge and decision.contracts < 0  # sells to reduce it

    def test_the_default_multiplier_is_off_during_the_session(self, es):
        """``in_session`` defaults True, so a caller that never passes it
        (as every pre-existing test in this file does) gets the intraday
        band unchanged -- the overnight widening is opt-in per call, not a
        silent global change."""
        cfg = HedgeConfig(target=0.0, band=10.0, overnight_band_multiplier=5.0)
        hedger = DeltaHedger(cfg, es)
        assert hedger.decide(12.0).should_hedge

    def test_a_multiplier_of_one_hedges_identically_around_the_clock(self, es):
        cfg = HedgeConfig(target=0.0, band=10.0, overnight_band_multiplier=1.0)
        hedger = DeltaHedger(cfg, es)
        day = hedger.decide(15.0, in_session=True)
        night = hedger.decide(15.0, in_session=False)
        assert day.should_hedge == night.should_hedge == True
        assert day.contracts == night.contracts

    def test_the_reason_names_the_overnight_band(self, neutral):
        decision = neutral.decide(15.0, in_session=False)
        assert "overnight" in decision.reason
