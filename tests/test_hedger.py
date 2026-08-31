"""Tests for the delta band.

The properties that matter most are the ones that keep the hedger from
misbehaving in production: it must never oscillate, never trade away from
the target, and never fire inside the band.
"""

import pytest

from deltahedger.config import HedgeConfig
from deltahedger.hedger import DeltaHedger


@pytest.fixture
def hedger(es):
    return DeltaHedger(HedgeConfig(), es)


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
        hedger = DeltaHedger(HedgeConfig(max_hedge_contracts=3), es)
        decision = hedger.decide(1000.0)
        assert decision.contracts == -3

    def test_respects_the_minimum_order_size(self, es):
        hedger = DeltaHedger(HedgeConfig(min_hedge_contracts=5), es)
        decision = hedger.decide(-30.0)  # wants +5 contracts
        assert decision.should_hedge and decision.contracts == 5
        assert not hedger.decide(-10.0).should_hedge  # wants +3, below the floor

    def test_respects_the_cooldown(self, es):
        hedger = DeltaHedger(HedgeConfig(min_seconds_between_hedges=60), es)
        assert not hedger.decide(200.0, seconds_since_last_hedge=10).should_hedge
        assert hedger.decide(200.0, seconds_since_last_hedge=90).should_hedge

    def test_a_zero_band_still_does_not_churn(self, es):
        hedger = DeltaHedger(HedgeConfig(band=0.0), es)
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
