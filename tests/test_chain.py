from datetime import date

import pytest

from deltahedger.chain import atm_strike, build_chain, select_atm_straddle, strike_grid
from deltahedger.config import VolConfig
from deltahedger.volsurface import FlatVolSurface, VolSurface

F = 5000.0
EXPIRY = date(2025, 6, 10)
T = 6.4 / 24 / 365


@pytest.fixture
def surface():
    return VolSurface(VolConfig())


class TestStrikeGrid:
    def test_uses_the_listed_increment(self, es):
        grid = strike_grid(F, es)
        assert all(round(k) % 5 == 0 for k in grid)
        assert grid[1] - grid[0] == pytest.approx(es.strike_increment)

    def test_brackets_the_future(self, es):
        grid = strike_grid(F, es, width_pct=0.05)
        assert grid[0] < F * 0.96 and grid[-1] > F * 1.04


class TestVolSurface:
    def test_puts_carry_more_vol_than_calls(self, surface):
        assert surface.iv(F, 4900.0, 0.15) > surface.iv(F, 5000.0, 0.15)
        assert surface.iv(F, 5100.0, 0.15) < surface.iv(F, 5000.0, 0.15)

    def test_atm_is_the_atm_level(self, surface):
        assert surface.iv(F, F, 0.15) == pytest.approx(0.15)

    def test_is_clipped_to_sane_bounds(self):
        surface = VolSurface(VolConfig(min_iv=0.05, max_iv=0.9, skew_slope=-50.0))
        assert surface.iv(F, 3000.0, 0.15) <= 0.9
        assert surface.iv(F, 9000.0, 0.15) >= 0.05

    def test_a_bad_atm_input_falls_back(self, surface):
        assert surface.iv(F, F, 0.0) == pytest.approx(VolConfig().fallback_atm_iv)
        assert surface.iv(F, F, float("nan")) == pytest.approx(VolConfig().fallback_atm_iv)

    def test_the_multiplier_scales_the_whole_surface(self):
        surface = VolSurface(VolConfig(iv_multiplier=2.0))
        assert surface.iv(F, F, 0.15) == pytest.approx(0.30)

    def test_flat_surface_ignores_the_strike(self):
        surface = FlatVolSurface(VolConfig())
        assert surface.iv(F, 4500.0, 0.15) == surface.iv(F, 5500.0, 0.15)


class TestAtmStrike:
    def test_picks_the_nearest_listed_strike(self, es):
        assert atm_strike(5002.0, es) == 5000.0
        assert atm_strike(5003.5, es) == 5005.0

    def test_a_tie_resolves_the_same_way_every_time(self, es):
        """Only a tie-break: at a 5-point increment both candidates are half
        a point from the money, well under one hedge contract of delta."""
        assert atm_strike(5002.5, es) == atm_strike(5002.5, es) == 5005.0

    def test_the_result_is_always_on_the_grid(self, es):
        for price in (4813.3, 4999.9, 5000.0, 5127.6):
            assert atm_strike(price, es) % es.strike_increment == 0


class TestStraddleSelection:
    def build(self, surface, es, future=F, t=T, iv=0.15):
        return select_atm_straddle(future, EXPIRY, t, iv, es, surface, 0.04)

    def test_it_returns_both_legs_on_one_strike(self, surface, es):
        quote = self.build(surface, es)
        assert quote is not None
        assert quote.call.strike == quote.put.strike == quote.strike
        assert quote.call.right == "C" and quote.put.right == "P"

    def test_the_strike_is_at_the_money(self, surface, es):
        assert abs(self.build(surface, es).strike - F) <= es.strike_increment / 2

    def test_the_premium_is_the_sum_of_the_legs(self, surface, es):
        quote = self.build(surface, es)
        assert quote.price == pytest.approx(quote.call.price + quote.put.price)

    def test_the_greeks_are_the_sum_of_the_legs(self, surface, es):
        quote = self.build(surface, es)
        assert quote.delta == pytest.approx(
            quote.call.greeks.delta + quote.put.greeks.delta
        )
        assert quote.gamma == pytest.approx(
            quote.call.greeks.gamma + quote.put.greeks.gamma
        )
        assert quote.vega == pytest.approx(quote.call.greeks.vega + quote.put.greeks.vega)
        assert quote.theta == pytest.approx(
            quote.call.greeks.theta + quote.put.greeks.theta
        )

    def test_an_atm_straddle_is_nearly_delta_neutral(self, surface, es):
        assert abs(self.build(surface, es).delta) < 0.15

    def test_the_quote_is_long_gamma_and_short_theta(self, surface, es):
        """Unsigned by position: the quote describes buying the pair, and a
        short position inherits the opposite signs from its quantity."""
        quote = self.build(surface, es)
        assert quote.gamma > 0
        assert quote.theta < 0

    def test_the_two_legs_carry_the_same_gamma(self, surface, es):
        """Put-call parity: the difference is linear in the forward."""
        quote = self.build(surface, es)
        assert quote.call.greeks.gamma == pytest.approx(
            quote.put.greeks.gamma, rel=1e-9
        )

    def test_the_skew_makes_the_put_leg_richer_in_vol(self, surface, es):
        """Both legs sit on the same strike, so their vols differ only
        because the strike is not exactly at the money."""
        quote = self.build(surface, es, future=5012.0)
        assert quote.put.iv == quote.call.iv  # same strike, same surface point
        assert quote.iv == pytest.approx(quote.call.iv)

    def test_it_declines_once_the_pair_has_no_premium_left(self, surface, es):
        """In the last seconds of a 0DTE session both legs collapse to
        intrinsic; there is no vol left to trade in either direction."""
        assert self.build(surface, es, t=0.0) is None


class TestChainBuilding:
    def test_it_builds_both_rights(self, surface, es):
        for right in ("C", "P"):
            chain = build_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right=right)
            assert chain and all(q.right == right for q in chain)

    def test_calls_carry_positive_delta_and_puts_negative(self, surface, es):
        calls = build_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right="C")
        puts = build_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right="P")
        assert all(q.greeks.delta >= 0 for q in calls)
        assert all(q.greeks.delta <= 0 for q in puts)

    def test_the_chain_spans_the_strike_grid(self, surface, es):
        chain = build_chain(F, EXPIRY, T, 0.15, es, surface, 0.04)
        assert [q.strike for q in chain] == strike_grid(F, es)
