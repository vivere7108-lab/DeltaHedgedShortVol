from datetime import date

import pytest

from deltahedger.chain import (
    build_option_chain, build_put_chain, select_short_option, select_short_put,
    strike_grid,
)
from deltahedger.config import StrategyConfig, VolConfig
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


class TestSelection:
    def build(self, surface, es, t=T, iv=0.15):
        return build_put_chain(F, EXPIRY, t, iv, es, surface, 0.04)

    def test_picks_a_strike_near_the_delta_target(self, surface, es):
        chosen = select_short_put(self.build(surface, es), StrategyConfig(), F)
        assert chosen is not None
        assert abs(chosen.abs_delta - 0.20) <= 0.10

    def test_picks_the_closest_available_delta(self, surface, es):
        chain = self.build(surface, es)
        chosen = select_short_put(chain, StrategyConfig(), F)
        best = min(abs(q.abs_delta - 0.20) for q in chain if q.price > 0)
        assert abs(chosen.abs_delta - 0.20) == pytest.approx(best)

    def test_the_chosen_put_is_out_of_the_money(self, surface, es):
        assert select_short_put(self.build(surface, es), StrategyConfig(), F).strike < F

    def test_declines_when_no_strike_is_close_enough(self, surface, es):
        """Late in a 0DTE session every strike is ~0 or ~1 delta."""
        chain = self.build(surface, es, t=3 / 60 / 24 / 365)
        assert select_short_put(chain, StrategyConfig(), F) is None

    def test_a_wider_tolerance_accepts_a_worse_strike(self, surface, es):
        chain = self.build(surface, es, t=3 / 60 / 24 / 365)
        cfg = StrategyConfig(short_put_delta_tolerance=1.0)
        assert select_short_put(chain, cfg, F) is not None

    def test_moneyness_mode_targets_a_percentage(self, surface, es):
        cfg = StrategyConfig(strike_mode="moneyness", short_put_otm_pct=0.02)
        chosen = select_short_put(self.build(surface, es), cfg, F)
        assert chosen.strike == pytest.approx(4900.0, abs=es.strike_increment)

    def test_a_higher_delta_target_picks_a_higher_strike(self, surface, es):
        chain = self.build(surface, es)
        low = select_short_put(chain, StrategyConfig(short_put_delta=0.10), F)
        high = select_short_put(chain, StrategyConfig(short_put_delta=0.40), F)
        assert high.strike > low.strike

    def test_an_empty_chain_returns_none(self):
        assert select_short_put([], StrategyConfig(), F) is None


class TestCallSelection:
    def build(self, surface, es, t=T, iv=0.15):
        return build_put_chain(F, EXPIRY, t, iv, es, surface, 0.04)

    def build_call(self, surface, es, t=T, iv=0.15):
        return build_option_chain(F, EXPIRY, t, iv, es, surface, 0.04, right="C")

    def test_a_short_call_has_positive_delta(self, surface, es):
        chosen = select_short_option(
            self.build_call(surface, es), StrategyConfig(), F, right="C"
        )
        assert chosen is not None
        assert chosen.greeks.delta > 0

    def test_the_chosen_call_is_out_of_the_money(self, surface, es):
        chosen = select_short_option(
            self.build_call(surface, es), StrategyConfig(), F, right="C"
        )
        assert chosen.strike > F

    def test_picks_a_strike_near_the_call_delta_target(self, surface, es):
        cfg = StrategyConfig(short_call_delta=0.25, short_call_delta_tolerance=0.10)
        chosen = select_short_option(self.build_call(surface, es), cfg, F, right="C")
        assert abs(chosen.abs_delta - 0.25) <= 0.10

    def test_put_and_call_targets_are_independent(self, surface, es):
        cfg = StrategyConfig(short_put_delta=0.10, short_call_delta=0.40)
        put = select_short_option(self.build(surface, es), cfg, F, right="P")
        call = select_short_option(self.build_call(surface, es), cfg, F, right="C")
        assert abs(put.abs_delta - 0.10) <= 0.10
        assert abs(call.abs_delta - 0.40) <= 0.10

    def test_a_symmetric_put_and_call_roughly_offset(self, surface, es):
        """The default short_call_delta matches short_put_delta, so a
        strangle's own delta is small before hedging (see strategy.py)."""
        cfg = StrategyConfig()  # short_put_delta == short_call_delta == 0.20
        put = select_short_option(self.build(surface, es), cfg, F, right="P")
        call = select_short_option(self.build_call(surface, es), cfg, F, right="C")
        assert abs(put.greeks.delta + call.greeks.delta) < 0.10

    def test_moneyness_mode_does_not_apply_to_calls(self, surface, es):
        """strike_mode is put-only; a call always selects by delta."""
        cfg = StrategyConfig(strike_mode="moneyness", short_put_otm_pct=0.01)
        chosen = select_short_option(self.build_call(surface, es), cfg, F, right="C")
        assert abs(chosen.abs_delta - cfg.short_call_delta) <= cfg.short_call_delta_tolerance

    def test_declines_when_no_call_strike_is_close_enough(self, surface, es):
        chain = self.build_call(surface, es, t=3 / 60 / 24 / 365)
        assert select_short_option(chain, StrategyConfig(), F, right="C") is None

    def test_select_short_option_defaults_to_put(self, surface, es):
        chain = self.build(surface, es)
        assert select_short_option(chain, StrategyConfig(), F) == select_short_put(
            chain, StrategyConfig(), F
        )


class TestBuildOptionChain:
    def test_right_p_matches_build_put_chain(self, surface, es):
        via_alias = build_put_chain(F, EXPIRY, T, 0.15, es, surface, 0.04)
        via_generic = build_option_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right="P")
        assert [q.strike for q in via_alias] == [q.strike for q in via_generic]
        assert [q.price for q in via_alias] == [q.price for q in via_generic]

    def test_a_call_chain_has_calls_not_puts(self, surface, es):
        chain = build_option_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right="C")
        assert chain and all(q.right == "C" for q in chain)

    def test_put_and_call_chains_share_the_vol_surface(self, surface, es):
        """Same strike, same implied vol either side -- VolSurface.iv takes
        no right argument, so this checks the chain builder doesn't
        accidentally introduce an asymmetry."""
        puts = build_option_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right="P")
        calls = build_option_chain(F, EXPIRY, T, 0.15, es, surface, 0.04, right="C")
        by_strike_p = {q.strike: q.iv for q in puts}
        by_strike_c = {q.strike: q.iv for q in calls}
        assert by_strike_p == pytest.approx(by_strike_c)
