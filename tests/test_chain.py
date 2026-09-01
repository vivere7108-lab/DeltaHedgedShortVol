from datetime import date

import pytest

from deltahedger.chain import build_put_chain, select_short_put, strike_grid
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
