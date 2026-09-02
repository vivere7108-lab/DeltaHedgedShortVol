import math

import pytest

from deltahedger.pricing import (
    MIN_YEARS,
    Greeks,
    black76,
    black76_gamma,
    implied_vol,
)

F, R = 5000.0, 0.04


class TestPutCallParity:
    @pytest.mark.parametrize("strike", [4500.0, 4900.0, 5000.0, 5100.0, 5600.0])
    @pytest.mark.parametrize("t", [1 / 365, 7 / 365, 0.25, 1.0])
    def test_parity_holds(self, strike, t):
        call = black76(F, strike, t, 0.18, R, "C").price
        put = black76(F, strike, t, 0.18, R, "P").price
        assert call - put == pytest.approx(math.exp(-R * t) * (F - strike), abs=1e-9)


class TestGreeks:
    def test_put_delta_is_negative_and_bounded(self):
        for strike in (4000.0, 5000.0, 6000.0):
            delta = black76(F, strike, 0.1, 0.2, R, "P").delta
            assert -1.0 <= delta <= 0.0

    def test_call_delta_is_positive_and_bounded(self):
        for strike in (4000.0, 5000.0, 6000.0):
            delta = black76(F, strike, 0.1, 0.2, R, "C").delta
            assert 0.0 <= delta <= 1.0

    def test_delta_is_monotone_in_strike(self):
        """A lower put strike is always less sensitive to the future."""
        deltas = [black76(F, k, 0.05, 0.2, R, "P").delta for k in range(4600, 5401, 50)]
        assert all(a >= b for a, b in zip(deltas, deltas[1:]))

    def test_gamma_and_vega_are_positive(self):
        greeks = black76(F, 4900.0, 0.05, 0.2, R, "P")
        assert greeks.gamma > 0
        assert greeks.vega > 0

    def test_short_dated_atm_theta_is_negative(self):
        assert black76(F, F, 4 / 24 / 365, 0.15, R, "P").theta < 0

    def test_delta_matches_a_numerical_bump(self):
        strike, t, vol, h = 4900.0, 0.05, 0.2, 0.01
        up = black76(F + h, strike, t, vol, R, "P").price
        down = black76(F - h, strike, t, vol, R, "P").price
        assert (up - down) / (2 * h) == pytest.approx(
            black76(F, strike, t, vol, R, "P").delta, abs=1e-6
        )

    def test_gamma_matches_a_numerical_bump(self):
        strike, t, vol, h = 4950.0, 0.05, 0.2, 0.5
        up = black76(F + h, strike, t, vol, R, "P").delta
        down = black76(F - h, strike, t, vol, R, "P").delta
        assert (up - down) / (2 * h) == pytest.approx(
            black76(F, strike, t, vol, R, "P").gamma, rel=1e-4
        )


class TestExpiryLimits:
    """0DTE lives here: the model must stay finite as T reaches zero."""

    @pytest.mark.parametrize("t", [0.0, MIN_YEARS / 2, -1.0])
    def test_at_expiry_a_put_is_worth_its_intrinsic(self, t):
        assert black76(F, 5100.0, t, 0.2, 0.0, "P").price == pytest.approx(100.0)
        assert black76(F, 4900.0, t, 0.2, 0.0, "P").price == pytest.approx(0.0)

    def test_at_expiry_delta_is_binary(self):
        assert black76(F, 5100.0, 0.0, 0.2, R, "P").delta == -1.0
        assert black76(F, 4900.0, 0.0, 0.2, R, "P").delta == 0.0
        assert black76(F, F, 0.0, 0.2, R, "P").delta == -0.5

    def test_greeks_stay_finite_approaching_expiry(self):
        for seconds in (3600, 600, 60, 10, 1, 0):
            greeks = black76(F, F, seconds / (365 * 24 * 3600), 0.15, R, "P")
            for value in (greeks.price, greeks.delta, greeks.gamma, greeks.vega, greeks.theta):
                assert math.isfinite(value), f"non-finite greek at {seconds}s"

    def test_zero_vol_degenerates_to_intrinsic(self):
        assert black76(F, 5100.0, 0.5, 0.0, 0.0, "P").price == pytest.approx(100.0)


class TestImpliedVol:
    @pytest.mark.parametrize("vol", [0.08, 0.15, 0.35, 0.90])
    @pytest.mark.parametrize("strike", [4700.0, 5000.0, 5300.0])
    def test_round_trips(self, vol, strike):
        price = black76(F, strike, 0.08, vol, R, "P").price
        assert implied_vol(price, F, strike, 0.08, R, "P") == pytest.approx(vol, abs=1e-5)

    def test_returns_none_below_intrinsic(self):
        assert implied_vol(1.0, F, 5500.0, 0.1, 0.0, "P") is None

    def test_returns_none_at_expiry(self):
        assert implied_vol(50.0, F, 5100.0, 0.0, R, "P") is None


class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        {"f": -1.0, "k": 5000.0}, {"f": 5000.0, "k": 0.0},
    ])
    def test_rejects_non_positive_inputs(self, kwargs):
        with pytest.raises(ValueError):
            black76(t=0.1, sigma=0.2, **kwargs)

    def test_rejects_an_unknown_right(self):
        with pytest.raises(ValueError, match="right must be"):
            black76(F, 5000.0, 0.1, 0.2, R, "X")


def test_scaled_greeks_multiply_through():
    scaled = Greeks(2.0, -0.2, 0.01, 0.5, -3.0).scaled(quantity=-10, multiplier=50.0)
    assert scaled.delta == pytest.approx(2.0)  # short 10 puts is long delta
    assert scaled.price == pytest.approx(-1000.0)


class TestVectorisedGamma:
    """``black76_gamma`` is a fast path for the GEX flip search, which
    reprices a whole chain across a grid of hypothetical spot levels on every
    bar. It must agree with ``black76`` exactly -- a fast path that disagrees
    with the model the book is marked at would put the regime and the greeks
    on different surfaces."""

    @pytest.mark.parametrize("strike", [4800.0, 4950.0, 5000.0, 5050.0, 5200.0])
    @pytest.mark.parametrize("t", [1.0 / 24 / 365, 6.4 / 24 / 365, 1.0 / 365])
    def test_it_matches_black76_exactly(self, strike, t):
        expected = black76(5000.0, strike, t, 0.18, 0.04, "P").gamma
        assert float(black76_gamma(5000.0, strike, t, 0.18, 0.04)) == pytest.approx(
            expected, rel=1e-12
        )

    def test_calls_and_puts_share_a_gamma(self):
        """Put-call parity: the two differ by a forward, which is linear."""
        call = black76(5000.0, 4980.0, 0.01, 0.18, 0.04, "C").gamma
        put = black76(5000.0, 4980.0, 0.01, 0.18, 0.04, "P").gamma
        assert call == pytest.approx(put, rel=1e-12)

    def test_it_broadcasts_over_a_spot_and_strike_grid(self):
        import numpy as np

        spots = np.array([[4950.0], [5000.0], [5050.0]])
        strikes = np.array([4975.0, 5000.0, 5025.0])
        grid = black76_gamma(spots, strikes, 0.01, 0.18, 0.04)
        assert grid.shape == (3, 3)
        for i, spot in enumerate(spots[:, 0]):
            for j, strike in enumerate(strikes):
                assert grid[i, j] == pytest.approx(
                    black76(spot, strike, 0.01, 0.18, 0.04, "P").gamma, rel=1e-12
                )

    def test_it_is_zero_at_the_expiry_floor(self):
        """Matching ``black76``: at expiry the delta is a step and gamma is a
        spike of zero width, which is not a number a hedger can act on."""
        assert float(black76_gamma(5000.0, 5000.0, 0.0, 0.18)) == 0.0
        assert black76(5000.0, 5000.0, 0.0, 0.18).gamma == 0.0

    def test_it_is_zero_at_the_volatility_floor(self):
        assert float(black76_gamma(5000.0, 4980.0, 0.01, 0.0)) == 0.0


class TestVectorisedVolSurface:
    """``iv_array`` feeds the same flip search and must agree with ``iv``."""

    def test_it_matches_the_scalar_surface(self):
        import numpy as np

        from deltahedger.config import VolConfig
        from deltahedger.volsurface import VolSurface

        surface = VolSurface(VolConfig())
        strikes = np.array([4800.0, 4950.0, 5000.0, 5100.0])
        vectorised = surface.iv_array(5000.0, strikes, 0.15)
        for strike, value in zip(strikes, vectorised):
            assert value == pytest.approx(surface.iv(5000.0, strike, 0.15), rel=1e-12)

    def test_moving_spot_recentres_the_skew(self):
        """The flip search moves spot across a grid; the ATM level must stay
        put while the log-moneyness re-anchors."""
        from deltahedger.config import VolConfig
        from deltahedger.volsurface import VolSurface

        surface = VolSurface(VolConfig())
        assert surface.iv_array(5000.0, 5000.0, 0.15) == pytest.approx(0.15)
        assert surface.iv_array(5100.0, 5100.0, 0.15) == pytest.approx(0.15)

    def test_the_flat_surface_ignores_the_strike(self):
        import numpy as np

        from deltahedger.config import VolConfig
        from deltahedger.volsurface import FlatVolSurface

        surface = FlatVolSurface(VolConfig())
        values = surface.iv_array(5000.0, np.array([4500.0, 5500.0]), 0.15)
        assert values[0] == values[1] == pytest.approx(0.15)
