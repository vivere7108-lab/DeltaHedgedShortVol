"""Volatility surface used to price strikes away from the money.

IBKR publishes an at-the-money implied-volatility series for the future
(``whatToShow="OPTION_IMPLIED_VOLATILITY"``) but not a strike-by-strike
surface, so out-of-the-money puts have to be extrapolated.  The model here
is a log-moneyness skew:

    iv(K) = clip( multiplier * (atm + slope*ln(K/F) + curvature*ln(K/F)^2) )

with ``slope < 0`` so lower strikes carry higher vol.  This is an
assumption, and ``VolConfig`` exposes the parameters; swap in a fitted
surface by overriding ``VolSurface.iv`` -- and ``iv_array`` with it, since
the GEX flip search uses the vectorised form.

The straddle is traded at the money, so the skew biases its *premium* far
less than it did when this system sold an out-of-the-money put.  It has not
stopped mattering, though: the whole GEX profile is built on gammas priced
off this surface, so the slope moves the gamma flip point and therefore
which side the strategy takes.  The approximation moved from the entry price
to the signal.
"""

from __future__ import annotations

import math

import numpy as np

from .config import VolConfig


class VolSurface:
    def __init__(self, cfg: VolConfig):
        self.cfg = cfg

    def iv(self, future: float, strike: float, atm_iv: float, t: float = 0.0) -> float:
        """Implied vol for ``strike`` given the ATM level."""
        if not math.isfinite(atm_iv) or atm_iv <= 0.0:
            atm_iv = self.cfg.fallback_atm_iv
        moneyness = math.log(strike / future)
        raw = (
            atm_iv
            + self.cfg.skew_slope * moneyness
            + self.cfg.skew_curvature * moneyness * moneyness
        )
        return min(max(raw * self.cfg.iv_multiplier, self.cfg.min_iv), self.cfg.max_iv)

    def iv_array(self, future, strike, atm_iv: float, t: float = 0.0) -> np.ndarray:
        """``iv`` broadcast over arrays of futures and strikes.

        The gamma-flip search reprices the whole chain at every point of a
        hypothetical-spot grid, so the surface is asked for a 2-D block of
        vols on every bar.  ``future`` re-anchors the skew, which is the
        behaviour we want: moving spot re-centres the log-moneyness, it does
        not move the ATM level.
        """
        if not math.isfinite(atm_iv) or atm_iv <= 0.0:
            atm_iv = self.cfg.fallback_atm_iv
        moneyness = np.log(np.asarray(strike, dtype=float) / np.asarray(future, dtype=float))
        raw = (
            atm_iv
            + self.cfg.skew_slope * moneyness
            + self.cfg.skew_curvature * moneyness * moneyness
        )
        return np.clip(raw * self.cfg.iv_multiplier, self.cfg.min_iv, self.cfg.max_iv)


class FlatVolSurface(VolSurface):
    """No skew -- every strike prices off the ATM level.

    Useful as a control when you want to see how much of a backtest result is
    coming from the assumed skew shape rather than the strategy.
    """

    def iv(self, future: float, strike: float, atm_iv: float, t: float = 0.0) -> float:
        if not math.isfinite(atm_iv) or atm_iv <= 0.0:
            atm_iv = self.cfg.fallback_atm_iv
        return min(max(atm_iv * self.cfg.iv_multiplier, self.cfg.min_iv), self.cfg.max_iv)

    def iv_array(self, future, strike, atm_iv: float, t: float = 0.0) -> np.ndarray:
        if not math.isfinite(atm_iv) or atm_iv <= 0.0:
            atm_iv = self.cfg.fallback_atm_iv
        level = min(
            max(atm_iv * self.cfg.iv_multiplier, self.cfg.min_iv), self.cfg.max_iv
        )
        shape = np.broadcast(
            np.asarray(future, dtype=float), np.asarray(strike, dtype=float)
        ).shape
        return np.full(shape, level)
