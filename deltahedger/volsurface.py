"""Volatility surface used to price strikes away from the money.

IBKR publishes an at-the-money implied-volatility series for the future
(``whatToShow="OPTION_IMPLIED_VOLATILITY"``) but not a strike-by-strike
surface, so out-of-the-money puts have to be extrapolated.  The model here
is a log-moneyness skew:

    iv(K) = clip( multiplier * (atm + slope*ln(K/F) + curvature*ln(K/F)^2) )

with ``slope < 0`` so lower strikes carry higher vol.  This is an assumption
and it is the single largest modelling approximation in the backtest -- a
short put is sold *on* the skew, so getting the slope wrong biases entry
credits directly.  ``VolConfig`` exposes the parameters; swap in a fitted
surface by subclassing ``VolSurface.iv``.
"""

from __future__ import annotations

import math

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


class FlatVolSurface(VolSurface):
    """No skew -- every strike prices off the ATM level.

    Useful as a control when you want to see how much of a backtest result is
    coming from the assumed skew shape rather than the strategy.
    """

    def iv(self, future: float, strike: float, atm_iv: float, t: float = 0.0) -> float:
        if not math.isfinite(atm_iv) or atm_iv <= 0.0:
            atm_iv = self.cfg.fallback_atm_iv
        return min(max(atm_iv * self.cfg.iv_multiplier, self.cfg.min_iv), self.cfg.max_iv)
