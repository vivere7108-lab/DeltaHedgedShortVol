"""Dealer gamma exposure: the flip point, and the regime it implies.

What this computes
------------------
GEX is an estimate of the gamma the option dealer community is carrying,
inferred from open interest.  The standard assumption -- and the one every
published GEX print uses -- is that the public buys puts and sells calls, so
the dealer is *long the calls and short the puts*::

    gex(K) = mult * S^2 * 0.01 * gamma(K) * (call_sign*OI_call + put_sign*OI_put)

The ``S^2 * 0.01`` turns per-point gamma into dollars of delta the dealer
must trade for a 1% move, which is the unit the number is quoted in.

Why it matters is entirely mechanical.  A dealer who is **short gamma**
(negative GEX) has to sell as the market falls and buy as it rises: their
hedging *adds* to the move.  A dealer who is **long gamma** (positive GEX)
does the opposite and damps it.  So the sign of GEX is a statement about
whether hedging flow will amplify or suppress realised volatility -- which
is exactly the variable a delta-hedged straddle is a bet on.

The **gamma flip point** is the spot level at which total GEX crosses zero.
It is found by repricing the whole chain's gamma across a grid of
hypothetical spot levels, holding open interest fixed, and interpolating the
crossing.  Above it dealers are long gamma, below it they are short.

What the strategy does with it
------------------------------
=================  ==================  ==============  ===================
GEX                dealer hedging      realised vol    the position
=================  ==================  ==============  ===================
negative           amplifies moves     runs above IV   LONG the straddle
positive           damps moves         runs below IV   SHORT the straddle
near zero / flip   about to change     unknown         stand aside
=================  ==================  ==============  ===================

Honest limits
-------------
1. **Open interest is not positioning.**  Who is long and who is short is
   not in the OI print; the call/put sign convention is an assumption, and
   it is the load-bearing one.  ``call_sign``/``put_sign`` are config so it
   can be varied rather than believed.
2. **OI is stale intraday.**  Exchange open interest is an end-of-previous-
   day figure.  Same-day 0DTE flow -- which is most of the flow -- is not in
   it.  This is the largest approximation in the GEX layer.
3. **0DTE gamma is a spike.**  As expiry approaches, gamma concentrates at
   the money and vanishes elsewhere, so the profile becomes dominated by two
   or three strikes and the flip point gets noisy.  ``min_hours_to_expiry``
   floors the tenor used for classification so the shape stays legible; it
   never touches the greeks the hedger acts on.
4. **The flip point moves with vol.**  It is computed off the same modelled
   surface used to price the book, so an error in the skew moves the flip
   point as well as the credit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Sequence

import numpy as np

from .config import GexConfig
from .instruments import RiskSource
from .pricing import black76_gamma
from .volsurface import VolSurface

log = logging.getLogger(__name__)

#: The three regimes. Strings rather than an enum so they land in a CSV and
#: a log line unchanged.
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"

#: What each regime says to trade. The sign is the straddle's quantity sign.
LONG_STRADDLE = 1
SHORT_STRADDLE = -1
STAND_ASIDE = 0

HOURS_PER_YEAR = 365.0 * 24.0


@dataclass(frozen=True)
class StrikeOpenInterest:
    """Open interest at one strike of one expiry."""

    strike: float
    call_oi: float
    put_oi: float


class OpenInterestProvider(Protocol):
    """Anything that can say what open interest sits on a chain.

    The backtest generates it, a CSV replays it, and the live path reads it
    from IBKR -- but ``GexCalculator`` sees the same list either way, which
    is what lets the forward test exercise the classification code that was
    measured historically.
    """

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]: ...


@dataclass(frozen=True)
class StrikeGex:
    """The dealer gamma one strike contributes, split by right."""

    strike: float
    call_oi: float
    put_oi: float
    gamma: float
    call_gex: float
    put_gex: float

    @property
    def net_gex(self) -> float:
        return self.call_gex + self.put_gex


@dataclass(frozen=True)
class GexProfile:
    """The whole picture at one spot level: the number, the flip, the call."""

    spot: float
    time_to_expiry: float
    total_gex: float
    #: Absolute gamma in the book, both rights summed unsigned. ``total_gex``
    #: measured against this is how *directional* dealer positioning is,
    #: which is what the neutral threshold is written in terms of.
    gross_gex: float
    call_gex: float
    put_gex: float
    flip_point: float | None
    regime: str
    reason: str
    by_strike: tuple[StrikeGex, ...] = ()

    @property
    def direction(self) -> int:
        """The straddle quantity sign this profile implies.

        Negative GEX -> dealers amplify moves -> we want gamma -> long.
        Positive GEX -> dealers damp moves -> we want theta -> short.
        """
        if self.regime == NEGATIVE:
            return LONG_STRADDLE
        if self.regime == POSITIVE:
            return SHORT_STRADDLE
        return STAND_ASIDE

    @property
    def above_flip(self) -> bool | None:
        if self.flip_point is None:
            return None
        return self.spot > self.flip_point

    @property
    def distance_to_flip(self) -> float | None:
        """Points from spot to the flip; positive means spot is above it."""
        if self.flip_point is None:
            return None
        return self.spot - self.flip_point

    @property
    def peak_strike(self) -> float | None:
        """The strike carrying the most absolute gamma -- the pin candidate."""
        if not self.by_strike:
            return None
        return max(self.by_strike, key=lambda s: abs(s.net_gex)).strike

    def describe(self) -> str:
        flip = f"{self.flip_point:,.1f}" if self.flip_point is not None else "none found"
        return (
            f"GEX {self.total_gex / 1e6:+,.1f}M/1% at {self.spot:,.2f}, "
            f"flip {flip}, regime {self.regime}"
        )

    def table(self, limit: int = 15) -> str:
        """The strikes carrying the most gamma, for eyeballing a live read."""
        rows = sorted(self.by_strike, key=lambda s: -abs(s.net_gex))[:limit]
        rows.sort(key=lambda s: s.strike)
        lines = [f"{'strike':>9} {'call OI':>9} {'put OI':>9} {'net GEX ($M/1%)':>17}"]
        for row in rows:
            lines.append(
                f"{row.strike:>9,.0f} {row.call_oi:>9,.0f} {row.put_oi:>9,.0f} "
                f"{row.net_gex / 1e6:>17,.2f}"
            )
        return "\n".join(lines)


class GexCalculator:
    """Turns open interest into a profile, a flip point and a regime."""

    def __init__(
        self,
        cfg: GexConfig,
        source: RiskSource,
        surface: VolSurface,
        risk_free_rate: float = 0.0,
    ):
        self.cfg = cfg
        self.source = source
        self.surface = surface
        self.risk_free_rate = risk_free_rate

    # -- the profile -----------------------------------------------------

    def profile(
        self,
        spot: float,
        open_interest: Sequence[StrikeOpenInterest],
        time_to_expiry: float,
        atm_iv: float,
    ) -> GexProfile:
        """Total GEX at ``spot``, the flip point, and the regime they imply."""
        strikes, calls, puts = self._arrays(spot, open_interest)
        t = self._effective_tenor(time_to_expiry)

        if strikes.size == 0:
            return GexProfile(
                spot=spot, time_to_expiry=t, total_gex=0.0, gross_gex=0.0,
                call_gex=0.0, put_gex=0.0, flip_point=None, regime=NEUTRAL,
                reason="no open interest inside the strike window",
            )

        gamma = black76_gamma(spot, strikes, t, self._vols(spot, strikes, atm_iv),
                              self.risk_free_rate)
        scale = self.source.option.multiplier * spot * spot * 0.01
        call_gex = scale * gamma * self.cfg.call_sign * calls
        put_gex = scale * gamma * self.cfg.put_sign * puts
        per_strike = call_gex + put_gex

        total = float(per_strike.sum())
        # Gross is the gamma in the book, summed per *leg* rather than per
        # strike. Summing net-per-strike would collapse to zero for a chain
        # with matched call and put interest -- which is a maximally
        # gamma-laden book, not an empty one -- and the neutral test below
        # divides by this.
        gross = float((np.abs(call_gex) + np.abs(put_gex)).sum())
        flip = self._flip_point(spot, strikes, calls, puts, t, atm_iv)
        regime, reason = self._classify(spot, total, gross, flip)

        return GexProfile(
            spot=spot,
            time_to_expiry=t,
            total_gex=total,
            gross_gex=gross,
            call_gex=float(call_gex.sum()),
            put_gex=float(put_gex.sum()),
            flip_point=flip,
            regime=regime,
            reason=reason,
            by_strike=tuple(
                StrikeGex(
                    strike=float(k), call_oi=float(c), put_oi=float(p),
                    gamma=float(g), call_gex=float(cg), put_gex=float(pg),
                )
                for k, c, p, g, cg, pg in zip(
                    strikes, calls, puts, gamma, call_gex, put_gex
                )
            ),
        )

    def total_at(
        self,
        hypothetical_spot: float,
        spot: float,
        open_interest: Sequence[StrikeOpenInterest],
        time_to_expiry: float,
        atm_iv: float,
    ) -> float:
        """Total GEX the current book would carry if spot were elsewhere."""
        strikes, calls, puts = self._arrays(spot, open_interest)
        if strikes.size == 0:
            return 0.0
        t = self._effective_tenor(time_to_expiry)
        return float(
            self._curve(np.array([hypothetical_spot]), strikes, calls, puts, t, atm_iv)[0]
        )

    # -- internals -------------------------------------------------------

    def _effective_tenor(self, time_to_expiry: float) -> float:
        """Tenor used for classification, floored (see the module docstring)."""
        return max(time_to_expiry, self.cfg.min_hours_to_expiry / HOURS_PER_YEAR)

    def _arrays(
        self, spot: float, open_interest: Sequence[StrikeOpenInterest]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Strikes inside the window, with their OI, as sorted arrays."""
        half = spot * self.cfg.strike_width_pct
        rows = sorted(
            (r for r in open_interest if abs(r.strike - spot) <= half and r.strike > 0),
            key=lambda r: r.strike,
        )
        if not rows:
            empty = np.zeros(0)
            return empty, empty, empty.copy()
        return (
            np.array([r.strike for r in rows], dtype=float),
            np.array([r.call_oi for r in rows], dtype=float),
            np.array([r.put_oi for r in rows], dtype=float),
        )

    def _vols(self, spot, strikes, atm_iv: float) -> np.ndarray:
        return self.surface.iv_array(spot, strikes, atm_iv)

    def _curve(
        self, spots: np.ndarray, strikes: np.ndarray, calls: np.ndarray,
        puts: np.ndarray, t: float, atm_iv: float,
    ) -> np.ndarray:
        """Total GEX at each of ``spots``, holding open interest fixed.

        One vectorised block rather than a loop: the flip search reprices
        every strike at every grid point on every bar, and doing that a
        scalar at a time dominates the whole backtest.
        """
        column = spots[:, None]
        vols = self.surface.iv_array(column, strikes[None, :], atm_iv)
        gamma = black76_gamma(column, strikes[None, :], t, vols, self.risk_free_rate)
        weight = self.cfg.call_sign * calls + self.cfg.put_sign * puts
        scale = self.source.option.multiplier * column * column * 0.01
        return (scale * gamma * weight[None, :]).sum(axis=1)

    def _flip_point(
        self, spot: float, strikes: np.ndarray, calls: np.ndarray,
        puts: np.ndarray, t: float, atm_iv: float,
    ) -> float | None:
        """The spot level where total GEX crosses zero, nearest to ``spot``.

        Returns ``None`` when the curve holds one sign across the whole
        search range -- a real answer ("there is no flip nearby"), not a
        failure, and the caller must not fabricate one from the endpoints.
        """
        half = spot * self.cfg.flip_search_pct
        grid = np.linspace(spot - half, spot + half, self.cfg.flip_search_steps)
        grid = grid[grid > 0.0]
        if grid.size < 2:
            return None

        curve = self._curve(grid, strikes, calls, puts, t, atm_iv)
        crossings: list[float] = []
        for i in range(len(grid) - 1):
            lo, hi = curve[i], curve[i + 1]
            if lo == 0.0:
                crossings.append(float(grid[i]))
            elif (lo < 0.0) != (hi < 0.0):
                # Linear interpolation between the bracketing grid points.
                crossings.append(float(grid[i] + (grid[i + 1] - grid[i]) * lo / (lo - hi)))
        if curve[-1] == 0.0:
            crossings.append(float(grid[-1]))
        if not crossings:
            return None
        return min(crossings, key=lambda level: abs(level - spot))

    def _classify(
        self, spot: float, total: float, gross: float, flip: float | None
    ) -> tuple[str, str]:
        """Regime, and the sentence explaining it, for the event log."""
        if gross <= 0.0:
            return NEUTRAL, "no gamma in the chain"

        if flip is not None and abs(spot - flip) <= spot * self.cfg.flip_proximity_pct:
            return NEUTRAL, (
                f"spot {spot:,.2f} is within {self.cfg.flip_proximity_pct:.2%} of the "
                f"gamma flip at {flip:,.2f}; the sign is about to change"
            )

        share = abs(total) / gross
        if share < self.cfg.neutral_gex_fraction:
            return NEUTRAL, (
                f"net GEX is only {share:.1%} of gross "
                f"(threshold {self.cfg.neutral_gex_fraction:.0%}); dealers are "
                "close to flat"
            )

        flip_text = f", flip {flip:,.1f}" if flip is not None else ""
        if total > 0.0:
            return POSITIVE, (
                f"GEX {total / 1e6:+,.1f}M/1% at {spot:,.2f}{flip_text}: dealers are "
                "long gamma and hedge against the move, damping realised vol"
            )
        return NEGATIVE, (
            f"GEX {total / 1e6:+,.1f}M/1% at {spot:,.2f}{flip_text}: dealers are "
            "short gamma and hedge with the move, amplifying realised vol"
        )
