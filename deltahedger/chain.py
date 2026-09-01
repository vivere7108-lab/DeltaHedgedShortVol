"""Option chain construction and strike selection.

In the backtest there is no real chain to read, so one is synthesised on the
listed strike grid (5-point increments for ES dailies) and priced with
Black-76 off the vol surface.  The live path builds the same
``OptionQuote`` objects from IBKR chain data, so ``select_short_put`` is
shared: the strike chosen in a forward test is chosen by the same code that
was validated historically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .config import StrategyConfig
from .instruments import RiskSource
from .pricing import Greeks, black76
from .volsurface import VolSurface


@dataclass(frozen=True)
class OptionQuote:
    """One option in the chain, marked and with greeks attached."""

    strike: float
    right: str
    expiry: date
    price: float
    iv: float
    greeks: Greeks
    time_to_expiry: float

    @property
    def abs_delta(self) -> float:
        return abs(self.greeks.delta)


def strike_grid(
    future: float,
    source: RiskSource,
    width_pct: float = 0.08,
) -> list[float]:
    """Listed strikes spanning +/- ``width_pct`` around the future."""
    step = source.strike_increment
    lo = math.floor(future * (1.0 - width_pct) / step) * step
    hi = math.ceil(future * (1.0 + width_pct) / step) * step
    count = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 10) for i in range(count)]


def build_put_chain(
    future: float,
    expiry: date,
    time_to_expiry: float,
    atm_iv: float,
    source: RiskSource,
    surface: VolSurface,
    risk_free_rate: float = 0.0,
    width_pct: float = 0.08,
) -> list[OptionQuote]:
    """Synthesise a put chain priced off the vol surface."""
    quotes: list[OptionQuote] = []
    for strike in strike_grid(future, source, width_pct):
        iv = surface.iv(future, strike, atm_iv, time_to_expiry)
        greeks = black76(future, strike, time_to_expiry, iv, risk_free_rate, "P")
        quotes.append(
            OptionQuote(
                strike=strike,
                right="P",
                expiry=expiry,
                price=greeks.price,
                iv=iv,
                greeks=greeks,
                time_to_expiry=time_to_expiry,
            )
        )
    return quotes


def select_short_put(
    chain: list[OptionQuote],
    cfg: StrategyConfig,
    future: float,
) -> OptionQuote | None:
    """Pick the put to sell.

    ``strike_mode == "delta"`` takes the strike whose delta is closest to
    ``short_put_delta`` and rejects the choice if no strike lands within
    ``short_put_delta_tolerance`` -- which is the common case late in a 0DTE
    session, when every listed strike is either ~0 or ~1 delta.  Returning
    ``None`` there is deliberate: no strike at the intended risk exists, so
    the strategy should stand aside rather than sell something else.
    """
    puts = [q for q in chain if q.right == "P" and q.price > 0.0]
    if not puts:
        return None

    if cfg.strike_mode == "moneyness":
        target_strike = future * (1.0 - cfg.short_put_otm_pct)
        return min(puts, key=lambda q: abs(q.strike - target_strike))

    best = min(puts, key=lambda q: abs(q.abs_delta - cfg.short_put_delta))
    if abs(best.abs_delta - cfg.short_put_delta) > cfg.short_put_delta_tolerance:
        return None
    return best
