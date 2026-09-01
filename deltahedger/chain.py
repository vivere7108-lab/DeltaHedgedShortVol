"""Option chain construction and strike selection.

In the backtest there is no real chain to read, so one is synthesised on the
listed strike grid (5-point increments for ES dailies) and priced with
Black-76 off the vol surface.  The live path builds the same
``OptionQuote`` objects from IBKR chain data, so ``select_short_option`` is
shared: the strike chosen in a forward test is chosen by the same code that
was validated historically.

Puts and calls share one vol surface.  ``VolSurface.iv`` is a function of
strike alone (no ``right`` argument), which is put-call parity's actual
statement for European-style pricing: a given strike carries one implied
vol regardless of which side you price at it.  ES options are American, so
this is an approximation already implicit in the put-only model this
extends -- stated here because it's easy to assume the skew is somehow
put-specific once a call chain exists alongside it.
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


def build_option_chain(
    future: float,
    expiry: date,
    time_to_expiry: float,
    atm_iv: float,
    source: RiskSource,
    surface: VolSurface,
    risk_free_rate: float = 0.0,
    width_pct: float = 0.08,
    right: str = "P",
) -> list[OptionQuote]:
    """Synthesise a put or call chain priced off the vol surface."""
    quotes: list[OptionQuote] = []
    for strike in strike_grid(future, source, width_pct):
        iv = surface.iv(future, strike, atm_iv, time_to_expiry)
        greeks = black76(future, strike, time_to_expiry, iv, risk_free_rate, right)
        quotes.append(
            OptionQuote(
                strike=strike,
                right=right,
                expiry=expiry,
                price=greeks.price,
                iv=iv,
                greeks=greeks,
                time_to_expiry=time_to_expiry,
            )
        )
    return quotes


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
    """Puts-only alias of ``build_option_chain``, kept for existing callers."""
    return build_option_chain(
        future, expiry, time_to_expiry, atm_iv, source, surface,
        risk_free_rate, width_pct, right="P",
    )


def select_short_option(
    chain: list[OptionQuote],
    cfg: StrategyConfig,
    future: float,
    right: str = "P",
) -> OptionQuote | None:
    """Pick the put or call to sell.

    Delta-targeted selection takes the strike whose delta is closest to the
    configured target for ``right`` and rejects the choice if no strike
    lands within the tolerance -- which is the common case late in a 0DTE
    session, when every listed strike is either ~0 or ~1 delta.  Returning
    ``None`` there is deliberate: no strike at the intended risk exists, so
    the strategy should stand aside rather than sell something else.

    ``strike_mode == "moneyness"`` is put-only (it targets a discount below
    the future, which has no natural symmetric meaning for a call without
    a second, independent parameter); a call always selects by delta.
    """
    candidates = [q for q in chain if q.right == right and q.price > 0.0]
    if not candidates:
        return None

    if right == "P" and cfg.strike_mode == "moneyness":
        target_strike = future * (1.0 - cfg.short_put_otm_pct)
        return min(candidates, key=lambda q: abs(q.strike - target_strike))

    target = cfg.short_put_delta if right == "P" else cfg.short_call_delta
    tolerance = (
        cfg.short_put_delta_tolerance if right == "P" else cfg.short_call_delta_tolerance
    )
    best = min(candidates, key=lambda q: abs(q.abs_delta - target))
    if abs(best.abs_delta - target) > tolerance:
        return None
    return best


def select_short_put(
    chain: list[OptionQuote],
    cfg: StrategyConfig,
    future: float,
) -> OptionQuote | None:
    """Puts-only alias of ``select_short_option``, kept for existing callers."""
    return select_short_option(chain, cfg, future, right="P")
