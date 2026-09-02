"""Option chain construction and strike selection.

In the backtest there is no real chain to read, so one is synthesised on the
listed strike grid (5-point increments for ES dailies) and priced with
Black-76 off the vol surface.  The live path builds the same
``OptionQuote`` objects from IBKR chain data, so ``select_atm_straddle`` is
shared: the strike chosen in a forward test is chosen by the same code that
was validated historically.

The traded unit is a ``StraddleQuote`` -- the call and the put on one
strike, quoted together.  Its greeks are the sum of the legs, which is what
makes the rest of the system regime-agnostic: the hedger, the sizing and the
P&L accounting see one instrument whose sign says whether we are buying
gamma or selling it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

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


@dataclass(frozen=True)
class StraddleQuote:
    """The call and the put on one strike, priced as a single instrument.

    Every greek here is *per straddle* and unsigned by position: the caller
    multiplies by a signed quantity.  Gamma is therefore always positive and
    theta always negative, and a short straddle inherits the opposite signs
    from its quantity rather than from the quote.
    """

    strike: float
    expiry: date
    call: OptionQuote
    put: OptionQuote
    time_to_expiry: float

    @property
    def price(self) -> float:
        """Premium of the pair: the debit to buy it, the credit to sell it."""
        return self.call.price + self.put.price

    @property
    def delta(self) -> float:
        return self.call.greeks.delta + self.put.greeks.delta

    @property
    def gamma(self) -> float:
        return self.call.greeks.gamma + self.put.greeks.gamma

    @property
    def vega(self) -> float:
        return self.call.greeks.vega + self.put.greeks.vega

    @property
    def theta(self) -> float:
        return self.call.greeks.theta + self.put.greeks.theta

    @property
    def iv(self) -> float:
        """Average of the two legs' vols; they differ under a skew."""
        return 0.5 * (self.call.iv + self.put.iv)

    def legs(self) -> tuple[OptionQuote, OptionQuote]:
        return self.call, self.put


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


def price_option(
    future: float,
    strike: float,
    right: str,
    expiry: date,
    time_to_expiry: float,
    atm_iv: float,
    surface: VolSurface,
    risk_free_rate: float = 0.0,
) -> OptionQuote:
    """One option, marked off the vol surface."""
    iv = surface.iv(future, strike, atm_iv, time_to_expiry)
    greeks = black76(future, strike, time_to_expiry, iv, risk_free_rate, right)
    return OptionQuote(
        strike=strike,
        right=right,
        expiry=expiry,
        price=greeks.price,
        iv=iv,
        greeks=greeks,
        time_to_expiry=time_to_expiry,
    )


def build_chain(
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
    """Synthesise a chain of one right, priced off the vol surface."""
    return [
        price_option(
            future, strike, right, expiry, time_to_expiry, atm_iv, surface,
            risk_free_rate,
        )
        for strike in strike_grid(future, source, width_pct)
    ]


def atm_strike(future: float, source: RiskSource) -> float:
    """The listed strike nearest the future.

    Ties go to the higher strike, which is only a tie-break: at a 5-point
    increment the two candidates are half a point from the money and the
    straddle's delta differs between them by well under one hedge contract.
    """
    step = source.strike_increment
    return round(math.floor(future / step + 0.5) * step, 10)


def select_atm_straddle(
    future: float,
    expiry: date,
    time_to_expiry: float,
    atm_iv: float,
    source: RiskSource,
    surface: VolSurface,
    risk_free_rate: float = 0.0,
) -> StraddleQuote | None:
    """Build the at-the-money straddle we trade in either regime.

    Returns ``None`` when the pair carries no premium at all, which happens
    in the last minutes of a 0DTE session once both legs have collapsed to
    intrinsic.  There is nothing to buy or sell there -- neither a gamma
    scalp nor a theta harvest exists in a zero-vega instrument -- so the
    strategy stands aside rather than paying commission for a stub.
    """
    strike = atm_strike(future, source)
    call = price_option(
        future, strike, "C", expiry, time_to_expiry, atm_iv, surface, risk_free_rate
    )
    put = price_option(
        future, strike, "P", expiry, time_to_expiry, atm_iv, surface, risk_free_rate
    )
    quote = StraddleQuote(
        strike=strike, expiry=expiry, call=call, put=put,
        time_to_expiry=time_to_expiry,
    )
    if quote.price <= 0.0 or quote.gamma <= 0.0:
        return None
    return quote
