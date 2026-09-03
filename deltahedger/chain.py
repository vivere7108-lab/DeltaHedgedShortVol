"""Option chain construction, tenor choice and strike selection.

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

Which *series* that straddle is on is decided here too, by ``TenorPolicy``
and ``select_straddle``.  The system used to trade today's expiry and
nothing else; it now trades a listed expiry a few sessions out and closes
the position before the tenor decays into the expiry-day gamma spike.  Two
things pushed the choice off 0DTE:

* an at-the-money 0DTE straddle's gamma diverges into the bell, so the
  hedger is fighting a position whose delta is unstable on a timescale
  shorter than the rebalance interval;
* open interest -- the only input GEX has -- is an end-of-previous-day
  figure, and on the 0DTE series it is *stalest exactly where most of the
  flow is*.  A series listed a few days out has had its open interest built
  over several sessions, so the same stale print describes far more of the
  book that is actually there.

Both arguments are about the tenor, not about the strike, which is why the
strike rule is unchanged.
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

    Returns ``None`` when the pair carries no premium or no gamma at all.
    At the tenor this system now trades that should never happen; it is
    reachable only if a caller asks for a series that has already settled.
    There is nothing to buy or sell there -- neither a gamma scalp nor a
    theta harvest exists in a zero-vega instrument -- so the strategy stands
    aside rather than paying commission for a stub.
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


@dataclass(frozen=True)
class TenorPolicy:
    """Which listed expiry to trade, and when to be out of it.

    All four numbers are in *trading* days (``session.trading_days_between``).
    ``min_days``/``max_days`` bound what may be entered, ``prefer_days``
    picks between the expiries inside that range, and ``close_days`` is the
    DTE at which an open position is closed regardless of P&L.

    ``close_days`` must sit strictly below ``min_days`` or a position would
    be opened already eligible to close.  With the shipped 2/5/(3,4)/1 the
    position is entered three or four sessions out and closed on the first
    bar of the session before expiry, so it is never carried into the last
    two sessions where an at-the-money straddle's gamma and pin risk both
    run away.
    """

    min_days: int = 2
    max_days: int = 5
    prefer_days: tuple[int, int] = (3, 4)
    close_days: int = 1

    def validate(self) -> None:
        if self.min_days < 0:
            raise ValueError("tenor min_days must be >= 0")
        if self.min_days > self.max_days:
            raise ValueError("tenor min_days > max_days")
        low, high = min(self.prefer_days), max(self.prefer_days)
        if not (self.min_days <= low <= high <= self.max_days):
            raise ValueError(
                "tenor prefer_days must lie inside [min_days, max_days]"
            )
        if self.close_days >= self.min_days:
            raise ValueError(
                "tenor close_days must be < min_days, or a position would be "
                "opened already eligible to close"
            )

    def should_close(self, days_to_expiry: int) -> bool:
        """Whether a position at ``days_to_expiry`` has reached the floor."""
        return days_to_expiry <= self.close_days


def select_expiry(clock, moment, policy: TenorPolicy) -> date | None:
    """The expiry ``policy`` says to trade at ``moment``, or ``None``.

    A thin seam over ``SessionClock.select_expiry`` so that callers -- the
    strategy, the live runner's vol read, the ``gex`` and ``doctor``
    commands -- all ask the question in exactly one way.  There is no
    fallback to a nearer or farther series: if nothing is listed inside the
    range the answer is "do not trade", which is the same answer the 0DTE
    version gave on a day with no daily series.
    """
    return clock.select_expiry(
        moment, policy.min_days, policy.max_days, policy.prefer_days
    )


def select_straddle(
    clock,
    moment,
    future: float,
    atm_iv: float,
    source: RiskSource,
    surface: VolSurface,
    policy: TenorPolicy,
    risk_free_rate: float = 0.0,
) -> tuple[date, float, StraddleQuote] | None:
    """Expiry, time-to-expiry and the ATM straddle on it, in one call.

    Returns ``None`` when no expiry is in range or the pair carries nothing
    tradeable.  Bundling the three together is what keeps the backtest and
    the live path on the same series: neither picks an expiry of its own.
    """
    expiry = select_expiry(clock, moment, policy)
    if expiry is None:
        return None
    t = clock.time_to_expiry(moment, expiry)
    quote = select_atm_straddle(
        future, expiry, t, atm_iv, source, surface, risk_free_rate
    )
    if quote is None:
        return None
    return expiry, t, quote
