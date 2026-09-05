"""The delta band.

Given the portfolio's current net delta -- and, now, its gamma and tenor --
decide whether to trade the hedge instrument and by how much.  Kept free of
any market or broker dependency so the band logic is directly testable and
identical in backtest and live.

Where the band comes from
-------------------------
The half-width is a function of the book rather than a number, by default:
the Whalley-Wilmott (1997) asymptotic no-transaction band for a hedger with
exponential utility facing proportional transaction costs::

    H = ( 3/2 * exp(-r*T) * k*S * Gamma^2 / gamma_ra ) ** (1/3)

``k*S`` is the cost of trading one unit of the underlying, ``Gamma`` is the
position's gamma in those units per dollar of the unit's price, and
``gamma_ra`` is the hedger's absolute risk aversion per dollar.  The result
is in units of the underlying either side of the Black-Scholes delta.  It
says three things a fixed band cannot:

* a book with more gamma is allowed to drift further *in delta* before it
  is touched (``H ~ Gamma^(2/3)``) -- but since delta moves faster on such
  a book, the band covers *fewer points* of underlying (``~ Gamma^(-1/3)``);
* cheaper hedging means hedging more often (``H ~ k^(1/3)``);
* a more risk-averse hedger hedges more often (``H ~ gamma_ra^(-1/3)``).

Everything is worked in **delta units** (1 unit == 1% of one ES contract,
see ``instruments``), and the conversion is the only subtle part::

    unit   = 1% of the reference future
    m_u    = dollars of P&L one unit earns per 1.00 point     ($0.50 on ES)
    c_u    = dollar cost of trading one unit, one way
           = cost per hedge contract / units per hedge contract
    G      = |position gamma| in units per point  (Portfolio.option_gamma_units)
    Gamma  = G / m_u                               (units per dollar)

    H_units = ( 3/2 * exp(-r*T) * c_u * Gamma^2 / gamma_ra ) ** (1/3)

``c_u`` plays the ``k*S`` role: Whalley-Wilmott's cost is proportional to
value traded, and the cost of one unit at the current price is exactly
that.  The formula is invariant to what one calls a "unit" -- redefining
the share as ``n`` units divides ``H`` by ``n`` -- so the band means the
same thing in MES contracts, ES contracts or delta units.

The ``fixed`` model is the old heuristic (``band`` delta units either side
of the target) and is kept as the control, and for ``deltahedger sweep
--bands``.

Granularity
-----------
One MES contract moves net delta by 10 units, and the rule only trades when
a whole contract lands *closer* to target -- so any half-width under 5 fires
on precisely the same bars as 5 and is inert.  A Whalley-Wilmott band that
comes out narrower than that (a small book, or a straddle far from its
strike) simply behaves as +/-5.  The rule is:

  * do nothing while net delta is inside the band;
  * once it leaves, trade the whole-contract quantity that lands *closest*
    to the target and stop.

That leaves a residual of up to half a contract (5 units) -- so the decision
is additionally gated on *strict improvement*: a hedge is only issued if it
moves net delta closer to the target than it already is.  Without that gate,
a breach that no whole-contract trade can fix would re-fire on every bar and
churn commissions.

Which of the two bounds actually binds depends on the width: a band wider
than half a contract is what caps the residual; a narrower one lets the
contract size cap it instead.  Both bounds are asserted in the tests.

The band is symmetric about the target and even in gamma (``Gamma^2``), so
the same rule is applied whether the book is long gamma or short it: a
comparison between the two regimes measures the signal rather than the
hedger.  What differs between them is only how much gamma each carries.

Outside the session
-------------------
``decide`` takes ``in_session`` and widens the band by
``overnight_band_multiplier`` when it is false: outside RTH only a *larger*
breach is hedged.  Under Whalley-Wilmott a wider overnight quote is a
larger ``k``, and a band ``m`` times wider stands for costs ``m^3`` times
higher -- the multiplier is a stand-in for that, not a second model.  The
hedge is never switched off, because a gap through the band is exactly the
event an unhedged straddle cannot survive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import BAND_FIXED, BAND_WHALLEY_WILMOTT, CostsConfig, HedgeConfig
from .instruments import RiskSource


def hedge_cost_per_contract(costs: CostsConfig, source: RiskSource) -> float:
    """Dollar cost of trading one hedge contract, one way: slippage plus fees.

    Read off the costs section whether or not ``costs.enabled`` is set --
    the band is a decision rule the live system runs, and a backtest that
    switches costs off to isolate the strategy's arithmetic should still
    hedge the way the live system will.
    """
    return (
        costs.hedge_slippage_ticks * source.hedge.tick_value
        + costs.hedge_fees_per_contract
    )


def whalley_wilmott_half_width(
    gamma_units_per_point: float,
    cost_per_unit: float,
    dollars_per_point_per_unit: float,
    risk_aversion: float,
    time_to_expiry: float = 0.0,
    risk_free_rate: float = 0.0,
) -> float:
    """The Whalley-Wilmott half-width, in delta units.

    ``gamma_units_per_point`` is the position's gamma as delta units gained
    per 1.00 move in the underlying (sign ignored); ``cost_per_unit`` the
    dollar cost of trading one delta unit one way; ``dollars_per_point_per_unit``
    what one unit earns per point.  See the module docstring for the
    derivation.  Zero gamma or zero cost gives a zero band, which is the
    continuous-hedging limit the formula is an expansion around.
    """
    if risk_aversion <= 0.0:
        raise ValueError("risk aversion must be > 0")
    gamma_per_dollar = abs(gamma_units_per_point) / dollars_per_point_per_unit
    if gamma_per_dollar <= 0.0 or cost_per_unit <= 0.0:
        return 0.0
    discount = math.exp(-risk_free_rate * max(time_to_expiry, 0.0))
    return (
        1.5 * discount * cost_per_unit * gamma_per_dollar ** 2 / risk_aversion
    ) ** (1.0 / 3.0)


@dataclass(frozen=True)
class HedgeDecision:
    """What the band says to do right now."""

    should_hedge: bool
    contracts: int  # signed: positive buys the hedge instrument
    net_delta_before: float
    net_delta_after: float
    target: float
    reason: str
    #: The half-width that applied, in delta units.
    band: float = 0.0

    @property
    def residual_delta(self) -> float:
        """Distance from target left after the (possible) hedge."""
        return self.net_delta_after - self.target


class DeltaHedger:
    """Turns net delta into hedge orders under a target band."""

    def __init__(
        self,
        cfg: HedgeConfig,
        source: RiskSource,
        cost_per_contract: float | None = None,
        risk_free_rate: float = 0.0,
    ):
        self.cfg = cfg
        self.source = source
        self.quantum = source.hedge_quantum
        if self.quantum <= 0:
            raise ValueError("hedge instrument carries no delta")
        #: Dollar cost of one hedge contract, one way, for the band.
        self.cost_per_contract = (
            cfg.hedge_cost_per_contract
            if cfg.hedge_cost_per_contract is not None
            else (cost_per_contract if cost_per_contract is not None else 0.0)
        )
        self.risk_free_rate = risk_free_rate

    # -- the band ----------------------------------------------------------

    @property
    def cost_per_unit(self) -> float:
        """Dollar cost of trading one delta unit, one way."""
        return self.cost_per_contract / self.quantum

    def half_width(
        self,
        gamma_units: float = 0.0,
        time_to_expiry: float = 0.0,
        in_session: bool = True,
    ) -> float:
        """The half-width that applies to a book with this gamma, right now."""
        if self.cfg.band_model == BAND_FIXED:
            half = self.cfg.band
        elif self.cfg.band_model == BAND_WHALLEY_WILMOTT:
            half = whalley_wilmott_half_width(
                gamma_units,
                self.cost_per_unit,
                self.source.dollars_per_point_per_delta_unit,
                self.cfg.risk_aversion,
                time_to_expiry,
                self.risk_free_rate,
            )
        else:  # pragma: no cover - validated in config
            raise ValueError(f"unknown band model {self.cfg.band_model!r}")
        if not in_session:
            half *= self.cfg.overnight_band_multiplier
        return half

    def bounds(
        self, gamma_units: float = 0.0, time_to_expiry: float = 0.0,
        in_session: bool = True,
    ) -> tuple[float, float]:
        half = self.half_width(gamma_units, time_to_expiry, in_session)
        return self.cfg.target - half, self.cfg.target + half

    def in_band(self, net_delta: float, half_width: float) -> bool:
        return abs(net_delta - self.cfg.target) <= half_width

    # -- the decision ------------------------------------------------------

    def decide(
        self,
        net_delta: float,
        seconds_since_last_hedge: float | None = None,
        in_session: bool = True,
        gamma_units: float = 0.0,
        time_to_expiry: float = 0.0,
    ) -> HedgeDecision:
        cfg = self.cfg
        half = self.half_width(gamma_units, time_to_expiry, in_session)
        low, high = cfg.target - half, cfg.target + half
        where = "" if in_session else " (overnight band)"
        no_trade = lambda reason: HedgeDecision(  # noqa: E731
            False, 0, net_delta, net_delta, cfg.target, reason, half
        )

        if self.in_band(net_delta, half):
            return no_trade(
                f"net delta {net_delta:.1f} inside [{low:.1f}, {high:.1f}]{where}"
            )

        if (
            seconds_since_last_hedge is not None
            and seconds_since_last_hedge < cfg.min_seconds_between_hedges
        ):
            return no_trade(
                f"hedged {seconds_since_last_hedge:.0f}s ago, cooldown is "
                f"{cfg.min_seconds_between_hedges:.0f}s"
            )

        # Whole-contract quantity landing closest to the target. Rounded away
        # from zero at the half so the choice never depends on parity.
        ideal = (cfg.target - net_delta) / self.quantum
        contracts = int(ideal + (0.5 if ideal >= 0 else -0.5))

        if contracts == 0:
            return no_trade(
                f"net delta {net_delta:.1f} is outside the band but no whole "
                f"{self.source.hedge.symbol} contract ({self.quantum:.0f} delta "
                f"units) lands closer to {cfg.target:.1f}"
            )

        if abs(contracts) < cfg.min_hedge_contracts:
            return no_trade(
                f"hedge of {abs(contracts)} contracts is below "
                f"min_hedge_contracts={cfg.min_hedge_contracts}"
            )

        if abs(contracts) > cfg.max_hedge_contracts:
            contracts = cfg.max_hedge_contracts * (1 if contracts > 0 else -1)

        net_after = net_delta + contracts * self.quantum
        if abs(net_after - cfg.target) >= abs(net_delta - cfg.target):
            return no_trade(
                f"hedging {contracts:+d} would not move net delta "
                f"{net_delta:.1f} closer to {cfg.target:.1f}"
            )

        side = "buy" if contracts > 0 else "sell"
        return HedgeDecision(
            should_hedge=True,
            contracts=contracts,
            net_delta_before=net_delta,
            net_delta_after=net_after,
            target=cfg.target,
            reason=(
                f"net delta {net_delta:.1f} outside [{low:.1f}, {high:.1f}]{where}"
                f"; {side} {abs(contracts)} {self.source.hedge.symbol} -> "
                f"{net_after:.1f}"
            ),
            band=half,
        )

    def flatten(self, net_delta: float) -> int:
        """Contracts needed to take the hedge leg's delta to zero."""
        ideal = -net_delta / self.quantum
        return int(ideal + (0.5 if ideal >= 0 else -0.5))
