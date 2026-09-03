"""The delta band.

Given the portfolio's current net delta, decide whether to trade the hedge
instrument and by how much.  Kept free of any market or broker dependency so
the band logic is directly testable and identical in backtest and live.

Granularity
-----------
The band is ``0 +/- 10`` delta units by default -- hold the straddle
delta-neutral -- and one MES contract moves net delta by 10 units.  The band
is therefore exactly one contract wide, which is the narrowest it can be and
still bind: the rule below only trades when a whole contract lands closer to
target, so any half-width under 5 fires on precisely the same bars as 5 and
is an inert parameter.  The rule this module implements is:

  * do nothing while net delta is inside the band;
  * once it leaves, trade the whole-contract quantity that lands *closest*
    to the target and stop.

That leaves a residual of up to half a contract (5 units) -- so the decision
is additionally gated on *strict improvement*: a hedge is only issued if it
moves net delta closer to the target than it already is.  Without that gate,
a breach that no whole-contract trade can fix would re-fire on every bar and
churn commissions.

Which of the two bounds actually binds depends on the width.  With the
shipped ``+/-10`` the band is wider than half a contract, so the band is what
caps the residual; with a half-width under 5 the contract size caps it
instead.  Both bounds are asserted in the tests.

The band is target-agnostic and symmetric, which matters here more than it
looks: the same width is applied whether the book is long gamma or short it,
so a comparison between the two regimes measures the signal rather than the
hedger.

Outside the session
-------------------
The position is held for several sessions now, so for most of its life the
regular session is closed.  ``decide`` therefore takes ``in_session`` and
widens the band by ``overnight_band_multiplier`` when it is false: outside
RTH only a *larger* breach is hedged.

The reason is not that overnight delta matters less -- it matters more.  It
is that overnight the two costs of hedging both go up while the information
in a breach goes down: the book is quoted wider, so each MES fill gives up
more; and a delta picked up on thin overnight volume is as likely to be
handed back by the open as to be realised.  Hedging is not switched off,
because a gap through the band is exactly the event an unhedged straddle
cannot survive.  Setting the multiplier to 1.0 hedges identically around the
clock, which is the control to compare against.

Only the *width* is session-dependent.  The target, the granularity rule and
the strict-improvement rule are the same at every hour, so the residual
bound stays whichever of half a contract or the effective band is wider.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import HedgeConfig
from .instruments import RiskSource


@dataclass(frozen=True)
class HedgeDecision:
    """What the band says to do right now."""

    should_hedge: bool
    contracts: int  # signed: positive buys the hedge instrument
    net_delta_before: float
    net_delta_after: float
    target: float
    reason: str

    @property
    def residual_delta(self) -> float:
        """Distance from target left after the (possible) hedge."""
        return self.net_delta_after - self.target


class DeltaHedger:
    """Turns net delta into hedge orders under a target band."""

    def __init__(self, cfg: HedgeConfig, source: RiskSource):
        self.cfg = cfg
        self.source = source
        self.quantum = source.hedge_quantum
        if self.quantum <= 0:
            raise ValueError("hedge instrument carries no delta")

    def decide(
        self,
        net_delta: float,
        seconds_since_last_hedge: float | None = None,
        in_session: bool = True,
    ) -> HedgeDecision:
        cfg = self.cfg
        low, high = cfg.bounds(in_session)
        where = "" if in_session else " (overnight band)"
        no_trade = lambda reason: HedgeDecision(  # noqa: E731
            False, 0, net_delta, net_delta, cfg.target, reason
        )

        if cfg.in_band(net_delta, in_session):
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
        )

    def flatten(self, net_delta: float) -> int:
        """Contracts needed to take the hedge leg's delta to zero."""
        ideal = -net_delta / self.quantum
        return int(ideal + (0.5 if ideal >= 0 else -0.5))
