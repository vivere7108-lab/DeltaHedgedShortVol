"""The delta band.

Given the portfolio's current net delta, decide whether to trade the hedge
instrument and by how much.  Kept free of any market or broker dependency so
the band logic is directly testable and identical in backtest and live.

Granularity
-----------
The band is ``20 +/- 3`` delta units by default, but one MES contract moves
net delta by 10 units.  A 6-unit-wide band is therefore narrower than the
smallest trade available, and no sequence of hedges can guarantee landing
inside it.  The rule this module implements is:

  * do nothing while net delta is inside the band;
  * once it leaves, trade the whole-contract quantity that lands *closest*
    to the target and stop.

That leaves a residual of up to half a contract (5 units), which may still
sit outside the band -- so the decision is additionally gated on *strict
improvement*: a hedge is only issued if it moves net delta closer to the
target than it already is.  Without that gate, a breach that no whole-
contract trade can fix would re-fire on every bar and churn commissions.
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
    ) -> HedgeDecision:
        cfg = self.cfg
        no_trade = lambda reason: HedgeDecision(  # noqa: E731
            False, 0, net_delta, net_delta, cfg.target, reason
        )

        if cfg.in_band(net_delta):
            return no_trade(f"net delta {net_delta:.1f} inside [{cfg.lower:.1f}, {cfg.upper:.1f}]")

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
                f"net delta {net_delta:.1f} outside [{cfg.lower:.1f}, {cfg.upper:.1f}]"
                f"; {side} {abs(contracts)} {self.source.hedge.symbol} -> "
                f"{net_after:.1f}"
            ),
        )

    def flatten(self, net_delta: float) -> int:
        """Contracts needed to take the hedge leg's delta to zero."""
        ideal = -net_delta / self.quantum
        return int(ideal + (0.5 if ideal >= 0 else -0.5))
