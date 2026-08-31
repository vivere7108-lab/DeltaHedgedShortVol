"""Position book and delta aggregation.

All exposure is reported in *delta units* (1 unit == 1% of one reference
future -- see ``instruments``), which is the unit the hedge band is written
in.  The same book is used by the backtest and the live runner; the live
runner reconciles it against IBKR positions rather than keeping a separate
representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from .instruments import RiskSource
from .pricing import Greeks


@dataclass
class OptionPosition:
    """A short (or long) option leg. ``quantity`` is signed: short is < 0."""

    strike: float
    expiry: date
    right: str
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_iv: float = 0.0
    entry_delta: float = 0.0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def unrealised(self, mark: float, multiplier: float) -> float:
        return self.quantity * (mark - self.entry_price) * multiplier

    def credit_received(self, multiplier: float) -> float:
        """Positive for a short position: the premium taken in."""
        return -self.quantity * self.entry_price * multiplier


@dataclass
class HedgePosition:
    """The futures hedge. ``quantity`` is signed: short is < 0."""

    quantity: int = 0
    avg_price: float = 0.0

    def unrealised(self, mark: float, multiplier: float) -> float:
        return self.quantity * (mark - self.avg_price) * multiplier

    def apply_fill(self, filled_qty: int, fill_price: float) -> float:
        """Book a fill; returns realised P&L from any quantity closed out.

        Handles the sign flip correctly: closing through zero realises P&L
        on the closed portion and re-bases the average price on the residual.
        """
        realised_points = 0.0
        old_qty, old_avg = self.quantity, self.avg_price
        new_qty = old_qty + filled_qty

        if old_qty == 0 or (old_qty > 0) == (filled_qty > 0):
            # Opening or adding: weighted-average the entry price.
            if new_qty != 0:
                self.avg_price = (old_qty * old_avg + filled_qty * fill_price) / new_qty
        else:
            closed = min(abs(filled_qty), abs(old_qty))
            direction = 1 if old_qty > 0 else -1
            realised_points = direction * closed * (fill_price - old_avg)
            if abs(filled_qty) > abs(old_qty):
                self.avg_price = fill_price  # flipped through zero
            elif new_qty == 0:
                self.avg_price = 0.0
        self.quantity = new_qty
        return realised_points


@dataclass
class Portfolio:
    """Cash, positions and the aggregated delta the hedger acts on."""

    starting_equity: float
    source: RiskSource
    #: Realised P&L split by leg, so results can attribute the short-vol
    #: premium separately from the cost of hedging it.
    option_realised: float = 0.0
    hedge_realised: float = 0.0
    fees_paid: float = 0.0
    option: OptionPosition | None = None
    hedge: HedgePosition = field(default_factory=HedgePosition)

    # -- delta ---------------------------------------------------------

    def option_delta_units(self, greeks: Greeks | None) -> float:
        if self.option is None or greeks is None:
            return 0.0
        per_contract = self.source.delta_units_per_contract(self.source.option)
        return self.option.quantity * greeks.delta * per_contract

    def hedge_delta_units(self) -> float:
        per_contract = self.source.delta_units_per_contract(self.source.hedge)
        return self.hedge.quantity * per_contract

    def net_delta_units(self, greeks: Greeks | None) -> float:
        return self.option_delta_units(greeks) + self.hedge_delta_units()

    def option_gamma_units(self, greeks: Greeks | None) -> float:
        """Delta units gained per 1.00 move in the future."""
        if self.option is None or greeks is None:
            return 0.0
        per_contract = self.source.delta_units_per_contract(self.source.option)
        return self.option.quantity * greeks.gamma * per_contract

    # -- valuation -----------------------------------------------------

    def unrealised_pnl(self, option_mark: float | None, hedge_mark: float) -> float:
        total = self.hedge.unrealised(hedge_mark, self.source.hedge.multiplier)
        if self.option is not None and option_mark is not None:
            total += self.option.unrealised(option_mark, self.source.option.multiplier)
        return total

    def equity(self, option_mark: float | None, hedge_mark: float) -> float:
        return (
            self.starting_equity
            + self.realised_pnl
            - self.fees_paid
            + self.unrealised_pnl(option_mark, hedge_mark)
        )

    # -- mutation ------------------------------------------------------

    def open_option(self, position: OptionPosition) -> None:
        if self.option is not None:
            raise RuntimeError("an option position is already open")
        self.option = position

    def close_option(self, exit_price: float) -> float:
        """Close the option leg and realise its P&L."""
        if self.option is None:
            return 0.0
        pnl = self.option.unrealised(exit_price, self.source.option.multiplier)
        self.option_realised += pnl
        self.option = None
        return pnl

    def apply_hedge_fill(self, filled_qty: int, fill_price: float) -> float:
        points = self.hedge.apply_fill(filled_qty, fill_price)
        pnl = points * self.source.hedge.multiplier
        self.hedge_realised += pnl
        return pnl

    def charge_fees(self, amount: float) -> None:
        self.fees_paid += amount

    @property
    def realised_pnl(self) -> float:
        return self.option_realised + self.hedge_realised

    @property
    def is_flat(self) -> bool:
        return self.option is None and self.hedge.quantity == 0
