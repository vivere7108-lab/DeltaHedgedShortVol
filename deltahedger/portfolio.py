"""Position book and delta aggregation.

All exposure is reported in *delta units* (1 unit == 1% of one reference
future -- see ``instruments``), which is the unit the hedge band is written
in.  The same book is used by the backtest and the live runner; the live
runner reconciles it against IBKR positions rather than keeping a separate
representation.

The option book can hold up to two legs at once, one put and one call,
keyed by ``right`` ("P"/"C") in ``Portfolio.legs``.  Selling both turns the
short put into a strangle: the delta target itself is unaffected (the
hedger still holds net portfolio delta -- option legs plus the futures
hedge -- at ``hedge.target`` regardless of how many option legs make it
up), what changes is the option book's shape, collecting theta on both
sides rather than one.
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

    def close_value(self, mark: float, multiplier: float) -> float:
        """Dollar cost to close this leg at ``mark`` right now.

        Positive for a short position -- the mirror of ``credit_received``,
        so a combined stop/target ratio (close value over credit received)
        reduces to the familiar ``mark / entry_price`` for a single leg and
        generalises correctly when a put and a call of different sizes are
        held together.
        """
        return -self.quantity * mark * multiplier


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
    #: Realised P&L split by leg family, so results can attribute the
    #: short-vol premium (both option legs combined) separately from the
    #: cost of hedging it.
    option_realised: float = 0.0
    hedge_realised: float = 0.0
    fees_paid: float = 0.0
    #: Open option legs, keyed by right ("P"/"C"). At most one per right.
    legs: dict[str, OptionPosition] = field(default_factory=dict)
    hedge: HedgePosition = field(default_factory=HedgePosition)

    @property
    def put(self) -> OptionPosition | None:
        return self.legs.get("P")

    @property
    def call(self) -> OptionPosition | None:
        return self.legs.get("C")

    @property
    def has_option(self) -> bool:
        return bool(self.legs)

    # -- delta ---------------------------------------------------------

    def option_delta_units(self, greeks_by_right: dict[str, Greeks]) -> float:
        per_contract = self.source.delta_units_per_contract(self.source.option)
        total = 0.0
        for right, position in self.legs.items():
            greeks = greeks_by_right.get(right)
            if greeks is not None:
                total += position.quantity * greeks.delta * per_contract
        return total

    def hedge_delta_units(self) -> float:
        per_contract = self.source.delta_units_per_contract(self.source.hedge)
        return self.hedge.quantity * per_contract

    def net_delta_units(self, greeks_by_right: dict[str, Greeks]) -> float:
        return self.option_delta_units(greeks_by_right) + self.hedge_delta_units()

    def option_gamma_units(self, greeks_by_right: dict[str, Greeks]) -> float:
        """Delta units gained per 1.00 move in the future, combined legs."""
        per_contract = self.source.delta_units_per_contract(self.source.option)
        total = 0.0
        for right, position in self.legs.items():
            greeks = greeks_by_right.get(right)
            if greeks is not None:
                total += position.quantity * greeks.gamma * per_contract
        return total

    # -- combined premium (for a shared stop / take-profit) -------------

    def combined_credit_received(self) -> float:
        mult = self.source.option.multiplier
        return sum(p.credit_received(mult) for p in self.legs.values())

    def combined_close_value(self, marks_by_right: dict[str, float]) -> float | None:
        """Dollar cost to close every open leg right now.

        ``None`` if any open leg is missing a mark -- a partial close value
        would misstate the position, so callers should skip the check
        rather than act on an incomplete number.
        """
        mult = self.source.option.multiplier
        total = 0.0
        for right, position in self.legs.items():
            mark = marks_by_right.get(right)
            if mark is None:
                return None
            total += position.close_value(mark, mult)
        return total

    # -- valuation -----------------------------------------------------

    def unrealised_pnl(self, marks_by_right: dict[str, float], hedge_mark: float) -> float:
        total = self.hedge.unrealised(hedge_mark, self.source.hedge.multiplier)
        for right, position in self.legs.items():
            mark = marks_by_right.get(right)
            if mark is not None:
                total += position.unrealised(mark, self.source.option.multiplier)
        return total

    def equity(self, marks_by_right: dict[str, float], hedge_mark: float) -> float:
        return (
            self.starting_equity
            + self.realised_pnl
            - self.fees_paid
            + self.unrealised_pnl(marks_by_right, hedge_mark)
        )

    # -- mutation ------------------------------------------------------

    def open_leg(self, position: OptionPosition) -> None:
        if position.right in self.legs:
            raise RuntimeError(f"a {position.right} position is already open")
        self.legs[position.right] = position

    def close_leg(self, right: str, exit_price: float) -> float:
        """Close one leg and realise its P&L. 0.0 if that leg isn't open."""
        position = self.legs.pop(right, None)
        if position is None:
            return 0.0
        pnl = position.unrealised(exit_price, self.source.option.multiplier)
        self.option_realised += pnl
        return pnl

    def close_all_legs(self, exit_prices: dict[str, float]) -> float:
        """Close every open leg at its price in ``exit_prices``.

        Returns the combined realised P&L. A leg missing from
        ``exit_prices`` is left open -- callers that need an all-or-nothing
        close should check ``exit_prices`` covers ``self.legs`` first.
        """
        total = 0.0
        for right in list(self.legs):
            price = exit_prices.get(right)
            if price is not None:
                total += self.close_leg(right, price)
        return total

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
        return not self.legs and self.hedge.quantity == 0
