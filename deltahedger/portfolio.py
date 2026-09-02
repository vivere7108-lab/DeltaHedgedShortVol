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

from .chain import StraddleQuote
from .instruments import RiskSource


@dataclass
class StraddlePosition:
    """An ATM straddle. ``quantity`` is signed: long is > 0, short is < 0.

    The sign is the whole strategy.  A positive quantity is the negative-GEX
    trade -- long gamma, scalped by the hedger; a negative quantity is the
    positive-GEX trade -- short gamma, collecting theta.  Everything
    downstream reads the sign rather than carrying a separate mode flag, so
    there is exactly one place the direction is decided (the GEX regime) and
    no way for two parts of the system to disagree about which way round the
    book is.
    """

    strike: float
    expiry: date
    quantity: int
    call_entry: float
    put_entry: float
    entry_time: datetime
    entry_future: float = 0.0
    entry_iv: float = 0.0
    entry_delta: float = 0.0
    #: The GEX regime that opened it, carried for reporting and for the
    #: regime-flip exit.
    regime: str = ""

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def direction(self) -> int:
        return 1 if self.quantity > 0 else -1 if self.quantity < 0 else 0

    @property
    def entry_premium(self) -> float:
        """Premium of one straddle at entry, in points."""
        return self.call_entry + self.put_entry

    def unrealised(self, call_mark: float, put_mark: float, multiplier: float) -> float:
        return self.quantity * (
            (call_mark + put_mark) - self.entry_premium
        ) * multiplier

    def debit_paid(self, multiplier: float) -> float:
        """Cash out the door at entry. Negative for a short (a credit in)."""
        return self.quantity * self.entry_premium * multiplier

    def premium_at_risk(self, multiplier: float) -> float:
        """Size of the entry premium, unsigned -- the yardstick both exit
        rules are written against."""
        return abs(self.quantity) * self.entry_premium * multiplier


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
    #: Realised P&L split by leg, so results can attribute what the straddle
    #: did separately from what hedging it did. For a long straddle those
    #: two numbers *are* the strategy: the option leg is the premium bled to
    #: theta, the hedge leg is the gamma scalped back.
    option_realised: float = 0.0
    hedge_realised: float = 0.0
    fees_paid: float = 0.0
    straddle: StraddlePosition | None = None
    hedge: HedgePosition = field(default_factory=HedgePosition)

    # -- delta ---------------------------------------------------------

    def option_delta_units(self, quote: StraddleQuote | None) -> float:
        if self.straddle is None or quote is None:
            return 0.0
        per_contract = self.source.delta_units_per_contract(self.source.option)
        return self.straddle.quantity * quote.delta * per_contract

    def hedge_delta_units(self) -> float:
        per_contract = self.source.delta_units_per_contract(self.source.hedge)
        return self.hedge.quantity * per_contract

    def net_delta_units(self, quote: StraddleQuote | None) -> float:
        return self.option_delta_units(quote) + self.hedge_delta_units()

    def option_gamma_units(self, quote: StraddleQuote | None) -> float:
        """Delta units gained per 1.00 move in the future.

        Positive when we are long the straddle -- the book gets longer as the
        market rises, which is the exposure the hedger converts into scalps.
        """
        if self.straddle is None or quote is None:
            return 0.0
        per_contract = self.source.delta_units_per_contract(self.source.option)
        return self.straddle.quantity * quote.gamma * per_contract

    def option_vega(self, quote: StraddleQuote | None) -> float:
        """Dollars per volatility point."""
        if self.straddle is None or quote is None:
            return 0.0
        return self.straddle.quantity * quote.vega * self.source.option.multiplier

    def option_theta(self, quote: StraddleQuote | None) -> float:
        """Dollars per calendar day. Negative when long the straddle."""
        if self.straddle is None or quote is None:
            return 0.0
        return self.straddle.quantity * quote.theta * self.source.option.multiplier

    # -- valuation -----------------------------------------------------

    def straddle_unrealised(self, quote: StraddleQuote | None) -> float:
        if self.straddle is None or quote is None:
            return 0.0
        return self.straddle.unrealised(
            quote.call.price, quote.put.price, self.source.option.multiplier
        )

    def unrealised_pnl(self, quote: StraddleQuote | None, hedge_mark: float) -> float:
        return self.straddle_unrealised(quote) + self.hedge.unrealised(
            hedge_mark, self.source.hedge.multiplier
        )

    def equity(self, quote: StraddleQuote | None, hedge_mark: float) -> float:
        return (
            self.starting_equity
            + self.realised_pnl
            - self.fees_paid
            + self.unrealised_pnl(quote, hedge_mark)
        )

    # -- mutation ------------------------------------------------------

    def open_straddle(self, position: StraddlePosition) -> None:
        if self.straddle is not None:
            raise RuntimeError("a straddle position is already open")
        self.straddle = position

    def close_straddle(self, call_price: float, put_price: float) -> float:
        """Close both legs and realise the straddle's P&L."""
        if self.straddle is None:
            return 0.0
        pnl = self.straddle.unrealised(
            call_price, put_price, self.source.option.multiplier
        )
        self.option_realised += pnl
        self.straddle = None
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
        return self.straddle is None and self.hedge.quantity == 0
