"""Execution interface.

The strategy never talks to a broker directly -- it asks an
``ExecutionHandler`` to trade and gets a ``Fill`` back.  The backtest
supplies a simulated handler, the live runner supplies an IBKR-backed one,
and the strategy code above them is byte-for-byte the same in both.  That is
the whole point of the seam: a forward test exercises the logic that was
validated historically, not a reimplementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..chain import OptionQuote


@dataclass(frozen=True)
class Fill:
    """The result of an order. ``quantity`` is signed: negative is a sell."""

    quantity: int
    price: float
    fees: float
    timestamp: datetime
    instrument: str  # "option" | "hedge"
    note: str = ""

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0


class ExecutionHandler(Protocol):
    def execute_option(
        self, quote: OptionQuote, quantity: int, moment: datetime
    ) -> Fill | None: ...

    def execute_hedge(
        self, quantity: int, reference_price: float, moment: datetime
    ) -> Fill | None: ...


class ExecutionError(RuntimeError):
    """An order could not be placed or filled."""
