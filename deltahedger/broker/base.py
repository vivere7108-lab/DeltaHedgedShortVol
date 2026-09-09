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

#: Hard ceiling on any single order, regardless of config. A backstop
#: against a sizing bug turning into a position nobody intended -- but it
#: is applied as ``min(MAX_ORDER_CONTRACTS, <config limit>)``, so it only
#: binds when it is the smaller of the two. ``SizingConfig.validate`` says
#: so when it is not.
MAX_ORDER_CONTRACTS = 500


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
    """An order could not be placed or filled.

    Raised for refusals that happen *before* anything reaches the exchange
    -- a size check, an unqualifiable contract -- so nothing has traded and
    the book is still correct.
    """


class OrderStateUnknown(ExecutionError):
    """An order was sent and its outcome could not be established.

    Deliberately distinct from ``ExecutionError``: "we refused to send it"
    and "we sent it and do not know what happened" call for opposite
    responses.  The first leaves the book correct and can simply be
    retried; the second means the account may now hold a position this
    process has no record of, and retrying is how one such position
    becomes several.  A caller that treats this as "did not fill" is making
    exactly the assumption that is not available to it.
    """
