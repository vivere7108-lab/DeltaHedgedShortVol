"""Simulated execution for backtesting.

Fills happen at the model price crossed by a fixed number of ticks, plus
per-contract fees.  Slippage is charged in the direction that hurts -- sells
fill below the mark, buys above -- so a strategy that trades more pays more,
which is the property that matters when tuning a hedge band.

This is an optimistic model in two ways worth naming.  It assumes the order
fills at all, in full, at that price -- and a 0DTE option during a fast move
is exactly when that assumption is worst.  It also fills both legs of a
straddle unconditionally, so the backtest never exercises the half-filled
case the live path is written to survive (see
``GexStraddleStrategy._open_legs``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..chain import OptionQuote
from ..config import CostsConfig
from ..instruments import RiskSource
from .base import Fill


@dataclass
class SimulatedExecution:
    costs: CostsConfig
    source: RiskSource

    def execute_option(
        self, quote: OptionQuote, quantity: int, moment: datetime
    ) -> Fill | None:
        if quantity == 0:
            return None
        spec = self.source.option
        price = quote.price
        fees = 0.0
        if self.costs.enabled:
            slip = self.costs.option_slippage_ticks * spec.tick_size
            price += slip if quantity > 0 else -slip
            price = max(price, 0.0)
            fees = abs(quantity) * self.costs.option_fees_per_contract
        return Fill(
            quantity=quantity,
            price=price,
            fees=fees,
            timestamp=moment,
            instrument="option",
        )

    def execute_hedge(
        self, quantity: int, reference_price: float, moment: datetime
    ) -> Fill | None:
        if quantity == 0:
            return None
        spec = self.source.hedge
        price = reference_price
        fees = 0.0
        if self.costs.enabled:
            slip = self.costs.hedge_slippage_ticks * spec.tick_size
            price += slip if quantity > 0 else -slip
            fees = abs(quantity) * self.costs.hedge_fees_per_contract
        return Fill(
            quantity=quantity,
            price=price,
            fees=fees,
            timestamp=moment,
            instrument="hedge",
        )
