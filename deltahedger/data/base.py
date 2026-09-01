"""Market data interface shared by every source.

The engine consumes an iterator of ``MarketBar``.  A source's only job is to
produce those in ascending time order with a timezone-aware timestamp and an
at-the-money implied vol, so the backtest does not care whether the bars
came from IBKR, a CSV, or a generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Protocol


@dataclass(frozen=True)
class MarketBar:
    """One bar of the underlying future, plus the ATM implied vol at its close."""

    timestamp: datetime  # timezone-aware, exchange local
    open: float
    high: float
    low: float
    close: float
    atm_iv: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("MarketBar.timestamp must be timezone-aware")


class DataSource(Protocol):
    """Anything that can produce an ordered stream of bars."""

    def bars(self) -> Iterable[MarketBar]: ...


def ensure_sorted(bars: Iterable[MarketBar]) -> Iterator[MarketBar]:
    """Pass bars through, rejecting out-of-order timestamps.

    Silently accepting unsorted bars produces a backtest that quietly
    time-travels, which is far worse than a loud failure.
    """
    previous: datetime | None = None
    for bar in bars:
        if previous is not None and bar.timestamp < previous:
            raise ValueError(
                f"bars are not in ascending time order: {bar.timestamp} follows {previous}"
            )
        previous = bar.timestamp
        yield bar
