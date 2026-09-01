"""Synthetic bar generator.

Exists so the whole system -- engine, hedger, sizing, reporting -- can be
exercised end to end without a TWS connection.  It is a test harness, not a
market model: results from synthetic data say the machinery works, never
that the strategy works.

The price path is GBM.  Implied vol is a mean-reverting series with a
negative return correlation, because short volatility with a flat vol
assumption would hide the exact risk the strategy carries -- vol rising as
the market falls.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Iterator
from zoneinfo import ZoneInfo

import numpy as np

from ..config import DataConfig
from ..instruments import RiskSource
from ..session import is_trading_day, next_trading_day
from .base import MarketBar

#: Bar sizes we know how to step, in seconds.
BAR_SECONDS = {
    "1 min": 60, "2 mins": 120, "3 mins": 180, "5 mins": 300,
    "10 mins": 600, "15 mins": 900, "30 mins": 1800, "1 hour": 3600,
}


def bar_seconds(bar_size: str) -> int:
    try:
        return BAR_SECONDS[bar_size]
    except KeyError:
        known = ", ".join(sorted(BAR_SECONDS, key=BAR_SECONDS.get))
        raise ValueError(f"unsupported bar size {bar_size!r}; known: {known}") from None


class SyntheticSource:
    def __init__(
        self,
        cfg: DataConfig,
        source: RiskSource,
        start: date | None = None,
        vol_of_vol: float = 2.0,
        vol_mean_reversion: float = 0.08,
        vol_return_beta: float = -8.0,
    ):
        self.cfg = cfg
        self.source = source
        self.tz = ZoneInfo(source.timezone)
        self.start = start or date(2025, 1, 2)
        self.vol_of_vol = vol_of_vol
        self.vol_mean_reversion = vol_mean_reversion
        self.vol_return_beta = vol_return_beta

    def bars(self) -> Iterator[MarketBar]:
        rng = np.random.default_rng(self.cfg.synthetic_seed)
        step = bar_seconds(self.cfg.bar_size)
        dt = step / (365.0 * 24.0 * 3600.0)

        open_h, open_m = (int(p) for p in self.source.session_open.split(":")[:2])
        close_h, close_m = (int(p) for p in self.source.session_close.split(":")[:2])
        bars_per_day = int(
            ((close_h * 60 + close_m) - (open_h * 60 + open_m)) * 60 / step
        )

        price = self.cfg.synthetic_start_price
        atm_iv = self.cfg.synthetic_annual_vol
        long_run_iv = self.cfg.synthetic_annual_vol
        drift = self.cfg.synthetic_annual_drift

        day = self.start
        if not is_trading_day(day):
            day = next_trading_day(day)

        for _ in range(self.cfg.synthetic_days):
            session_start = datetime(
                day.year, day.month, day.day, open_h, open_m, tzinfo=self.tz
            )
            for i in range(bars_per_day):
                # Realised move uses the current implied level as its vol.
                shock = rng.standard_normal()
                ret = (drift - 0.5 * atm_iv**2) * dt + atm_iv * math.sqrt(dt) * shock
                new_price = price * math.exp(ret)

                # Vol mean-reverts and leans against the return.
                iv_shock = rng.standard_normal() * self.vol_of_vol * math.sqrt(dt)
                atm_iv = max(
                    0.03,
                    atm_iv
                    + self.vol_mean_reversion * (long_run_iv - atm_iv) * dt * 252
                    + iv_shock
                    + self.vol_return_beta * ret,
                )

                low, high = sorted((price, new_price))
                wiggle = price * atm_iv * math.sqrt(dt) * 0.5
                yield MarketBar(
                    timestamp=session_start + timedelta(seconds=step * (i + 1)),
                    open=price,
                    high=high + abs(rng.standard_normal()) * wiggle,
                    low=max(low - abs(rng.standard_normal()) * wiggle, 0.01),
                    close=new_price,
                    atm_iv=atm_iv,
                    volume=float(rng.integers(500, 5000)),
                )
                price = new_price
            day = next_trading_day(day)
