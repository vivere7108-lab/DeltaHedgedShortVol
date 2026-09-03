"""Synthetic bar generator.

Exists so the whole system -- engine, hedger, sizing, reporting -- can be
exercised end to end without a TWS connection.  It is a test harness, not a
market model: results from synthetic data say the machinery works, never
that the strategy works.

The price path is GBM.  Implied vol is a mean-reverting series with a
negative return correlation, because trading volatility against a flat vol
assumption would hide the exact risk the position carries -- vol moving
while the book is long or short vega.

That realism has a consequence worth stating loudly, because it changes how
a generated result should be read.  Instantaneously the generator draws
returns at the vol it reports as implied, so it has no *gamma* edge.  Over a
holding period it is a different matter: implied vol wanders after entry, so
a straddle -- which is a large vega position, unlike the single
out-of-the-money put this system used to trade -- picks up a systematic
vega P&L that has nothing to do with the strategy.  Set
``synthetic_vol_of_vol``, ``synthetic_vol_mean_reversion`` and
``synthetic_vol_return_beta`` to zero for a genuinely neutral control; that
is what the correctness tests use, and what ``configs/es_zero_edge.yaml``
ships.

Overnight is a step too
------------------------
This only bit once positions started being held past the closing bell.  The
generator only ever produced bars *inside* the session, and the loop simply
carried the last close of one day into the first bar of the next with no
step in between -- as if no time passed overnight at all.  A 0DTE position
never noticed: it was always flat before the gap existed.  A position held
across sessions does notice, and badly: the option pricer's clock counts
the real overnight hours (correctly -- theta owes for the calendar time
that actually elapsed), while the underlying carried zero variance over
that same stretch.  The result was a straddle that decayed every night
against a future that provably could not have moved, which is not
"realised vol below implied" -- it is realised vol of *exactly zero* for a
fraction of the option's life that pricing charges for in full.  A short
straddle collected that decay for nothing; a long one paid it for nothing;
and because a run holds more of one side than the other on any given
sample, the zero-edge control came out biased rather than merely noisy.
``bars()`` now takes one GBM step across the overnight gap -- weekends and
holidays included, at whatever real calendar time
``session.trading_days_between`` implies -- before the next session's first
bar, so the vol the market realises matches the vol the pricer is told is
implied on the same clock the option actually decays on.
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
        vol_of_vol: float | None = None,
        vol_mean_reversion: float | None = None,
        vol_return_beta: float | None = None,
    ):
        self.cfg = cfg
        self.source = source
        self.tz = ZoneInfo(source.timezone)
        self.start = start or date(2025, 1, 2)
        # Config supplies these; the arguments stay for tests that want to
        # vary them without building a whole Config.
        self.vol_of_vol = (
            cfg.synthetic_vol_of_vol if vol_of_vol is None else vol_of_vol
        )
        self.vol_mean_reversion = (
            cfg.synthetic_vol_mean_reversion
            if vol_mean_reversion is None
            else vol_mean_reversion
        )
        self.vol_return_beta = (
            cfg.synthetic_vol_return_beta if vol_return_beta is None else vol_return_beta
        )

    def _step(self, price: float, atm_iv: float, dt: float, rng) -> tuple[float, float, float]:
        """One GBM/vol-of-vol step of ``dt`` years. Shared by the intraday
        loop and the overnight gap so both realise vol on the same rule."""
        long_run_iv = self.cfg.synthetic_annual_vol
        drift = self.cfg.synthetic_annual_drift

        shock = rng.standard_normal()
        ret = (drift - 0.5 * atm_iv**2) * dt + atm_iv * math.sqrt(dt) * shock
        new_price = price * math.exp(ret)

        # Vol mean-reverts and leans against the return.
        iv_shock = rng.standard_normal() * self.vol_of_vol * math.sqrt(dt)
        new_iv = max(
            0.03,
            atm_iv
            + self.vol_mean_reversion * (long_run_iv - atm_iv) * dt * 252
            + iv_shock
            + self.vol_return_beta * ret,
        )
        return new_price, new_iv, ret

    def bars(self) -> Iterator[MarketBar]:
        # Separate streams for the overnight step and the intraday ones,
        # rather than one shared stream consumed sequentially. Sharing one
        # would make the overnight path depend on how many intraday draws
        # ``bar_size`` happens to take per session, so the same seed at "5
        # mins" and "15 mins" would gap through different overnight moves --
        # silently breaking any comparison across bar sizes, which is
        # exactly the comparison the frequency-scaling correctness test
        # makes.
        rng = np.random.default_rng(self.cfg.synthetic_seed)
        overnight_rng = np.random.default_rng([self.cfg.synthetic_seed, 0x0FF5E77])
        step = bar_seconds(self.cfg.bar_size)
        seconds_per_year = 365.0 * 24.0 * 3600.0
        dt = step / seconds_per_year

        open_h, open_m = (int(p) for p in self.source.session_open.split(":")[:2])
        close_h, close_m = (int(p) for p in self.source.session_close.split(":")[:2])
        bars_per_day = int(
            ((close_h * 60 + close_m) - (open_h * 60 + open_m)) * 60 / step
        )

        price = self.cfg.synthetic_start_price
        atm_iv = self.cfg.synthetic_annual_vol

        day = self.start
        if not is_trading_day(day):
            day = next_trading_day(day)

        previous_close: datetime | None = None
        for _ in range(self.cfg.synthetic_days):
            session_start = datetime(
                day.year, day.month, day.day, open_h, open_m, tzinfo=self.tz
            )

            # The gap since the previous session's last bar -- a weekend, a
            # holiday, or an ordinary overnight, whichever it is -- carries
            # exactly as much price and vol variance as its real elapsed
            # time implies. Skipping this (as if no time passed overnight)
            # is what let a straddle held across sessions decay every night
            # against a future that could not have moved: see the module
            # docstring.
            if previous_close is not None:
                overnight_dt = (session_start - previous_close).total_seconds() / seconds_per_year
                if overnight_dt > 0.0:
                    price, atm_iv, _ = self._step(price, atm_iv, overnight_dt, overnight_rng)

            for i in range(bars_per_day):
                new_price, atm_iv, _ = self._step(price, atm_iv, dt, rng)

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
            previous_close = session_start + timedelta(seconds=step * bars_per_day)
            day = next_trading_day(day)
