"""Historical bars from IBKR (TWS or IB Gateway).

Two series are pulled for the same window and merged on timestamp:

  * ``TRADES``                     -- the future's price path
  * ``OPTION_IMPLIED_VOLATILITY``  -- the at-the-money implied vol path

Strike-level prices are *not* requested.  Per-contract option history for
0DTE ES options is sparse, slow to page, and needs market-data permissions
most accounts do not carry; the strategy instead prices strikes off the ATM
vol series through ``VolSurface``.  That is the central approximation in
this backtest and it is stated again here because it is easy to forget when
reading the results.

Requests are chunked (IBKR caps intraday history per request), paced (the
API allows ~60 requests per 10 minutes), and cached to disk, so a re-run of
the same window costs nothing.
"""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from ..config import Config
from ..instruments import RiskSource
from .base import MarketBar

log = logging.getLogger(__name__)

#: Days of intraday history to ask for per request. IBKR rejects larger
#: windows for sub-daily bars and the limit varies by bar size.
CHUNK_DAYS = 20

#: Seconds to wait between historical requests, to stay inside the
#: ~60-requests-per-10-minutes pacing limit.
PACING_SECONDS = 11.0


class IbkrHistorySource:
    """Downloads and caches ES bars + ATM IV from IBKR."""

    def __init__(self, cfg: Config, source: RiskSource):
        self.cfg = cfg
        self.source = source
        self.tz = ZoneInfo(source.timezone)
        self.cache_dir = Path(cfg.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public --------------------------------------------------------

    def bars(self) -> Iterator[MarketBar]:
        frame = self.load()
        for row in frame.itertuples(index=False):
            yield MarketBar(
                timestamp=row.timestamp.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                atm_iv=float(row.atm_iv),
                volume=float(row.volume),
            )

    def load(self) -> pd.DataFrame:
        """Return the merged frame, downloading only what isn't cached."""
        path = self._cache_path()
        if path.exists():
            log.info("using cached history at %s", path)
            frame = pd.read_csv(path, parse_dates=["timestamp"])
            frame["timestamp"] = pd.to_datetime(
                frame["timestamp"], utc=True
            ).dt.tz_convert(self.tz)
            return frame

        frame = self.download()
        frame.to_csv(path, index=False)
        log.info("cached %d bars to %s", len(frame), path)
        return frame

    def download(self) -> pd.DataFrame:
        """Fetch price and IV history from a running TWS / Gateway."""
        from ib_async import ContFuture, IB  # imported lazily: optional dependency

        start, end = self._window()
        ib = IB()
        log.info(
            "connecting to IBKR at %s:%d (clientId=%d)",
            self.cfg.ibkr.host, self.cfg.ibkr.port, self.cfg.ibkr.client_id,
        )
        ib.connect(
            self.cfg.ibkr.host,
            self.cfg.ibkr.port,
            clientId=self.cfg.ibkr.client_id,
            timeout=self.cfg.ibkr.connect_timeout,
            readonly=True,
        )
        try:
            contract = ContFuture(
                symbol=self.source.future.symbol,
                exchange=self.source.future.exchange,
                currency=self.source.future.currency,
            )
            (qualified,) = ib.qualifyContracts(contract)
            log.info("qualified %s", qualified.localSymbol or qualified.symbol)

            prices = self._fetch_series(ib, qualified, "TRADES", start, end)
            if prices.empty:
                raise RuntimeError(
                    "IBKR returned no TRADES bars. Check the date range, the "
                    "bar size, and that the account has CME market data."
                )
            vols = self._fetch_series(
                ib, qualified, "OPTION_IMPLIED_VOLATILITY", start, end
            )
        finally:
            ib.disconnect()

        return self._merge(prices, vols)

    # -- internals -----------------------------------------------------

    def _window(self) -> tuple[datetime, datetime]:
        end = (
            datetime.fromisoformat(self.cfg.end_date).replace(tzinfo=self.tz)
            if self.cfg.end_date
            else datetime.now(self.tz)
        )
        if self.cfg.start_date:
            start = datetime.fromisoformat(self.cfg.start_date).replace(tzinfo=self.tz)
        else:
            start = end - timedelta(days=30)
        if start >= end:
            raise ValueError(f"start_date {start} is not before end_date {end}")
        return start, end

    def _fetch_series(
        self, ib, contract, what_to_show: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Page backwards through the window, one chunk per request."""
        from ib_async import util

        frames: list[pd.DataFrame] = []
        cursor = end
        request_count = 0
        while cursor > start:
            span_days = min(CHUNK_DAYS, max((cursor - start).days, 1))
            if request_count:
                time_module.sleep(PACING_SECONDS)
            log.info(
                "requesting %s %s ending %s",
                span_days, what_to_show, cursor.date(),
            )
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=cursor,
                    durationStr=f"{span_days} D",
                    barSizeSetting=self.cfg.data.bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=2,  # UTC epoch, no ambiguous local strings
                )
            except Exception as exc:  # noqa: BLE001 - surface, then continue
                log.warning("%s request failed: %s", what_to_show, exc)
                bars = None
            request_count += 1

            if not bars:
                if what_to_show == "OPTION_IMPLIED_VOLATILITY":
                    log.warning(
                        "no implied-vol history returned; falling back to "
                        "data.default_atm_iv=%.3f for the affected bars",
                        self.cfg.data.default_atm_iv,
                    )
                    break
                log.warning("empty %s chunk ending %s", what_to_show, cursor.date())
                cursor -= timedelta(days=span_days)
                continue

            chunk = util.df(bars)
            frames.append(chunk)
            earliest = pd.to_datetime(chunk["date"].min(), utc=True)
            new_cursor = earliest.to_pydatetime().astimezone(self.tz)
            if new_cursor >= cursor:  # no progress; stop rather than spin
                break
            cursor = new_cursor

        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["date"], utc=True).dt.tz_convert(
            self.tz
        )
        merged = (
            merged.drop(columns=["date"])
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return merged[(merged["timestamp"] >= start) & (merged["timestamp"] <= end)]

    def _merge(self, prices: pd.DataFrame, vols: pd.DataFrame) -> pd.DataFrame:
        frame = prices[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        if vols.empty:
            frame["atm_iv"] = self.cfg.data.default_atm_iv
            return frame

        iv = vols[["timestamp", "close"]].rename(columns={"close": "atm_iv"})
        # merge_asof carries the last known IV forward into price bars whose
        # IV bar is missing, rather than dropping the price bar.
        frame = pd.merge_asof(
            frame.sort_values("timestamp"),
            iv.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta("1h"),
        )
        missing = int(frame["atm_iv"].isna().sum())
        if missing:
            log.warning(
                "%d of %d bars had no implied vol within 1h; using %.3f",
                missing, len(frame), self.cfg.data.default_atm_iv,
            )
        frame["atm_iv"] = frame["atm_iv"].fillna(self.cfg.data.default_atm_iv)
        frame.loc[frame["atm_iv"] <= 0, "atm_iv"] = self.cfg.data.default_atm_iv
        return frame

    def _cache_path(self) -> Path:
        start, end = self._window()
        slug = self.cfg.data.bar_size.replace(" ", "")
        return self.cache_dir / (
            f"{self.source.name}_{slug}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        )
