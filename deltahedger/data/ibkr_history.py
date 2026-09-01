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

Front-month stitching
---------------------
IBKR refuses ``endDateTime`` on a continuous future ("Error 10339: Setting
end date/time for continuous future security type is not allowed"), so a
``ContFuture`` can only ever hand back the most recent bars -- useless for
paging through a historical window.  History is therefore assembled from the
*concrete* quarterly contracts that were front month at each point in time,
each queried over the window during which it led, then concatenated.

The stitch is deliberately **not** back-adjusted.  Those were the prices that
actually traded, and the strategy selects strikes off the price level, so
shifting the series to remove roll gaps would put the backtest on strikes
that never existed.  The strategy is flat overnight, so a gap across a roll
costs nothing -- no position spans it.

IBKR only resolves expired futures back a limited distance -- in practice
roughly a year, and it varies by account.  ``reqContractDetails`` and
``qualifyContracts`` simply omit anything older; there is no error, just an
absence.  When the earliest contract this account can resolve is not old
enough to reach the requested start, that contract cannot be *verified* as
front month for the uncovered span -- fetching it anyway would silently pull
a forward-quarter contract's prices and label them as the front month's.
``download`` refuses that case rather than guessing.

Requests are chunked (IBKR caps intraday history per request), paced (the
API allows ~60 requests per 10 minutes), and cached to disk, so a re-run of
the same window costs nothing.
"""

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
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


@dataclass(frozen=True)
class ContractWindow:
    """A concrete future and the span over which it was front month."""

    expiry: date
    start: datetime
    end: datetime
    contract: Any = None
    #: False when nothing in the input expiries bounds this window's start --
    #: there is no earlier known contract to hand over from, so ``start`` was
    #: assumed to equal the requested window's start rather than derived from
    #: a real roll date.  Only the earliest window a stitch produces can ever
    #: be unverified; every later one is bounded by the contract before it.
    verified_start: bool = True

    @property
    def label(self) -> str:
        local = getattr(self.contract, "localSymbol", None)
        return local or self.expiry.strftime("%Y%m")


def front_month_windows(
    expiries: list[date],
    start: datetime,
    end: datetime,
    roll_days: int = 8,
    tz: ZoneInfo | None = None,
) -> list[ContractWindow]:
    """Slice ``[start, end]`` into the spans each contract led.

    A contract is front month from the previous contract's roll date until
    its own, where the roll date is ``roll_days`` before expiry.  Only spans
    that intersect the requested window are returned, clipped to it.

    The very first window returned can have ``verified_start=False``: if
    ``expiries`` has nothing earlier than the earliest contract that
    intersects the window, there is no real roll date to bound its start, so
    ``start`` is used as a placeholder rather than derived.  That happens
    when the caller's source (IBKR's expired-contract lookup, in practice)
    simply does not go back far enough -- callers must check this flag
    before trusting the window; see ``IbkrHistorySource.download``.

    Pure and IBKR-free so the slicing can be tested directly -- getting it
    wrong silently produces a price series stitched from the wrong contracts,
    which is the kind of error a backtest happily runs on.
    """
    if start >= end:
        raise ValueError(f"start {start} is not before end {end}")

    ordered = sorted(set(expiries))
    windows: list[ContractWindow] = []
    previous_roll: datetime | None = None
    #: Whether ordered[-1]'s own start was bounded by a real predecessor,
    #: tracked independently of whether a window was actually emitted for it
    #: (it may have been skipped inside the loop and only picked up again by
    #: the trailing extension below).
    last_verified = False

    for expiry in ordered:
        roll = datetime.combine(
            expiry - timedelta(days=roll_days), datetime.min.time(), tzinfo=tz
        )
        verified = previous_roll is not None
        span_start = previous_roll if previous_roll is not None else start
        previous_roll = roll
        last_verified = verified
        if roll <= start or span_start >= end:
            continue  # entirely before or after the requested window
        windows.append(
            ContractWindow(
                expiry=expiry,
                start=max(span_start, start),
                end=min(roll, end),
                verified_start=verified,
            )
        )

    # The window can extend past the last listed roll -- there is no further
    # contract to hand over to, so the newest one covers the remainder.
    # Extend its existing span rather than emitting a second entry for it.
    if ordered and previous_roll is not None and previous_roll < end:
        if windows and windows[-1].expiry == ordered[-1]:
            windows[-1] = ContractWindow(
                expiry=windows[-1].expiry, start=windows[-1].start, end=end,
                verified_start=windows[-1].verified_start,
            )
        else:
            windows.append(
                ContractWindow(
                    expiry=ordered[-1], start=max(previous_roll, start), end=end,
                    verified_start=last_verified,
                )
            )
    return [w for w in windows if w.start < w.end]


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

    def cache_path(self) -> Path:
        start, end = self._window()
        slug = self.cfg.data.bar_size.replace(" ", "")
        return self.cache_dir / (
            f"{self.source.name}_{slug}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        )

    def load(self) -> pd.DataFrame:
        """Return the merged frame, downloading only what isn't cached."""
        path = self.cache_path()
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
        from ib_async import IB  # imported lazily: optional dependency

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
            windows = self._resolve_windows(ib, start, end)
            if not windows:
                raise RuntimeError(
                    f"no {self.source.future.symbol} contracts were listed over "
                    f"{start.date()}..{end.date()}. Check the date range."
                )
            self._check_coverage(windows, start)
            log.info(
                "stitching %d front-month contract(s): %s",
                len(windows),
                ", ".join(
                    f"{w.label} {w.start.date()}..{w.end.date()}" for w in windows
                ),
            )

            prices = self._fetch_windows(ib, windows, "TRADES")
            if prices.empty:
                raise RuntimeError(
                    "IBKR returned no TRADES bars for "
                    f"{start.date()}..{end.date()}. Check that the account has "
                    "CME market data, that the range is not older than IBKR's "
                    "retention for this bar size, and that the range covers "
                    "trading days."
                )
            vols = self._fetch_windows(ib, windows, "OPTION_IMPLIED_VOLATILITY")
        finally:
            ib.disconnect()

        return self._merge(prices, vols)

    # -- contract resolution --------------------------------------------

    def _resolve_windows(
        self, ib, start: datetime, end: datetime
    ) -> list[ContractWindow]:
        """Find the concrete contracts that were front month over the window."""
        from ib_async import Future

        spec = self.source.future
        query = Future(
            symbol=spec.symbol,
            exchange=spec.exchange,
            currency=spec.currency,
            includeExpired=True,  # required: the window is usually in the past
        )
        details = ib.reqContractDetails(query)
        if not details:
            raise RuntimeError(
                f"IBKR returned no contract details for {spec.symbol} on "
                f"{spec.exchange}. Check the symbol and exchange."
            )

        by_expiry: dict[date, Any] = {}
        for detail in details:
            expiry = _parse_expiry(detail.contract.lastTradeDateOrContractMonth)
            if expiry is not None:
                by_expiry.setdefault(expiry, detail.contract)

        windows = front_month_windows(
            list(by_expiry),
            start,
            end,
            roll_days=self.cfg.data.roll_days_before_expiry,
            tz=self.tz,
        )
        return [
            ContractWindow(
                w.expiry, w.start, w.end, by_expiry[w.expiry],
                verified_start=w.verified_start,
            )
            for w in windows
        ]

    def _check_coverage(self, windows: list[ContractWindow], start: datetime) -> None:
        """Refuse a stitch whose earliest span cannot be verified.

        IBKR simply omits contracts it cannot resolve -- there is no error,
        just an absence -- so an unverified first window is silently the
        wrong contract's prices rather than a missing one.  Failing loudly
        here is the whole point of tracking ``verified_start`` at all.
        """
        earliest = windows[0]
        if earliest.verified_start:
            return
        roll_days = self.cfg.data.roll_days_before_expiry
        roll_in = datetime.combine(
            earliest.expiry - timedelta(days=roll_days), datetime.min.time(), tzinfo=self.tz
        )
        base = (
            f"cannot verify {self.source.future.symbol} front-month coverage "
            f"for {start.date()}..{earliest.end.date()}: the earliest contract "
            f"this connection could resolve is {earliest.label} (expires "
            f"{earliest.expiry}), and nothing in IBKR's expired-contract "
            f"lookup goes back far enough to bound when its own front-month "
            f"span began. Fetching it anyway would silently label a "
            f"forward-quarter contract's prices as the front month's for "
            f"that whole span.\n"
            f"This is IBKR's expired-contract retention on this account/"
            f"connection, not the requested date range being invalid."
        )
        if roll_in < earliest.end:
            raise RuntimeError(
                f"{base} Retry with --start on or after {roll_in.date()} (this "
                f"contract's own roll-in date) to fetch the recoverable portion "
                f"of the range, or use a data source with longer historical "
                f"contract retention for the rest."
            )
        raise RuntimeError(
            f"{base} No part of the requested range is recoverable this way: "
            f"this contract only becomes verifiably front month on "
            f"{roll_in.date()}, which is after the requested end "
            f"({windows[-1].end.date()}). Use a data source with longer "
            f"historical contract retention."
        )

    # -- fetching --------------------------------------------------------

    def _fetch_windows(self, ib, windows: list[ContractWindow], what_to_show: str):
        frames = []
        for window in windows:
            frame = self._fetch_series(
                ib, window.contract, what_to_show, window.start, window.end
            )
            if frame.empty:
                log.warning(
                    "no %s bars for %s over %s..%s",
                    what_to_show, window.label, window.start.date(), window.end.date(),
                )
                continue
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _fetch_series(
        self, ib, contract, what_to_show: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Page backwards through one contract's window, a chunk per request."""
        from ib_async import util

        frames: list[pd.DataFrame] = []
        cursor = end
        request_count = 0
        last_error: str | None = None

        while cursor > start:
            span_days = min(CHUNK_DAYS, max((cursor - start).days, 1))
            if request_count:
                time_module.sleep(PACING_SECONDS)
            log.info(
                "requesting %dD %s of %s ending %s",
                span_days, what_to_show,
                getattr(contract, "localSymbol", "?"), cursor.date(),
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
            except Exception as exc:  # noqa: BLE001 - record, then keep paging
                last_error = str(exc)
                log.warning("%s request failed: %s", what_to_show, exc)
                bars = None
            request_count += 1

            if not bars:
                if what_to_show == "OPTION_IMPLIED_VOLATILITY":
                    log.warning(
                        "no implied-vol history returned%s; falling back to "
                        "data.default_atm_iv=%.3f for the affected bars",
                        f" ({last_error})" if last_error else "",
                        self.cfg.data.default_atm_iv,
                    )
                    break
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
            if last_error and what_to_show == "TRADES":
                log.error("every %s request failed; last error: %s",
                          what_to_show, last_error)
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

    # -- assembly --------------------------------------------------------

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

    def _merge(self, prices: pd.DataFrame, vols: pd.DataFrame) -> pd.DataFrame:
        frame = prices[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        if vols.empty:
            log.warning(
                "no implied-vol series available; every bar will use "
                "data.default_atm_iv=%.3f. The backtest becomes a constant-vol "
                "study -- see the README on modelled option prices.",
                self.cfg.data.default_atm_iv,
            )
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


def _parse_expiry(value: str) -> date | None:
    """Parse IBKR's lastTradeDateOrContractMonth (YYYYMMDD or YYYYMM)."""
    text = (value or "").strip()
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    log.warning("could not parse a contract expiry from %r", value)
    return None
