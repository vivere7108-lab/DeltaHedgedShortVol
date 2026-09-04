"""The intraday flow nowcast: CME Globex MDP 3.0 option trades, via Databento.

See ``config.NowcastConfig`` for what this is *for* and the honesty case
for ``dealer_share`` -- this module is only concerned with getting from raw
trade prints to a per-strike signed-volume table, live or replayed from a
local backfill.

Two independent things live here, both producing the same output shape
(``StrikeFlow``, aggregated per strike since some reference time) so the
strategy cannot tell which one it is reading from -- the same symmetry
``OpenInterestProvider`` already has across its synthetic/CSV/IBKR
implementations:

``DatabentoLiveFlowFeed``
    Subscribes to CME Globex ES options trades and definitions over
    Databento's live gateway and keeps a rolling, per-trade log in memory
    (not just a running total -- see below for why).

``DatabentoHistoricalFlowSource``
    Backfills 6-12 months of the same trades and definitions from
    Databento's historical API, resolves each trade against the
    definitions that were live at the time, and caches the result as one
    Parquet file per UTC day: ``ts_event, expiry, strike, right,
    signed_size``.  That per-trade cache -- not a pre-aggregated one -- is
    what makes the nowcast *backtestable at any window*, not merely
    observable live: a bar loop that asks "what was signed flow between
    10:14 and 10:34" gets an answer built from the same rows a live feed
    would have produced, replayed rather than approximated.

Why a per-trade log rather than a running total
-------------------------------------------------
The strategy asks for flow *since the last OI print*, refreshed on its own
throttled cadence (``nowcast.refresh_seconds``).  A single running counter
reset at some remembered instant would work only if the reset always landed
exactly on the OI print; keeping every trade with its timestamp instead
means the exact same window-aggregation function -- ``since()`` for the live
feed, ``flow_since()`` for the historical one -- answers "since when" for
any two timestamps, live or replayed, without having to get a reset call
site right.  ES options trade in the tens of thousands of prints a day, so
keeping a session's worth of them in memory is not a sizing concern; the
live feed prunes anything older than ``retain`` regardless.

Signing a trade
----------------
CME/Databento's normalised trade schema carries the *aggressor* side
directly (``Side.ASK`` -- traded at the ask, a buy initiator; ``Side.BID`` --
traded at the bid, a sell initiator), so no tick rule is needed.  Reading
"aggressor" as "customer" is the assumption ``dealer_share`` exists to
scale rather than trust -- see ``config.NowcastConfig``.

Databento itself is imported lazily, inside the methods that need it, since
it is an optional dependency (the ``databento`` extra) -- a backtest run
against synthetic or CSV data never has to have it installed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from ..gex import StrikeFlow
from ..session import is_trading_day, next_trading_day

log = logging.getLogger(__name__)

#: Aggressor-side characters, matching ``databento.Side.ASK``/``BID``.value
#: without importing the package at module load time (it is an optional
#: dependency -- see ``build_flow_feed``/``build_flow_source`` below).
SIDE_BUY = "A"
SIDE_SELL = "B"


@dataclass(frozen=True)
class InstrumentInfo:
    """What a strike is, resolved once from a Databento instrument definition."""

    instrument_id: int
    expiry: date
    strike: float
    right: str  # "C" | "P", matching OptionQuote.right elsewhere in the codebase

    @classmethod
    def from_definition(cls, record: object) -> "InstrumentInfo | None":
        """Build one from an ``InstrumentDefMsg`` (or a duck-typed stand-in).

        Returns ``None`` for anything that is not a call or a put -- the
        underlying future itself, a spread, or any other instrument class
        the parent symbology might ever surface.  Reads the ``pretty_*``
        convenience properties Databento's DBN bindings provide when
        present, and falls back to the raw scaled fields otherwise, so a
        hand-built test record does not have to reproduce the fixed-point
        scaling to be usable.
        """
        right = _instrument_right(record)
        if right is None:
            return None
        strike = _attr_float(record, "pretty_strike_price", "strike_price", 1e9)
        expiry = _attr_date(record, "pretty_expiration", "expiration")
        if strike is None or expiry is None:
            return None
        return cls(
            instrument_id=int(record.instrument_id),
            expiry=expiry,
            strike=float(strike),
            right=right,
        )


def _instrument_right(record: object) -> str | None:
    raw = getattr(record, "instrument_class", None)
    value = getattr(raw, "value", raw)
    if value in ("C", "P"):
        return value
    return None


def _attr_float(record: object, pretty: str, raw: str, scale: float) -> float | None:
    value = getattr(record, pretty, None)
    if value is not None:
        return float(value)
    value = getattr(record, raw, None)
    if value is None:
        return None
    return float(value) / scale


def _attr_date(record: object, pretty: str, raw: str) -> date | None:
    value = getattr(record, pretty, None)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    value = getattr(record, raw, None)
    if not value:
        return None
    return datetime.fromtimestamp(int(value) / 1e9, tz=timezone.utc).date()


def _signed_size(side: object, size: float) -> float:
    """Aggressor-signed size: positive for a buy, negative for a sell."""
    value = getattr(side, "value", side)
    if value == SIDE_BUY:
        return float(size)
    if value == SIDE_SELL:
        return -float(size)
    return 0.0  # Side.NONE (or a record with no side): not usable


class InstrumentMap:
    """``instrument_id -> InstrumentInfo``, built from definition records."""

    def __init__(self) -> None:
        self._by_id: dict[int, InstrumentInfo] = {}

    def add_definition(self, record: object) -> InstrumentInfo | None:
        info = InstrumentInfo.from_definition(record)
        if info is not None:
            self._by_id[info.instrument_id] = info
        return info

    def get(self, instrument_id: int) -> InstrumentInfo | None:
        return self._by_id.get(instrument_id)

    def __len__(self) -> int:
        return len(self._by_id)


def _aggregate(rows: Sequence[tuple[float, str, float]]) -> tuple[StrikeFlow, ...]:
    """``(strike, right, signed_size)`` rows, summed per strike."""
    by_strike: dict[float, list[float]] = {}
    for strike, right, signed_size in rows:
        bucket = by_strike.setdefault(strike, [0.0, 0.0])
        bucket[0 if right == "C" else 1] += signed_size
    return tuple(
        StrikeFlow(strike=k, call_signed_volume=v[0], put_signed_volume=v[1])
        for k, v in sorted(by_strike.items())
    )


@dataclass(frozen=True)
class _Trade:
    ts_event: datetime
    strike: float
    right: str
    signed_size: float


class FlowAccumulator:
    """Thread-safe per-trade log, aggregated into ``StrikeFlow`` on demand.

    Fed from the live callback thread (Databento's client runs it in the
    background); read from the poll loop's thread. One lock, held only
    across the list append/copy, keeps that safe without serialising the
    aggregation itself.
    """

    def __init__(self, retain: timedelta = timedelta(days=3)):
        self.retain = retain
        self._lock = threading.Lock()
        self._by_expiry: dict[date, list[_Trade]] = {}

    def record(self, info: InstrumentInfo, ts_event: datetime, side: object, size: float) -> None:
        signed = _signed_size(side, size)
        if signed == 0.0:
            return
        trade = _Trade(ts_event, info.strike, info.right, signed)
        with self._lock:
            self._by_expiry.setdefault(info.expiry, []).append(trade)

    def since(self, expiry: date, start: datetime, end: datetime) -> tuple[StrikeFlow, ...]:
        with self._lock:
            trades = list(self._by_expiry.get(expiry, ()))
        rows = [
            (t.strike, t.right, t.signed_size)
            for t in trades
            if start <= t.ts_event < end
        ]
        return _aggregate(rows)

    def prune(self, now: datetime) -> None:
        """Drop trades older than ``retain``. Cheap insurance against an
        unattended live session growing its trade log without bound."""
        cutoff = now - self.retain
        with self._lock:
            for expiry in list(self._by_expiry):
                kept = [t for t in self._by_expiry[expiry] if t.ts_event >= cutoff]
                if kept:
                    self._by_expiry[expiry] = kept
                else:
                    del self._by_expiry[expiry]


class DatabentoLiveFlowFeed:
    """Live CME Globex ES options trades and definitions, over Databento.

    Subscribes to the whole options complex via the parent symbol
    (``nowcast.parent_symbol``, typically ``"ES.OPT"``) so no strike list
    has to be maintained by hand as the chain rolls. Definitions are
    subscribed alongside trades on the same session: Databento resends the
    live definition set on connect and pushes updates as new instruments
    list, which is what keeps ``InstrumentMap`` current without a separate
    poll.
    """

    def __init__(self, cfg, api_key: str | None = None):
        self.cfg = cfg
        self.api_key = api_key
        self.instruments = InstrumentMap()
        self.flow = FlowAccumulator()
        self._client = None
        self._unmapped_warned: set[int] = set()

    def start(self) -> None:
        import databento as db  # imported lazily: optional dependency

        client = db.Live(key=self.api_key)
        client.subscribe(
            dataset=self.cfg.dataset,
            schema=db.Schema.DEFINITION,
            symbols=self.cfg.parent_symbol,
            stype_in=db.SType.PARENT,
        )
        client.subscribe(
            dataset=self.cfg.dataset,
            schema=db.Schema.TRADES,
            symbols=self.cfg.parent_symbol,
            stype_in=db.SType.PARENT,
        )
        client.add_callback(record_callback=self._on_record)
        client.start()
        self._client = client
        log.info(
            "subscribed to %s trades + definitions on %s",
            self.cfg.dataset, self.cfg.parent_symbol,
        )

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()
            self._client = None

    def flow_since(
        self, moment: datetime, expiry: date, since: datetime
    ) -> tuple[StrikeFlow, ...]:
        self.flow.prune(moment)
        return self.flow.since(expiry, since, moment)

    # -- internals ---------------------------------------------------------

    def _on_record(self, record: object) -> None:
        if hasattr(record, "instrument_class"):
            self.instruments.add_definition(record)
            return
        if not hasattr(record, "side"):
            return  # some other schema's record; not expected, not fatal
        instrument_id = int(record.instrument_id)
        info = self.instruments.get(instrument_id)
        if info is None:
            if instrument_id not in self._unmapped_warned:
                self._unmapped_warned.add(instrument_id)
                log.debug(
                    "trade for unmapped instrument_id=%d (definition not seen "
                    "yet); dropped", instrument_id,
                )
            return
        ts_event = datetime.fromtimestamp(int(record.ts_event) / 1e9, tz=timezone.utc)
        self.flow.record(info, ts_event, record.side, float(record.size))


class DatabentoHistoricalFlowSource:
    """Backfilled trades and definitions, replayed for a backtest.

    Caches Databento's own DBN encoding, one definitions file and one
    trades file per UTC calendar day -- the same bytes ``deltahedger
    nowcast-backfill`` downloads are what a later ``flow_since`` call
    replays, through the *identical* ``InstrumentInfo.from_definition`` /
    ``FlowAccumulator.record`` path the live feed uses.  A day is cached
    once by its raw DBN, not distilled into a derived format up front, so
    "6-12 months backfilled" means the actual trade tape is sitting on
    disk, replayable at any window a backtest asks for -- not a
    pre-aggregated summary that only answers the questions someone thought
    to ask when building the cache.

    Bucketed by UTC calendar day rather than by exchange session: simpler
    to cache and to reason about, at the cost of a session's overnight
    trades being split across two cache files. ``flow_since`` reads
    whichever files the requested window touches, so this only matters for
    how the cache is laid out on disk, never for what a query returns.
    """

    def __init__(self, cfg, cache_dir: str | Path | None = None, api_key: str | None = None):
        self.cfg = cfg
        self.api_key = api_key
        self.cache_dir = Path(cache_dir if cache_dir is not None else cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._accumulators: dict[date, FlowAccumulator] = {}

    # -- backfill ------------------------------------------------------

    def backfill(self, start: date, end: date, force: bool = False) -> list[date]:
        """Download every trading day in ``[start, end]`` not already cached.

        Returns the days actually fetched, so a caller (the CLI command)
        can report what it did rather than a silent no-op on a day that was
        already there.
        """
        fetched: list[date] = []
        day = start if is_trading_day(start) else next_trading_day(start)
        while day <= end:
            if force or not self._is_cached(day):
                self._backfill_day(day)
                fetched.append(day)
            day = next_trading_day(day)
        return fetched

    def _is_cached(self, day: date) -> bool:
        return self._definitions_path(day).exists() and self._trades_path(day).exists()

    def _backfill_day(self, day: date) -> None:
        import databento as db

        client = db.Historical(key=self.api_key)
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        log.info("backfilling %s %s for %s", self.cfg.dataset, self.cfg.parent_symbol, day)

        definitions = client.timeseries.get_range(
            dataset=self.cfg.dataset,
            schema=db.Schema.DEFINITION,
            symbols=self.cfg.parent_symbol,
            stype_in=db.SType.PARENT,
            start=start,
            end=end,
        )
        definitions.to_file(self._definitions_path(day))

        trades = client.timeseries.get_range(
            dataset=self.cfg.dataset,
            schema=db.Schema.TRADES,
            symbols=self.cfg.parent_symbol,
            stype_in=db.SType.PARENT,
            start=start,
            end=end,
        )
        trades.to_file(self._trades_path(day))

    def _definitions_path(self, day: date) -> Path:
        return self.cache_dir / f"definitions-{day.isoformat()}.dbn.zst"

    def _trades_path(self, day: date) -> Path:
        return self.cache_dir / f"trades-{day.isoformat()}.dbn.zst"

    # -- replay ----------------------------------------------------------

    def flow_since(
        self, moment: datetime, expiry: date, since: datetime
    ) -> tuple[StrikeFlow, ...]:
        rows: list[tuple[float, str, float]] = []
        for day in self._days_touching(since, moment):
            accumulator = self._day_accumulator(day)
            for strike_flow in accumulator.since(expiry, since, moment):
                rows.append((strike_flow.strike, "C", strike_flow.call_signed_volume))
                rows.append((strike_flow.strike, "P", strike_flow.put_signed_volume))
        return _aggregate(rows)

    def _days_touching(self, since: datetime, moment: datetime) -> list[date]:
        days = []
        day = since.astimezone(timezone.utc).date()
        last = moment.astimezone(timezone.utc).date()
        while day <= last:
            days.append(day)
            day += timedelta(days=1)
        return days

    def _day_accumulator(self, day: date) -> FlowAccumulator:
        """The day's trades, replayed once and memoised.

        A backtest re-derives the same window on every throttled refresh
        (``nowcast.refresh_seconds``), so replaying a day's DBN files
        afresh on every call would mean re-parsing the same trade tape
        dozens of times over a session; each day is resolved once and kept.
        """
        cached = self._accumulators.get(day)
        if cached is not None:
            return cached

        import databento as db

        if not self._is_cached(day):
            raise FileNotFoundError(
                f"no backfilled nowcast data for {day} in {self.cache_dir}; run "
                "`deltahedger nowcast-backfill` for this date range first"
            )

        instruments = InstrumentMap()
        db.DBNStore.from_file(self._definitions_path(day)).replay(
            instruments.add_definition
        )

        accumulator = FlowAccumulator(retain=timedelta(days=400))
        def _on_trade(record: object) -> None:
            if not hasattr(record, "side"):
                return
            info = instruments.get(int(record.instrument_id))
            if info is None:
                return
            ts_event = datetime.fromtimestamp(
                int(record.ts_event) / 1e9, tz=timezone.utc
            )
            accumulator.record(info, ts_event, record.side, float(record.size))

        db.DBNStore.from_file(self._trades_path(day)).replay(_on_trade)
        self._accumulators[day] = accumulator
        return accumulator
