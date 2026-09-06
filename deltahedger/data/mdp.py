"""Intraday open interest from CME's MDP 3.0 feed.

Why this exists
---------------
GEX has one input, open interest, and the 0DTE read is only defensible on
open interest that describes the book that is *there* -- not the previous
session's close.  IBKR's generic tick 101 (``broker.ibkr.IbkrOpenInterestProvider``)
is the previous session's close; the exchange's own MDP 3.0 feed publishes
open interest intraday, as a statistics message per instrument.  This module
is the consumer of that feed the rest of the system reads through.

Two pieces, deliberately separate:

``MdpOpenInterest``
    The provider the strategy sees.  It reads a **snapshot file** -- one
    JSON document holding, per expiry, per strike, the latest call and put
    open interest and the time it was published -- and refuses to serve a
    snapshot older than ``data.oi_max_age_seconds``.  A stale feed reads as
    "no open interest", which the strategy treats as a reason to stand
    aside; it never quietly falls back to yesterday's print, because a walk
    that did so would look exactly like a walk on the feed it claims.

``DatabentoOpenInterestFeed``
    Writes that snapshot, from Databento's redistribution of the MDP 3.0
    feed (dataset ``GLBX.MDP3``): the ``definition`` schema to map an
    instrument id to its expiry, strike and right, the ``statistics``
    schema for the open-interest updates.  It is a separate long-running
    process (``deltahedger mdp-feed``) so a hiccup in the feed cannot take
    the hedger with it, and so the snapshot is one file the strategy reads
    rather than a socket it depends on.  Any other MDP 3.0 consumer -- a
    direct multicast handler, another vendor -- can write the same file;
    the schema is ``write_snapshot``'s and nothing in the strategy knows
    where it came from.

The Databento adapter is written against their published Python API and
has not been run against the live service inside this repository's test
suite -- the tests here cover the snapshot format, the staleness rule and
the record handling with synthetic records.  Run ``deltahedger doctor``
against a live snapshot before trusting a walk to it.

Snapshot format::

    {
      "as_of":   "2026-09-08T14:32:10-04:00",   # when the newest update landed
      "source":  "GLBX.MDP3 via databento",
      "expiries": {
        "2026-09-08": {"5000.0": [1234, 5678], ...},   # strike: [call_oi, put_oi]
        "2026-09-09": {...}
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from ..config import DataConfig
from ..gex import StrikeOpenInterest
from ..instruments import RiskSource

log = logging.getLogger(__name__)

#: Databento's ``stat_type`` for open interest, per the DBN specification.
STAT_OPEN_INTEREST = 9
#: The Databento dataset carrying CME Globex MDP 3.0.
DATASET = "GLBX.MDP3"


# -- the snapshot file -------------------------------------------------------


def write_snapshot(
    path: str | Path,
    as_of: datetime,
    expiries: dict[date, dict[float, tuple[float, float]]],
    source: str = "",
) -> None:
    """Write the whole open-interest snapshot atomically.

    Written to a sibling temp file and renamed into place, so a reader never
    sees half a document and a writer killed mid-write leaves the previous
    snapshot standing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "as_of": as_of.isoformat(),
        "source": source,
        "expiries": {
            expiry.isoformat(): {
                f"{strike:g}": [float(call), float(put)]
                for strike, (call, put) in sorted(rows.items())
            }
            for expiry, rows in sorted(expiries.items())
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_snapshot(path: str | Path) -> dict[str, Any] | None:
    """The snapshot document, or ``None`` if there is none or it is unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error("could not read the open-interest snapshot at %s (%s)", path, exc)
        return None


def snapshot_age_seconds(document: dict[str, Any], now: datetime) -> float | None:
    """Seconds between the snapshot's ``as_of`` and ``now``; None if unreadable."""
    raw = document.get("as_of")
    if not raw:
        return None
    try:
        as_of = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=now.tzinfo or timezone.utc)
    return (now - as_of).total_seconds()


class MdpOpenInterest:
    """The strategy's view of the MDP 3.0 open interest: the snapshot file.

    Re-read on every call -- the strategy already rate-limits the calls on
    ``gex.refresh_seconds``, and a JSON document of a few hundred strikes
    costs nothing next to the flip search that follows it.
    """

    def __init__(self, cfg: DataConfig, source: RiskSource):
        if not cfg.oi_mdp_path:
            raise ValueError(
                "data.oi_mdp_path must be set when data.open_interest == 'mdp': "
                "it is the snapshot `deltahedger mdp-feed` writes"
            )
        self.path = Path(cfg.oi_mdp_path)
        self.max_age = float(cfg.oi_max_age_seconds)
        self.source = source
        self._warned_stale = False

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        document = read_snapshot(self.path)
        if document is None:
            log.warning(
                "no open-interest snapshot at %s; is `deltahedger mdp-feed` running? "
                "GEX cannot be computed and the strategy will stand aside", self.path,
            )
            return []
        age = snapshot_age_seconds(document, moment)
        if age is None or age > self.max_age:
            if not self._warned_stale:
                log.warning(
                    "the open-interest snapshot at %s is %s; the feed is stale and "
                    "the strategy will stand aside rather than read yesterday's book",
                    self.path,
                    f"{age / 60:.0f} minutes old (limit {self.max_age / 60:.0f})"
                    if age is not None else "undated",
                )
                self._warned_stale = True
            return []
        self._warned_stale = False
        rows = document.get("expiries", {}).get(expiry.isoformat(), {})
        return [
            StrikeOpenInterest(
                strike=float(strike), call_oi=float(pair[0]), put_oi=float(pair[1])
            )
            for strike, pair in rows.items()
            if len(pair) == 2 and (float(pair[0]) or float(pair[1]))
        ]


# -- the feed ------------------------------------------------------------------


class OpenInterestBook:
    """Open interest by instrument, assembled from definitions and statistics.

    Pure record handling, with no dependency on the feed library, so it can
    be driven with synthetic records in the tests.  ``on_definition`` learns
    what an instrument id *is*; ``on_statistic`` learns the open interest
    on it; ``expiries`` folds the two into the snapshot's shape.
    """

    def __init__(self, tz: ZoneInfo, underlying: str = "ES"):
        self.tz = tz
        self.underlying = underlying.upper()
        #: instrument id -> (expiry, strike, right)
        self.definitions: dict[int, tuple[date, float, str]] = {}
        #: instrument id -> open interest
        self.open_interest: dict[int, float] = {}
        self.last_update: datetime | None = None

    def on_definition(
        self, instrument_id: int, expiration_ns: int, strike: float, right: str,
        underlying: str = "",
    ) -> None:
        if underlying and underlying.upper() != self.underlying:
            return
        right = str(right).upper()[:1]
        if right not in ("C", "P"):
            return  # a future, a spread -- not an option leg
        expiry = (
            datetime.fromtimestamp(expiration_ns / 1e9, tz=timezone.utc)
            .astimezone(self.tz)
            .date()
        )
        self.definitions[int(instrument_id)] = (expiry, float(strike), right)

    def on_statistic(
        self, instrument_id: int, stat_type: int, quantity: float, ts_event_ns: int
    ) -> None:
        if int(stat_type) != STAT_OPEN_INTEREST:
            return
        self.open_interest[int(instrument_id)] = float(quantity)
        moment = datetime.fromtimestamp(ts_event_ns / 1e9, tz=timezone.utc).astimezone(self.tz)
        if self.last_update is None or moment > self.last_update:
            self.last_update = moment

    def expiries(self) -> dict[date, dict[float, tuple[float, float]]]:
        out: dict[date, dict[float, list[float]]] = {}
        for instrument_id, quantity in self.open_interest.items():
            definition = self.definitions.get(instrument_id)
            if definition is None:
                continue  # statistics can arrive before the definition
            expiry, strike, right = definition
            pair = out.setdefault(expiry, {}).setdefault(strike, [0.0, 0.0])
            pair[0 if right == "C" else 1] = quantity
        return {
            expiry: {strike: (pair[0], pair[1]) for strike, pair in rows.items()}
            for expiry, rows in out.items()
        }


def _field(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    return default


def apply_record(book: OpenInterestBook, record: Any) -> None:
    """Feed one Databento record (definition or statistic) into the book."""
    kind = type(record).__name__
    if kind == "InstrumentDefMsg":
        strike = _field(record, "pretty_strike_price")
        if strike is None:
            raw = _field(record, "strike_price", default=0)
            strike = float(raw) / 1e9
        book.on_definition(
            instrument_id=_field(record, "instrument_id", default=0),
            expiration_ns=_field(record, "expiration", default=0),
            strike=float(strike),
            right=str(_field(record, "instrument_class", default="")),
            underlying=str(_field(record, "underlying", default="") or ""),
        )
    elif kind == "StatMsg":
        book.on_statistic(
            instrument_id=_field(record, "instrument_id", default=0),
            stat_type=int(_field(record, "stat_type", default=-1)),
            quantity=float(_field(record, "quantity", default=0.0)),
            ts_event_ns=int(_field(record, "ts_event", "ts_recv", default=0)),
        )


class DatabentoOpenInterestFeed:
    """Subscribes to the MDP 3.0 feed via Databento and writes the snapshot.

    Needs the ``databento`` package and an API key in ``DATABENTO_API_KEY``
    (or passed in).  Subscribes to every option on the risk source's
    future -- ``ES.OPT`` in parent symbology covers the dailies, weeklies
    and quarterlies alike -- and rewrites the snapshot every
    ``write_seconds`` while updates are arriving.
    """

    def __init__(
        self,
        source: RiskSource,
        path: str | Path,
        api_key: str | None = None,
        write_seconds: float = 30.0,
    ):
        self.source = source
        self.path = Path(path)
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        self.write_seconds = write_seconds
        self.book = OpenInterestBook(ZoneInfo(source.timezone), source.option.symbol)
        self._stop = False

    def request_stop(self, *_: object) -> None:
        self._stop = True

    def _client(self):
        try:
            import databento as db
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise RuntimeError(
                "the MDP feed needs the `databento` package: pip install databento"
            ) from exc
        if not self.api_key:
            raise RuntimeError("set DATABENTO_API_KEY (or pass api_key) for the MDP feed")
        client = db.Live(key=self.api_key)
        parent = f"{self.source.option.symbol}.OPT"
        client.subscribe(dataset=DATASET, schema="definition", stype_in="parent", symbols=parent)
        client.subscribe(dataset=DATASET, schema="statistics", stype_in="parent", symbols=parent)
        return client

    def run(self, records: Iterable[Any] | None = None) -> None:
        """Consume records until stopped, writing the snapshot as they land.

        ``records`` lets a test (or another consumer) hand records in
        directly; without it the Databento live client is opened.
        """
        stream = records if records is not None else self._client()
        last_write = 0.0
        dirty = False
        log.info("MDP open-interest feed writing %s", self.path)
        for record in stream:
            if self._stop:
                break
            apply_record(self.book, record)
            dirty = True
            now = time.monotonic()
            if now - last_write >= self.write_seconds:
                self.flush()
                last_write, dirty = now, False
        if dirty:
            self.flush()

    def flush(self) -> None:
        as_of = self.book.last_update or datetime.now(self.book.tz)
        write_snapshot(self.path, as_of, self.book.expiries(), source=f"{DATASET} via databento")
        log.debug(
            "snapshot: %d instruments with open interest across %d expiries",
            len(self.book.open_interest), len(self.book.expiries()),
        )


def describe_snapshot(path: str | Path, now: datetime, expiry: date | None = None) -> str:
    """One line on the snapshot's health, for ``deltahedger doctor``."""
    document = read_snapshot(path)
    if document is None:
        return f"no snapshot at {path} -- is `deltahedger mdp-feed` running?"
    age = snapshot_age_seconds(document, now)
    expiries = document.get("expiries", {})
    text = (
        f"{len(expiries)} expiries, "
        f"{'undated' if age is None else f'{timedelta(seconds=int(age))} old'}"
    )
    if expiry is not None:
        rows = expiries.get(expiry.isoformat(), {})
        total = sum(float(c) + float(p) for c, p in rows.values())
        text += f"; {expiry}: {len(rows)} strikes, {total:,.0f} contracts"
    return text
