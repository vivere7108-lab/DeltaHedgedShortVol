"""Where the option tape -- and therefore the measured dealer sign -- comes from.

Three providers, all producing the same ``OptionTrade`` list so
``DealerFlowBook`` cannot tell them apart, plus a fourth (``IbkrTradeFeed``)
that lives in ``broker.ibkr`` with the rest of the live path:

``CsvTradeFeed``
    Replays a real tape.  The columns it needs are
    ``timestamp,expiry,strike,right,price,size``; the ones that make the
    classification better are ``bid,ask`` (the Lee-Ready quote rule),
    ``aggressor`` (MDP 3.0 tag 5797, taken verbatim: 1 buy, 2 sell, 0 none)
    and ``bid_size_delta,ask_size_delta`` (the MBO book-state change).  A
    file with only the required columns still classifies, on the tick test
    alone, and the rule counts will say so.

``SyntheticTradeFeed``
    Generates a tape for the backtest.  It is a **harness, not a market
    model**: it exists so the classification path is exercised end to end by
    a generated run rather than only by unit tests, and it says the
    machinery works, never that the signal works.  It is deliberately built
    to agree with ``SyntheticOpenInterest`` -- both read the same per-expiry
    ``call_share`` draw -- because a generator whose flow contradicted its
    own open interest would produce a confidence gate firing constantly for
    reasons that have nothing to do with any market.

The two live feeds live with their connections -- ``IbkrTradeFeed`` in
``broker.ibkr`` and ``DatabentoTradeFeed`` in ``databento_source`` -- and
they are not equivalent.  Databento reads MDP 3.0's aggressor flag, so
rule 1 of the classification chain resolves the tape outright; IBKR relays
no such flag, so its feed falls back to the Lee-Ready quote and tick rules.
Same interface, materially different evidence, and
``DealerFlowBook.rule_counts`` is what says which one a given read got.

``NullTradeFeed``
    No tape at all, which is the default.  Every strike then falls back to
    ``gex.call_sign``/``gex.put_sign`` and the system behaves exactly as it
    did before flow was wired in.  This is a real answer, not a degraded
    one: a deployment either has an aggressor-carrying feed or it does not,
    and the wrong response to not having one is to invent it.

A note on the window
--------------------
Feeds are pulled with the interval since the last pull, ``(start, end]``,
and every provider must honour that half-open convention.  A feed that
returned a trade twice would double-count it into the dealer position at
that strike -- silently, since nothing downstream sees individual trades --
and one that dropped a boundary trade would lose it entirely.  ``end`` is
the bar's own timestamp, so a trade that prints exactly on a bar boundary
belongs to that bar and to no other.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from ..config import FlowConfig
from ..flow import CALL, PUT, OptionTrade, aggressor_from_mdp
from ..instruments import RiskSource

log = logging.getLogger(__name__)


class NullTradeFeed:
    """No tape. Every sign falls back to the configured prior."""

    def trades(
        self, start: datetime, end: datetime, expiry: date
    ) -> Sequence[OptionTrade]:
        return ()


class CsvTradeFeed:
    """Replays an option tape from CSV, keyed by expiry.

    The file is read once and indexed by expiry, then bisected per pull, so
    replaying a session's tape across a few hundred bars does not rescan it
    a few hundred times.
    """

    REQUIRED = ("timestamp", "expiry", "strike", "right", "price", "size")
    OPTIONAL = ("bid", "ask", "aggressor", "bid_size_delta", "ask_size_delta")

    def __init__(self, cfg: FlowConfig, source: RiskSource, tz=None):
        if not cfg.csv_path:
            raise ValueError("flow.csv_path must be set when flow.source == 'csv'")
        self.path = Path(cfg.csv_path)
        self.source = source
        self.tz = tz
        self._by_expiry: dict[date, list[OptionTrade]] | None = None
        self._stamps: dict[date, list[datetime]] = {}

    def _load(self) -> dict[date, list[OptionTrade]]:
        import pandas as pd

        if not self.path.exists():
            raise FileNotFoundError(f"no option trade CSV at {self.path}")
        frame = pd.read_csv(self.path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        missing = [c for c in self.REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{self.path} is missing column(s): {', '.join(missing)}; "
                f"found {', '.join(frame.columns)}"
            )
        stamps = pd.to_datetime(frame["timestamp"])
        if getattr(stamps.dt, "tz", None) is None:
            if self.tz is None:
                raise ValueError(
                    f"{self.path} has naive timestamps and no timezone was "
                    "supplied to localise them with. The strategy compares "
                    "them against timezone-aware bar times, so a naive tape "
                    "would silently classify nothing -- give the column an "
                    "offset, or build the feed with a tz."
                )
            stamps = stamps.dt.tz_localize(self.tz)
        frame["timestamp"] = stamps
        frame["expiry"] = pd.to_datetime(frame["expiry"]).dt.date
        for column in self.OPTIONAL:
            if column not in frame.columns:
                frame[column] = None

        rows: dict[date, list[OptionTrade]] = {}
        for record in frame.sort_values("timestamp").itertuples(index=False):
            right = str(record.right).strip().upper()[:1]
            if right not in (CALL, PUT):
                continue
            rows.setdefault(record.expiry, []).append(
                OptionTrade(
                    timestamp=record.timestamp.to_pydatetime(),
                    expiry=record.expiry,
                    strike=float(record.strike),
                    right=right,
                    price=float(record.price),
                    size=float(record.size),
                    bid=_optional_float(record.bid),
                    ask=_optional_float(record.ask),
                    aggressor=aggressor_from_mdp(record.aggressor),
                    bid_size_delta=_optional_float(record.bid_size_delta),
                    ask_size_delta=_optional_float(record.ask_size_delta),
                )
            )
        for expiry, trades in rows.items():
            self._stamps[expiry] = [t.timestamp for t in trades]
        return rows

    def trades(
        self, start: datetime, end: datetime, expiry: date
    ) -> Sequence[OptionTrade]:
        if self._by_expiry is None:
            self._by_expiry = self._load()
        rows = self._by_expiry.get(expiry)
        if not rows:
            return ()
        stamps = self._stamps[expiry]
        lo = bisect.bisect_right(stamps, start)
        hi = bisect.bisect_right(stamps, end)
        return rows[lo:hi]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class SyntheticTradeFeed:
    """A generated tape, for exercising the classification path.

    What it produces per pull, per expiry: a handful of executions spread
    over the strikes near spot, priced against a synthetic quote, some
    carrying an MDP 3.0 aggressor flag and the rest left for the Lee-Ready
    rules -- so a generated backtest walks the whole precedence chain rather
    than only its first link.

    The **direction** of the generated flow is not free.  It is derived from
    the same per-expiry ``call_share`` draw ``SyntheticOpenInterest`` uses,
    mapped so that a call-heavy chain is one where customers were selling
    calls to dealers (dealers long, positive GEX) and a put-heavy one is
    where customers were buying puts from them (dealers short, negative
    GEX).  That is not a claim about how markets work -- it is the *minimum*
    consistency requirement for a harness: a generator whose tape said the
    opposite of its own open interest would make the measured sign fight the
    prior at every strike and the confidence gate fire all day, and the
    resulting backtest would be measuring the disagreement between two
    generators rather than any part of the system.
    """

    def __init__(
        self, cfg: FlowConfig, source: RiskSource, oi_provider=None,
        anchor: float | None = None,
    ):
        self.cfg = cfg
        self.source = source
        #: The open-interest generator to stay consistent with. Optional so
        #: the feed can be built standalone in a test; without it the flow
        #: is balanced and every strike stays near its prior.
        self.oi = oi_provider
        #: Strike level to centre the generated tape on when there is no OI
        #: generator to take it from.
        self.anchor = anchor

    def _call_share(self, expiry: date) -> float:
        if self.oi is None or not hasattr(self.oi, "call_share"):
            return 0.5
        return float(self.oi.call_share(expiry))

    def _anchor(self, expiry: date) -> float | None:
        """Where to centre the generated strikes for this expiry.

        Taken from the level ``SyntheticOpenInterest`` froze for that expiry,
        which is what its open interest is built around: a tape centred
        anywhere else would put its measured signs at strikes the generated
        book carries no open interest at, and so measure nothing.  That
        level is set the first time the expiry's open interest is read,
        which is why the strategy reads open interest before draining the
        tape -- and why an expiry whose surface has not been built yet
        yields no trades rather than an arbitrarily centred handful.
        """
        if self.oi is not None:
            frozen = getattr(self.oi, "_anchors", {}).get(expiry)
            if frozen is not None:
                return float(frozen)
        return self.anchor

    def trades(
        self, start: datetime, end: datetime, expiry: date
    ) -> Sequence[OptionTrade]:
        if end <= start:
            return ()
        share = self._call_share(expiry)
        # Map a call share in [0, 1] onto a dealer lean in [-1, 1]: 0.5 is
        # balanced flow, above it dealers are being sold calls, below it
        # they are being bought puts from.
        lean = max(-1.0, min(1.0, (share - 0.5) * 4.0))
        anchor = self._anchor(expiry)
        if anchor is None:
            log.debug(
                "no anchor for %s yet; the generated tape starts once the "
                "open-interest surface for that expiry has been built", expiry
            )
            return ()

        step = self.source.strike_increment
        tick = self.source.option.tick_size
        per_bar = self.cfg.synthetic_contracts_per_bar
        strikes = [anchor + step * i for i in range(-4, 5)]
        span = (end - start).total_seconds() or 1.0

        trades: list[OptionTrade] = []
        for index, strike in enumerate(strikes):
            for right in (CALL, PUT):
                draw = _unit_draw(
                    self.cfg.synthetic_seed, expiry.toordinal(),
                    int(start.timestamp()), index, right,
                )
                size = max(round(per_bar / (2 * len(strikes)) * (0.5 + draw)), 1)
                # A dealer-long lean means customers are selling: the trade
                # goes off at the bid. The two rights lean opposite ways,
                # which is what makes the net read directional at all.
                dealer_long = lean if right == CALL else -lean
                sells = 0.5 + 0.5 * dealer_long  # share of size hitting the bid
                quote_mid = max(tick * 4.0, tick * (2 + index))
                bid, ask = quote_mid - tick, quote_mid + tick
                flagged = draw < self.cfg.synthetic_flagged_share
                for side_share, at_price, aggressor in (
                    (sells, bid, "sell"),
                    (1.0 - sells, ask, "buy"),
                ):
                    lots = round(size * side_share)
                    if lots <= 0:
                        continue
                    offset = (index + (right == PUT)) / (2.0 * len(strikes))
                    trades.append(
                        OptionTrade(
                            timestamp=start + (end - start) * min(offset, 0.999),
                            expiry=expiry,
                            strike=strike,
                            right=right,
                            price=at_price,
                            size=float(lots),
                            bid=bid,
                            ask=ask,
                            aggressor=aggressor if flagged else "unknown",
                        )
                    )
        trades.sort(key=lambda t: t.timestamp)
        log.debug(
            "synthetic tape: %d trades for %s over %.0fs", len(trades), expiry, span
        )
        return trades


def _unit_draw(*parts: object) -> float:
    """Deterministic pseudo-random in [0, 1), the same trick the OI uses."""
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / float(1 << 64)


def build_trade_feed(cfg, source: RiskSource, open_interest=None, tz=None):
    # ``tz`` localises a replayed tape whose timestamps carry no offset.
    # The strategy compares trade times against timezone-aware bar times,
    # so a naive tape has to be given a zone or refused -- comparing the
    # two raises, which the strategy would catch and log as a feed failure
    # once per bar while quietly classifying nothing.
    """Construct the feed named by ``cfg.flow.source``.

    ``ibkr`` is not constructible here -- it needs a live connection -- so
    the live runner builds it and a backtest refuses it loudly rather than
    quietly substituting a generated tape for a real one, which is the same
    rule the open-interest factory follows and for the same reason.
    """
    kind = cfg.flow.source.lower()
    if kind == "none":
        return NullTradeFeed()
    if kind == "csv":
        return CsvTradeFeed(cfg.flow, source, tz=tz)
    if kind == "synthetic":
        return SyntheticTradeFeed(cfg.flow, source, open_interest)
    if kind in ("ibkr", "databento"):
        raise ValueError(
            f"flow.source == {kind!r} needs a live connection; it is "
            "available in `deltahedger live`, not in a backtest. Use 'csv' to "
            "replay a real option tape historically."
        )
    raise ValueError(
        f"unknown flow source {cfg.flow.source!r}; use 'none', 'csv', "
        "'synthetic', 'ibkr' or 'databento'"
    )
