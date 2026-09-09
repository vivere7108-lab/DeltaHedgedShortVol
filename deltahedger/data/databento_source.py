"""Open interest read live off CME's MDP 3.0 feed, via Databento.

Two providers, both producing the same ``StrikeOpenInterest`` list the
synthetic, CSV and IBKR providers do -- see ``data.openinterest`` -- so
``GexCalculator`` cannot tell them apart.

``DatabentoOpenInterestProvider``
    The exchange's open interest, read directly off the feed instead of
    through IBKR's relay. This is **not** a fix for staleness: open
    interest is an exchange-computed, once-a-session figure everywhere --
    CME publishes it after overnight clearing, and no feed, this one
    included, makes it move intraday. What a direct read buys over IBKR's
    generic tick 101 is not freshness; it is fewer hops between the
    exchange and the number this system trades on.

``DatabentoFlowAdjustedOpenInterestProvider``
    The same base print, plus cumulative signed trade volume on each
    instrument since that print arrived -- a same-day proxy, not a
    measurement. See its docstring for exactly what it does and does not
    claim.

There is no single parent symbol for this instrument family
--------------------------------------------------------------
Confirmed empirically, 2026-09-08: Databento's parent-symbol grouping
"ES.OPT" resolves to **zero** instruments. CME lists ES's 0DTE weekly
options under a *different Globex root per weekday* -- the series traded
that day was root "E2B", not "ES" (`Historical.symbology.resolve` on
"ES.OPT" returned nothing; on "E2B.OPT" it returned 1,604 instruments
matching IBKR's own qualified contracts symbol for symbol). A hardcoded
weekday-to-root table would work today and silently go stale the day CME
changes the rotation -- exactly the kind of un-checked assumption this
system elsewhere refuses to bake in (see the README's "Known
approximations"). So this module does not guess the root: it asks IBKR,
which already resolves it correctly on every qualified contract (that is
where the "E2B" above came from in the first place -- `doctor`'s log). See
``DatabentoSession.ensure_subscribed``.

Session model
-------------
One ``DatabentoSession`` owns a single ``databento.Live`` connection and
feeds both providers, across as many CME roots as the walk trades across
its lifetime (today's series and, in the roll window, tomorrow's). Each
newly-seen expiry triggers one IBKR contract qualification to learn its
root, then one additional Databento subscription for that root; both are
cached, so a root is only resolved and subscribed once no matter how many
times ``open_interest`` is polled for the expiries that share it. Databento
runs the connection's I/O on a background thread it manages internally
(see ``databento.Live``'s docstring); this module's job is to keep the
caches consistent under a lock while that thread writes to them and the
live runner's poll loop reads them.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from ..config import Config
from ..gex import StrikeOpenInterest
from ..instruments import RiskSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Definition:
    strike: float
    right: str  # "C" | "P"
    expiry: date


class DatabentoSession:
    """Owns the live connection; both providers read its caches."""

    def __init__(self, cfg: Config, source: RiskSource):
        self.cfg = cfg.databento
        self.source = source
        self._lock = threading.Lock()
        self._definitions: dict[int, _Definition] = {}
        self._oi: dict[int, tuple[float, int]] = {}  # instrument_id -> (qty, ts_ref)
        self._flow: dict[int, float] = {}  # instrument_id -> signed qty since ts_ref
        self._root_by_expiry: dict[date, str] = {}
        self._subscribed_roots: set[str] = set()
        self._live = None
        self._started = False
        self._warned_expiries: set[date] = set()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Validate credentials and prepare the connection.

        Nothing is subscribed yet: there is no single parent symbol for
        this instrument family (see the module docstring), so the actual
        TCP session opens lazily on the first ``ensure_subscribed`` call,
        once an expiry -- and therefore a root -- is actually known.
        """
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(
                f"set {self.cfg.api_key_env} to a Databento API key with "
                f"{self.cfg.dataset} entitlement"
            )
        # Imported here, after the key check, so a missing key is reported
        # plainly even when the optional `databento` extra isn't installed.
        import databento as db

        self._live = db.Live(
            key=key,
            reconnect_policy="reconnect" if self.cfg.reconnect else "none",
        )
        self._live.add_callback(self._on_record, self._on_error)

    def close(self) -> None:
        if self._live is not None and self._started:
            self._live.stop()
        self._live = None
        self._started = False

    def ensure_subscribed(self, expiry: date, connection: Any) -> None:
        """Make sure the root that serves ``expiry`` is subscribed.

        ``connection`` is the live runner's current ``IbkrConnection`` --
        used only to qualify one contract and read back its
        ``tradingClass``, which is the root Databento needs. Resolved once
        per expiry and cached; a second call for an expiry that shares its
        week's root (or a repeat call for the same expiry) does nothing.

        ``databento.parent_symbol`` in config, if set, skips IBKR
        resolution entirely and is used for every expiry -- for a risk
        source that does not have this weekly-root complexity.
        """
        with self._lock:
            if expiry in self._root_by_expiry:
                return

        if self.cfg.parent_symbol:
            root = self.cfg.parent_symbol.removesuffix(".OPT")
        else:
            strike = (
                round(connection.future_price() / connection.source.strike_increment)
                * connection.source.strike_increment
            )
            contract = connection.option_contract(expiry, strike, "C")
            root = getattr(contract, "tradingClass", None)
            if not root:
                raise RuntimeError(
                    f"IBKR returned no tradingClass qualifying {expiry} "
                    f"{strike:g}C; cannot resolve the Databento root for "
                    "this expiry"
                )

        with self._lock:
            self._root_by_expiry[expiry] = root
            if root in self._subscribed_roots:
                return
            self._subscribed_roots.add(root)
        self._subscribe_root(root)

    def _subscribe_root(self, root: str) -> None:
        parent = f"{root}.OPT"
        for schema in ("definition", "statistics", "trades"):
            self._live.subscribe(
                dataset=self.cfg.dataset,
                schema=schema,
                stype_in="parent",
                symbols=parent,
            )
        # .start() may only be called once, after at least one subscription
        # exists; later roots just add more subscriptions to the same
        # already-running session.
        if not self._started:
            self._live.start()
            self._started = True
        log.info("databento subscribed to %s (dataset=%s)", parent, self.cfg.dataset)

    # -- record handling ----------------------------------------------------

    def _on_error(self, exc: Exception) -> None:
        log.warning("databento live session error: %s", exc)

    def _on_record(self, record: object) -> None:
        import databento_dbn as dbn

        if isinstance(record, dbn.InstrumentDefMsg):
            self._on_definition(record)
        elif isinstance(record, dbn.StatMsg):
            self._on_stat(record)
        elif isinstance(record, dbn.TradeMsg):
            self._on_trade(record)

    def _on_definition(self, record: object) -> None:
        import databento_dbn as dbn

        if record.instrument_class == dbn.InstrumentClass.CALL:
            right = "C"
        elif record.instrument_class == dbn.InstrumentClass.PUT:
            right = "P"
        else:
            return
        definition = _Definition(
            strike=float(record.pretty_strike_price),
            right=right,
            expiry=record.pretty_expiration.date(),
        )
        with self._lock:
            self._definitions[record.instrument_id] = definition

    def _on_stat(self, record: object) -> None:
        import databento_dbn as dbn

        if record.stat_type != dbn.StatType.OPEN_INTEREST:
            return
        if record.quantity == dbn.UNDEF_STAT_QUANTITY:
            return
        with self._lock:
            self._oi[record.instrument_id] = (float(record.quantity), record.ts_ref)
            # A fresh print already embeds every trade up to ts_ref; flow
            # accumulated against the old print no longer applies.
            self._flow[record.instrument_id] = 0.0

    def _on_trade(self, record: object) -> None:
        import databento_dbn as dbn

        if record.side == dbn.Side.BID:
            signed = float(record.size)
        elif record.side == dbn.Side.ASK:
            signed = -float(record.size)
        else:
            return
        with self._lock:
            base = self._oi.get(record.instrument_id)
            if base is None:
                return  # no OI print yet for this instrument
            _, ts_ref = base
            if record.ts_event <= ts_ref:
                return  # this trade predates the current print
            self._flow[record.instrument_id] = (
                self._flow.get(record.instrument_id, 0.0) + signed
            )

    # -- reads --------------------------------------------------------------

    def rows(self, expiry: date, adjusted: bool) -> list[StrikeOpenInterest]:
        """Every strike with a known OI print for ``expiry``.

        A strike with a definition but no OI print yet is left out rather
        than reported as zero -- ``CsvOpenInterest`` makes the same choice
        for the same reason: a fabricated zero and "no data yet" mean
        different things to the confidence gate.
        """
        with self._lock:
            by_strike: dict[float, dict[str, float]] = {}
            matched = 0
            for instrument_id, definition in self._definitions.items():
                if definition.expiry != expiry:
                    continue
                base = self._oi.get(instrument_id)
                if base is None:
                    continue
                matched += 1
                quantity, _ = base
                if adjusted:
                    quantity = max(quantity + self._flow.get(instrument_id, 0.0), 0.0)
                slot = by_strike.setdefault(definition.strike, {"C": 0.0, "P": 0.0})
                slot[definition.right] = quantity
            root = self._root_by_expiry.get(expiry)

        if matched == 0 and expiry not in self._warned_expiries:
            self._warned_expiries.add(expiry)
            log.warning(
                "databento: no open interest for expiry %s yet (root %s, %d "
                "instruments known for it total) -- if this persists past "
                "the first few minutes of a session, check that root is "
                "actually what CME lists this expiry's series under",
                expiry, root, len(self._definitions),
            )

        return [
            StrikeOpenInterest(strike=strike, call_oi=sides["C"], put_oi=sides["P"])
            for strike, sides in sorted(by_strike.items())
        ]


class DatabentoOpenInterestProvider:
    """The exchange's open interest, read directly off MDP 3.0.

    See the module docstring for what this does and does not fix relative
    to the IBKR path: it is a more direct read of the same once-a-session
    figure, not an intraday one. ``connection`` is the live runner's
    current ``IbkrConnection``, used only to resolve which CME root serves
    a given expiry -- see ``DatabentoSession.ensure_subscribed``.
    """

    def __init__(self, session: DatabentoSession, connection: Any):
        self._session = session
        self._connection = connection

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        self._session.ensure_subscribed(expiry, self._connection)
        return self._session.rows(expiry, adjusted=False)


class DatabentoFlowAdjustedOpenInterestProvider:
    """The exchange's open interest, plus signed flow since the print.

    A proxy for same-day positioning change, not a measurement of it.
    Real open interest cannot update intraday -- see
    ``DatabentoOpenInterestProvider`` -- so this adds cumulative
    aggressor-side trade volume on each instrument since its last OI print:
    buyer-initiated trades add, seller-initiated trades subtract. A trade's
    aggressor side says which side was more eager to trade, not whether the
    trade opened or closed a position, so on a day with a lot of two-sided
    unwinding this over-states the change, and on a day that is mostly
    opening flow it under-states it. Floored at zero: a claimed negative
    open interest is a bug in this provider, not a number to report.
    """

    def __init__(self, session: DatabentoSession, connection: Any):
        self._session = session
        self._connection = connection

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        self._session.ensure_subscribed(expiry, self._connection)
        return self._session.rows(expiry, adjusted=True)
