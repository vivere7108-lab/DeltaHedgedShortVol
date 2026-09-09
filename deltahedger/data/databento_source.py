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

``DatabentoTradeFeed``
    Not an open-interest provider at all: it hands ``deltahedger.flow``
    the individual executions, each carrying MDP 3.0's own aggressor side,
    so the dealer *sign* at each strike is measured rather than assumed.
    This is the one live path on which rule 1 of the classification chain
    actually fires -- IBKR relays no aggressor flag, so the IBKR feed falls
    back to Lee-Ready.  The session already decodes every ``TradeMsg`` for
    the flow-adjusted provider above; this only keeps them.

    The two are different quantities off the same messages and it is worth
    being explicit about which is which.  The flow-adjusted provider uses
    aggressor side to guess how much open interest has been *added* at a
    strike since the print.  The trade feed uses it to say who ended up
    *holding* what -- the dealer is the resting side.  One adjusts a
    magnitude, the other decides a sign, and neither substitutes for the
    other.

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

Every subscription asks for **intraday replay** (``start=0``: everything
the gateway still holds, up to a day). The open-interest print is a
once-a-session message, published after overnight clearing; a session
subscribed from "now" -- which is what a runner restarted at 09:49 does --
would never see it, and the strategy would read zero GEX, stand aside all
day, and look in the log exactly like a market reading neutral. Replay
hands the morning's print, the definitions and the day's tape to a
process that started after them. Databento only honours ``start`` on
subscriptions made before the session starts, so a root that turns up
later -- tomorrow's series, at the roll -- is handled by stopping the
session and opening a new one with every root subscribed from the start
again (``_restart_with_replay``). The replay covers the gap that costs;
the flow and the buffered tape are cleared first so nothing the old
session already counted is counted again when it is replayed, while the
open-interest prints and definitions are kept, because a replayed print
simply overwrites the one held and the read never goes dark.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..config import Config, _parse_time
from ..flow import BUY, CALL, PUT, SELL, UNKNOWN, OptionTrade
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
        #: instrument_id -> (open interest, ns timestamp the print covers
        #: trades up to: the close of its trade date, not its ts_ref).
        self._oi: dict[int, tuple[float, int]] = {}
        #: instrument_id -> signed aggressor volume since that moment.
        self._flow: dict[int, float] = {}
        #: instrument_id -> [(ts_event, signed size)] for trades seen before
        #: any print for that instrument. A replay delivers the day's tape
        #: before its statistics, and a live session can see a trade before
        #: the morning's print lands; either way the trade is not yet
        #: attributable to "since the print" until the print says what it
        #: covers. Folded into ``_flow`` when it arrives, bounded meanwhile.
        self._pending: dict[int, list[tuple[int, float]]] = {}
        self._close_time = _parse_time(cfg.databento.trade_date_close)
        self._tz = ZoneInfo(source.timezone)
        self._root_by_expiry: dict[date, str] = {}
        self._subscribed_roots: set[str] = set()
        self._live = None
        self._key: str | None = None
        self._started = False
        self._warned_expiries: set[date] = set()
        #: Executions kept for ``DatabentoTradeFeed``, per expiry, oldest
        #: first. Off unless something asks for them: the session decodes
        #: every trade anyway for the flow-adjusted provider, but keeping
        #: them costs memory that a walk not measuring the dealer sign has
        #: no use for.
        self._capture_trades = False
        self._trades: dict[date, list[OptionTrade]] = {}
        self._dropped_trades = 0

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
        self._key = key
        self._live = self._new_client()

    def _new_client(self):
        """A fresh ``databento.Live`` with this session's callbacks attached.

        One place, because the session builds a client twice: at ``start``
        and again in ``_restart_with_replay`` when a new root has to be
        subscribed with replay after the first client is already running.
        """
        # Imported here, after the key check in ``start``, so a missing key
        # is reported plainly even when the optional `databento` extra
        # isn't installed.
        import databento as db

        live = db.Live(
            key=self._key,
            reconnect_policy="reconnect" if self.cfg.reconnect else "none",
        )
        live.add_callback(self._on_record, self._on_error)
        return live

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

    #: Seconds to wait for the old client to close in a restart before the
    #: new one is started on top of it.
    RESTART_TIMEOUT = 10.0

    def _subscribe_root(self, root: str) -> None:
        if self._started:
            # Databento honours ``start`` only on subscriptions made before
            # the session starts, and this root's print is already in the
            # past -- so the session is rebuilt with every root replayed
            # rather than this one subscribed from now and left blind.
            self._restart_with_replay(root)
            return
        self._subscribe(self._live, root)
        self._live.start()
        self._started = True
        log.info(
            "databento subscribed to %s.OPT with intraday replay (dataset=%s)",
            root, self.cfg.dataset,
        )

    def _subscribe(self, live: Any, root: str) -> None:
        parent = f"{root}.OPT"
        for schema in ("definition", "statistics", "trades"):
            live.subscribe(
                dataset=self.cfg.dataset,
                schema=schema,
                stype_in="parent",
                symbols=parent,
                # Everything the gateway still holds, up to a day: the
                # morning's open-interest print above all, which a session
                # opened after it would otherwise never see.
                start=0,
            )

    def _restart_with_replay(self, root: str) -> None:
        """Replace the running client with one that replays every root.

        The old client is stopped *before* the caches are touched, so a
        trade it delivers on the way out is either counted once by it or
        once by the replay, never by both.  What the replay will rebuild --
        the signed flow since each print and the buffered tape -- is
        cleared; what it will merely overwrite -- the prints and the
        definitions -- is kept, so ``rows`` never reads empty across the
        switch.  The replay then re-delivers everything from the gateway's
        buffer, the gap included.
        """
        old = self._live
        try:
            old.stop()
            old.block_for_close(timeout=self.RESTART_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - the old client is being discarded
            log.warning("databento: the old session did not close cleanly (%s)", exc)
        with self._lock:
            roots = sorted(self._subscribed_roots)
            self._flow.clear()
            self._pending.clear()
            self._trades.clear()
            self._dropped_trades = 0
        live = self._new_client()
        self._live = live
        for each in roots:
            self._subscribe(live, each)
        live.start()
        log.info(
            "databento session restarted with intraday replay for %s to add "
            "%s.OPT (dataset=%s)", ", ".join(f"{r}.OPT" for r in roots), root,
            self.cfg.dataset,
        )

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
        covers = self._print_covers_until(record.ts_ref)
        with self._lock:
            held = self._oi.get(record.instrument_id)
            self._oi[record.instrument_id] = (float(record.quantity), covers)
            if held is not None and held[1] == covers:
                # The same print again -- a correction, or a replay of one
                # already held. The flow counted since it still stands.
                return
            # A new print embeds every trade up to the close of its trade
            # date; flow accumulated against the old one no longer applies.
            # Trades seen before any print for this instrument are folded in
            # now that what the print covers is known.
            self._flow[record.instrument_id] = sum(
                signed
                for ts_event, signed in self._pending.pop(record.instrument_id, ())
                if ts_event > covers
            )

    def _print_covers_until(self, ts_ref: int) -> int:
        """The last nanosecond of trading an open-interest print embeds.

        ``ts_ref`` names the trade date the print describes, as midnight UTC
        of that date.  The print is struck after that date's close, so it
        contains the whole of that date's session -- and comparing trades
        against midnight, as an earlier revision did, counted the entire
        session before the print a second time.
        """
        trade_date = datetime.fromtimestamp(ts_ref / 1e9, tz=timezone.utc).date()
        close = datetime.combine(trade_date, self._close_time, tzinfo=self._tz)
        return int(close.timestamp() * 1e9)

    #: Per-instrument cap on trades held before a print arrives for it.
    MAX_PENDING_TRADES = 20_000

    #: Per-expiry cap on kept executions. A drain that stops happening --
    #: the poll loop wedged, the strategy standing aside all day -- must
    #: not turn into unbounded memory on a walk that runs for days. Past
    #: the cap the oldest are dropped and the count is reported, because
    #: silently keeping the *newest* would bias the measured sign towards
    #: whatever the last few minutes did.
    MAX_BUFFERED_TRADES = 20_000

    def enable_trade_capture(self) -> None:
        """Start keeping decoded executions for ``DatabentoTradeFeed``."""
        with self._lock:
            self._capture_trades = True

    def _on_trade(self, record: object) -> None:
        if self._capture_trades:
            self._capture(record)

        side = _aggressor_side(record.side)
        if side == BUY:
            signed = float(record.size)
        elif side == SELL:
            signed = -float(record.size)
        else:
            return  # no aggressor named: an implied or administrative fill
        with self._lock:
            base = self._oi.get(record.instrument_id)
            if base is None:
                # No print yet for this instrument, so nothing says what
                # "since the print" means. Hold it until one does.
                kept = self._pending.setdefault(record.instrument_id, [])
                kept.append((record.ts_event, signed))
                if len(kept) > self.MAX_PENDING_TRADES:
                    del kept[: len(kept) - self.MAX_PENDING_TRADES]
                return
            _, covers = base
            if record.ts_event <= covers:
                return  # inside the print's own trade date: already counted
            self._flow[record.instrument_id] = (
                self._flow.get(record.instrument_id, 0.0) + signed
            )

    def _capture(self, record: object) -> None:
        """Keep one execution for the dealer-sign measurement.

        Deliberately *before* the two early returns the flow-adjusted
        provider makes below.  Those are right for adjusting an open-
        interest print -- a trade that predates the print is already inside
        it, and a strike with no print yet has nothing to adjust -- and
        wrong here.  Who took which side of an execution is evidence about
        dealer positioning whether or not there is an open-interest figure
        at that strike to attach it to, and discarding it would make the
        measured sign depend on the arrival order of an unrelated message.
        """
        with self._lock:
            definition = self._definitions.get(record.instrument_id)
        if definition is None:
            return  # no strike/right yet; the definition message is late
        side = _aggressor_side(record.side)
        trade = OptionTrade(
            timestamp=datetime.fromtimestamp(
                record.ts_event / 1e9, tz=timezone.utc
            ),
            expiry=definition.expiry,
            strike=definition.strike,
            right=definition.right,
            price=float(record.pretty_price),
            size=float(record.size),
            # MDP 3.0 states the aggressor outright, so rule 1 of the
            # classification chain resolves this and the Lee-Ready rules
            # never run. No quote is attached because none is needed and
            # the trades schema does not carry one -- attaching a stale
            # top-of-book from another subscription would add a worse
            # answer underneath a better one.
            aggressor=side,
        )
        with self._lock:
            kept = self._trades.setdefault(definition.expiry, [])
            kept.append(trade)
            if len(kept) > self.MAX_BUFFERED_TRADES:
                overflow = len(kept) - self.MAX_BUFFERED_TRADES
                del kept[:overflow]
                self._dropped_trades += overflow

    # -- reads --------------------------------------------------------------

    def take_trades(
        self, expiry: date, start: datetime, end: datetime
    ) -> list[OptionTrade]:
        """Executions in ``(start, end]``, removing what is now in the past.

        Half-open, like every other feed: a trade folded into the dealer
        position twice would move a strike's sign with nothing downstream
        able to see that it had.  Anything at or before ``end`` is dropped
        on the way out, because a window is only ever asked for once and
        keeping it would only grow the buffer.
        """
        with self._lock:
            rows = self._trades.get(expiry)
            if not rows:
                return []
            kept = [t for t in rows if start < t.timestamp <= end]
            self._trades[expiry] = [t for t in rows if t.timestamp > end]
            dropped, self._dropped_trades = self._dropped_trades, 0
        if dropped:
            log.warning(
                "databento: dropped %d buffered executions for %s before they "
                "were read (buffer cap %d) -- the measured dealer sign for "
                "that expiry is missing them",
                dropped, expiry, self.MAX_BUFFERED_TRADES,
            )
        return kept

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


def _aggressor_side(side: object) -> str:
    """MDP 3.0's aggressor side, as ``deltahedger.flow`` spells it.

    Databento reports the side that *initiated* the trade: ``Bid`` for a
    buy aggressor, ``Ask`` for a sell aggressor, and ``None`` where the
    match names no aggressor (an implied or administrative fill).  That
    last case is left ``UNKNOWN`` and falls through to the rest of the
    classification chain rather than being guessed at.

    This is the single definition of that mapping.  ``_on_trade`` above
    uses the same convention to sign its open-interest adjustment, and two
    copies of a direction this load-bearing is two chances to invert one.
    """
    import databento_dbn as dbn

    if side == dbn.Side.BID:
        return BUY
    if side == dbn.Side.ASK:
        return SELL
    return UNKNOWN


class DatabentoTradeFeed:
    """The option tape off MDP 3.0, carrying its own aggressor flag.

    An ``OptionTradeFeed`` (see ``deltahedger.flow``) reading the executions
    the shared ``DatabentoSession`` has already decoded.  Construction turns
    capture on; before that the session throws each trade away once it has
    adjusted its open-interest figure with it.

    This is the feed the flow layer was written for.  Every other live path
    infers the aggressor -- IBKR relays no flag, so it runs Lee-Ready
    against the prevailing quote -- and inference has an error rate that is
    worse for wide quotes and thin books and is not symmetric between the
    two sides.  Here the exchange states the aggressor in the trade summary
    message and rule 1 resolves essentially the whole tape, which is
    visible as ``aggressor_flag`` dominating ``DealerFlowBook.rule_counts``.

    It subscribes nothing of its own: the roots are already subscribed by
    whichever open-interest provider is running, so a walk wanting measured
    signs off this feed needs ``data.open_interest`` on one of the Databento
    providers too.  ``build_trade_feed`` says so rather than silently
    delivering an empty tape.
    """

    def __init__(self, session: DatabentoSession, connection: Any):
        self._session = session
        self._connection = connection
        session.enable_trade_capture()

    def trades(
        self, start: datetime, end: datetime, expiry: date
    ) -> list[OptionTrade]:
        # Same lazy subscription the open-interest providers make: an
        # expiry entering the blend has to have its root resolved and
        # subscribed before anything arrives for it.
        self._session.ensure_subscribed(expiry, self._connection)
        return self._session.take_trades(expiry, start, end)


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
