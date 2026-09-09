"""Live / forward-testing runner.

Polls IBKR, synthesises a ``MarketBar`` from the current market, and hands
it to the same ``GexStraddleStrategy`` the backtest drives.  The strategy
does not know which runner it is under; that is what makes a forward test
evidence about the validated logic rather than about a second
implementation of it.

Differences from the backtest that are worth being explicit about:

  * bars are *polls*, not completed bars, so the strategy sees the market as
    of each poll rather than a settled OHLC;
  * fills come from the exchange and can be partial or missing entirely.
    The strategy books what came back rather than what was asked for, and
    squares a half-filled straddle before it records anything.  On top of
    that the book is checked against IBKR's own positions at connect
    (``_reconcile``, which adopts them) and then on the
    ``live.reconcile_seconds`` timer (``_verify_positions``, which halts
    entries on any disagreement).  That second check is the one that
    matters: between two connects the book is only this process's record
    of its own fills, and an order IBKR fills *after*
    ``IbkrExecution._send`` has given up waiting and cancelled it is
    invisible to that record;
  * the ATM implied vol comes from the live chain rather than a historical
    series;
  * open interest is the exchange's, read through
    ``IbkrOpenInterestProvider``, rather than generated.  A forward test
    with a generated OI surface would be measuring the generator, so the
    live path refuses to fall back to one;
  * **the loop does not stop at the bell.**  The 0DTE position is rolled
    into tomorrow's series a quarter of an hour before settlement and that
    position is carried overnight, so the runner keeps polling while
    anything is open, under the widened overnight band -- and honours a
    pre-market event blackout on the way.  This is the one thing the
    backtest cannot check: its bar sources are RTH-only, so in a backtest
    an overnight move arrives whole on the next session's first bar, and
    no hedge happens inside it.  The forward walk is where that part of
    the system is actually exercised.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from datetime import date, datetime

from ..broker.ibkr import (
    IbkrChainProvider,
    IbkrConnection,
    IbkrExecution,
    IbkrOpenInterestProvider,
    WhatIfMarginModel,
)
from ..config import Config
from ..data.base import MarketBar
from ..data.databento_source import (
    DatabentoFlowAdjustedOpenInterestProvider,
    DatabentoOpenInterestProvider,
    DatabentoSession,
)
from ..instruments import RiskSource
from ..portfolio import StraddlePosition
from ..sizing import build_margin_model
from ..strategy import GexStraddleStrategy
from .journal import JournallingStrategy, SessionJournal

log = logging.getLogger(__name__)

#: The regime an adopted position is booked under. It is not a GEX read --
#: this process never made one for it -- and giving it a real regime name
#: would put P&L the runner cannot attribute into a bucket that is supposed
#: to answer whether reading GEX paid.
ADOPTED = "adopted"


class ReconciliationError(RuntimeError):
    """The broker holds something the strategy cannot safely manage.

    Deliberately not a ``ConnectionError``: reconnecting cannot change what
    the account holds, so the reconnect loop must let this one out rather
    than retrying it forever against an unchanging answer.
    """


@dataclass(frozen=True)
class _PositionRow:
    """One position the broker reports, in the terms the book works in."""

    quantity: int
    entry_price: float
    expiry: date | None = None
    strike: float = 0.0
    right: str = ""


def _parse_expiry(value: str) -> date:
    """IBKR's ``YYYYMMDD`` contract month, as a date.

    A row we cannot parse is a reconciliation failure rather than a
    programming one: it has to surface as ``ReconciliationError`` so the
    reconnect loop lets it out. Raising anything else would put it back in
    the retry path, where reconnecting cannot change the answer and the
    runner would spin on it exactly as it used to on the old refusal.
    """
    text = str(value)[:8]
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError as exc:
        raise ReconciliationError(
            f"cannot read an expiry from the contract month {value!r}: {exc}"
        ) from exc


def _straddle_from_rows(
    rows: list[_PositionRow], source: RiskSource, adopted_at: datetime
) -> StraddlePosition | None:
    """The straddle these option rows describe, or ``None`` when flat.

    Raises ``ReconciliationError`` for anything that is not one matched
    straddle.  The strategy has exactly one shape of option position and no
    way to represent a lone leg, two strikes or two expiries, so adopting
    such a book would mean hedging a delta it had computed wrongly -- which
    is the failure the old blanket refusal was reaching for.  The change
    here is only that a *matched* straddle is now adopted instead of
    refused along with everything else.
    """
    if not rows:
        return None

    def described() -> str:
        return ", ".join(
            f"{row.quantity:+d} {row.expiry} {row.strike:g}{row.right}"
            for row in sorted(rows, key=lambda r: (str(r.expiry), r.strike, r.right))
        )

    expiries = {row.expiry for row in rows}
    strikes = {row.strike for row in rows}
    rights = {row.right for row in rows}
    if len(rows) != 2 or len(expiries) != 1 or len(strikes) != 1 or rights != {"C", "P"}:
        raise ReconciliationError(
            f"the account holds {len(rows)} {source.option.symbol} option "
            f"position(s) that are not one matched straddle ({described()}). "
            "The strategy can only represent a call and a put on one strike and "
            "one expiry, so this cannot be adopted -- close it, or restart once "
            "flat."
        )

    call = next(row for row in rows if row.right == "C")
    put = next(row for row in rows if row.right == "P")
    if call.quantity != put.quantity:
        raise ReconciliationError(
            f"the {source.option.symbol} legs are not the same size "
            f"({described()}); that is a naked option, not a straddle, and "
            "adopting it would leave the hedger working from the wrong delta."
        )

    return StraddlePosition(
        strike=call.strike,
        expiry=call.expiry,
        quantity=call.quantity,
        call_entry=call.entry_price,
        put_entry=put.entry_price,
        # The real entry time is not recoverable from a position row. This
        # is when the book picked it up, which is also when its P&L
        # baselines start, so the two agree rather than quietly disagreeing.
        entry_time=adopted_at,
        regime=ADOPTED,
    )


class LiveRunner:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.source = cfg.source
        self.connection = IbkrConnection(cfg, self.source)
        self.strategy: GexStraddleStrategy | None = None
        self.journal = (
            SessionJournal(cfg.live.journal_dir) if cfg.live.journal else None
        )
        self._driver = None
        self._cycles = 0
        self._stop = False
        # Independent of the IBKR connection's reconnect cycle -- Databento
        # manages its own session, so it is started once and outlives any
        # number of IBKR gateway restarts.
        self._databento: DatabentoSession | None = None

    def request_stop(self, *_: object) -> None:
        log.info("stop requested; finishing the current cycle")
        self._stop = True

    def run(self, max_cycles: int | None = None) -> GexStraddleStrategy:
        """Poll until stopped, surviving disconnections.

        The outer loop exists because IBKR force-restarts the gateway once a
        day and drops every API connection with it.  Without it a forward
        walk goes quiet after its first night and keeps logging exceptions
        into a dead socket, which looks exactly like a working run until you
        read the log.

        Each reconnection rebuilds the strategy and re-reconciles against
        the broker's positions rather than resuming the in-memory book: the
        book may be minutes or hours stale by then, and the broker is the
        only thing that knows what is actually open.  It is also the only
        thing that can say so -- the journal records fills but not which
        strike, right or expiry they were on, so it cannot rebuild a
        position; what it carries across the gap is the decision history,
        not the book.

        A ``ReconciliationError`` is deliberately not caught here.  It is
        not a connection failure, and retrying it would mean asking the
        same question of the same account and getting the same answer
        until the process is killed.
        """
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        backoff = self.cfg.live.reconnect_backoff_seconds
        attempts = 0

        try:
            while not self._stop:
                cycles_before = self._cycles
                try:
                    self._run_connected(max_cycles)
                    backoff = self.cfg.live.reconnect_backoff_seconds
                    attempts = 0
                except ReconciliationError:
                    # Not a connection failure, so the reconnect loop must
                    # not treat it as one. Reconnecting cannot change what
                    # the account holds: every retry would fail identically,
                    # forever at the shipped max_reconnect_attempts of null,
                    # while whatever is open goes unhedged the whole time.
                    # Dying here is loud, and a supervisor restart is honest.
                    log.error(
                        "cannot reconcile against the broker; stopping rather "
                        "than retrying an answer that will not change"
                    )
                    raise
                except Exception as exc:  # noqa: BLE001 - the point is to survive it
                    if self._stop:
                        break
                    if not self.cfg.live.reconnect:
                        raise
                    # A session that actually polled before dropping was a
                    # healthy one, so the failure budget starts over. Without
                    # this, a walk that reconnects cleanly every night still
                    # exhausts max_reconnect_attempts after that many days and
                    # dies for having worked.
                    if self._cycles > cycles_before:
                        attempts = 0
                        backoff = self.cfg.live.reconnect_backoff_seconds
                    attempts += 1
                    limit = self.cfg.live.max_reconnect_attempts
                    if limit is not None and attempts >= limit:
                        log.error(
                            "giving up after %d consecutive connection failures: %s",
                            attempts, exc,
                        )
                        raise
                    log.warning(
                        "connection lost (%s); reconnecting in %.0fs (attempt %d%s)",
                        exc, backoff, attempts,
                        f" of {limit}" if limit else "",
                    )
                    self._sleep(backoff)
                    backoff = min(
                        backoff * 2.0, self.cfg.live.max_reconnect_backoff_seconds
                    )
                    continue

                # A clean return means the cycle budget ran out or we were
                # asked to stop -- neither is a reason to reconnect.
                break
        finally:
            # Independent of the IBKR connection, so it needs its own
            # shutdown regardless of which path out of the loop was taken.
            if self._databento is not None:
                self._databento.close()

        if self.strategy is None:
            raise RuntimeError("the runner never established a session")
        if self.journal is not None:
            log.info("journal written to %s (%s)", self.journal.directory,
                     ", ".join(f"{n} {k}" for k, n in self.journal.counts().items()))
        return self.strategy

    def _run_connected(self, max_cycles: int | None) -> None:
        """One connected session: connect, reconcile, poll until it ends."""
        with self.connection as conn:
            fallback = build_margin_model(
                self.cfg.sizing, self.source, self.cfg.risk_free_rate
            )
            margin_model = (
                WhatIfMarginModel(conn, fallback)
                if self.cfg.ibkr.use_whatif_margin
                else fallback
            )
            strategy = GexStraddleStrategy(
                self.cfg,
                self.source,
                margin_model,
                open_interest=self._open_interest_provider(conn),
            )
            self.strategy = strategy
            driver = strategy
            if self.journal is not None:
                driver = JournallingStrategy(strategy, self.journal)
            self._driver = driver

            execution = IbkrExecution(conn, self.cfg, dry_run=self.dry_run)
            chain_provider = IbkrChainProvider(conn, self.cfg)
            self._reconcile(conn)
            last_reconcile = time.monotonic()

            log.info(
                "live runner started on %s (%s), polling every %.1fs%s",
                conn.account,
                self.source.name,
                self.cfg.ibkr.poll_seconds,
                " [DRY RUN]" if self.dry_run else "",
            )

            last_heartbeat = time.monotonic()
            while not self._stop and (
                max_cycles is None or self._cycles < max_cycles
            ):
                if not conn.ib.isConnected():
                    raise ConnectionError("the IBKR API connection dropped")
                self._cycle(conn, chain_provider, execution)
                self._cycles += 1
                if self._stop:
                    break

                now = time.monotonic()
                every = self.cfg.live.reconcile_seconds
                if every is not None and now - last_reconcile >= every:
                    self._verify_positions(conn)
                    self._check_broker_margin(conn)
                    self._rebase_equity(conn)
                    last_reconcile = now
                if now - last_heartbeat >= self.cfg.live.heartbeat_seconds:
                    log.info(
                        "alive: %d cycles, %d events, %d fills",
                        self._cycles, len(strategy.events), len(strategy.fills),
                    )
                    last_heartbeat = now
                conn.ib.sleep(self.cfg.ibkr.poll_seconds)

            log.info("live runner stopped after %d cycles", self._cycles)

    def _open_interest_provider(self, conn):
        """Build the OI provider named by ``cfg.data.open_interest``.

        ``databento``/``databento_flow`` share one ``DatabentoSession``,
        started once and cached on ``self`` so it survives IBKR reconnects
        rather than tearing down and rebuilding a separate live connection
        every time the gateway does its daily restart.
        """
        kind = self.cfg.data.open_interest.lower()
        if kind == "ibkr":
            return IbkrOpenInterestProvider(conn, self.cfg)
        if kind in ("databento", "databento_flow"):
            if self._databento is None:
                self._databento = DatabentoSession(self.cfg, self.source)
                self._databento.start()
            if kind == "databento":
                return DatabentoOpenInterestProvider(self._databento, conn)
            return DatabentoFlowAdjustedOpenInterestProvider(self._databento, conn)
        raise ValueError(
            f"data.open_interest == {self.cfg.data.open_interest!r} is not a "
            "live source; use 'ibkr', 'databento' or 'databento_flow'"
        )

    def _now(self) -> datetime:
        """The exchange-local moment, from the strategy's own clock.

        Read through the strategy rather than from ``datetime`` directly so
        the tests that freeze the clock freeze this too.
        """
        assert self.strategy is not None
        return datetime.now(self.strategy.clock.tz)

    def _holding(self) -> bool:
        """Whether there is anything to hedge right now.

        The rolled position is open through the night and the runner has
        to keep polling to hedge it -- the widened overnight band is a
        *wider* band, not an absent hedger, and a gap through it is exactly
        what an unhedged straddle cannot survive.  When the book is flat
        there is nothing outside the session worth waking up for: entries
        are blocked by the entry window regardless, and the end-of-day
        roll happens inside the session.
        """
        book = self.strategy.portfolio if self.strategy else None
        if book is None:
            return False
        return book.straddle is not None or book.hedge.quantity != 0

    def _sleep(self, seconds: float) -> None:
        """Sleep in slices so a stop signal is not swallowed by a backoff."""
        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    # -- internals ------------------------------------------------------

    def _cycle(self, conn, chain_provider, execution) -> None:
        """One poll. Raises only when the connection itself is gone.

        A bad tick, a missing strike or a chain that will not qualify are
        all ordinary and must not end the run -- but they are logged rather
        than swallowed silently, because a poll that fails every time is
        indistinguishable from a working one in an empty log.
        """
        assert self.strategy is not None
        now = datetime.now(self.strategy.clock.tz)
        if not self.strategy.clock.in_session(now) and not self._holding():
            log.debug("outside the session at %s and flat; idling", now)
            return

        try:
            self._poll(conn, chain_provider, execution, now)
        except Exception:  # noqa: BLE001 - a bad poll must not kill the run
            if not conn.ib.isConnected():
                raise  # the outer loop reconnects
            log.exception("poll failed; continuing")

    def _poll(self, conn, chain_provider, execution, now: datetime) -> None:
        future_price = conn.future_price()
        atm_iv = self._atm_iv(conn, chain_provider, future_price, now)
        bar = MarketBar(
            timestamp=now,
            open=future_price,
            high=future_price,
            low=future_price,
            close=future_price,
            atm_iv=atm_iv,
        )
        state = self._driver.on_bar(bar, execution)
        gex = (
            f"{state.gex_total / 1e6:+,.0f}M" if state.gex_total is not None else "n/a"
        )
        flip = f"{state.gex_flip:,.1f}" if state.gex_flip is not None else "-"
        log.info(
            "%s%s%s F=%.2f IV=%.3f | GEX %s (%s->%s, flip %s) | straddle=%+d @ %s "
            "%s | hedge=%+d | net delta %+.1f (target %.1f +/- %.1f) | equity %s",
            now.strftime("%H:%M:%S"), "" if state.in_session else " [overnight]",
            " [event blackout]" if state.event_blackout else "",
            state.future, state.atm_iv,
            gex, state.gex_regime, state.confirmed_regime, flip,
            state.straddle_contracts,
            f"{state.strike:g}" if state.strike else "-",
            f"{state.days_to_expiry}DTE" if state.days_to_expiry is not None else "-",
            state.hedge_contracts, state.net_delta_units, self.cfg.hedge.target,
            state.band_half_width, f"${state.equity:,.0f}",
        )

    def _atm_iv(self, conn, chain_provider, future_price: float, now: datetime) -> float:
        """Read ATM implied vol off the live chain, on the traded series.

        Averaged across the call and the put at the money rather than taken
        from one right: a single stale leg moves the level enough to change
        what the whole book is marked at.

        The series asked for is the one the tenor policy selects -- or the
        one an open position is already on -- so the vol marking the book is
        read at the tenor the book is actually carrying, not at the front
        month's.
        """
        expiry = self.strategy._traded_expiry(now)
        if expiry is None:
            return self.cfg.data.default_atm_iv
        t = self.strategy.clock.time_to_expiry(now, expiry)
        try:
            straddle = chain_provider.straddle(future_price, expiry, t)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read ATM vol from the chain (%s); using default", exc)
            return self.cfg.data.default_atm_iv
        if straddle is None:
            return self.cfg.data.default_atm_iv
        return straddle.iv

    def _reconcile(self, conn) -> None:
        """Take the broker's positions as the book, at the start of a session.

        Starting a live session with a stale in-memory book is how a hedger
        ends up doubling a position, so the runner trusts the broker rather
        than itself.

        The hedge leg is adopted outright.  The option leg is adopted only
        when it is a *matched* straddle -- one expiry, one strike, a call
        and a put in the same signed size -- because that is the only shape
        the strategy can represent, and anything else genuinely is the
        half-known book the old refusal was written to avoid.

        Why the broker rather than the journal.  Everything that drives a
        decision survives the trip: the expiry and strike come off the
        contract, and ``avgCost`` divided by the multiplier recovers each
        leg's entry price, which is the entry premium both exit rules are
        written against.  What is lost is attribution only -- the entry
        time, the entry vol, and which GEX regime opened it -- so an
        adopted position is booked under the ``adopted`` regime rather than
        pretending to a read this process never made.

        One consequence worth stating: the P&L baselines start at
        adoption, so the long branch's stop measures the scalp from the
        restart rather than from the original entry.  That understates a
        position that has already moved, which is the safe direction for a
        stop and the wrong one for a target.
        """
        assert self.strategy is not None
        hedge_rows, option_rows = self._read_positions(conn)
        book = self.strategy.portfolio

        for row in hedge_rows:
            book.hedge.quantity = row.quantity
            book.hedge.avg_price = row.entry_price
            log.warning(
                "adopted an existing hedge position: %+d %s @ %.2f",
                row.quantity, self.source.hedge.symbol, row.entry_price,
            )

        straddle = _straddle_from_rows(option_rows, self.source, self._now())
        if straddle is not None:
            book.open_straddle(straddle)
            log.warning(
                "adopted an existing %+d %s %g straddle @ %.2f (C %.2f / P %.2f). "
                "P&L for it is measured from now, not from when it was opened.",
                straddle.quantity, straddle.expiry, straddle.strike,
                straddle.entry_premium, straddle.call_entry, straddle.put_entry,
            )
        if not hedge_rows and straddle is None:
            log.info("no existing positions to adopt")

        # After adoption, so the guard inside it sees the book the account
        # actually left us with. What decides whether the arithmetic is
        # exact is whether *the account* holds a straddle, not whether the
        # book happened to be empty a moment ago.
        self._rebase_equity(conn)

    def _verify_positions(self, conn) -> None:
        """Re-check the broker against the book, mid-session.

        Between two connects the book is only this process's record of its
        own fills, and that record can be wrong in the one direction that
        matters: an order the runner gave up waiting on, cancelled, and
        reported as unfilled can still be filled by the exchange.  The
        strategy then believes it is flat, sizes a fresh entry against the
        full budget, and does it again on the next poll.

        So this compares the two and, on any disagreement about the option
        leg, halts entries rather than trading on a number it cannot
        trust.  It does *not* silently re-adopt: a position that appeared
        without the runner placing it is exactly the moment a person should
        look, and quietly absorbing it would erase the only evidence that
        anything went wrong.  Hedging and the exits carry on, because
        whatever is open still has to be managed.
        """
        assert self.strategy is not None
        strategy = self.strategy
        if strategy.halted:
            return
        try:
            hedge_rows, option_rows = self._read_positions(conn)
            broker_straddle = _straddle_from_rows(option_rows, self.source, self._now())
        except ReconciliationError as exc:
            # Not a shape the strategy can hold, so it certainly is not the
            # one the book thinks it holds.
            strategy.halt_entries(str(exc))
            return

        book = strategy.portfolio
        broker_quantity = broker_straddle.quantity if broker_straddle else 0
        book_quantity = book.straddle.quantity if book.straddle else 0
        if broker_quantity != book_quantity:
            strategy.halt_entries(
                f"the broker reports {broker_quantity:+d} straddles and the book "
                f"holds {book_quantity:+d}. The book is this process's record of "
                "its own fills and something has filled outside it; entries stop "
                "until a person has looked."
            )
            return

        broker_hedge = hedge_rows[0].quantity if hedge_rows else 0
        if broker_hedge != book.hedge.quantity:
            # The hedge is adopted rather than halted on: it is a single
            # signed number with no shape to get wrong, and the band puts
            # it right on the next pass.
            log.warning(
                "hedge drift: the broker reports %+d %s, the book held %+d; "
                "taking the broker's",
                broker_hedge, self.source.hedge.symbol, book.hedge.quantity,
            )
            book.hedge.quantity = broker_hedge
            if hedge_rows:
                book.hedge.avg_price = hedge_rows[0].entry_price

    def _rebase_equity(self, conn) -> None:
        """Size against the account's own value, not against the config.

        ``starting_equity`` is a number in a YAML file, and the book's
        equity is that number plus whatever this *process* has realised.
        A reconnect rebuilds the strategy, so the realised part resets to
        zero -- after a drawdown and the nightly gateway restart, the
        strategy sizes its next entry as though the loss had not happened.

        Only done while the option book is flat, which is when it can be
        exact: ``NetLiquidation`` already values open positions at market,
        so re-basing on top of a position would count its open P&L twice
        -- once inside the broker's number and again in ``unrealised``.
        Flat, the only unrealised is the hedge leg's, and that is
        subtractable because the hedge has a mark. Entries only happen
        when flat, so sizing always sees the broker's figure.
        """
        assert self.strategy is not None
        book = self.strategy.portfolio
        if book.straddle is not None:
            return
        try:
            nav = conn.net_liquidation()
        except Exception as exc:  # noqa: BLE001 - never block on an account read
            log.warning("could not read the account value (%s); keeping %s",
                        exc, f"${book.starting_equity:,.0f}")
            return
        if nav is None:
            log.warning(
                "IBKR reported no NetLiquidation; sizing stays on the "
                "configured $%s", f"{book.starting_equity:,.0f}",
            )
            return

        hedge_mark = conn.hedge_price() if book.hedge.quantity else 0.0
        unrealised = book.hedge.unrealised(hedge_mark, self.source.hedge.multiplier)
        rebased = nav - book.realised_pnl + book.fees_paid - unrealised
        if abs(rebased - book.starting_equity) < 1.0:
            return
        log.info(
            "sizing against the account: NetLiquidation $%s (the book was "
            "carrying $%s)", f"{nav:,.0f}", f"{book.starting_equity:,.0f}",
        )
        book.starting_equity = rebased

    def _check_broker_margin(self, conn) -> None:
        """Compare what the broker is holding with what was budgeted.

        The one check that does not go through the margin model, and so
        the only one that can catch the model itself being wrong.  That is
        not hypothetical: a ``future_initial_margin`` carrying the micro
        contract's figure made every short straddle look a tenth as
        expensive as CME charges, and the book was sized several times too
        large with the sizing arithmetic, the entry log and the equity
        curve all internally consistent and all wrong.  Nothing derived
        from the model could have noticed.  This would have, within one
        reconcile interval, because it asks the account.
        """
        assert self.strategy is not None
        if self.strategy.halted:
            return
        try:
            values = conn.account_values()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read the account's margin (%s)", exc)
            return
        held = values.get("FullInitMarginReq")
        nav = values.get("NetLiquidation")
        if not held or not nav or nav <= 0:
            return

        limit = self.cfg.sizing.buying_power_pct * nav
        if held <= limit:
            return
        self.strategy.halt_entries(
            f"the broker is holding ${held:,.0f} of initial margin against a "
            f"${nav:,.0f} account -- {held / nav:.0%}, past the "
            f"{self.cfg.sizing.buying_power_pct:.0%} buying-power limit the book "
            "is sized to. Either the margin model understates what this "
            "position costs or something was opened outside it; both mean the "
            "next entry would be sized off a number that is not true."
        )

    def _read_positions(self, conn) -> tuple[list["_PositionRow"], list["_PositionRow"]]:
        """The account's positions in our two instruments, split by leg."""
        hedge_rows: list[_PositionRow] = []
        option_rows: list[_PositionRow] = []
        for position in conn.ib.positions(conn.account):
            contract = position.contract
            quantity = int(position.position)
            if quantity == 0:
                continue  # IBKR reports closed positions as zero rows
            if contract.secType == "FUT" and contract.symbol == self.source.hedge.symbol:
                hedge_rows.append(
                    _PositionRow(
                        quantity=quantity,
                        entry_price=float(position.avgCost)
                        / (self.source.hedge.multiplier or 1.0),
                    )
                )
            elif (
                contract.secType == "FOP"
                and contract.symbol == self.source.option.symbol
            ):
                try:
                    strike = float(contract.strike)
                except (TypeError, ValueError) as exc:
                    raise ReconciliationError(
                        f"cannot read a strike from {contract.strike!r}: {exc}"
                    ) from exc
                option_rows.append(
                    _PositionRow(
                        quantity=quantity,
                        entry_price=float(position.avgCost)
                        / (self.source.option.multiplier or 1.0),
                        expiry=_parse_expiry(contract.lastTradeDateOrContractMonth),
                        strike=strike,
                        right=str(contract.right).upper()[:1],
                    )
                )
        return hedge_rows, option_rows


def run_live(cfg: Config, dry_run: bool = False, max_cycles: int | None = None):
    return LiveRunner(cfg, dry_run=dry_run).run(max_cycles=max_cycles)
