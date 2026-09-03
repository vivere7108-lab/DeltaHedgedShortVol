"""Live / forward-testing runner.

Polls IBKR, synthesises a ``MarketBar`` from the current market, and hands
it to the same ``GexStraddleStrategy`` the backtest drives.  The strategy
does not know which runner it is under; that is what makes a forward test
evidence about the validated logic rather than about a second
implementation of it.

Differences from the backtest that are worth being explicit about:

  * bars are *polls*, not completed bars, so the strategy sees the market as
    of each poll rather than a settled OHLC;
  * fills come from the exchange and can be partial or missing entirely, so
    the position is reconciled against IBKR on every cycle;
  * the ATM implied vol comes from the live chain rather than a historical
    series;
  * open interest is the exchange's, read through
    ``IbkrOpenInterestProvider``, rather than generated.  A forward test
    with a generated OI surface would be measuring the generator, so the
    live path refuses to fall back to one;
  * **the loop does not stop at the bell.**  The tenor is multi-session, so
    the book is carried overnight and the runner keeps polling while
    anything is open, under the widened overnight band.  This is the one
    thing the backtest cannot check: its bar sources are RTH-only, so in a
    backtest an overnight move arrives whole on the next session's first
    bar, and no hedge happens inside it.  The forward walk is where that
    part of the system is actually exercised.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime

from ..broker.ibkr import (
    IbkrChainProvider,
    IbkrConnection,
    IbkrExecution,
    IbkrOpenInterestProvider,
    WhatIfMarginModel,
)
from ..config import Config
from ..data.base import MarketBar
from ..sizing import build_margin_model
from ..strategy import GexStraddleStrategy
from .journal import JournallingStrategy, SessionJournal

log = logging.getLogger(__name__)


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
        only thing that knows what is actually open.  The journal is what
        carries the record across the gap.
        """
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        backoff = self.cfg.live.reconnect_backoff_seconds
        attempts = 0

        while not self._stop:
            cycles_before = self._cycles
            try:
                self._run_connected(max_cycles)
                backoff = self.cfg.live.reconnect_backoff_seconds
                attempts = 0
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

            # A clean return means the cycle budget ran out or we were asked
            # to stop -- neither is a reason to reconnect.
            break

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
                open_interest=IbkrOpenInterestProvider(conn, self.cfg),
            )
            self.strategy = strategy
            driver = strategy
            if self.journal is not None:
                driver = JournallingStrategy(strategy, self.journal)
            self._driver = driver

            execution = IbkrExecution(conn, self.cfg, dry_run=self.dry_run)
            chain_provider = IbkrChainProvider(conn, self.cfg)
            self._reconcile(conn)

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
                if now - last_heartbeat >= self.cfg.live.heartbeat_seconds:
                    log.info(
                        "alive: %d cycles, %d events, %d fills",
                        self._cycles, len(strategy.events), len(strategy.fills),
                    )
                    last_heartbeat = now
                conn.ib.sleep(self.cfg.ibkr.poll_seconds)

            log.info("live runner stopped after %d cycles", self._cycles)

    def _holding(self) -> bool:
        """Whether there is anything to hedge right now.

        The tenor is multi-session, so the book is open through the night
        and the runner has to keep polling to hedge it -- the widened
        overnight band is a *wider* band, not an absent hedger, and a gap
        through it is exactly what an unhedged straddle cannot survive.
        When the book is flat there is nothing outside the session worth
        waking up for: entries are blocked by the entry window regardless.
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
        band = self.cfg.hedge.effective_band(state.in_session)
        log.info(
            "%s%s F=%.2f IV=%.3f | GEX %s (%s->%s, flip %s) | straddle=%+d @ %s "
            "%s | hedge=%+d | net delta %+.1f (target %.1f +/- %.1f) | equity %s",
            now.strftime("%H:%M:%S"), "" if state.in_session else " [overnight]",
            state.future, state.atm_iv,
            gex, state.gex_regime, state.confirmed_regime, flip,
            state.straddle_contracts,
            f"{state.strike:g}" if state.strike else "-",
            f"{state.days_to_expiry}DTE" if state.days_to_expiry is not None else "-",
            state.hedge_contracts, state.net_delta_units, self.cfg.hedge.target,
            band, f"${state.equity:,.0f}",
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
        """Adopt whatever positions IBKR already reports for our contracts.

        Starting a live session with a stale in-memory book is how a hedger
        ends up doubling a position, so the runner trusts the broker, not
        itself.
        """
        assert self.strategy is not None
        positions = conn.ib.positions(conn.account)
        hedge_symbol = self.source.hedge.symbol
        adopted = 0
        for position in positions:
            contract = position.contract
            if contract.secType == "FUT" and contract.symbol == hedge_symbol:
                self.strategy.portfolio.hedge.quantity = int(position.position)
                self.strategy.portfolio.hedge.avg_price = float(position.avgCost) / (
                    self.source.hedge.multiplier or 1.0
                )
                adopted += 1
                log.warning(
                    "adopted an existing hedge position: %+d %s @ %.2f",
                    self.strategy.portfolio.hedge.quantity, hedge_symbol,
                    self.strategy.portfolio.hedge.avg_price,
                )
            elif contract.secType == "FOP" and contract.symbol == self.source.option.symbol:
                log.error(
                    "an existing %s option position is open (%+d %s %g%s). This "
                    "runner will not adopt option legs it did not open -- close it "
                    "or restart once flat. Adopting a half-known straddle is how a "
                    "book ends up long gamma while the strategy believes it is "
                    "short it.",
                    contract.symbol, int(position.position),
                    contract.lastTradeDateOrContractMonth, contract.strike, contract.right,
                )
                raise RuntimeError("refusing to start with an unmanaged option position")
        if not adopted:
            log.info("no existing positions to adopt")


def run_live(cfg: Config, dry_run: bool = False, max_cycles: int | None = None):
    return LiveRunner(cfg, dry_run=dry_run).run(max_cycles=max_cycles)
