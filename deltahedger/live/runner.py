"""Live / forward-testing runner.

Polls IBKR, synthesises a ``MarketBar`` from the current market, and hands
it to the same ``ShortVolStrategy`` the backtest drives.  The strategy does
not know which runner it is under; that is what makes a forward test
evidence about the validated logic rather than about a second
implementation of it.

Differences from the backtest that are worth being explicit about:

  * bars are *polls*, not completed bars, so the strategy sees the market as
    of each poll rather than a settled OHLC;
  * fills come from the exchange and can be partial or missing entirely, so
    the position is reconciled against IBKR on every cycle;
  * the ATM implied vol comes from the live chain rather than a historical
    series.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime

from ..broker.ibkr import IbkrChainProvider, IbkrConnection, IbkrExecution, WhatIfMarginModel
from ..config import Config
from ..data.base import MarketBar
from ..sizing import build_margin_model
from ..strategy import ShortVolStrategy

log = logging.getLogger(__name__)


class LiveRunner:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.source = cfg.source
        self.connection = IbkrConnection(cfg, self.source)
        self.strategy: ShortVolStrategy | None = None
        self._stop = False

    def request_stop(self, *_: object) -> None:
        log.info("stop requested; finishing the current cycle")
        self._stop = True

    def run(self, max_cycles: int | None = None) -> ShortVolStrategy:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        with self.connection as conn:
            fallback = build_margin_model(
                self.cfg.sizing, self.source, self.cfg.risk_free_rate
            )
            margin_model = (
                WhatIfMarginModel(conn, fallback)
                if self.cfg.ibkr.use_whatif_margin
                else fallback
            )
            self.strategy = ShortVolStrategy(self.cfg, self.source, margin_model)
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

            cycles = 0
            while not self._stop and (max_cycles is None or cycles < max_cycles):
                try:
                    self._cycle(conn, chain_provider, execution)
                except Exception:  # noqa: BLE001 - a bad poll must not kill the run
                    log.exception("cycle failed; continuing")
                cycles += 1
                if self._stop:
                    break
                conn.ib.sleep(self.cfg.ibkr.poll_seconds)

            log.info("live runner stopped after %d cycles", cycles)
        return self.strategy

    # -- internals ------------------------------------------------------

    def _cycle(self, conn, chain_provider, execution) -> None:
        assert self.strategy is not None
        now = datetime.now(self.strategy.clock.tz)
        if not self.strategy.clock.in_session(now):
            log.debug("outside the session at %s; idling", now)
            return

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
        state = self.strategy.on_bar(bar, execution)
        log.info(
            "%s F=%.2f IV=%.3f | opt=%d @ %s hedge=%+d | net delta %+.1f "
            "(target %.1f) | equity %s",
            now.strftime("%H:%M:%S"), state.future, state.atm_iv,
            state.option_contracts,
            f"{state.strike:g}P" if state.strike else "-",
            state.hedge_contracts, state.net_delta_units, self.cfg.hedge.target,
            f"${state.equity:,.0f}",
        )

    def _atm_iv(self, conn, chain_provider, future_price: float, now: datetime) -> float:
        """Read ATM implied vol off the live chain."""
        expiries = self.strategy.clock.candidate_expiries(
            now, self.cfg.strategy.max_days_to_expiry
        )
        if not expiries:
            return self.cfg.data.default_atm_iv
        t = self.strategy.clock.time_to_expiry(now, expiries[0])
        try:
            quotes = chain_provider.put_chain(future_price, expiries[0], t, width_pct=0.005)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read ATM vol from the chain (%s); using default", exc)
            return self.cfg.data.default_atm_iv
        if not quotes:
            return self.cfg.data.default_atm_iv
        atm = min(quotes, key=lambda q: abs(q.strike - future_price))
        return atm.iv

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
            if not position.position:
                continue  # closed-out legs still surface in some feeds
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
                    "or restart once flat.",
                    contract.symbol, int(position.position),
                    contract.lastTradeDateOrContractMonth, contract.strike, contract.right,
                )
                raise RuntimeError("refusing to start with an unmanaged option position")
        if not adopted:
            log.info("no existing positions to adopt")


def run_live(cfg: Config, dry_run: bool = False, max_cycles: int | None = None):
    return LiveRunner(cfg, dry_run=dry_run).run(max_cycles=max_cycles)
