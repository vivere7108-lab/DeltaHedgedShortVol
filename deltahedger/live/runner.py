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
from datetime import date, datetime

from ..broker.ibkr import IbkrChainProvider, IbkrConnection, IbkrExecution, WhatIfMarginModel
from ..config import Config
from ..data.base import MarketBar
from ..portfolio import OptionPosition
from ..sizing import build_margin_model
from ..strategy import ShortVolStrategy

log = logging.getLogger(__name__)


def _parse_ibkr_expiry(value: str) -> date:
    """Parse a contract's lastTradeDateOrContractMonth (YYYYMMDD or YYYYMM).

    Raises rather than returning ``None`` on a bad parse: an adopted option
    leg with an unknown expiry can't be risk-managed (the close-before-
    expiry check has nothing to compare against), so failing loudly here
    beats silently adopting a leg the runner then can't close on time.
    """
    text = (value or "").strip()
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"could not parse an expiry from lastTradeDateOrContractMonth={value!r}")


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
        itself -- this includes option legs, not just the hedge. A process
        restart (a crash, a redeploy, or IB Gateway's own mandatory nightly
        restart under IBC) must not require a human to notice and manually
        re-seed the position, or "unattended forward test" stops being
        true. Only a *foreign* position -- one whose strike or expiry the
        strategy did not itself choose and so cannot risk-manage against --
        is worth refusing to start over; anything shaped like our own put
        (and call, if selling one) is adopted instead.

        What is necessarily approximate about an adopted leg: IBKR's
        position feed has no entry timestamp, and its avgCost is a cost
        basis, not the exact per-unit premium our own fills would have
        recorded -- and whether IBKR reports avgCost signed by long/short or
        as an unsigned magnitude is not something this codebase can verify
        without a live account, so entry_price and avg_price both take
        abs(avgCost): a per-unit price must come out positive either way,
        and everything downstream (unrealised P&L, the stop/target ratio)
        already assumes a positive entry_price the way a normal fill
        produces one. entry_time, entry_iv and entry_delta are placeholders
        (informational only -- nothing downstream keys exit decisions off
        them). Log the derived entry_price on every adoption
        specifically so this is auditable rather than silently trusted.
        """
        assert self.strategy is not None
        positions = conn.ib.positions(conn.account)
        hedge_symbol = self.source.hedge.symbol
        option_symbol = self.source.option.symbol
        option_mult = self.source.option.multiplier
        adopted_legs: dict[str, OptionPosition] = {}
        now = datetime.now(self.strategy.clock.tz)

        for position in positions:
            contract = position.contract
            if not position.position:
                continue  # closed-out legs still surface in some feeds

            if contract.secType == "FUT" and contract.symbol == hedge_symbol:
                self.strategy.portfolio.hedge.quantity = int(position.position)
                # abs(): avgCost is a per-contract cost-basis magnitude, not
                # signed by long/short -- a price must come out positive
                # regardless of which convention this account's feed uses.
                self.strategy.portfolio.hedge.avg_price = abs(float(position.avgCost)) / (
                    self.source.hedge.multiplier or 1.0
                )
                log.warning(
                    "adopted an existing hedge position: %+d %s @ %.2f",
                    self.strategy.portfolio.hedge.quantity, hedge_symbol,
                    self.strategy.portfolio.hedge.avg_price,
                )

            elif contract.secType == "FOP" and contract.symbol == option_symbol:
                right = contract.right[:1].upper()  # IBKR sends "P"/"PUT" inconsistently
                entry_price = abs(float(position.avgCost)) / option_mult
                expiry = _parse_ibkr_expiry(contract.lastTradeDateOrContractMonth)
                leg = OptionPosition(
                    strike=float(contract.strike),
                    expiry=expiry,
                    right=right,
                    quantity=int(position.position),
                    entry_price=entry_price,
                    entry_time=now,  # unknown; IBKR positions carry no timestamp
                    entry_iv=0.0,  # informational only -- not used in exit decisions
                    entry_delta=0.0,
                )
                self.strategy.portfolio.open_leg(leg)
                adopted_legs[right] = leg
                log.warning(
                    "adopted an existing %s option position: %+d %g%s exp %s "
                    "@ derived entry price %.2f (from avgCost %.2f) -- verify "
                    "this looks right before trusting the stop/target off it",
                    option_symbol, leg.quantity, leg.strike, right, expiry,
                    entry_price, float(position.avgCost),
                )

        if len(adopted_legs) == 2:
            put, call = adopted_legs.get("P"), adopted_legs.get("C")
            if put and call and put.expiry != call.expiry:
                log.error(
                    "adopted put (exp %s) and call (exp %s) have different "
                    "expiries -- they should never diverge if this runner "
                    "opened both together. Proceeding, but the close-before-"
                    "expiry check only looks at one leg's expiry and the "
                    "other may be closed later than intended. Do not trade "
                    "manually in an account this bot manages.",
                    put.expiry, call.expiry,
                )

        if not self.strategy.portfolio.legs and not self.strategy.portfolio.hedge.quantity:
            log.info("no existing positions to adopt")


def run_live(cfg: Config, dry_run: bool = False, max_cycles: int | None = None):
    return LiveRunner(cfg, dry_run=dry_run).run(max_cycles=max_cycles)
