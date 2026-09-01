"""The delta-hedged short-volatility strategy.

One bar (or one live poll) at a time, in this order:

  1. mark the open option and recompute greeks
  2. check exits -- stop, target, time, daily loss limit
  3. check entry -- if flat and inside the entry window
  4. check the delta band and hedge

Exits run before entry so a stop and a re-entry can happen on the same bar
when ``reenter_after_exit`` is set, and hedging runs last so it sees the
delta the other two steps left behind.

This class holds no market-data or broker dependency: it is handed a
``MarketBar`` and an ``ExecutionHandler``.  The backtest loop and the live
runner both call ``on_bar`` and differ only in where those two come from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .broker.base import ExecutionHandler, Fill
from .chain import OptionQuote, build_put_chain, select_short_put
from .config import Config
from .data.base import MarketBar
from .hedger import DeltaHedger
from .instruments import RiskSource
from .portfolio import OptionPosition, Portfolio
from .pricing import Greeks, black76
from .session import SessionClock
from .sizing import MarginModel, build_margin_model, size_short_puts
from .volsurface import VolSurface

log = logging.getLogger(__name__)


@dataclass
class StrategyEvent:
    """Something worth recording: a fill, a skipped entry, a band breach."""

    timestamp: datetime
    kind: str
    detail: str
    net_delta: float = 0.0
    equity: float = 0.0


@dataclass
class BarState:
    """Everything the strategy computed for one bar, for logging and reporting."""

    timestamp: datetime
    future: float
    atm_iv: float
    time_to_expiry: float
    option_mark: float | None
    option_greeks: Greeks | None
    option_delta_units: float
    hedge_delta_units: float
    net_delta_units: float
    gamma_units: float
    hedge_contracts: int
    option_contracts: int
    strike: float | None
    equity: float
    realised_pnl: float
    fees_paid: float
    in_band: bool


class ShortVolStrategy:
    def __init__(
        self,
        cfg: Config,
        source: RiskSource | None = None,
        margin_model: MarginModel | None = None,
    ):
        self.cfg = cfg
        self.source = source or cfg.source
        self.clock = SessionClock(self.source)
        self.surface = VolSurface(cfg.vol)
        self.hedger = DeltaHedger(cfg.hedge, self.source)
        self.margin_model = margin_model or build_margin_model(
            cfg.sizing, self.source, cfg.risk_free_rate
        )
        self.portfolio = Portfolio(cfg.starting_equity, self.source)

        self.events: list[StrategyEvent] = []
        self.bar_states: list[BarState] = []
        self.fills: list[Fill] = []
        self._last_hedge_time: datetime | None = None
        self._session_date: date | None = None
        self._session_start_equity = cfg.starting_equity
        self._exited_this_session = False
        self._halted_for_session = False

    # -- main loop ------------------------------------------------------

    def on_bar(self, bar: MarketBar, execution: ExecutionHandler) -> BarState:
        moment = self.clock.localize(bar.timestamp)
        self._roll_session(moment, bar.close)

        quote = self._mark_open_option(bar, moment)
        option_mark = quote.price if quote is not None else None
        greeks = quote.greeks if quote is not None else None

        self._check_exits(bar, moment, quote, execution)

        if self.portfolio.option is None:
            self._try_entry(bar, moment, execution)
            quote = self._mark_open_option(bar, moment)
            option_mark = quote.price if quote is not None else None
            greeks = quote.greeks if quote is not None else None

        self._hedge(bar, moment, greeks, execution)

        state = self._snapshot(bar, moment, option_mark, greeks)
        self.bar_states.append(state)
        return state

    # -- session bookkeeping --------------------------------------------

    def _roll_session(self, moment: datetime, future_price: float) -> None:
        """Reset per-session state at the first bar of a new trading day.

        The session's opening equity has to be marked at a real price: a
        hedge carried overnight (``flatten_hedge_on_exit: false``) would be
        valued against zero otherwise, and the daily loss limit would fire
        on the first bar of every day.
        """
        day = moment.date()
        if day == self._session_date:
            return
        self._session_date = day
        option_mark = None
        if self.portfolio.option is not None:
            option_mark = self._mark_open_option(
                MarketBar(moment, future_price, future_price, future_price,
                          future_price, self.cfg.data.default_atm_iv),
                moment,
            ).price
        self._session_start_equity = self.portfolio.equity(option_mark, future_price)
        self._exited_this_session = False
        self._halted_for_session = False

    def _record(self, moment: datetime, kind: str, detail: str, net_delta: float = 0.0) -> None:
        self.events.append(
            StrategyEvent(
                timestamp=moment,
                kind=kind,
                detail=detail,
                net_delta=net_delta,
                equity=self.portfolio.starting_equity + self.portfolio.realised_pnl
                - self.portfolio.fees_paid,
            )
        )
        log.debug("%s | %s | %s", moment, kind, detail)

    # -- marking ---------------------------------------------------------

    def _mark_open_option(self, bar: MarketBar, moment: datetime) -> OptionQuote | None:
        """Reprice the open position at this bar's future and vol."""
        position = self.portfolio.option
        if position is None:
            return None
        t = self.clock.time_to_expiry(moment, position.expiry)
        iv = self.surface.iv(bar.close, position.strike, bar.atm_iv, t)
        greeks = black76(bar.close, position.strike, t, iv, self.cfg.risk_free_rate, position.right)
        return OptionQuote(
            strike=position.strike,
            right=position.right,
            expiry=position.expiry,
            price=greeks.price,
            iv=iv,
            greeks=greeks,
            time_to_expiry=t,
        )

    # -- entry -----------------------------------------------------------

    def _try_entry(self, bar: MarketBar, moment: datetime, execution: ExecutionHandler) -> None:
        cfg = self.cfg.strategy
        if self._halted_for_session:
            return
        if self._exited_this_session and not cfg.reenter_after_exit:
            return

        local = moment.timetz().replace(tzinfo=None)
        if not (cfg.entry_time <= local <= cfg.entry_cutoff_time):
            return

        expiries = self.clock.candidate_expiries(moment, cfg.max_days_to_expiry)
        expiries = [
            e for e in expiries if (e - moment.date()).days >= cfg.min_days_to_expiry
        ]
        if not expiries:
            self._record(moment, "entry_skipped", "no listed expiry inside the DTE window")
            return
        expiry = expiries[0]

        t = self.clock.time_to_expiry(moment, expiry)
        cutoff = timedelta(minutes=cfg.close_before_expiry_minutes)
        if self.clock.seconds_to_expiry(moment, expiry) <= cutoff.total_seconds():
            return

        chain = build_put_chain(
            bar.close, expiry, t, bar.atm_iv, self.source, self.surface,
            self.cfg.risk_free_rate,
        )
        quote = select_short_put(chain, cfg, bar.close)
        if quote is None:
            self._record(
                moment, "entry_skipped",
                f"no strike within {cfg.short_put_delta_tolerance:.2f} of "
                f"{cfg.short_put_delta:.2f} delta (T={t*365*24:.1f}h, IV={bar.atm_iv:.3f})",
            )
            return

        equity = self.portfolio.equity(None, bar.close)
        sizing = size_short_puts(
            equity, quote, bar.close, self.cfg.sizing, self.source, self.margin_model
        )
        if not sizing.ok:
            self._record(moment, "entry_skipped", sizing.reason)
            return

        fill = execution.execute_option(quote, -sizing.contracts, moment)
        if fill is None:
            self._record(moment, "entry_failed", "execution handler returned no fill")
            return

        self.fills.append(fill)
        self.portfolio.charge_fees(fill.fees)
        self.portfolio.open_option(
            OptionPosition(
                strike=quote.strike,
                expiry=expiry,
                right="P",
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_time=moment,
                entry_iv=quote.iv,
                entry_delta=quote.greeks.delta,
            )
        )
        credit = -fill.quantity * fill.price * self.source.option.multiplier
        self._record(
            moment, "entry",
            f"sold {sizing.contracts} {expiry} {quote.strike:g}P @ {fill.price:.2f} "
            f"(delta {quote.greeks.delta:+.3f}, IV {quote.iv:.3f}) for ${credit:,.0f} "
            f"credit; margin ${sizing.total_margin:,.0f} of ${sizing.budget:,.0f} budget",
        )

    # -- exits -------------------------------------------------------------

    def _check_exits(
        self, bar: MarketBar, moment: datetime, quote: OptionQuote | None,
        execution: ExecutionHandler,
    ) -> None:
        position = self.portfolio.option
        if position is None or quote is None:
            return
        cfg = self.cfg.strategy

        seconds_left = self.clock.seconds_to_expiry(moment, position.expiry)
        reason: str | None = None

        if seconds_left <= cfg.close_before_expiry_minutes * 60:
            reason = f"{cfg.close_before_expiry_minutes}m to expiry"
        elif (
            cfg.stop_loss_premium_multiple is not None
            and position.entry_price > 0
            and quote.price >= position.entry_price * cfg.stop_loss_premium_multiple
        ):
            reason = (
                f"stop: mark {quote.price:.2f} >= {cfg.stop_loss_premium_multiple:g}x "
                f"entry {position.entry_price:.2f}"
            )
        elif (
            cfg.take_profit_pct is not None
            and position.entry_price > 0
            and quote.price <= position.entry_price * (1.0 - cfg.take_profit_pct)
        ):
            reason = (
                f"target: captured {cfg.take_profit_pct:.0%} of the "
                f"{position.entry_price:.2f} credit"
            )
        else:
            limit = cfg.daily_loss_limit_pct
            if limit is not None:
                equity = self.portfolio.equity(quote.price, bar.close)
                drawdown = self._session_start_equity - equity
                if drawdown >= limit * self._session_start_equity:
                    reason = (
                        f"daily loss limit: -${drawdown:,.0f} vs "
                        f"{limit:.0%} of ${self._session_start_equity:,.0f}"
                    )
                    self._halted_for_session = True

        if reason is None:
            return
        self._close_position(bar, moment, quote, execution, reason)

    def _close_position(
        self, bar: MarketBar, moment: datetime, quote: OptionQuote,
        execution: ExecutionHandler, reason: str,
    ) -> None:
        position = self.portfolio.option
        assert position is not None
        fill = execution.execute_option(quote, -position.quantity, moment)
        if fill is None:
            self._record(moment, "exit_failed", f"could not close: {reason}")
            return
        self.fills.append(fill)
        self.portfolio.charge_fees(fill.fees)
        pnl = self.portfolio.close_option(fill.price)
        self._exited_this_session = True
        self._record(
            moment, "exit",
            f"bought back {abs(position.quantity)} {position.strike:g}P @ "
            f"{fill.price:.2f} ({reason}); option P&L ${pnl:,.0f}",
        )

        if self.cfg.hedge.flatten_hedge_on_exit and self.portfolio.hedge.quantity:
            self._flatten_hedge(bar, moment, execution)

    def _flatten_hedge(
        self, bar: MarketBar, moment: datetime, execution: ExecutionHandler
    ) -> None:
        quantity = -self.portfolio.hedge.quantity
        fill = execution.execute_hedge(quantity, bar.close, moment)
        if fill is None:
            return
        self.fills.append(fill)
        self.portfolio.charge_fees(fill.fees)
        pnl = self.portfolio.apply_hedge_fill(fill.quantity, fill.price)
        self._record(
            moment, "hedge_flatten",
            f"closed {abs(quantity)} {self.source.hedge.symbol} @ {fill.price:.2f}; "
            f"hedge P&L ${pnl:,.0f}",
        )

    # -- hedging -----------------------------------------------------------

    def _hedge(
        self, bar: MarketBar, moment: datetime, greeks: Greeks | None,
        execution: ExecutionHandler,
    ) -> None:
        if self.portfolio.option is None and self.portfolio.hedge.quantity == 0:
            return

        net_delta = self.portfolio.net_delta_units(greeks)
        elapsed = (
            (moment - self._last_hedge_time).total_seconds()
            if self._last_hedge_time
            else None
        )
        decision = self.hedger.decide(net_delta, elapsed)
        if not decision.should_hedge:
            return

        fill = execution.execute_hedge(decision.contracts, bar.close, moment)
        if fill is None:
            self._record(moment, "hedge_failed", decision.reason, net_delta)
            return

        self.fills.append(fill)
        self.portfolio.charge_fees(fill.fees)
        realised = self.portfolio.apply_hedge_fill(fill.quantity, fill.price)
        self._last_hedge_time = moment
        self._record(
            moment, "hedge",
            f"{decision.reason} @ {fill.price:.2f}"
            + (f"; realised ${realised:,.0f}" if realised else ""),
            decision.net_delta_after,
        )

    # -- reporting ---------------------------------------------------------

    def _snapshot(
        self, bar: MarketBar, moment: datetime, option_mark: float | None,
        greeks: Greeks | None,
    ) -> BarState:
        position = self.portfolio.option
        option_delta = self.portfolio.option_delta_units(greeks)
        hedge_delta = self.portfolio.hedge_delta_units()
        net = option_delta + hedge_delta
        return BarState(
            timestamp=moment,
            future=bar.close,
            atm_iv=bar.atm_iv,
            time_to_expiry=(
                self.clock.time_to_expiry(moment, position.expiry) if position else 0.0
            ),
            option_mark=option_mark,
            option_greeks=greeks,
            option_delta_units=option_delta,
            hedge_delta_units=hedge_delta,
            net_delta_units=net,
            gamma_units=self.portfolio.option_gamma_units(greeks),
            hedge_contracts=self.portfolio.hedge.quantity,
            option_contracts=position.quantity if position else 0,
            strike=position.strike if position else None,
            equity=self.portfolio.equity(option_mark, bar.close),
            realised_pnl=self.portfolio.realised_pnl,
            fees_paid=self.portfolio.fees_paid,
            in_band=self.cfg.hedge.in_band(net) if position or hedge_delta else True,
        )
