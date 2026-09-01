"""The delta-hedged short-volatility strategy.

One bar (or one live poll) at a time, in this order:

  1. mark every open option leg and recompute greeks
  2. check exits -- stop, target, time, daily loss limit
  3. check entry -- if flat and inside the entry window
  4. check the delta band and hedge

Exits run before entry so a stop and a re-entry can happen on the same bar
when ``reenter_after_exit`` is set, and hedging runs last so it sees the
delta the other two steps left behind.

The option book can hold a put alone, or a put and a call together when
``strategy.sell_call`` is set (a strangle -- see ``portfolio.py``).  Both
legs share one expiry, are sized together off their combined margin, and
share one stop/target computed off their combined premium; the *delta*
target is unaffected either way, since the hedger holds net portfolio
delta at ``hedge.target`` regardless of how many option legs feed it.

This class holds no market-data or broker dependency: it is handed a
``MarketBar`` and an ``ExecutionHandler``.  The backtest loop and the live
runner both call ``on_bar`` and differ only in where those two come from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .broker.base import ExecutionHandler, Fill
from .chain import OptionQuote, build_option_chain, select_short_option
from .config import Config
from .data.base import MarketBar
from .hedger import DeltaHedger
from .instruments import RiskSource
from .portfolio import OptionPosition, Portfolio
from .pricing import Greeks, black76
from .session import SessionClock
from .sizing import MarginModel, build_margin_model, size_short_option_position
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
    put_strike: float | None
    put_mark: float | None
    put_delta: float | None
    put_contracts: int
    call_strike: float | None
    call_mark: float | None
    call_delta: float | None
    call_contracts: int
    option_delta_units: float  # combined across all open legs
    hedge_delta_units: float
    net_delta_units: float
    gamma_units: float  # combined
    hedge_contracts: int
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
        self._session_date = None
        self._session_start_equity = cfg.starting_equity
        self._exited_this_session = False
        self._halted_for_session = False

    # -- main loop ------------------------------------------------------

    def on_bar(self, bar: MarketBar, execution: ExecutionHandler) -> BarState:
        moment = self.clock.localize(bar.timestamp)
        self._roll_session(moment, bar.close)

        quotes = self._mark_open_legs(bar, moment)
        self._check_exits(bar, moment, quotes, execution)

        if not self.portfolio.has_option:
            self._try_entry(bar, moment, execution)
            quotes = self._mark_open_legs(bar, moment)

        greeks_by_right = {right: q.greeks for right, q in quotes.items()}
        self._hedge(bar, moment, greeks_by_right, execution)

        state = self._snapshot(bar, moment, quotes, greeks_by_right)
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
        marks: dict[str, float] = {}
        if self.portfolio.has_option:
            marker_bar = MarketBar(
                moment, future_price, future_price, future_price,
                future_price, self.cfg.data.default_atm_iv,
            )
            quotes = self._mark_open_legs(marker_bar, moment)
            marks = {right: q.price for right, q in quotes.items()}
        self._session_start_equity = self.portfolio.equity(marks, future_price)
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

    def _mark_open_legs(self, bar: MarketBar, moment: datetime) -> dict[str, OptionQuote]:
        """Reprice every currently open leg at this bar's future and vol."""
        quotes: dict[str, OptionQuote] = {}
        for right, position in self.portfolio.legs.items():
            t = self.clock.time_to_expiry(moment, position.expiry)
            iv = self.surface.iv(bar.close, position.strike, bar.atm_iv, t)
            greeks = black76(
                bar.close, position.strike, t, iv, self.cfg.risk_free_rate, position.right
            )
            quotes[right] = OptionQuote(
                strike=position.strike,
                right=position.right,
                expiry=position.expiry,
                price=greeks.price,
                iv=iv,
                greeks=greeks,
                time_to_expiry=t,
            )
        return quotes

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

        rights = ("P", "C") if cfg.sell_call else ("P",)
        quotes: dict[str, OptionQuote] = {}
        for right in rights:
            chain = build_option_chain(
                bar.close, expiry, t, bar.atm_iv, self.source, self.surface,
                self.cfg.risk_free_rate, right=right,
            )
            quote = select_short_option(chain, cfg, bar.close, right)
            if quote is None:
                target = cfg.short_put_delta if right == "P" else cfg.short_call_delta
                tolerance = (
                    cfg.short_put_delta_tolerance if right == "P"
                    else cfg.short_call_delta_tolerance
                )
                self._record(
                    moment, "entry_skipped",
                    f"no {right} strike within {tolerance:.2f} of {target:.2f} "
                    f"delta (T={t*365*24:.1f}h, IV={bar.atm_iv:.3f})",
                )
                return
            quotes[right] = quote

        equity = self.portfolio.equity({}, bar.close)
        sizing = size_short_option_position(
            equity, list(quotes.values()), bar.close, self.cfg.sizing,
            self.source, self.margin_model,
        )
        if not sizing.ok:
            self._record(moment, "entry_skipped", sizing.reason)
            return

        fills = self._fill_legs(quotes, sizing.contracts, moment, execution)
        if fills is None:
            return  # already recorded and, if needed, unwound

        for right, fill in fills.items():
            self.fills.append(fill)
            self.portfolio.charge_fees(fill.fees)
            quote = quotes[right]
            self.portfolio.open_leg(
                OptionPosition(
                    strike=quote.strike,
                    expiry=expiry,
                    right=right,
                    quantity=fill.quantity,
                    entry_price=fill.price,
                    entry_time=moment,
                    entry_iv=quote.iv,
                    entry_delta=quote.greeks.delta,
                )
            )

        credit = sum(
            -f.quantity * f.price * self.source.option.multiplier for f in fills.values()
        )
        legs_desc = ", ".join(
            f"{sizing.contracts} {expiry} {quotes[r].strike:g}{r} @ {fills[r].price:.2f} "
            f"(delta {quotes[r].greeks.delta:+.3f})"
            for r in fills
        )
        self._record(
            moment, "entry",
            f"sold {legs_desc}, IV {bar.atm_iv:.3f}, for ${credit:,.0f} combined "
            f"credit; margin ${sizing.total_margin:,.0f} of ${sizing.budget:,.0f} budget",
        )

    def _fill_legs(
        self, quotes: dict[str, OptionQuote], contracts: int, moment: datetime,
        execution: ExecutionHandler,
    ) -> dict[str, Fill] | None:
        """Fill every leg of a new entry, unwinding on a partial failure.

        A strangle is opened as one decision, so a fill failing partway
        through (the put goes through, the call doesn't) must not leave a
        naked leg nobody chose to carry -- the already-filled leg is bought
        straight back rather than left on the book.
        """
        fills: dict[str, Fill] = {}
        for right, quote in quotes.items():
            fill = execution.execute_option(quote, -contracts, moment)
            if fill is None:
                self._record(
                    moment, "entry_failed",
                    f"execution handler returned no fill for the {right} leg"
                    + (f"; unwinding the filled {'/'.join(fills)} leg(s)" if fills else ""),
                )
                self._unwind_fills(fills, quotes, moment, execution)
                return None
            fills[right] = fill
        return fills

    def _unwind_fills(
        self, fills: dict[str, Fill], quotes: dict[str, OptionQuote], moment: datetime,
        execution: ExecutionHandler,
    ) -> None:
        for right, fill in fills.items():
            self.fills.append(fill)
            self.portfolio.charge_fees(fill.fees)
            unwind = execution.execute_option(quotes[right], -fill.quantity, moment)
            if unwind is None:
                log.error(
                    "could not unwind the %s leg after a partial entry failure; "
                    "%d contracts are now open unintentionally",
                    right, fill.quantity,
                )
                continue
            self.fills.append(unwind)
            self.portfolio.charge_fees(unwind.fees)
            pnl = unwind.quantity * (unwind.price - fill.price) * self.source.option.multiplier
            self.portfolio.option_realised += pnl

    # -- exits -------------------------------------------------------------

    def _check_exits(
        self, bar: MarketBar, moment: datetime, quotes: dict[str, OptionQuote],
        execution: ExecutionHandler,
    ) -> None:
        if not self.portfolio.has_option or not quotes:
            return
        cfg = self.cfg.strategy

        any_position = next(iter(self.portfolio.legs.values()))
        seconds_left = self.clock.seconds_to_expiry(moment, any_position.expiry)
        marks = {right: q.price for right, q in quotes.items()}
        combined_credit = self.portfolio.combined_credit_received()
        combined_close = self.portfolio.combined_close_value(marks)
        reason: str | None = None

        if seconds_left <= cfg.close_before_expiry_minutes * 60:
            reason = f"{cfg.close_before_expiry_minutes}m to expiry"
        elif (
            cfg.stop_loss_premium_multiple is not None
            and combined_credit > 0
            and combined_close is not None
            and combined_close >= combined_credit * cfg.stop_loss_premium_multiple
        ):
            reason = (
                f"stop: combined mark ${combined_close:,.0f} >= "
                f"{cfg.stop_loss_premium_multiple:g}x combined credit ${combined_credit:,.0f}"
            )
        elif (
            cfg.take_profit_pct is not None
            and combined_credit > 0
            and combined_close is not None
            and combined_close <= combined_credit * (1.0 - cfg.take_profit_pct)
        ):
            reason = (
                f"target: captured {cfg.take_profit_pct:.0%} of the "
                f"${combined_credit:,.0f} combined credit"
            )
        else:
            limit = cfg.daily_loss_limit_pct
            if limit is not None:
                equity = self.portfolio.equity(marks, bar.close)
                drawdown = self._session_start_equity - equity
                if drawdown >= limit * self._session_start_equity:
                    reason = (
                        f"daily loss limit: -${drawdown:,.0f} vs "
                        f"{limit:.0%} of ${self._session_start_equity:,.0f}"
                    )
                    self._halted_for_session = True

        if reason is None:
            return
        self._close_all_legs(bar, moment, quotes, execution, reason)

    def _close_all_legs(
        self, bar: MarketBar, moment: datetime, quotes: dict[str, OptionQuote],
        execution: ExecutionHandler, reason: str,
    ) -> None:
        closed: list[tuple[str, OptionPosition, Fill]] = []
        exit_prices: dict[str, float] = {}
        for right, position in list(self.portfolio.legs.items()):
            quote = quotes.get(right)
            if quote is None:
                self._record(moment, "exit_failed", f"no quote to close the {right} leg")
                continue
            fill = execution.execute_option(quote, -position.quantity, moment)
            if fill is None:
                self._record(moment, "exit_failed", f"could not close the {right} leg: {reason}")
                continue
            self.fills.append(fill)
            self.portfolio.charge_fees(fill.fees)
            exit_prices[right] = fill.price
            closed.append((right, position, fill))

        if not closed:
            return

        total_pnl = self.portfolio.close_all_legs(exit_prices)
        self._exited_this_session = True
        legs_desc = ", ".join(
            f"{abs(position.quantity)} {position.strike:g}{right} @ {fill.price:.2f}"
            for right, position, fill in closed
        )
        self._record(
            moment, "exit",
            f"bought back {legs_desc} ({reason}); option P&L ${total_pnl:,.0f}",
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
        self, bar: MarketBar, moment: datetime, greeks_by_right: dict[str, Greeks],
        execution: ExecutionHandler,
    ) -> None:
        if not self.portfolio.has_option and self.portfolio.hedge.quantity == 0:
            return

        net_delta = self.portfolio.net_delta_units(greeks_by_right)
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
        self, bar: MarketBar, moment: datetime, quotes: dict[str, OptionQuote],
        greeks_by_right: dict[str, Greeks],
    ) -> BarState:
        put_pos, call_pos = self.portfolio.put, self.portfolio.call
        put_quote, call_quote = quotes.get("P"), quotes.get("C")
        option_delta = self.portfolio.option_delta_units(greeks_by_right)
        hedge_delta = self.portfolio.hedge_delta_units()
        net = option_delta + hedge_delta
        marks = {right: q.price for right, q in quotes.items()}
        any_position = put_pos or call_pos
        return BarState(
            timestamp=moment,
            future=bar.close,
            atm_iv=bar.atm_iv,
            time_to_expiry=(
                self.clock.time_to_expiry(moment, any_position.expiry) if any_position else 0.0
            ),
            put_strike=put_pos.strike if put_pos else None,
            put_mark=put_quote.price if put_quote else None,
            put_delta=put_quote.greeks.delta if put_quote else None,
            put_contracts=put_pos.quantity if put_pos else 0,
            call_strike=call_pos.strike if call_pos else None,
            call_mark=call_quote.price if call_quote else None,
            call_delta=call_quote.greeks.delta if call_quote else None,
            call_contracts=call_pos.quantity if call_pos else 0,
            option_delta_units=option_delta,
            hedge_delta_units=hedge_delta,
            net_delta_units=net,
            gamma_units=self.portfolio.option_gamma_units(greeks_by_right),
            hedge_contracts=self.portfolio.hedge.quantity,
            equity=self.portfolio.equity(marks, bar.close),
            realised_pnl=self.portfolio.realised_pnl,
            fees_paid=self.portfolio.fees_paid,
            in_band=self.cfg.hedge.in_band(net) if any_position or hedge_delta else True,
        )
