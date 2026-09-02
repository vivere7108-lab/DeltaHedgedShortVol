"""The GEX-directed, delta-hedged straddle strategy.

One bar (or one live poll) at a time, in this order:

  1. mark the open straddle and recompute greeks
  2. read GEX at the current spot -- total, flip point, regime
  3. check exits -- time, regime flip, directional stop/target, loss limit
  4. check entry -- if flat and inside the entry window, take the side the
     regime implies
  5. check the delta band and hedge

The direction is not a parameter.  It is whatever dealer positioning says::

    negative GEX  -> dealers hedge WITH the move, amplifying it
                  -> realised vol should exceed implied
                  -> LONG the ATM straddle, and scalp the gamma

    positive GEX  -> dealers hedge AGAINST the move, damping it
                  -> realised vol should fall short of implied
                  -> SHORT the ATM straddle, and let theta run

    near the flip -> the sign is about to change; stand aside

Exits run before entry so a regime flip and the re-entry on the other side
can happen on the same bar, and hedging runs last so it sees the delta the
other two steps left behind.

This class holds no market-data or broker dependency: it is handed a
``MarketBar``, an ``OpenInterestProvider`` and an ``ExecutionHandler``.  The
backtest loop and the live runner both call ``on_bar`` and differ only in
where those come from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

from .broker.base import ExecutionHandler, Fill
from .chain import StraddleQuote, price_option, select_atm_straddle
from .config import Config
from .data.base import MarketBar
from .gex import (
    NEUTRAL,
    GexCalculator,
    GexProfile,
    OpenInterestProvider,
    StrikeOpenInterest,
)
from .hedger import DeltaHedger
from .instruments import RiskSource
from .portfolio import Portfolio, StraddlePosition
from .session import SessionClock
from .sizing import MarginModel, build_margin_model, size_straddles
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
    regime: str = ""


@dataclass
class BarState:
    """Everything the strategy computed for one bar, for logging and reporting."""

    timestamp: datetime
    future: float
    atm_iv: float
    time_to_expiry: float
    straddle_mark: float | None
    call_mark: float | None
    put_mark: float | None
    option_delta_units: float
    hedge_delta_units: float
    net_delta_units: float
    gamma_units: float
    vega_dollars: float
    theta_dollars: float
    hedge_contracts: int
    straddle_contracts: int
    direction: int
    strike: float | None
    equity: float
    realised_pnl: float
    fees_paid: float
    in_band: bool
    # -- the GEX read that drove the bar ------------------------------
    gex_total: float | None
    gex_flip: float | None
    gex_regime: str
    distance_to_flip: float | None


class GexStraddleStrategy:
    def __init__(
        self,
        cfg: Config,
        source: RiskSource | None = None,
        margin_model: MarginModel | None = None,
        open_interest: OpenInterestProvider | None = None,
    ):
        self.cfg = cfg
        self.source = source or cfg.source
        self.clock = SessionClock(self.source)
        self.surface = VolSurface(cfg.vol)
        self.hedger = DeltaHedger(cfg.hedge, self.source)
        self.margin_model = margin_model or build_margin_model(
            cfg.sizing, self.source, cfg.risk_free_rate
        )
        self.gex = GexCalculator(
            cfg.gex, self.source, self.surface, cfg.risk_free_rate
        )
        self.open_interest = open_interest
        self.portfolio = Portfolio(cfg.starting_equity, self.source)

        self.events: list[StrategyEvent] = []
        self.bar_states: list[BarState] = []
        self.fills: list[Fill] = []
        #: Position P&L realised at each close, keyed by the regime that
        #: opened it. This is the number that says whether reading GEX paid.
        self.regime_pnl: dict[str, float] = {}
        self.regime_trades: dict[str, int] = {}

        self._last_hedge_time: datetime | None = None
        self._session_date: date | None = None
        self._session_start_equity = cfg.starting_equity
        self._entries_this_session = 0
        self._halted_for_session = False
        self._profile: GexProfile | None = None
        #: Cached open interest, and when it was read. OI is an end-of-day
        #: figure, so it is re-read on a timer while the *profile* is
        #: recomputed at the live spot on every bar.
        self._oi: list[StrikeOpenInterest] | None = None
        self._oi_read_at: datetime | None = None
        self._oi_expiry: date | None = None
        # Baselines for measuring one position's P&L, set at each entry.
        self._hedge_realised_at_entry = 0.0
        self._fees_at_entry = 0.0

    # -- main loop ------------------------------------------------------

    def on_bar(self, bar: MarketBar, execution: ExecutionHandler) -> BarState:
        moment = self.clock.localize(bar.timestamp)
        self._roll_session(moment, bar.close)

        quote = self._mark_open_straddle(bar, moment)
        profile = self._read_gex(bar, moment)

        self._check_exits(bar, moment, quote, profile, execution)

        if self.portfolio.straddle is None:
            self._try_entry(bar, moment, profile, execution)
            quote = self._mark_open_straddle(bar, moment)

        self._hedge(bar, moment, quote, execution)

        state = self._snapshot(bar, moment, quote, profile)
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
        quote = None
        if self.portfolio.straddle is not None:
            quote = self._mark_open_straddle(
                MarketBar(moment, future_price, future_price, future_price,
                          future_price, self.cfg.data.default_atm_iv),
                moment,
            )
        self._session_start_equity = self.portfolio.equity(quote, future_price)
        self._entries_this_session = 0
        self._halted_for_session = False

    def _record(
        self, moment: datetime, kind: str, detail: str, net_delta: float = 0.0,
        regime: str = "",
    ) -> None:
        self.events.append(
            StrategyEvent(
                timestamp=moment,
                kind=kind,
                detail=detail,
                net_delta=net_delta,
                equity=self.portfolio.starting_equity + self.portfolio.realised_pnl
                - self.portfolio.fees_paid,
                regime=regime,
            )
        )
        log.debug("%s | %s | %s", moment, kind, detail)

    # -- the 0DTE series -------------------------------------------------

    def _todays_expiry(self, moment: datetime) -> date | None:
        """The 0DTE expiry, or ``None`` when there is not one to trade.

        ``max_days_to_expiry`` defaults to 0, so this is today's series
        while it is still listed and nothing at all once it has settled.
        Standing aside is the correct answer there -- the strategy is a
        statement about 0DTE dealer hedging, and there is no 0DTE.
        """
        cfg = self.cfg.strategy
        expiries = [
            e
            for e in self.clock.candidate_expiries(moment, cfg.max_days_to_expiry)
            if (e - moment.date()).days >= cfg.min_days_to_expiry
        ]
        return expiries[0] if expiries else None

    # -- GEX -------------------------------------------------------------

    def _read_gex(self, bar: MarketBar, moment: datetime) -> GexProfile | None:
        """The GEX profile at this bar's spot, or ``None`` if unavailable.

        Open interest is cached on ``gex.refresh_seconds`` because it is an
        end-of-day figure that does not move intraday.  The *profile* is
        rebuilt every bar regardless: the regime is a statement about where
        spot sits relative to the flip point, and reusing a stale spot would
        mean never seeing the crossing the strategy exists to trade.
        """
        if not self.cfg.gex.enabled or self.open_interest is None:
            return None
        expiry = self._todays_expiry(moment)
        if expiry is None:
            return None

        stale = (
            self._oi is None
            or self._oi_expiry != expiry
            or self._oi_read_at is None
            or (moment - self._oi_read_at).total_seconds()
            >= self.cfg.gex.refresh_seconds
        )
        if stale:
            try:
                self._oi = list(
                    self.open_interest.open_interest(moment, bar.close, expiry)
                )
            except Exception as exc:  # noqa: BLE001 - a bad read must not halt the run
                log.warning("could not read open interest (%s); holding the last read", exc)
                if self._oi is None:
                    return None
            else:
                self._oi_read_at = moment
                self._oi_expiry = expiry

        t = self.clock.time_to_expiry(moment, expiry)
        self._profile = self.gex.profile(bar.close, self._oi, t, bar.atm_iv)
        return self._profile

    # -- marking ---------------------------------------------------------

    def _mark_open_straddle(
        self, bar: MarketBar, moment: datetime
    ) -> StraddleQuote | None:
        """Reprice the open position at this bar's future and vol."""
        position = self.portfolio.straddle
        if position is None:
            return None
        t = self.clock.time_to_expiry(moment, position.expiry)
        legs = {
            right: price_option(
                bar.close, position.strike, right, position.expiry, t, bar.atm_iv,
                self.surface, self.cfg.risk_free_rate,
            )
            for right in ("C", "P")
        }
        return StraddleQuote(
            strike=position.strike,
            expiry=position.expiry,
            call=legs["C"],
            put=legs["P"],
            time_to_expiry=t,
        )

    # -- entry -----------------------------------------------------------

    def _try_entry(
        self, bar: MarketBar, moment: datetime, profile: GexProfile | None,
        execution: ExecutionHandler,
    ) -> None:
        cfg = self.cfg.strategy
        if self._halted_for_session:
            return
        if self._entries_this_session >= 1 and not cfg.reenter_after_exit:
            return
        if self._entries_this_session >= cfg.max_entries_per_session:
            return

        local = moment.timetz().replace(tzinfo=None)
        if not (cfg.entry_time <= local <= cfg.entry_cutoff_time):
            return

        expiry = self._todays_expiry(moment)
        if expiry is None:
            self._record(moment, "entry_skipped", "no 0DTE series is listed")
            return

        if self.clock.seconds_to_expiry(moment, expiry) <= (
            cfg.close_before_expiry_minutes * 60
        ):
            return

        if profile is None:
            self._record(
                moment, "entry_skipped",
                "no GEX profile: open interest is unavailable for this expiry",
            )
            return
        direction = profile.direction
        if direction == 0:
            self._record(moment, "entry_skipped", profile.reason, regime=profile.regime)
            return

        t = self.clock.time_to_expiry(moment, expiry)
        quote = select_atm_straddle(
            bar.close, expiry, t, bar.atm_iv, self.source, self.surface,
            self.cfg.risk_free_rate,
        )
        if quote is None:
            self._record(
                moment, "entry_skipped",
                f"the {expiry} ATM straddle carries no premium or gamma left "
                f"(T={t * 365 * 24:.2f}h)",
                regime=profile.regime,
            )
            return

        equity = self.portfolio.equity(None, bar.close)
        sizing = size_straddles(
            equity, quote, bar.close, direction, self.cfg.sizing, self.source,
            self.margin_model,
        )
        if not sizing.ok:
            self._record(moment, "entry_skipped", sizing.reason, regime=profile.regime)
            return

        quantity = direction * sizing.contracts
        fills = self._open_legs(quote, quantity, moment, execution)
        if fills is None:
            return
        call_fill, put_fill = fills

        self._hedge_realised_at_entry = self.portfolio.hedge_realised
        self._fees_at_entry = self.portfolio.fees_paid - call_fill.fees - put_fill.fees
        self.portfolio.open_straddle(
            StraddlePosition(
                strike=quote.strike,
                expiry=expiry,
                quantity=quantity,
                call_entry=call_fill.price,
                put_entry=put_fill.price,
                entry_time=moment,
                entry_future=bar.close,
                entry_iv=quote.iv,
                entry_delta=quote.delta,
                regime=profile.regime,
            )
        )
        self._entries_this_session += 1
        self.regime_trades[profile.regime] = self.regime_trades.get(profile.regime, 0) + 1

        side = "bought" if direction > 0 else "sold"
        cash = abs(quantity) * (call_fill.price + put_fill.price) * self.source.option.multiplier
        intent = "scalp gamma" if direction > 0 else "collect theta"
        self._record(
            moment, "entry",
            f"{side} {sizing.contracts} {expiry} {quote.strike:g} straddle "
            f"@ {call_fill.price + put_fill.price:.2f} "
            f"(C {call_fill.price:.2f} / P {put_fill.price:.2f}, IV {quote.iv:.3f}) "
            f"for ${cash:,.0f} {'debit' if direction > 0 else 'credit'}; "
            f"{sizing.requirement_kind} ${sizing.total_margin:,.0f} of "
            f"${sizing.budget:,.0f} budget -- {profile.reason}, so {intent}",
            regime=profile.regime,
        )

    def _open_legs(
        self, quote: StraddleQuote, quantity: int, moment: datetime,
        execution: ExecutionHandler,
    ) -> tuple[Fill, Fill] | None:
        """Fill both legs, or leave the book flat.

        A straddle with one leg on is a naked option, not a straddle -- it
        carries the wrong sign of delta and none of the gamma exposure the
        regime called for.  If the second leg does not fill, the first is
        unwound immediately rather than held.  In the backtest this cannot
        happen; in live it can, which is the case worth writing for.
        """
        call_fill = execution.execute_option(quote.call, quantity, moment)
        if call_fill is None:
            self._record(moment, "entry_failed", "the call leg did not fill")
            return None
        self.fills.append(call_fill)
        self.portfolio.charge_fees(call_fill.fees)

        put_fill = execution.execute_option(quote.put, quantity, moment)
        if put_fill is not None:
            self.fills.append(put_fill)
            self.portfolio.charge_fees(put_fill.fees)
            return call_fill, put_fill

        unwind = execution.execute_option(quote.call, -call_fill.quantity, moment)
        if unwind is None:
            self._record(
                moment, "entry_failed",
                f"the put leg did not fill and the {call_fill.quantity:+d} call leg "
                "could not be unwound -- the book is holding a naked option and "
                "needs manual attention",
            )
            return None
        self.fills.append(unwind)
        self.portfolio.charge_fees(unwind.fees)
        self.portfolio.option_realised += (
            call_fill.quantity * (unwind.price - call_fill.price)
            * self.source.option.multiplier
        )
        self._record(
            moment, "entry_failed",
            "the put leg did not fill; the call leg was unwound and the book is flat",
        )
        return None

    # -- exits -------------------------------------------------------------

    def _position_pnl(self, quote: StraddleQuote | None, future_price: float) -> float:
        """P&L of the open position since entry: straddle plus its hedge.

        The two legs only mean something together.  A long 0DTE straddle is
        *supposed* to bleed on the mark -- that is theta -- and make it back
        through hedge realisations as the underlying moves.  Judging either
        leg alone would stop every long trade out on the first hour of decay
        and let every short trade run through an adverse move it was already
        losing on.
        """
        book = self.portfolio
        return (
            book.straddle_unrealised(quote)
            + book.hedge.unrealised(future_price, self.source.hedge.multiplier)
            + (book.hedge_realised - self._hedge_realised_at_entry)
            - (book.fees_paid - self._fees_at_entry)
        )

    def _check_exits(
        self, bar: MarketBar, moment: datetime, quote: StraddleQuote | None,
        profile: GexProfile | None, execution: ExecutionHandler,
    ) -> None:
        position = self.portfolio.straddle
        if position is None or quote is None:
            return
        cfg = self.cfg.strategy

        seconds_left = self.clock.seconds_to_expiry(moment, position.expiry)
        pnl = self._position_pnl(quote, bar.close)
        premium = position.premium_at_risk(self.source.option.multiplier)
        reason: str | None = None

        if seconds_left <= cfg.close_before_expiry_minutes * 60:
            reason = f"{cfg.close_before_expiry_minutes}m to expiry"
        elif (
            cfg.exit_on_regime_flip
            and profile is not None
            and profile.direction != 0
            and profile.direction != position.direction
        ):
            reason = (
                f"GEX flipped to {profile.regime}: {profile.reason}. The position "
                f"is on the wrong side of dealer hedging"
            )
        elif position.is_long:
            reason = self._long_exit_reason(cfg, pnl, premium)
        else:
            reason = self._short_exit_reason(cfg, position, quote)

        if reason is None:
            limit = cfg.daily_loss_limit_pct
            if limit is not None:
                equity = self.portfolio.equity(quote, bar.close)
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

    def _long_exit_reason(self, cfg, pnl: float, premium: float) -> str | None:
        """Stops for the long (negative-GEX) side, measured on position P&L."""
        if premium <= 0.0:
            return None
        if cfg.long_stop_loss_pct is not None and pnl <= -cfg.long_stop_loss_pct * premium:
            return (
                f"stop: the scalp is -${-pnl:,.0f}, past {cfg.long_stop_loss_pct:.0%} "
                f"of the ${premium:,.0f} debit -- realised vol is not paying for "
                "the gamma"
            )
        if (
            cfg.long_take_profit_pct is not None
            and pnl >= cfg.long_take_profit_pct * premium
        ):
            return (
                f"target: the scalp is +${pnl:,.0f}, {cfg.long_take_profit_pct:.0%} "
                f"of the ${premium:,.0f} debit"
            )
        return None

    def _short_exit_reason(self, cfg, position, quote: StraddleQuote) -> str | None:
        """Stops for the short (positive-GEX) side, measured on premium.

        The short side is judged on the mark rather than on position P&L
        because the risk being managed is different: what ends a short
        straddle badly is the premium running away, and that has to be cut
        on the premium itself, before the hedge has finished paying for it.
        """
        entry = position.entry_premium
        if entry <= 0.0:
            return None
        mark = quote.price
        if (
            cfg.short_stop_loss_premium_multiple is not None
            and mark >= entry * cfg.short_stop_loss_premium_multiple
        ):
            return (
                f"stop: mark {mark:.2f} >= "
                f"{cfg.short_stop_loss_premium_multiple:g}x entry {entry:.2f}"
            )
        if (
            cfg.short_take_profit_pct is not None
            and mark <= entry * (1.0 - cfg.short_take_profit_pct)
        ):
            return (
                f"target: captured {cfg.short_take_profit_pct:.0%} of the "
                f"{entry:.2f} credit"
            )
        return None

    def _close_position(
        self, bar: MarketBar, moment: datetime, quote: StraddleQuote,
        execution: ExecutionHandler, reason: str,
    ) -> None:
        position = self.portfolio.straddle
        assert position is not None
        regime = position.regime
        quantity = position.quantity

        call_fill = execution.execute_option(quote.call, -quantity, moment)
        put_fill = execution.execute_option(quote.put, -quantity, moment)
        if call_fill is None or put_fill is None:
            filled = "call" if call_fill is not None else "put" if put_fill is not None else "neither"
            self._record(
                moment, "exit_failed",
                f"could not close ({reason}); {filled} leg filled",
                regime=regime,
            )
            return
        for fill in (call_fill, put_fill):
            self.fills.append(fill)
            self.portfolio.charge_fees(fill.fees)

        # Attribute the whole position -- straddle and hedge -- before the
        # book is torn down, so the regime that opened it is charged with
        # what it actually made.
        position_pnl = self._position_pnl(quote, bar.close)
        option_pnl = self.portfolio.close_straddle(call_fill.price, put_fill.price)
        self._record(
            moment, "exit",
            f"closed {abs(quantity)} {position.strike:g} straddle @ "
            f"{call_fill.price + put_fill.price:.2f} ({reason}); "
            f"straddle P&L ${option_pnl:,.0f}, position P&L ${position_pnl:,.0f}",
            regime=regime,
        )

        if self.cfg.hedge.flatten_hedge_on_exit and self.portfolio.hedge.quantity:
            self._flatten_hedge(bar, moment, execution)
            position_pnl = (
                self.portfolio.hedge_realised - self._hedge_realised_at_entry
                + option_pnl
                - (self.portfolio.fees_paid - self._fees_at_entry)
            )
        self.regime_pnl[regime] = self.regime_pnl.get(regime, 0.0) + position_pnl

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
        self, bar: MarketBar, moment: datetime, quote: StraddleQuote | None,
        execution: ExecutionHandler,
    ) -> None:
        if self.portfolio.straddle is None and self.portfolio.hedge.quantity == 0:
            return

        net_delta = self.portfolio.net_delta_units(quote)
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
        self, bar: MarketBar, moment: datetime, quote: StraddleQuote | None,
        profile: GexProfile | None,
    ) -> BarState:
        position = self.portfolio.straddle
        option_delta = self.portfolio.option_delta_units(quote)
        hedge_delta = self.portfolio.hedge_delta_units()
        net = option_delta + hedge_delta
        return BarState(
            timestamp=moment,
            future=bar.close,
            atm_iv=bar.atm_iv,
            time_to_expiry=(
                self.clock.time_to_expiry(moment, position.expiry) if position else 0.0
            ),
            straddle_mark=quote.price if quote else None,
            call_mark=quote.call.price if quote else None,
            put_mark=quote.put.price if quote else None,
            option_delta_units=option_delta,
            hedge_delta_units=hedge_delta,
            net_delta_units=net,
            gamma_units=self.portfolio.option_gamma_units(quote),
            vega_dollars=self.portfolio.option_vega(quote),
            theta_dollars=self.portfolio.option_theta(quote),
            hedge_contracts=self.portfolio.hedge.quantity,
            straddle_contracts=position.quantity if position else 0,
            direction=position.direction if position else 0,
            strike=position.strike if position else None,
            equity=self.portfolio.equity(quote, bar.close),
            realised_pnl=self.portfolio.realised_pnl,
            fees_paid=self.portfolio.fees_paid,
            in_band=self.cfg.hedge.in_band(net) if position or hedge_delta else True,
            gex_total=profile.total_gex if profile else None,
            gex_flip=profile.flip_point if profile else None,
            gex_regime=profile.regime if profile else NEUTRAL,
            distance_to_flip=profile.distance_to_flip if profile else None,
        )
