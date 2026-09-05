"""The GEX-directed, delta-hedged straddle strategy.

One bar (or one live poll) at a time, in this order:

  1. mark the open straddle and recompute greeks
  2. read GEX at the current spot -- total, flip point, regime -- across the
     front expiries, and update the persistence streak
  3. check exits -- the DTE floor, a *confirmed* regime flip, the
     directional stop/target, the daily loss limit
  4. check entry -- if flat, inside the entry window and past every gate,
     take the side the regime implies
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

The tenor, and the end of the day
---------------------------------
The traded series is today's expiry, chosen by ``StrategyConfig.tenor()``.
The end of the day is where the rules concentrate, and they are all hard
exits -- nothing in the gate machinery can delay any of them:

  * ``close_before_expiry_minutes`` before settlement the 0DTE position is
    closed: the last quarter hour is where an ATM straddle's gamma diverges
    and the hedger cannot keep up;
  * in that same window the *next* session's series is eligible to be
    opened (``roll_at_expiry``), outside the entry window but through every
    GEX gate, and carried overnight to become tomorrow's 0DTE position --
    so ``_roll_session`` re-marks the book at the new day's first price
    rather than assuming it starts flat, and the hedge band is
    session-aware;
  * a series across a weekend or holiday is never entered and a position
    on one is closed at the buffer on the last session before the gap
    (``hold_over_weekends``): a gap with no session in it cannot be hedged;
  * inside the blackout around a scheduled event -- an FOMC statement, a
    CPI print -- the position is closed and nothing is opened
    (``events``, ``event_blackout_minutes_*``), for the same reason.

The GEX read is a blend over the front expiries out to the traded one, so
during the roll window a bar reads open interest for both today's and
tomorrow's series rather than one.

The gates
---------
Four of them, described in ``GatesConfig``.  Two are inside the profile
(confidence, distance to the flip), one is checked here against the
calculator (the ensemble), and one is purely local (persistence).  The entry
window is the fifth thing that can block an entry and is checked here as
well, in ``_try_entry`` rather than in the bar loop -- so the backtest and
the live runner inherit it identically instead of each deciding for itself
which bars to offer.

Every block is recorded as an event carrying the gate that caused it, which
is what makes ``deltahedger sweep --gates`` and the live journal able to say
*why* nothing was traded rather than only that nothing was.

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
from .chain import StraddleQuote, TenorPolicy, price_option, select_expiry, select_atm_straddle
from .config import Config
from .data.base import MarketBar
from .events import EventCalendar
from .gex import (
    GATE_ENSEMBLE,
    GATE_ENTRY_WINDOW,
    GATE_PERSISTENCE,
    NEUTRAL,
    ExpiryBook,
    GexCalculator,
    GexProfile,
    OpenInterestProvider,
    StrikeOpenInterest,
)
from .hedger import DeltaHedger, hedge_cost_per_contract
from .instruments import RiskSource
from .portfolio import Portfolio, StraddlePosition
from .session import SessionClock, is_trading_day
from .sizing import MarginModel, build_margin_model, size_straddles
from .volsurface import VolSurface

log = logging.getLogger(__name__)

#: Two further reasons an entry is refused, named in the event log the way
#: the GEX gates are so the journal and the sweep can count them by cause.
#: They are risk rules rather than gates -- neither is switchable from
#: ``gates:`` and neither can be swept -- but a stand-aside is a stand-aside
#: and the attribution should say which rule it was.
BLOCK_EVENT_BLACKOUT = "event_blackout"
BLOCK_WEEKEND_GAP = "weekend_gap"


@dataclass
class StrategyEvent:
    """Something worth recording: a fill, a skipped entry, a band breach.

    ``gate`` names the gate responsible when the event is a block -- an
    entry that was not taken, or a flip that was not acted on.  It is empty
    for everything else.  Recording it here rather than only in the prose of
    ``detail`` is what lets the sweep and the live journal count blocks by
    cause without parsing sentences.
    """

    timestamp: datetime
    kind: str
    detail: str
    net_delta: float = 0.0
    equity: float = 0.0
    regime: str = ""
    gate: str = ""


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
    #: |total|/gross GEX: how directional the read was, on a 0-1 scale.
    gex_confidence: float | None = None
    #: The gate that forced a NEUTRAL profile, empty when the read stood.
    gex_gate: str = ""
    #: The regime after the persistence filter -- what the strategy is
    #: entitled to act on, as opposed to what this bar happened to read.
    confirmed_regime: str = NEUTRAL
    #: Trading days to expiry of the traded series (of the series that
    #: *would* be traded, when flat), or None when none is listed.
    days_to_expiry: int | None = None
    #: Whether the bar fell inside the regular session, which decides which
    #: band width applied to it.
    in_session: bool = True
    #: The band half-width that applied to this bar, in delta units. Under
    #: Whalley-Wilmott it is a function of the book's gamma and changes bar
    #: to bar; this is the number the "in_band" flag was judged against.
    band_half_width: float = 0.0
    #: The scheduled event whose blackout this bar fell inside, if any.
    event_blackout: str = ""


class GexStraddleStrategy:
    def __init__(
        self,
        cfg: Config,
        source: RiskSource | None = None,
        margin_model: MarginModel | None = None,
        open_interest: OpenInterestProvider | None = None,
        events: EventCalendar | None = None,
    ):
        self.cfg = cfg
        self.source = source or cfg.source
        self.clock = SessionClock(self.source)
        self.surface = VolSurface(cfg.vol)
        self.hedger = DeltaHedger(
            cfg.hedge, self.source,
            cost_per_contract=hedge_cost_per_contract(cfg.costs, self.source),
            risk_free_rate=cfg.risk_free_rate,
        )
        self.margin_model = margin_model or build_margin_model(
            cfg.sizing, self.source, cfg.risk_free_rate
        )
        self.gex = GexCalculator(
            cfg.gex, self.source, self.surface, cfg.risk_free_rate, cfg.gates
        )
        self.tenor: TenorPolicy = cfg.strategy.tenor()
        #: The event blackout calendar. ``self.events`` is the decision log.
        self.calendar: EventCalendar = (
            events if events is not None
            else EventCalendar.from_config(cfg.strategy, self.clock.tz)
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
        #: Cached open interest per expiry, and when it was read. OI is an
        #: end-of-day figure, so it is re-read on a timer while the
        #: *profile* is recomputed at the live spot on every bar.
        self._oi: dict[date, list[StrikeOpenInterest]] = {}
        self._oi_read_at: datetime | None = None
        self._oi_expiries: tuple[date, ...] = ()
        #: The books the last profile was built from, kept so the ensemble
        #: gate can re-price the same input without re-reading it.
        self._books: list[ExpiryBook] = []
        #: The persistence streak: the regime the last few bars have read,
        #: and how many in a row. ``_confirmed_regime`` is the one the
        #: strategy is allowed to act on.
        self._streak_regime: str = ""
        self._streak_bars: int = 0
        self._confirmed_regime: str = NEUTRAL
        # Baselines for measuring one position's P&L, set at each entry.
        self._hedge_realised_at_entry = 0.0
        self._fees_at_entry = 0.0

    # -- main loop ------------------------------------------------------

    def on_bar(self, bar: MarketBar, execution: ExecutionHandler) -> BarState:
        moment = self.clock.localize(bar.timestamp)
        self._roll_session(moment, bar.close)

        quote = self._mark_open_straddle(bar, moment)
        profile = self._read_gex(bar, moment)
        self._update_persistence(profile)

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
        regime: str = "", gate: str = "",
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
                gate=gate,
            )
        )
        log.debug("%s | %s | %s", moment, kind, detail)

    # -- the traded series -----------------------------------------------

    def _traded_expiry(self, moment: datetime) -> date | None:
        """The expiry the tenor policy says to trade, or ``None``.

        When a position is open this is the series it is *on*, not the one
        that would be chosen afresh -- otherwise the exit checks would
        measure the DTE of a series the book does not hold.
        """
        position = self.portfolio.straddle
        if position is not None:
            return position.expiry
        return select_expiry(self.clock, moment, self.tenor)

    def _todays_expiry(self, moment: datetime) -> date | None:
        """Today's series, if today is a session and it has not settled."""
        day = moment.date()
        if not is_trading_day(day) or moment >= self.clock.expiry_datetime(day):
            return None
        return day

    def _in_buffer(self, moment: datetime) -> bool:
        """Whether ``moment`` is inside today's pre-settlement buffer.

        This is the window in which today's series is no longer eligible
        and, with ``roll_at_expiry`` on, tomorrow's may be opened in its
        place regardless of the entry window.
        """
        today = self._todays_expiry(moment)
        if today is None:
            return False
        return self.clock.seconds_to_expiry(moment, today) <= self.tenor.buffer_seconds

    def _in_roll_window(self, moment: datetime) -> bool:
        return self.cfg.strategy.roll_at_expiry and self._in_buffer(moment)

    def _classification_expiries(self, moment: datetime) -> list[date]:
        """The series the GEX read is built from.

        Normally the front expiries out to the traded one.  When nothing is
        eligible to trade -- a Friday afternoon, say, with today's series
        inside the buffer and Monday's across the weekend -- the read is
        still worth having, for the journal and for the persistence streak
        that carries into the next entry, so it falls back to the listed
        series inside the tenor's range with nothing traded against it.
        """
        traded = self._traded_expiry(moment)
        if traded is not None:
            return self._blend_expiries(moment, traded)
        listed = self.clock.candidate_expiries(moment, self.tenor.max_days)
        if not listed:
            return []
        if not self.cfg.gex.blend_front_expiries:
            return [listed[0]]
        return listed[: self.cfg.gex.blend_max_expiries]

    def _blend_expiries(self, moment: datetime, traded: date) -> list[date]:
        """0DTE out to the traded series, nearest first.

        This is the set of expiries whose gamma a dealer is carrying in
        front of them right now.  It is capped by ``blend_max_expiries``
        because every entry costs an open-interest read, which is a
        subscription per listed strike per right on the live path.
        """
        if not self.cfg.gex.blend_front_expiries:
            return [traded]
        listed = self.clock.candidate_expiries(
            moment, max(self.clock.days_to_expiry(moment, traded), 0)
        )
        expiries = [e for e in listed if e <= traded]
        if traded not in expiries:
            expiries.append(traded)
        return sorted(expiries)[: self.cfg.gex.blend_max_expiries]

    # -- GEX -------------------------------------------------------------

    def _read_gex(self, bar: MarketBar, moment: datetime) -> GexProfile | None:
        """The GEX profile at this bar's spot, or ``None`` if unavailable.

        Open interest is cached on ``gex.refresh_seconds`` because it is an
        end-of-day figure that does not move intraday.  The *profile* is
        rebuilt every bar regardless: the regime is a statement about where
        spot sits relative to the flip point, and reusing a stale spot would
        mean never seeing the crossing the strategy exists to trade.

        A read that fails for one expiry does not discard the others.  The
        blend is an aggregate, and an aggregate missing one series is a
        worse answer than the full one but a far better answer than none --
        the alternative is standing aside for the rest of the session
        because one strike would not quote.
        """
        if not self.cfg.gex.enabled or self.open_interest is None:
            return None
        expiries = self._classification_expiries(moment)
        if not expiries:
            return None

        stale = (
            not self._oi
            or self._oi_expiries != tuple(expiries)
            or self._oi_read_at is None
            or (moment - self._oi_read_at).total_seconds()
            >= self.cfg.gex.refresh_seconds
        )
        if stale:
            fresh: dict[date, list[StrikeOpenInterest]] = {}
            for expiry in expiries:
                try:
                    fresh[expiry] = list(
                        self.open_interest.open_interest(moment, bar.close, expiry)
                    )
                except Exception as exc:  # noqa: BLE001 - a bad read must not halt the run
                    log.warning(
                        "could not read open interest for %s (%s); holding the "
                        "last read for that expiry", expiry, exc,
                    )
                    if expiry in self._oi:
                        fresh[expiry] = self._oi[expiry]
            if not fresh:
                return None
            self._oi = fresh
            self._oi_read_at = moment
            self._oi_expiries = tuple(expiries)

        self._books = [
            ExpiryBook.of(
                expiry,
                self.clock.time_to_expiry(moment, expiry),
                self._oi.get(expiry, []),
                self.clock.days_to_expiry(moment, expiry),
            )
            for expiry in expiries
        ]
        self._profile = self.gex.blended_profile(bar.close, self._books, bar.atm_iv)
        return self._profile

    # -- persistence -----------------------------------------------------

    def _update_persistence(self, profile: GexProfile | None) -> None:
        """Advance the streak, and confirm a regime once it has held.

        Without the gate the confirmed regime is simply the current one, so
        every downstream check reads the same field whether or not
        persistence is switched on -- there is no second code path that
        could behave differently.

        A bar with no profile at all resets the streak rather than extending
        it: a gap in the read is not evidence that the regime held through
        it.
        """
        if profile is None:
            self._streak_regime, self._streak_bars = "", 0
            self._confirmed_regime = NEUTRAL
            return
        if not self.cfg.gates.persistence:
            self._streak_regime, self._streak_bars = profile.regime, 1
            self._confirmed_regime = profile.regime
            return

        if profile.regime == self._streak_regime:
            self._streak_bars += 1
        else:
            self._streak_regime, self._streak_bars = profile.regime, 1
        if self._streak_bars >= self.cfg.gates.persistence_bars:
            self._confirmed_regime = profile.regime

    def _confirmed_direction(self) -> int:
        """The side the confirmed regime implies, 0 for none."""
        from .gex import LONG_STRADDLE, NEGATIVE, POSITIVE, SHORT_STRADDLE, STAND_ASIDE

        if self._confirmed_regime == NEGATIVE:
            return LONG_STRADDLE
        if self._confirmed_regime == POSITIVE:
            return SHORT_STRADDLE
        return STAND_ASIDE

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

        # The entry window is checked here rather than in the bar loop, so
        # the backtest and the live runner cannot disagree about which bars
        # are eligible. The end-of-day roll is the one exemption: inside
        # today's pre-settlement buffer the next series may be opened
        # whatever the window says, because that is the only moment it can
        # be. Exits are never windowed.
        local = moment.timetz().replace(tzinfo=None)
        in_window = cfg.entry_time <= local <= cfg.entry_cutoff_time
        if self.cfg.gates.entry_window and not in_window and not self._in_roll_window(moment):
            return

        # The event blackout is a risk rule, not a gate: it is checked before
        # the read is even consulted, and it is recorded so a quiet
        # afternoon around an FOMC statement is attributable afterwards.
        event = self.calendar.blackout(moment)
        if event is not None:
            self._record(
                moment, "entry_skipped",
                f"inside the blackout around {event} "
                f"(-{cfg.event_blackout_minutes_before}m/+"
                f"{cfg.event_blackout_minutes_after}m)",
                gate=BLOCK_EVENT_BLACKOUT,
            )
            return

        expiry = self._traded_expiry(moment)
        if expiry is None:
            self._record_no_expiry(moment)
            return

        # With the roll off, nothing is opened inside today's buffer even
        # when the entry window would allow it: the book stays flat from
        # the buffer to the next session's window.
        if not cfg.roll_at_expiry and self._in_buffer(moment) and expiry != moment.date():
            self._record(
                moment, "entry_skipped",
                f"inside today's {cfg.close_before_expiry_minutes}m pre-settlement "
                "buffer and roll_at_expiry is off; nothing is opened until the "
                "next session",
            )
            return

        days_left = self.clock.days_to_expiry(moment, expiry)
        if self.tenor.should_close(days_left):
            # Nothing listed far enough out to be worth opening: entering
            # here would open a position already eligible for the DTE exit.
            self._record(
                moment, "entry_skipped",
                f"the {expiry} series is {days_left}DTE, at or below the "
                f"{self.tenor.close_days}DTE close-out floor",
            )
            return

        if profile is None:
            self._record(
                moment, "entry_skipped",
                "no GEX profile: open interest is unavailable for these expiries",
            )
            return
        if profile.direction == 0:
            self._record(
                moment, "entry_skipped", profile.reason,
                regime=profile.regime, gate=profile.gate,
            )
            return

        direction = self._confirmed_direction()
        if direction == 0 or direction != profile.direction:
            self._record(
                moment, "entry_skipped",
                f"the {profile.regime} read has held {self._streak_bars} of the "
                f"{self.cfg.gates.persistence_bars} bars it needs before it "
                "counts as the regime rather than as spot crossing a level",
                regime=profile.regime, gate=GATE_PERSISTENCE,
            )
            return

        if self.cfg.gates.ensemble:
            ensemble = self.gex.ensemble(bar.close, self._books, bar.atm_iv)
            if not ensemble.unanimous or ensemble.regime != profile.regime:
                self._record(
                    moment, "entry_skipped", ensemble.detail,
                    regime=profile.regime, gate=GATE_ENSEMBLE,
                )
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
            f"{side} {sizing.contracts} {expiry} ({days_left}DTE) "
            f"{quote.strike:g} straddle "
            f"@ {call_fill.price + put_fill.price:.2f} "
            f"(C {call_fill.price:.2f} / P {put_fill.price:.2f}, IV {quote.iv:.3f}) "
            f"for ${cash:,.0f} {'debit' if direction > 0 else 'credit'}; "
            f"{sizing.requirement_kind} ${sizing.total_margin:,.0f} of "
            f"${sizing.budget:,.0f} budget -- {profile.reason}, so {intent}",
            regime=profile.regime,
        )

    def _record_no_expiry(self, moment: datetime) -> None:
        """Say *why* nothing is eligible, since two rules can be the cause.

        The weekend rule is the one worth naming: the series exists, it is
        inside the tenor, and it was refused because it sits on the far
        side of a gap.  That is a decision the journal should be able to
        count, not a listing problem.
        """
        if not self.tenor.hold_over_weekends:
            across = self.clock.select_expiry(
                moment, self.tenor.min_days, self.tenor.max_days,
                self.tenor.prefer_days,
                min_seconds_to_expiry=self.tenor.buffer_seconds,
                hold_over_gaps=True,
            )
            if across is not None and self.clock.gap_before(moment, across):
                self._record(
                    moment, "entry_skipped",
                    f"the {across} series is on the far side of a weekend or "
                    "holiday; no positions are held over a gap",
                    gate=BLOCK_WEEKEND_GAP,
                )
                return
        self._record(
            moment, "entry_skipped",
            f"no expiry eligible between {self.tenor.min_days} and "
            f"{self.tenor.max_days} trading days out (a series inside the "
            f"{self.tenor.close_before_expiry_minutes}m pre-settlement buffer "
            "does not count)",
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
        days_left = self.clock.days_to_expiry(moment, position.expiry)
        pnl = self._position_pnl(quote, bar.close)
        premium = position.premium_at_risk(self.source.option.multiplier)
        reason: str | None = None

        # The hard exits lead the ladder and none of them is gated. In
        # order: the pre-settlement buffer (the last quarter hour is where an
        # ATM straddle's gamma diverges), the DTE floor for a multi-session
        # tenor, the weekend rule, and the event blackout -- the last two
        # because a gap with no session in it cannot be hedged.
        if seconds_left <= self.tenor.buffer_seconds:
            reason = f"{cfg.close_before_expiry_minutes}m to expiry"
        elif self.tenor.should_close(days_left):
            reason = (
                f"{days_left}DTE, at the {self.tenor.close_days}DTE close-out floor"
            )
        elif (gap := self._gap_exit_reason(moment, position)) is not None:
            reason = gap
        elif (event := self.calendar.blackout(moment)) is not None:
            reason = (
                f"inside the blackout around {event} "
                f"(-{cfg.event_blackout_minutes_before}m/+"
                f"{cfg.event_blackout_minutes_after}m)"
            )
        elif cfg.exit_on_regime_flip and profile is not None:
            reason = self._flip_exit_reason(bar, moment, position, profile)

        if reason is None:
            reason = (
                self._long_exit_reason(cfg, pnl, premium)
                if position.is_long
                else self._short_exit_reason(cfg, position, quote)
            )

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

    def _gap_exit_reason(self, moment: datetime, position) -> str | None:
        """Close before a weekend or holiday a position would otherwise span.

        Fires on the last session before a gap, at the same pre-settlement
        buffer the 0DTE exit uses, for a position whose series is on the
        far side of that gap.  At the shipped tenor this cannot happen --
        such a series is never entered -- so this is the safety net for a
        wider tenor, or a config that switched the weekend rule on with a
        position already open.
        """
        if self.tenor.hold_over_weekends:
            return None
        today = self._todays_expiry(moment)
        if today is None or position.expiry <= today or not self.clock.gap_after(moment):
            return None
        if self.clock.seconds_to_expiry(moment, today) > self.tenor.buffer_seconds:
            return None
        return (
            f"the {position.expiry} series is on the far side of a weekend or "
            "holiday; no positions are held over a gap"
        )

    def _flip_exit_reason(
        self, bar: MarketBar, moment: datetime, position, profile: GexProfile
    ) -> str | None:
        """Close only on a flip the gates are willing to stand behind.

        A regime that has not yet held ``persistence_bars`` is spot crossing
        a level rather than positioning changing, and closing on it churns
        the book at exactly the wrong moments -- open interest, the only
        input, has not moved at all.  The same is true of a flip only part
        of the ensemble agrees with.

        A blocked flip is recorded rather than silently dropped: "the
        position stayed open through an opposing read" is a decision, and it
        is one worth being able to count afterwards.  The hard exits above
        are unaffected -- a gate can delay a side change, never an exit.
        """
        if profile.direction == 0 or profile.direction == position.direction:
            return None

        if self._confirmed_direction() != profile.direction:
            self._record(
                moment, "exit_deferred",
                f"GEX reads {profile.regime} against the open position, but "
                f"only for {self._streak_bars} of the "
                f"{self.cfg.gates.persistence_bars} consecutive bars a flip "
                "needs; holding",
                regime=position.regime, gate=GATE_PERSISTENCE,
            )
            return None

        if self.cfg.gates.ensemble:
            ensemble = self.gex.ensemble(bar.close, self._books, bar.atm_iv)
            if not ensemble.unanimous or ensemble.regime != profile.regime:
                self._record(
                    moment, "exit_deferred",
                    f"GEX reads {profile.regime} against the open position but "
                    f"{ensemble.detail}; holding",
                    regime=position.regime, gate=GATE_ENSEMBLE,
                )
                return None

        return (
            f"GEX flipped to {profile.regime}: {profile.reason}. The position "
            f"is on the wrong side of dealer hedging"
        )

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
        """Take the hedge leg to zero, in orders no larger than the cap.

        A book sized to the margin limit can be carrying well over the
        per-order cap in MES by the time it is closed -- an ATM straddle's
        delta runs to +/-100 units per contract near the bell -- and the
        live broker refuses any single order past it.  So the flatten is
        sent as a sequence of capped orders rather than one, and stops at
        the first that does not fill; whatever is left is an orphaned
        hedge, which the band (zero-width with no straddle behind it)
        closes on the following passes.
        """
        wanted = -self.portfolio.hedge.quantity
        cap = self.cfg.hedge.max_hedge_contracts
        remaining, closed, pnl, orders, notional = wanted, 0, 0.0, 0, 0.0
        while remaining != 0:
            chunk = max(-cap, min(cap, remaining))
            fill = execution.execute_hedge(chunk, bar.close, moment)
            if fill is None or fill.quantity == 0:
                break
            self.fills.append(fill)
            self.portfolio.charge_fees(fill.fees)
            pnl += self.portfolio.apply_hedge_fill(fill.quantity, fill.price)
            remaining -= fill.quantity
            closed += fill.quantity
            notional += abs(fill.quantity) * fill.price
            orders += 1
        if orders == 0:
            self._record(
                moment, "hedge_failed",
                f"could not flatten {abs(wanted)} {self.source.hedge.symbol}; the "
                "band will close it on the next pass",
            )
            return
        average = notional / abs(closed)
        self._record(
            moment, "hedge_flatten",
            f"closed {abs(closed)} {self.source.hedge.symbol} @ {average:.2f}"
            + (f" in {orders} orders" if orders > 1 else "")
            + f"; hedge P&L ${pnl:,.0f}"
            + (f"; {abs(remaining)} left for the band" if remaining else ""),
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
        decision = self.hedger.decide(
            net_delta, elapsed, self.clock.in_session(moment),
            gamma_units=self.portfolio.option_gamma_units(quote),
            time_to_expiry=quote.time_to_expiry if quote else 0.0,
        )
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
        in_session = self.clock.in_session(moment)
        expiry = position.expiry if position else self._traded_expiry(moment)
        gamma_units = self.portfolio.option_gamma_units(quote)
        band = self.hedger.half_width(
            gamma_units, quote.time_to_expiry if quote else 0.0, in_session
        )
        event = self.calendar.blackout(moment)
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
            gamma_units=gamma_units,
            vega_dollars=self.portfolio.option_vega(quote),
            theta_dollars=self.portfolio.option_theta(quote),
            hedge_contracts=self.portfolio.hedge.quantity,
            straddle_contracts=position.quantity if position else 0,
            direction=position.direction if position else 0,
            strike=position.strike if position else None,
            equity=self.portfolio.equity(quote, bar.close),
            realised_pnl=self.portfolio.realised_pnl,
            fees_paid=self.portfolio.fees_paid,
            in_band=(
                self.hedger.in_band(net, band)
                if position or hedge_delta
                else True
            ),
            gex_total=profile.total_gex if profile else None,
            gex_flip=profile.flip_point if profile else None,
            gex_regime=profile.regime if profile else NEUTRAL,
            distance_to_flip=profile.distance_to_flip if profile else None,
            gex_confidence=profile.confidence if profile else None,
            gex_gate=profile.gate if profile else "",
            confirmed_regime=self._confirmed_regime,
            days_to_expiry=(
                self.clock.days_to_expiry(moment, expiry) if expiry else None
            ),
            in_session=in_session,
            band_half_width=band,
            event_blackout=str(event) if event is not None else "",
        )
