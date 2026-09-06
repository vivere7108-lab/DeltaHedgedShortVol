"""The GEX-directed, delta-hedged straddle strategy.

One bar (or one live poll) at a time, in this order:

  1. mark the open straddle and recompute greeks
  2. read GEX at the current spot -- total, flip point, regime -- across the
     front expiries, and update the persistence streak
  3. check exits -- the DTE floor, a *confirmed* regime flip, the
     directional stop/target, the daily loss limit
  4. check entry -- if flat, inside the entry window and past every gate,
     take the side the regime implies, sized to whichever of the risk
     budget, the gamma ceiling and the capital cap allows least
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
  * once today's series has settled, in the ``roll_window_minutes`` after
    the bell, the *next* session's series is eligible to be opened
    (``roll_at_expiry``), outside the entry window but through every GEX
    gate, and carried overnight to become tomorrow's 0DTE position --
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
from datetime import date, datetime, timedelta

from .broker.base import ExecutionHandler, Fill
from .chain import (
    OptionQuote,
    StraddleQuote,
    TenorPolicy,
    price_option,
    select_atm_straddle,
    select_expiry,
)
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
from .sizing import BINDING_NAMES, MarginModel, build_margin_model, size_straddles
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
        #: Which sizing constraint decided each entry's size. A book that is
        #: always bound by the capital cap is one whose stop ladder cannot
        #: fire, which is what the risk budget exists to prevent.
        self.sizing_binds: dict[str, int] = {}

        self._last_hedge_time: datetime | None = None
        self._last_moment: datetime | None = None
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
        #: The persistence streak: the regime the recent bars have read,
        #: when that streak began, and how many bars it has run for (the
        #: count is for the log). ``_confirmed_regime`` is the one the
        #: strategy is allowed to act on.
        self._streak_regime: str = ""
        self._streak_started: datetime | None = None
        self._streak_bars: int = 0
        self._confirmed_regime: str = NEUTRAL
        # Baselines for measuring one position's P&L, set at each entry.
        self._hedge_realised_at_entry = 0.0
        self._option_realised_at_entry = 0.0
        self._fees_at_entry = 0.0

    # -- main loop ------------------------------------------------------

    def on_bar(self, bar: MarketBar, execution: ExecutionHandler) -> BarState:
        moment = self.clock.localize(bar.timestamp)
        self._last_moment = moment
        self._roll_session(moment, bar.close, bar.atm_iv)

        quote = self._mark_open_straddle(bar, moment)
        profile = self._read_gex(bar, moment)
        self._update_persistence(profile, moment)

        self._check_exits(bar, moment, quote, profile, execution)

        if self.portfolio.straddle is None:
            self._try_entry(bar, moment, profile, execution)
            quote = self._mark_open_straddle(bar, moment)

        self._hedge(bar, moment, quote, execution)

        state = self._snapshot(bar, moment, quote, profile)
        self.bar_states.append(state)
        return state

    # -- session bookkeeping --------------------------------------------

    def _roll_session(
        self, moment: datetime, future_price: float, atm_iv: float
    ) -> None:
        """Reset per-session state at the first bar of a new trading day.

        The session's opening equity has to be marked at a real price *and*
        a real vol.  A hedge carried overnight would be valued against zero
        without the price; a rolled straddle marked at a default vol rather
        than the market's would start the day with a phantom gain or loss
        the size of the vol gap -- tens of thousands of dollars on a book
        sized to the margin limit -- and the daily loss limit would fire
        (or fail to fire) on the first bar for a move that never happened.
        """
        day = moment.date()
        if day == self._session_date:
            return
        self._session_date = day
        quote = None
        if self.portfolio.straddle is not None:
            quote = self._mark_open_straddle(
                MarketBar(moment, future_price, future_price, future_price,
                          future_price, atm_iv),
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
        """Whether ``moment`` is inside the post-settlement roll window.

        The window opens when today's series settles and runs for
        ``roll_window_minutes``.  It is the one stretch outside the entry
        window in which a series may be opened, because the book was taken
        off at the buffer and this is the first moment the read is built on
        tomorrow's book alone.
        """
        if not self.cfg.strategy.roll_at_expiry:
            return False
        day = moment.date()
        if not is_trading_day(day):
            return False
        settled = self.clock.expiry_datetime(day)
        window_end = settled + timedelta(minutes=self.cfg.strategy.roll_window_minutes)
        return settled <= moment <= window_end

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
        # A series inside its own pre-settlement buffer is dropped unless it
        # is the one the book is on: its gamma is a spike that stops existing
        # within the quarter hour, and it says nothing about what dealers
        # will be hedging once the bell has gone -- which is the question
        # the read has to answer for a position that is about to be carried
        # overnight. Keeping it would let the expiring leg outweigh
        # tomorrow's seven to one and set the overnight side.
        expiries = [
            e for e in listed
            if e <= traded and (
                e == traded
                or self.clock.seconds_to_expiry(moment, e) > self.tenor.buffer_seconds
            )
        ]
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

    def _update_persistence(
        self, profile: GexProfile | None, moment: datetime
    ) -> None:
        """Advance the streak, and confirm a regime once it has held.

        The window is wall-clock time from the first bar of the streak, not
        a bar count: the backtest offers a bar every five minutes and the
        live runner one every few seconds, and a count would make the same
        setting mean fifteen minutes in one and fifteen seconds in the
        other -- on exactly the gate that exists to stop churn.

        Without the gate the confirmed regime is simply the current one, so
        every downstream check reads the same field whether or not
        persistence is switched on -- there is no second code path that
        could behave differently.

        A bar with no profile at all resets the streak rather than extending
        it: a gap in the read is not evidence that the regime held through
        it.
        """
        if profile is None:
            self._streak_regime, self._streak_started, self._streak_bars = "", None, 0
            self._confirmed_regime = NEUTRAL
            return
        if not self.cfg.gates.persistence:
            self._streak_regime, self._streak_started, self._streak_bars = (
                profile.regime, moment, 1
            )
            self._confirmed_regime = profile.regime
            return

        if profile.regime == self._streak_regime and self._streak_started is not None:
            self._streak_bars += 1
        else:
            self._streak_regime, self._streak_started, self._streak_bars = (
                profile.regime, moment, 1
            )
        if self._streak_seconds(moment) >= self.cfg.gates.persistence_seconds:
            self._confirmed_regime = profile.regime

    def _streak_seconds(self, moment: datetime) -> float:
        """How long the current regime has been read, in seconds."""
        if self._streak_started is None:
            return 0.0
        return max((moment - self._streak_started).total_seconds(), 0.0)

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
        # the window after today's settlement the next series may be opened
        # whatever the entry window says, because that is the only moment it
        # can be. Exits are never windowed.
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

        # Nothing is opened inside today's pre-settlement buffer, whatever
        # the entry window says. Today's series is settling and tomorrow's
        # waits for the bell: the roll happens in the window after it, on a
        # read built from the book that will still exist tonight.
        if self._in_buffer(moment) and expiry != moment.date():
            self._record(
                moment, "entry_skipped",
                f"inside today's {cfg.close_before_expiry_minutes}m pre-settlement "
                "buffer; " + (
                    f"the {expiry} series is opened once today's has settled"
                    if cfg.roll_at_expiry else
                    "roll_at_expiry is off, so nothing is opened until the next "
                    "session"
                ),
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
                f"the {profile.regime} read has held for "
                f"{self._streak_seconds(moment) / 60:.1f} of the "
                f"{self.cfg.gates.persistence_seconds / 60:.1f} minutes it needs "
                f"({self._streak_bars} bars) before it counts as the regime "
                "rather than as spot crossing a level",
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
            self.margin_model, stop_fraction=self._stop_fraction(direction),
        )
        if not sizing.ok:
            self._record(moment, "entry_skipped", sizing.reason, regime=profile.regime)
            return

        # Baselines are taken before the legs are sent, so the entry's own
        # fees and any unwind of a mismatched leg count against the position
        # they belong to.
        hedge_realised_before = self.portfolio.hedge_realised
        option_realised_before = self.portfolio.option_realised
        fees_before = self.portfolio.fees_paid

        wanted = direction * sizing.contracts
        opened = self._open_legs(quote, wanted, moment, execution)
        if opened is None:
            return
        call_fill, put_fill, quantity = opened

        self._hedge_realised_at_entry = hedge_realised_before
        self._option_realised_at_entry = option_realised_before
        self._fees_at_entry = fees_before
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
        if sizing.binding:
            self.sizing_binds[sizing.binding] = (
                self.sizing_binds.get(sizing.binding, 0) + 1
            )

        side = "bought" if direction > 0 else "sold"
        cash = abs(quantity) * (call_fill.price + put_fill.price) * self.source.option.multiplier
        intent = "scalp gamma" if direction > 0 else "collect theta"
        partial = (
            f" (asked for {sizing.contracts}; the legs filled short)"
            if abs(quantity) != sizing.contracts else ""
        )
        self._record(
            moment, "entry",
            f"{side} {abs(quantity)}{partial} {expiry} ({days_left}DTE) "
            f"{quote.strike:g} straddle "
            f"@ {call_fill.price + put_fill.price:.2f} "
            f"(C {call_fill.price:.2f} / P {put_fill.price:.2f}, IV {quote.iv:.3f}) "
            f"for ${cash:,.0f} {'debit' if direction > 0 else 'credit'}; "
            f"{sizing.requirement_kind} ${sizing.total_margin:,.0f}, "
            f"${sizing.contracts * sizing.risk_per_straddle:,.0f} at risk to its "
            f"stop, bound by {sizing.binding} ({sizing.describe_limits()}) "
            f"-- {profile.reason}, so {intent}",
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
    ) -> tuple[Fill, Fill, int] | None:
        """Fill both legs at a matched size, or leave the book flat.

        A straddle with one leg on is a naked option, not a straddle -- it
        carries the wrong sign of delta and none of the gamma exposure the
        regime called for.  If the second leg does not fill, the first is
        unwound immediately rather than held.  If the two legs fill at
        *different* sizes -- a partial on one of them -- the excess on the
        larger leg is unwound and the position is booked at the matched
        size, because the book can only ever describe matched straddles
        and booking the requested size would hedge and later close a
        position that does not exist.  In the backtest none of this can
        happen; in live it can, which is the case worth writing for.

        Returns the two fills and the signed quantity actually on.
        """
        call_fill = execution.execute_option(quote.call, quantity, moment)
        if call_fill is None or call_fill.quantity == 0:
            self._record(moment, "entry_failed", "the call leg did not fill")
            return None
        self.fills.append(call_fill)
        self.portfolio.charge_fees(call_fill.fees)

        put_fill = execution.execute_option(quote.put, quantity, moment)
        if put_fill is not None and put_fill.quantity != 0:
            self.fills.append(put_fill)
            self.portfolio.charge_fees(put_fill.fees)
            matched = min(abs(call_fill.quantity), abs(put_fill.quantity))
            direction = 1 if quantity > 0 else -1
            for leg, fill in (("call", call_fill), ("put", put_fill)):
                excess = abs(fill.quantity) - matched
                if excess:
                    self._unwind_excess(
                        quote.call if leg == "call" else quote.put, leg,
                        direction * excess, fill.price, moment, execution,
                    )
            return call_fill, put_fill, direction * matched

        unwind = execution.execute_option(quote.call, -call_fill.quantity, moment)
        if unwind is None or unwind.quantity == 0:
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

    def _unwind_excess(
        self, leg_quote: OptionQuote, leg: str, held: int, held_price: float,
        moment: datetime, execution: ExecutionHandler,
    ) -> None:
        """Trade away ``held`` contracts of one leg that have no partner.

        ``held`` is signed the way the excess is held (positive long).  The
        round trip's P&L is realised on the option leg; a failure is
        recorded loudly, because the book then carries a naked option the
        portfolio cannot see.
        """
        fill = execution.execute_option(leg_quote, -held, moment)
        if fill is None or fill.quantity == 0:
            self._record(
                moment, "entry_failed",
                f"the legs filled at different sizes and the {held:+d} excess "
                f"{leg} contracts could not be unwound -- the book is holding a "
                "naked option and needs manual attention",
            )
            return
        self.fills.append(fill)
        self.portfolio.charge_fees(fill.fees)
        self.portfolio.option_realised += (
            held * (fill.price - held_price) * self.source.option.multiplier
        )
        self._record(
            moment, "entry_trimmed",
            f"the legs filled at different sizes; unwound the {held:+d} excess "
            f"{leg} contracts @ {fill.price:.2f}",
        )

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
            + (book.option_realised - self._option_realised_at_entry)
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

        Fires on the last session before a gap, from the same pre-settlement
        buffer the 0DTE exit uses onwards (the bell included), for a
        position whose series is on the far side of that gap.  At the shipped tenor this cannot happen --
        such a series is never entered -- so this is the safety net for a
        wider tenor, or a config that switched the weekend rule on with a
        position already open.
        """
        if self.tenor.hold_over_weekends:
            return None
        today = moment.date()
        if (
            not is_trading_day(today)
            or position.expiry <= today
            or not self.clock.gap_after(moment)
        ):
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

        A regime that has not yet held ``persistence_seconds`` is spot crossing
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
                f"only for {self._streak_seconds(moment) / 60:.1f} of the "
                f"{self.cfg.gates.persistence_seconds / 60:.1f} minutes a flip "
                f"needs ({self._streak_bars} bars); holding",
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

    def _stop_fraction(self, direction: int) -> float | None:
        """Fraction of the entry premium one straddle loses at its stop.

        This is what the risk budget divides into, so it has to be read off
        the very rules the sizing is trying to make reachable: the long
        stop is a fraction of the debit, the short stop a multiple of the
        credit (so the *loss* is that multiple less one). ``None`` when the
        branch has no stop -- the sizing then falls back to the per-straddle
        requirement, which bounds the same loss.
        """
        cfg = self.cfg.strategy
        if direction > 0:
            return cfg.long_stop_loss_pct
        multiple = cfg.short_stop_loss_premium_multiple
        return None if multiple is None else multiple - 1.0

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
        """Close the straddle, booking exactly what filled.

        Both legs are sent for the whole position.  What comes back may be
        less: one leg may not fill, or may fill short.  The book can only
        describe matched straddles, so the matched part is realised and
        taken off, any leg closed *beyond* the match is put back on so the
        remainder is still a straddle, and the remainder stays open for the
        exit to try again on the next bar.  Retrying the whole close on a
        half-closed book -- the alternative -- would trade the already-
        closed leg a second time.
        """
        position = self.portfolio.straddle
        assert position is not None
        regime = position.regime
        quantity = position.quantity
        direction = position.direction
        held = abs(quantity)

        call_fill = execution.execute_option(quote.call, -quantity, moment)
        put_fill = execution.execute_option(quote.put, -quantity, moment)
        closed = {"call": 0, "put": 0}
        prices = {"call": quote.call.price, "put": quote.put.price}
        for leg, fill in (("call", call_fill), ("put", put_fill)):
            if fill is None or fill.quantity == 0:
                continue
            self.fills.append(fill)
            self.portfolio.charge_fees(fill.fees)
            closed[leg] = min(abs(fill.quantity), held)
            prices[leg] = fill.price

        matched = min(closed["call"], closed["put"])
        for leg in ("call", "put"):
            excess = closed[leg] - matched
            if excess:
                # The leg was closed past its partner: put the excess back
                # so what remains on the book is still a straddle.
                self._restore_leg(
                    quote.call if leg == "call" else quote.put, leg,
                    direction * excess, prices[leg], moment, execution,
                )

        if matched == 0:
            filled = ", ".join(f"{leg} {n}" for leg, n in closed.items() if n) or "neither leg"
            self._record(
                moment, "exit_failed",
                f"could not close ({reason}); {filled} filled and was put back",
                regime=regime,
            )
            return

        option_pnl = self.portfolio.close_straddle(
            prices["call"], prices["put"], matched
        )
        if matched < held:
            self._record(
                moment, "exit_partial",
                f"closed {matched} of {held} {position.strike:g} straddles @ "
                f"{prices['call'] + prices['put']:.2f} ({reason}); straddle P&L "
                f"${option_pnl:,.0f}; {held - matched} remain and the exit will "
                "retry",
                regime=regime,
            )
            return

        # Attribute the whole position -- straddle and hedge -- to the
        # regime that opened it, after the hedge leg has been dealt with so
        # the flatten's fees and realisations are inside the number.
        self._record(
            moment, "exit",
            f"closed {held} {position.strike:g} straddle @ "
            f"{prices['call'] + prices['put']:.2f} ({reason}); "
            f"straddle P&L ${option_pnl:,.0f}, position P&L "
            f"${self._position_pnl(None, bar.close):,.0f}",
            regime=regime,
        )
        if self.cfg.hedge.flatten_hedge_on_exit and self.portfolio.hedge.quantity:
            self._flatten_hedge(bar, moment, execution)
        position_pnl = self._position_pnl(None, bar.close)
        self.regime_pnl[regime] = self.regime_pnl.get(regime, 0.0) + position_pnl

    def _restore_leg(
        self, leg_quote: OptionQuote, leg: str, excess: int, closed_price: float,
        moment: datetime, execution: ExecutionHandler,
    ) -> None:
        """Re-open ``excess`` contracts of a leg closed past its partner.

        ``excess`` is signed the way the position holds the leg.  The round
        trip is realised on the option leg; a failure leaves a naked option
        the portfolio cannot see, and says so.
        """
        fill = execution.execute_option(leg_quote, excess, moment)
        if fill is None or fill.quantity == 0:
            self._record(
                moment, "exit_failed",
                f"the {leg} leg closed {abs(excess)} contracts past the other and "
                "they could not be put back -- the book is holding a naked "
                "option and needs manual attention",
            )
            return
        self.fills.append(fill)
        self.portfolio.charge_fees(fill.fees)
        self.portfolio.option_realised += (
            excess * (closed_price - fill.price) * self.source.option.multiplier
        )
        self._record(
            moment, "exit_trimmed",
            f"the legs closed at different sizes; put back {abs(excess)} {leg} "
            f"contracts @ {fill.price:.2f}",
        )

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

    # -- persistence across a restart ---------------------------------------

    def snapshot(self, moment: datetime | None = None) -> dict:
        """Everything a restarted process needs to carry this book on.

        The runner rebuilds the strategy on every reconnection -- the daily
        gateway restart included -- and the rolled position is open through
        exactly that.  What the broker can say is which contracts are held;
        what only this process knows is what they were opened for, which
        regime opened them, the hedge P&L scalped against them so far, and
        how the session's loss limit and entry count stand.  All of it is
        here, and none of it is a decision: ``restore`` puts it back only
        when the broker's positions match what it describes.
        """
        position = self.portfolio.straddle
        return {
            "version": 1,
            "as_of": (moment or self._last_moment).isoformat()
            if (moment or self._last_moment) else None,
            "straddle": None if position is None else {
                "strike": position.strike,
                "expiry": position.expiry.isoformat(),
                "quantity": int(position.quantity),
                "call_entry": float(position.call_entry),
                "put_entry": float(position.put_entry),
                "entry_time": position.entry_time.isoformat(),
                "entry_future": float(position.entry_future),
                "entry_iv": float(position.entry_iv),
                "entry_delta": float(position.entry_delta),
                "regime": position.regime,
            },
            "hedge": {
                "quantity": int(self.portfolio.hedge.quantity),
                "avg_price": float(self.portfolio.hedge.avg_price),
            },
            "realised": {
                "option": float(self.portfolio.option_realised),
                "hedge": float(self.portfolio.hedge_realised),
                "fees": float(self.portfolio.fees_paid),
            },
            "baselines": {
                "option_realised": float(self._option_realised_at_entry),
                "hedge_realised": float(self._hedge_realised_at_entry),
                "fees": float(self._fees_at_entry),
            },
            "session": {
                "date": self._session_date.isoformat() if self._session_date else None,
                "start_equity": float(self._session_start_equity),
                "entries": int(self._entries_this_session),
                "halted": bool(self._halted_for_session),
            },
            "streak": {
                "regime": self._streak_regime,
                "started": self._streak_started.isoformat() if self._streak_started else None,
                "bars": int(self._streak_bars),
                "confirmed": self._confirmed_regime,
            },
            "last_hedge_time": (
                self._last_hedge_time.isoformat() if self._last_hedge_time else None
            ),
            "regime_pnl": dict(self.regime_pnl),
            "regime_trades": dict(self.regime_trades),
        }

    def restore(
        self, state: dict, moment: datetime, adopt_straddle: bool = True
    ) -> None:
        """Put a ``snapshot`` back, for a book the broker confirms is open.

        ``adopt_straddle`` is the caller's statement that the broker's
        option positions match the recorded straddle exactly; without it
        only the tallies come back (realised P&L, fees, the session's loss
        limit and entry count), so a restart mid-session does not hand the
        strategy a fresh loss budget.  Session state is restored only when
        the record is from today's session; the persistence streak only
        when it is, too -- a regime read yesterday says nothing about
        whether it held overnight.
        """
        book = self.portfolio
        realised = state.get("realised", {})
        book.option_realised = float(realised.get("option", 0.0))
        book.hedge_realised = float(realised.get("hedge", 0.0))
        book.fees_paid = float(realised.get("fees", 0.0))
        self.regime_pnl = {k: float(v) for k, v in state.get("regime_pnl", {}).items()}
        self.regime_trades = {k: int(v) for k, v in state.get("regime_trades", {}).items()}

        recorded = state.get("straddle")
        if adopt_straddle and recorded:
            if book.straddle is not None:
                raise RuntimeError("cannot restore a straddle over an open one")
            book.straddle = StraddlePosition(
                strike=float(recorded["strike"]),
                expiry=date.fromisoformat(recorded["expiry"]),
                quantity=int(recorded["quantity"]),
                call_entry=float(recorded["call_entry"]),
                put_entry=float(recorded["put_entry"]),
                entry_time=self.clock.localize(datetime.fromisoformat(recorded["entry_time"])),
                entry_future=float(recorded.get("entry_future", 0.0)),
                entry_iv=float(recorded.get("entry_iv", 0.0)),
                entry_delta=float(recorded.get("entry_delta", 0.0)),
                regime=str(recorded.get("regime", "")),
            )
            baselines = state.get("baselines", {})
            self._option_realised_at_entry = float(baselines.get("option_realised", 0.0))
            self._hedge_realised_at_entry = float(baselines.get("hedge_realised", 0.0))
            self._fees_at_entry = float(baselines.get("fees", 0.0))
            hedge = state.get("hedge", {})
            book.hedge.quantity = int(hedge.get("quantity", 0))
            book.hedge.avg_price = float(hedge.get("avg_price", 0.0))

        today = self.clock.localize(moment).date()
        session = state.get("session", {})
        if session.get("date") == today.isoformat():
            self._session_date = today
            self._session_start_equity = float(session.get("start_equity", book.starting_equity))
            self._entries_this_session = int(session.get("entries", 0))
            self._halted_for_session = bool(session.get("halted", False))
            streak = state.get("streak", {})
            started = streak.get("started")
            if started:
                self._streak_regime = str(streak.get("regime", ""))
                self._streak_started = self.clock.localize(datetime.fromisoformat(started))
                self._streak_bars = int(streak.get("bars", 0))
                self._confirmed_regime = str(streak.get("confirmed", NEUTRAL))
        last_hedge = state.get("last_hedge_time")
        if last_hedge:
            self._last_hedge_time = self.clock.localize(datetime.fromisoformat(last_hedge))

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
