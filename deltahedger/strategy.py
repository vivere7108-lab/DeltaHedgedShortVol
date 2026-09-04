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

The tenor
---------
The traded series is a listed expiry two to five sessions out, chosen by
``StrategyConfig.tenor()`` and closed once it decays to the DTE floor.  That
has three consequences visible in this file:

  * the position is carried **overnight and across sessions**, so
    ``_roll_session`` re-marks the book at the new day's first price rather
    than assuming it starts flat, and the hedge band is session-aware;
  * the exit ladder is led by the **DTE floor**, not by minutes-to-expiry;
  * the GEX read is a blend over the front expiries, so a bar reads open
    interest for several series rather than one.

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
from .gex import (
    GATE_ENSEMBLE,
    GATE_ENTRY_WINDOW,
    GATE_NOWCAST,
    GATE_PERSISTENCE,
    NEUTRAL,
    ExpiryBook,
    GexCalculator,
    GexProfile,
    NowcastProvider,
    OpenInterestProvider,
    StrikeFlow,
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
    # -- the nowcast read, when the flow feed is on --------------------
    #: The flow-corrected total, ``None`` unless ``nowcast.enabled`` and a
    #: read has landed. Reported for eyeballing against ``gex_total``;
    #: never fed to anything downstream by itself.
    nowcast_total: float | None = None
    nowcast_regime: str = ""


class GexStraddleStrategy:
    def __init__(
        self,
        cfg: Config,
        source: RiskSource | None = None,
        margin_model: MarginModel | None = None,
        open_interest: OpenInterestProvider | None = None,
        nowcast: NowcastProvider | None = None,
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
            cfg.gex, self.source, self.surface, cfg.risk_free_rate, cfg.gates
        )
        self.tenor: TenorPolicy = cfg.strategy.tenor()
        self.open_interest = open_interest
        #: The flow nowcast (config.NowcastConfig). ``None`` -- the default,
        #: same as ``open_interest`` -- means the feature is off regardless
        #: of ``cfg.nowcast.enabled``; a config that turns it on without a
        #: provider wired in just never gets a read, the same way GEX itself
        #: goes quiet without an ``open_interest`` provider.
        self.nowcast = nowcast
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
        #: The nowcast read, refreshed on its own timer (nowcast.refresh_
        #: seconds), independent of gex.refresh_seconds. See _read_nowcast.
        self._nowcast_profile: GexProfile | None = None
        self._nowcast_read_at: datetime | None = None
        #: When the current session's OI print became the standing one --
        #: the "since" boundary flow is measured from, and the reference
        #: point the next morning's reconciliation check measures against.
        #: Approximated as session open: real settlement timing is not
        #: exposed by an OI read, and the entry window already exists to
        #: keep entries off a still-preliminary print -- see NowcastConfig.
        self._session_open_at: datetime | None = None
        #: What the flow implied the book looked like at the moment the OI
        #: print most recently refreshed to a new day's value, kept so the
        #: *next* refresh can reconcile against it. {expiry: {strike: (call, put)}}.
        self._nowcast_reconciliation_baseline: dict[date, dict[float, tuple[float, float]]] = {}
        #: The flow_by_expiry the last nowcast read used, kept only so the
        #: session-roll reconciliation snapshot can be built from it without
        #: re-reading flow for a session that is already ending.
        self._last_nowcast_flow: dict[date, tuple[StrikeFlow, ...]] | None = None
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
        self._read_nowcast(bar, moment)

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
        self._capture_nowcast_reconciliation_baseline()
        self._session_date = day
        self._session_open_at = moment
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
        traded = self._traded_expiry(moment)
        if traded is None:
            return None
        expiries = self._blend_expiries(moment, traded)

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
            self._reconcile_nowcast(moment, fresh)

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

    # -- the flow nowcast --------------------------------------------------

    def _read_nowcast(self, bar: MarketBar, moment: datetime) -> None:
        """The flow correction, refreshed on its own (much coarser) timer.

        Independent of ``gex.refresh_seconds``: the print's staleness and
        the flow's noise floor are different clocks, and conflating them
        would either re-read the print needlessly often or smooth the flow
        too little -- see ``NowcastConfig``. A bar with no blend yet (no OI
        read has landed) has nothing to correct, so this is a no-op until
        ``_read_gex`` has populated ``self._books`` at least once.
        """
        cfg = self.cfg.nowcast
        if not cfg.enabled or self.nowcast is None or not self._books:
            self._nowcast_profile = None
            return

        stale = (
            self._nowcast_read_at is None
            or (moment - self._nowcast_read_at).total_seconds() >= cfg.refresh_seconds
        )
        if not stale:
            return

        since = self._session_open_at or moment
        flow_by_expiry: dict[date, tuple[StrikeFlow, ...]] = {}
        for book in self._books:
            try:
                flow_by_expiry[book.expiry] = self.nowcast.flow_since(
                    moment, book.expiry, since
                )
            except Exception as exc:  # noqa: BLE001 - a bad read must not halt the run
                log.warning(
                    "could not read flow for %s (%s); treating as no flow "
                    "since the print", book.expiry, exc,
                )

        self._last_nowcast_flow = flow_by_expiry
        self._nowcast_profile = self.gex.nowcast_profile(
            bar.close, self._books, flow_by_expiry, bar.atm_iv, cfg.dealer_share,
        )
        self._nowcast_read_at = moment

    def _capture_nowcast_reconciliation_baseline(self) -> None:
        """Snapshot what the flow implied the book looked like, at the last
        moment ``self._books``/``self._nowcast_profile`` still describe the
        session that is about to roll over.

        Called from ``_roll_session``, before anything for the new day
        resets. What is captured here is compared, once the new session's
        OI print actually refreshes, against the print that comes back --
        see ``_reconcile_nowcast``.
        """
        cfg = self.cfg.nowcast
        if not (cfg.enabled and cfg.reconciliation_enabled):
            return
        if self.nowcast is None or self._last_nowcast_flow is None or not self._books:
            return
        adjusted = self.gex.nowcast_books(
            self._books, self._last_nowcast_flow, cfg.dealer_share
        )
        self._nowcast_reconciliation_baseline = {
            book.expiry: {row.strike: (row.call_oi, row.put_oi) for row in book.rows}
            for book in adjusted
        }

    def _reconcile_nowcast(
        self, moment: datetime, fresh: dict[date, list[StrikeOpenInterest]]
    ) -> None:
        """Compare the prior session's flow-implied close against the OI
        print that actually lands, per strike, and write down the gap.

        This is the whole ongoing check on whether ``dealer_share`` is any
        good -- and it is free: both numbers already exist, this only has
        to notice when a fresh print makes the comparison possible. Runs
        from every OI refresh but the baseline is captured only once, at
        the session roll, so in practice this does something once per
        session: the first time the new day's print is read.
        """
        if not self._nowcast_reconciliation_baseline:
            return
        baseline = self._nowcast_reconciliation_baseline
        self._nowcast_reconciliation_baseline = {}  # consume once, hit or miss

        for expiry, predicted in baseline.items():
            rows = fresh.get(expiry)
            if not rows:
                continue
            actual = {row.strike: (row.call_oi, row.put_oi) for row in rows}
            strikes = sorted(set(predicted) | set(actual))
            if not strikes:
                continue
            call_errors = [
                actual.get(k, (0.0, 0.0))[0] - predicted.get(k, (0.0, 0.0))[0]
                for k in strikes
            ]
            put_errors = [
                actual.get(k, (0.0, 0.0))[1] - predicted.get(k, (0.0, 0.0))[1]
                for k in strikes
            ]
            mean_abs_call = sum(abs(e) for e in call_errors) / len(call_errors)
            mean_abs_put = sum(abs(e) for e in put_errors) / len(put_errors)
            self._record(
                moment, "nowcast_reconciliation",
                f"{expiry}: mean |predicted - actual| OI error "
                f"{mean_abs_call:,.1f} calls / {mean_abs_put:,.1f} puts across "
                f"{len(strikes)} strikes (dealer_share="
                f"{self.cfg.nowcast.dealer_share:g})",
            )

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

    # -- the flow nowcast's authority: veto, exit, size haircut ----------
    #
    # Narrow on purpose (see NowcastConfig): the daily OI blend remains the
    # only thing that picks a side. The three methods below only ever say
    # "not this one", "not any more" or "not this much" -- never "this one".

    def _nowcast_veto_reason(self, direction: int) -> str | None:
        """Why the flow read objects to an entry the print already picked,
        or ``None`` if it does not.

        Fires only when the nowcast has an opinion (``direction != 0``) and
        that opinion disagrees with the print's. A neutral flow read -- no
        corroboration, but no objection either -- is never a veto.
        """
        cfg = self.cfg.nowcast
        if not (cfg.enabled and cfg.veto_enabled) or self._nowcast_profile is None:
            return None
        nc = self._nowcast_profile
        if nc.direction == 0 or nc.direction == direction:
            return None
        return (
            f"flow since the print reads {nc.regime} ({nc.confidence:.0%} "
            f"confidence), against the side the daily print picked"
        )

    def _nowcast_size_multiplier(self) -> float:
        """1.0 when flow corroborates the print or the feature is off; the
        configured haircut when flow has no opinion at all.

        A disagreeing flow read never reaches this call -- it is vetoed
        first, in ``_try_entry`` -- so the only two cases here are
        "agrees" and "no opinion".
        """
        cfg = self.cfg.nowcast
        if not (cfg.enabled and cfg.size_haircut_enabled) or self._nowcast_profile is None:
            return 1.0
        if self._nowcast_profile.direction == 0:
            return cfg.size_haircut_when_unconfirmed
        return 1.0

    def _nowcast_exit_reason(self, position: StraddlePosition) -> str | None:
        """Close early when flow since the print argues against an open
        position the print itself still likes.

        Checked after the daily-OI flip ladder in ``_check_exits``, so this
        only fires when the print agrees with the position but the trades
        since it do not -- the same disagreement rule as the veto, applied
        to what is already open rather than to a prospective entry.
        """
        cfg = self.cfg.nowcast
        if not (cfg.enabled and cfg.exit_enabled) or self._nowcast_profile is None:
            return None
        nc = self._nowcast_profile
        if nc.direction == 0 or nc.direction == position.direction:
            return None
        side = "long" if position.is_long else "short"
        return (
            f"flow since the print reads {nc.regime} ({nc.confidence:.0%} "
            f"confidence), against the open {side} position"
        )

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
        # are eligible. Its purpose is the open-interest print: the exchange
        # publishes final open interest for the previous session during the
        # morning, and an entry before that lands is taken on preliminary
        # numbers. Exits are never windowed.
        local = moment.timetz().replace(tzinfo=None)
        if self.cfg.gates.entry_window and not (
            cfg.entry_time <= local <= cfg.entry_cutoff_time
        ):
            return

        expiry = self._traded_expiry(moment)
        if expiry is None:
            self._record(
                moment, "entry_skipped",
                f"no expiry listed between {self.tenor.min_days} and "
                f"{self.tenor.max_days} trading days out",
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
        if self.clock.seconds_to_expiry(moment, expiry) <= (
            cfg.close_before_expiry_minutes * 60
        ):
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

        veto = self._nowcast_veto_reason(direction)
        if veto is not None:
            self._record(
                moment, "entry_skipped", veto, regime=profile.regime, gate=GATE_NOWCAST,
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
        size_multiplier = self._nowcast_size_multiplier()
        sizing = size_straddles(
            equity, quote, bar.close, direction, self.cfg.sizing, self.source,
            self.margin_model, size_multiplier,
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
        haircut_note = (
            f" ({size_multiplier:.0%} sized: flow since the print has no opinion)"
            if size_multiplier != 1.0 else ""
        )
        self._record(
            moment, "entry",
            f"{side} {sizing.contracts} {expiry} ({days_left}DTE) "
            f"{quote.strike:g} straddle "
            f"@ {call_fill.price + put_fill.price:.2f} "
            f"(C {call_fill.price:.2f} / P {put_fill.price:.2f}, IV {quote.iv:.3f}) "
            f"for ${cash:,.0f} {'debit' if direction > 0 else 'credit'}; "
            f"{sizing.requirement_kind} ${sizing.total_margin:,.0f} of "
            f"${sizing.budget:,.0f} budget{haircut_note} -- {profile.reason}, so {intent}",
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
        days_left = self.clock.days_to_expiry(moment, position.expiry)
        pnl = self._position_pnl(quote, bar.close)
        premium = position.premium_at_risk(self.source.option.multiplier)
        reason: str | None = None

        # The DTE floor leads the ladder and is never gated. It is the whole
        # point of moving off 0DTE: the position comes off before the tenor
        # decays into the range where gamma, pin risk and the staleness of
        # the open-interest print all get worse at once.
        if self.tenor.should_close(days_left):
            reason = (
                f"{days_left}DTE, at the {self.tenor.close_days}DTE close-out floor"
            )
        elif seconds_left <= cfg.close_before_expiry_minutes * 60:
            reason = f"{cfg.close_before_expiry_minutes}m to expiry"
        elif cfg.exit_on_regime_flip and profile is not None:
            reason = self._flip_exit_reason(bar, moment, position, profile)

        if reason is None:
            reason = self._nowcast_exit_reason(position)

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
        decision = self.hedger.decide(
            net_delta, elapsed, self.clock.in_session(moment)
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
            in_band=(
                self.cfg.hedge.in_band(net, in_session)
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
            nowcast_total=(
                self._nowcast_profile.total_gex if self._nowcast_profile else None
            ),
            nowcast_regime=self._nowcast_profile.regime if self._nowcast_profile else "",
        )
