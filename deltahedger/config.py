"""Configuration for the GEX-directed delta-hedged straddle system.

One ``Config`` object drives both the backtest and the live runner, so a
forward test routes the same parameters that were validated historically.
Load from YAML with ``Config.from_yaml`` or build in code.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import time
from pathlib import Path
from typing import Any

import yaml

from .instruments import RiskSource, get_risk_source

log = logging.getLogger(__name__)


def _parse_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        parts = [int(p) for p in value.split(":")]
        while len(parts) < 3:
            parts.append(0)
        return time(*parts[:3])
    raise TypeError(f"cannot read a time-of-day from {value!r}")


#: The two ways the band half-width can be set.
BAND_WHALLEY_WILMOTT = "whalley_wilmott"
BAND_FIXED = "fixed"


@dataclass
class HedgeConfig:
    """The delta band: where it sits and how wide it is.

    ``target`` is in delta units (1 unit == 1% of one ES contract).  The
    position is an ATM straddle, so the target is 0: hold the book
    delta-neutral and let the straddle express the gamma view.

    The half-width comes from one of two models, chosen by ``band_model``:

    ``whalley_wilmott`` (the default)
        The Whalley-Wilmott (1997) asymptotic no-transaction band for a
        hedger with exponential utility and proportional costs::

            H = ( 3/2 * exp(-r*T) * k * S * Gamma^2 / gamma_ra ) ** (1/3)

        where ``k*S`` is the cost of trading one unit of the underlying,
        ``Gamma`` is the position's gamma in those units per dollar, and
        ``gamma_ra`` is ``risk_aversion`` -- the absolute risk aversion of
        the hedger, *per dollar of wealth*.  The band is therefore not a
        number but a function of the book: it widens as ``Gamma^(2/3)``, so
        a big book near the money is allowed to drift further in delta
        units (though less far in *points*) before it is rebalanced, and it
        narrows as costs fall or risk aversion rises, each as a cube root.
        ``hedger.py`` spells out the unit conversion into delta units.

        ``risk_aversion`` is dollar-denominated, and that matters for
        reading the number: ``0.01`` per dollar is a very risk-averse
        hedger -- on a $250k book it is a relative risk aversion in the
        thousands -- and it gives a band of a few tens of delta units for
        a book of a few straddles, a couple of hundred for a book sized to
        the margin limit.  Raise it to hedge less often.

    ``fixed``
        The old heuristic: ``band`` delta units either side of the target,
        whatever the book carries.  Kept as the control -- ``deltahedger
        sweep --bands`` uses it -- and because a hedger that ignores gamma
        is the right thing to compare a gamma-aware one against.

    Either way, one MES contract moves net delta by 10 units, so a
    half-width under 5 cannot bind: the hedger only trades when a whole
    contract lands closer to target (see ``hedger.py``).  A Whalley-Wilmott
    band that comes out narrower than that -- a tiny book, or a straddle
    far from its strike -- simply behaves as +/-5.

    ``overnight_band_multiplier`` widens whichever half-width applies
    outside the regular session.  The hours outside RTH are not the same
    market: the book is quoted wider and thinner, and a delta picked up on
    thin overnight volume is as likely to be handed back by the open as
    realised.  Under Whalley-Wilmott a wider quote is a larger ``k``, and a
    band ``m`` times wider corresponds to costs ``m^3`` times higher -- so
    keep the multiplier modest.  The hedge is never switched off: an
    overnight gap is exactly when an unhedged straddle does the most damage.
    Set it to 1.0 to hedge identically around the clock.

    ``hedge_cost_per_contract`` is the ``k*S`` term: the dollar cost of
    trading one hedge contract, one way.  ``None`` derives it from the
    costs section (slippage in ticks plus fees) -- and does so whether or
    not ``costs.enabled`` is on, because the band is a decision rule the
    live system will run, and a backtest that switches costs off to read
    the strategy's arithmetic should still hedge the way the live system
    will.
    """

    target: float = 0.0
    #: ``"whalley_wilmott"`` or ``"fixed"``.
    band_model: str = BAND_WHALLEY_WILMOTT
    #: Whalley-Wilmott absolute risk aversion, per dollar of wealth.
    risk_aversion: float = 0.01
    #: Dollar cost of one hedge contract traded one way, for the band. None
    #: derives it from ``costs`` (hedge slippage ticks x tick value + fees).
    hedge_cost_per_contract: float | None = None
    #: Half-width in delta units when ``band_model == "fixed"``.
    band: float = 10.0
    #: Widen the band by this factor outside the regular session. 1.0 hedges
    #: overnight exactly as it does intraday.
    overnight_band_multiplier: float = 2.5
    #: Don't send a hedge smaller than this many contracts.
    min_hedge_contracts: int = 1
    #: Cap on a single hedge order, as a guard against a data glitch. A
    #: book sized to the margin limit can legitimately need several
    #: hundred MES in one go near expiry; anything the cap holds back is
    #: sent on the next pass.
    max_hedge_contracts: int = 500
    #: Seconds to wait between hedges; suppresses churn on noisy quotes.
    min_seconds_between_hedges: float = 0.0
    #: Flatten the hedge in the same order as the straddle, rather than
    #: leaving it for the band. Against a neutral target this is a smaller
    #: switch than it looks: an orphaned hedge is a naked directional
    #: position, so the band closes anything larger than itself on the very
    #: next pass either way. What the flag actually governs is whether a
    #: sub-band residual -- under one hedge contract -- is left behind.
    flatten_hedge_on_exit: bool = True

    @property
    def is_fixed(self) -> bool:
        return self.band_model == BAND_FIXED

    def validate(self) -> None:
        if self.band_model not in (BAND_WHALLEY_WILMOTT, BAND_FIXED):
            raise ValueError(
                f"hedge.band_model must be '{BAND_WHALLEY_WILMOTT}' or "
                f"'{BAND_FIXED}', got {self.band_model!r}"
            )
        if self.risk_aversion <= 0.0:
            raise ValueError("hedge.risk_aversion must be > 0")
        if self.hedge_cost_per_contract is not None and self.hedge_cost_per_contract < 0:
            raise ValueError("hedge.hedge_cost_per_contract must be >= 0 or null")
        if self.band < 0:
            raise ValueError("hedge.band must be >= 0")
        if self.overnight_band_multiplier < 1.0:
            raise ValueError(
                "hedge.overnight_band_multiplier must be >= 1.0: hedging more "
                "tightly overnight than intraday is churn, not risk control"
            )
        if self.min_hedge_contracts < 1:
            raise ValueError("hedge.min_hedge_contracts must be >= 1")
        if self.max_hedge_contracts < self.min_hedge_contracts:
            raise ValueError("hedge.max_hedge_contracts < min_hedge_contracts")


@dataclass
class SizingConfig:
    """How much of the account to commit to the straddle.

    The book is sized to the margin limit, less a buffer.  ``buying_power_pct``
    is the share of equity the strategy may commit as buying power -- margin
    for a short straddle, the debit for a long one, and the reserve for the
    hedge leg all come out of it -- and the remainder is the buffer that
    absorbs variation margin and a margin call on a bad day.  The default
    leaves 20% untouched.

    The straddle count is what that allocation buys at the *all-in*
    per-straddle requirement: the option leg's margin or debit, plus the
    margin on the futures the strategy will need to hedge it.  A straddle
    cannot be carried without its hedge, so the two are one requirement.
    """

    #: Fraction of portfolio equity to allocate as buying power. It covers
    #: margin for a short straddle and the debit for a long one. 1 minus
    #: this is the buffer the strategy never touches.
    buying_power_pct: float = 0.80
    #: Hard cap on straddles regardless of buying power. A backstop against
    #: a sizing bug rather than a sizing rule -- the buying-power budget is
    #: what decides the count. Matches the per-order hard ceiling in
    #: ``broker.ibkr.MAX_ORDER_CONTRACTS``.
    max_straddles: int = 500
    #: Never open a position smaller than this.
    min_straddles: int = 1
    #: Deprecated, and ignored. It held back a flat fraction of the budget
    #: for the hedge and never compared it with the hedge it stood behind,
    #: so it was decoration in one direction and a silent cap in the other.
    #: The hedge is now charged per straddle at what it will actually cost
    #: -- see ``sizing.size_straddles`` -- which needs no fraction to be
    #: chosen and cannot be mis-set. Left in place so existing configs load.
    hedge_margin_reserve_pct: float | None = None
    #: Margin model: "span_scan", "reg_t" or "fixed". See ``sizing.py`` --
    #: "span_scan" reproduces CME SPAN methodology and is the right default
    #: for futures options; "reg_t" is the equity-option rule and will
    #: badly overstate futures margin.
    margin_model: str = "span_scan"
    #: Used when margin_model == "fixed": USD initial margin per short leg.
    fixed_margin_per_contract: float = 2000.0
    #: span_scan: scale the price scan range derived from the risk source's
    #: outright future margin. 1.0 means "scan the move CME scans".
    span_scan_multiplier: float = 1.0
    #: span_scan: relative volatility bump, 0.30 == scan vol +/- 30%.
    span_vol_scan_pct: float = 0.30
    #: span_scan: short option minimum charge per contract, USD.
    span_short_option_minimum: float = 250.0
    #: reg_t coefficients: margin = premium + max(a*notional - otm, b*strike)
    reg_t_a: float = 0.15
    reg_t_b: float = 0.10

    def validate(self) -> None:
        if not 0.0 < self.buying_power_pct <= 1.0:
            raise ValueError("sizing.buying_power_pct must be in (0, 1]")
        if self.hedge_margin_reserve_pct is not None:
            log.warning(
                "sizing.hedge_margin_reserve_pct (%.2f) is deprecated and is "
                "being ignored. The hedge is now charged per straddle at what "
                "it will actually cost and budgeted alongside the option leg, "
                "so there is no fraction left to choose. Remove it from the "
                "config; buying_power_pct is the lever.",
                self.hedge_margin_reserve_pct,
            )
            self.hedge_margin_reserve_pct = None
        if self.margin_model not in ("span_scan", "reg_t", "fixed"):
            raise ValueError(
                "sizing.margin_model must be one of 'span_scan', 'reg_t', 'fixed'"
            )
        if self.max_straddles < self.min_straddles:
            raise ValueError("sizing.max_straddles < min_straddles")
        # The per-order hard ceiling in broker.ibkr is applied as
        # min(MAX_ORDER_CONTRACTS, max_straddles), so it can only ever bind
        # when it is the smaller of the two. At equal values it is inert,
        # and a backstop nobody can see is not one -- say so rather than
        # leaving it to be discovered.
        from .broker.base import MAX_ORDER_CONTRACTS

        if self.max_straddles >= MAX_ORDER_CONTRACTS:
            log.warning(
                "sizing.max_straddles is %d and the per-order hard ceiling is "
                "%d, so the ceiling can never bind and the only cap on an "
                "order is the config one. Lower sizing.max_straddles to give "
                "the backstop something to catch.",
                self.max_straddles, MAX_ORDER_CONTRACTS,
            )


@dataclass
class GexConfig:
    """Dealer gamma exposure: the flip point and the regime it implies.

    GEX is a *positioning* estimate, not an observable.  It assumes the
    dealer is on the other side of the public's option book -- long the
    calls, short the puts -- and asks what that inventory forces them to do
    when spot moves.  Short gamma (negative GEX) means dealers hedge with
    the move and amplify it; long gamma (positive GEX) means they hedge
    against it and suppress it.  The strategy trades alongside that
    mechanic: buy the straddle when dealers must chase, sell it when they
    must dampen.

    The sign convention is a modelling choice, so it is a parameter rather
    than a constant.  ``call_sign``/``put_sign`` of ``+1``/``-1`` is the
    standard assumption and what every published GEX print uses -- but it
    is only a *prior* now.  With a trade feed attached (``flow:``) the sign
    at each strike is measured from the tape's aggressor side rather than
    assumed, and these two numbers are what a strike with no measured flow
    falls back to.  See ``flow.py`` and ``use_flow_signs`` below.
    """

    enabled: bool = True
    #: Prior dealer inventory signs, applied to open interest at a strike
    #: where the tape has said nothing. ``+1``/``-1`` is the standard
    #: assumption -- dealers long the calls, short the puts.
    call_sign: float = 1.0
    put_sign: float = -1.0

    # -- measured signs, from classified trades ---------------------------
    #: Use the dealer sign measured from classified option trades in place
    #: of the prior, where there is flow to measure it from. Off restores
    #: the static convention exactly, which is what the system did before
    #: the tape was wired in and what a control run should use.
    use_flow_signs: bool = True
    #: Classified contracts at which a strike's measured sign carries half
    #: the weight, the prior the other half. The blend is
    #: ``w*measured + (1-w)*prior`` with ``w = n / (n + this)``, so it is
    #: the number that says how much tape it takes to overrule the
    #: assumption. Too small and three prints redefine a strike; too large
    #: and a well-traded strike still reports the assumption back at you.
    flow_confidence_contracts: float = 250.0
    #: Strikes included in the profile, as +/- a fraction of spot. Most of
    #: the gamma in the front expiries sits inside 2%; widening it costs
    #: live market-data lines on every expiry in the blend, which is the
    #: binding constraint rather than the arithmetic.
    strike_width_pct: float = 0.02
    #: Hypothetical-spot grid for the flip search: half-width and resolution.
    flip_search_pct: float = 0.03
    flip_search_steps: int = 61
    #: Spot within this fraction of the flip point reads as no clear regime:
    #: right at the flip the sign is about to change and the classification
    #: is not information. Toggled by ``gates.flip_distance``.
    flip_proximity_pct: float = 0.0015
    #: Floor on the time-to-expiry used for the profile, in hours. An
    #: expiring series' gamma collapses to a zero-width spike at the bell,
    #: which would let the 0DTE leg of the blend dominate everything else;
    #: the floor keeps the shape of the surface visible. It affects
    #: classification only, never the greeks the hedger acts on.
    min_hours_to_expiry: float = 0.5
    #: How often to re-read open interest, in seconds. OI is an end-of-day
    #: figure intraday, so re-reading it every bar buys nothing; the profile
    #: itself is recomputed at the live spot on every bar regardless.
    refresh_seconds: float = 900.0

    # -- the front-expiry blend ------------------------------------------
    #: Read the regime off the aggregate of the front expiries rather than
    #: off the traded series alone. What dealers hedge is one book, not one
    #: series, and at a 3-4 DTE tenor the traded series is a minority of the
    #: gamma sitting in front of it. Off means "classify on the traded
    #: expiry only", which is what the 0DTE version did.
    blend_front_expiries: bool = True
    #: Cap on how many expiries enter the blend, counting from 0DTE
    #: outwards. This bounds the live cost: every expiry in the blend is a
    #: separate open-interest read, and each one subscribes two market-data
    #: lines per listed strike.
    blend_max_expiries: int = 4

    def validate(self) -> None:
        if self.strike_width_pct <= 0.0:
            raise ValueError("gex.strike_width_pct must be > 0")
        if self.flip_search_steps < 3:
            raise ValueError("gex.flip_search_steps must be >= 3")
        if self.flip_search_pct <= 0.0:
            raise ValueError("gex.flip_search_pct must be > 0")
        if self.min_hours_to_expiry < 0.0:
            raise ValueError("gex.min_hours_to_expiry must be >= 0")
        if self.blend_max_expiries < 1:
            raise ValueError("gex.blend_max_expiries must be >= 1")
        if self.flow_confidence_contracts <= 0.0:
            raise ValueError("gex.flow_confidence_contracts must be > 0")


@dataclass
class GatesConfig:
    """Four independent reasons to stand aside, each one switchable.

    None of these makes a new statement about the market; each one refuses
    to act on a statement the GEX read is not entitled to make.  They are
    separate flags rather than one "be careful" switch so that a sweep can
    price each of them on its own -- a gate that costs more in missed trades
    than it saves in bad ones should be visible as such rather than hidden
    inside a bundle.  ``deltahedger sweep --gates`` runs exactly that
    comparison, and the journal records which gate blocked each would-be
    action so the attribution survives into a live walk.

    1. **confidence** -- ``|total GEX| / gross GEX`` is how *directional*
       dealer positioning is, on a 0-1 scale.  A book with matched call and
       put gamma nets to nothing, and its sign is then decided by noise in
       the open-interest print.  Below ``min_confidence_ratio`` the sign is
       not information.
    2. **flip_distance** -- the pre-existing fixed test: spot within
       ``gex.flip_proximity_pct`` of the gamma flip is about to change sign.
       It is kept separate from (1) because they fail differently: a book
       can be strongly directional *and* sitting on its flip, or flat and
       far from one.
    3. **ensemble** -- recompute the regime over a small grid of skew,
       sign-prior and flow-trust perturbations and trade only if every
       member agrees.  This is the only gate that tests the *model* rather
       than the data: all three perturbed inputs are assumptions the README
       flags as load-bearing, and a regime that reverses under a plausible
       variation of any of them was never a reading of the market.  With a
       trade feed attached the sign axis matters less at a heavily traded
       strike -- the sign is measured there, not assumed -- and the flow
       axis matters more, which is the gate correctly following where the
       uncertainty went.
    4. **persistence** -- a regime must hold ``persistence_bars``
       consecutive bars before it is acted on.  Open interest does not move
       intraday, so a regime that flickers bar to bar is spot crossing a
       level rather than positioning changing, and trading it churns.

    Exits on the hard rules -- the DTE floor, the stops, the daily loss
    limit -- are never gated.  A gate can stop the system taking a position
    or delay it changing sides; it can never stop it getting out.
    """

    #: (1) |total|/gross GEX below this reads as no usable direction.
    confidence: bool = True
    min_confidence_ratio: float = 0.15
    #: (2) the fixed distance-to-flip test.
    flip_distance: bool = True
    #: (3) unanimity across perturbed models.
    ensemble: bool = True
    #: Added to ``vol.skew_slope`` to make the ensemble members. The base
    #: surface must be in here (a 0.0 delta) or the ensemble is testing a
    #: model the system does not trade.
    ensemble_skew_slope_deltas: list[float] = field(
        default_factory=lambda: [-0.5, 0.0, 0.5]
    )
    #: ``[call_sign, put_sign]`` pairs. These are re-weightings of the
    #: standard convention, not inversions of it: inverting the sign
    #: inverts the answer by construction, so unanimity across an inverted
    #: member is unreachable and would only ever mean "never trade". What
    #: is being tested is whether the read survives dealers being somewhat
    #: less long the calls, or somewhat less short the puts, than assumed.
    ensemble_sign_conventions: list[list[float]] = field(
        default_factory=lambda: [[1.0, -1.0], [1.0, -0.8], [0.8, -1.0]]
    )
    #: Multipliers on ``gex.flow_confidence_contracts`` -- how much
    #: classified tape it takes to overrule the prior. This is the third
    #: perturbed axis and it exists because measuring the sign did not
    #: remove the assumption, it replaced it: "the flow I have seen at this
    #: strike represents the book standing at it" is now the load-bearing
    #: claim, and a scale of 0.5 trusts the tape twice as readily as 2.0
    #: does. Where a strike is heavily traded every member agrees and the
    #: axis costs nothing; where the read rests on a handful of prints they
    #: diverge and the gate blocks, which is the whole point. Must contain
    #: 1.0 -- the traded configuration has to be an ensemble member -- and
    #: it collapses to ``[1.0]`` automatically when there is no flow at all,
    #: so a run without a trade feed pays nothing for it.
    ensemble_flow_confidence_scales: list[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0]
    )
    #: (4) consecutive bars a regime must hold before it is acted on.
    persistence: bool = True
    persistence_bars: int = 3
    #: The entry window, ``strategy.entry_time`` to
    #: ``strategy.entry_cutoff_time``. Off means entries may be taken at any
    #: point in the session; the times themselves live in StrategyConfig
    #: because they are also what the backtest reports against.
    entry_window: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence_ratio < 1.0:
            raise ValueError("gates.min_confidence_ratio must be in [0, 1)")
        if self.persistence_bars < 1:
            raise ValueError("gates.persistence_bars must be >= 1")
        if not self.ensemble_skew_slope_deltas:
            raise ValueError("gates.ensemble_skew_slope_deltas must not be empty")
        if 0.0 not in [float(d) for d in self.ensemble_skew_slope_deltas]:
            raise ValueError(
                "gates.ensemble_skew_slope_deltas must include 0.0 -- the "
                "traded surface has to be one of the ensemble members"
            )
        if not self.ensemble_sign_conventions:
            raise ValueError("gates.ensemble_sign_conventions must not be empty")
        for pair in self.ensemble_sign_conventions:
            if len(pair) != 2:
                raise ValueError(
                    "each gates.ensemble_sign_conventions entry must be "
                    f"[call_sign, put_sign]; got {pair!r}"
                )
        if not self.ensemble_flow_confidence_scales:
            raise ValueError("gates.ensemble_flow_confidence_scales must not be empty")
        if any(float(s) <= 0.0 for s in self.ensemble_flow_confidence_scales):
            raise ValueError("gates.ensemble_flow_confidence_scales must all be > 0")
        if 1.0 not in [float(s) for s in self.ensemble_flow_confidence_scales]:
            raise ValueError(
                "gates.ensemble_flow_confidence_scales must include 1.0 -- the "
                "traded configuration has to be one of the ensemble members"
            )

    def sign_conventions(self) -> list[tuple[float, float]]:
        return [(float(a), float(b)) for a, b in self.ensemble_sign_conventions]

    def flow_confidence_scales(self) -> list[float]:
        return [float(s) for s in self.ensemble_flow_confidence_scales]


@dataclass
class StrategyConfig:
    """Tenor, entry, strike selection and exit rules for the GEX straddle.

    The position is always an at-the-money straddle.  Its *direction* is not
    a parameter -- it is whatever the GEX regime says: long when dealers are
    short gamma, short when they are long it.

    The tenor
    ---------
    The traded series is **today's expiry**, and the position is rolled into
    tomorrow's at the end of the day.  Concretely:

    * during the session the ATM straddle on the 0DTE series is traded;
    * ``close_before_expiry_minutes`` before settlement it is closed, whatever
      it is worth -- the last minutes of an expiring straddle's life are
      where its gamma diverges and the hedger cannot keep up;
    * at that moment the next session's series (1DTE) may be opened in its
      place and carried overnight, becoming the 0DTE position the next
      morning (``roll_at_expiry``);
    * unless the next session is across a weekend or a holiday, in which
      case nothing is opened and the book is flat over the gap
      (``hold_over_weekends``);
    * and never inside the blackout around a scheduled high-volatility
      event: the position is closed ``event_blackout_minutes_before`` the
      event and nothing is opened until ``event_blackout_minutes_after`` it
      (``events`` / ``events_path``).

    The four ``*_days_to_expiry`` numbers are what make that a *policy*
    rather than a hard-coded rule: they bound the expiry that may be
    entered, in trading days (``session.trading_days_between``), and
    ``prefer_*`` picks between the ones in range.  ``0 / 1 / (0, 0)`` reads
    "today's series, or tomorrow's once today's is inside the pre-expiry
    buffer".  The old multi-session tenor is still reachable by widening
    them, and ``close_at_days_to_expiry`` -- the DTE floor that closed that
    tenor early -- is kept for it, disabled by default.
    """

    #: Bounds on the expiry that may be entered, in trading days.
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 1
    #: Inside those bounds, prefer the expiry closest to this window.
    prefer_min_days_to_expiry: int = 0
    prefer_max_days_to_expiry: int = 0
    #: Close the position once it has decayed to this DTE, whatever it is
    #: worth. ``None`` disables the floor. At the shipped 0/1 DTE tenor it
    #: has no work to do -- the pre-expiry buffer below is the exit -- but
    #: a multi-session tenor wants it (1 with a 2-5 DTE range keeps the
    #: book out of the last two sessions).
    close_at_days_to_expiry: int | None = None
    #: Earliest time of day to open a position (exchange local time). The
    #: first minutes after the open are skipped: the opening auction leaves
    #: quotes wide and the chain's vol unsettled. Gated by
    #: ``gates.entry_window``; the end-of-day roll is exempt (see
    #: ``roll_at_expiry``).
    entry_time: time = time(9, 35)
    #: Latest time of day to open a position. A same-day straddle entered
    #: late has little premium left and a gamma the hedger will be fighting
    #: within the hour.
    entry_cutoff_time: time = time(14, 30)
    #: Close the position this many minutes before its series settles. This
    #: leads the exit ladder and nothing can delay it: the last quarter hour
    #: of an ATM straddle is where its gamma diverges. The same buffer
    #: decides which series is *entered* -- one inside it is never opened.
    close_before_expiry_minutes: int = 15
    #: When today's series is closed at the buffer, allow the next session's
    #: series to be opened in its place -- outside the entry window, but
    #: still subject to every GEX gate, the weekend rule and the event
    #: blackout. Off means nothing is opened inside the buffer, so with the
    #: entry-window gate on the book is flat from the buffer to the next
    #: morning's window (with the gate off, an entry after the bell is what
    #: the config asked for).
    roll_at_expiry: bool = True
    #: Carry a position across a weekend or an exchange holiday. Off (the
    #: default) means a series on the far side of any calendar gap is never
    #: entered, and a position already on one is closed at the buffer on
    #: the last session before the gap: an unhedgeable gap is exactly what
    #: a delta-hedged straddle cannot survive.
    hold_over_weekends: bool = False
    #: Scheduled high-volatility events, exchange-local time, as
    #: ``"YYYY-MM-DD HH:MM label"`` strings (see ``events.py``). The
    #: position is closed before each and nothing is opened until after.
    events: list[Any] = field(default_factory=list)
    #: A text file of the same, one per line. ``configs/events.txt`` ships
    #: the FOMC statement times.
    events_path: str | None = None
    #: The blackout either side of an event, in minutes.
    event_blackout_minutes_before: int = 15
    event_blackout_minutes_after: int = 15
    #: SHORT straddle (positive GEX): buy it back if the premium reaches
    #: this multiple of the entry credit. ``None`` disables the stop.
    short_stop_loss_premium_multiple: float | None = 2.5
    #: SHORT straddle: buy it back once this fraction of the credit has
    #: decayed away. ``None`` holds to the timed exit.
    short_take_profit_pct: float | None = 0.60
    #: LONG straddle (negative GEX): exits are measured on *position* P&L --
    #: the straddle mark plus the gamma scalped by the hedge -- as a
    #: fraction of the debit paid. A premium-decay stop would be wrong here:
    #: a long straddle is supposed to bleed on the mark and make it back on
    #: the hedge.
    long_stop_loss_pct: float | None = 0.50
    long_take_profit_pct: float | None = 1.00
    #: Stop trading for the day after a loss this large, as a fraction of
    #: the session's opening equity. ``None`` disables.
    daily_loss_limit_pct: float | None = 0.05
    #: Close the position when the GEX regime flips against it. This is the
    #: whole premise of the strategy -- we hold the side dealers are forced
    #: to take -- so it defaults on.
    exit_on_regime_flip: bool = True
    #: Allow another entry after an exit, so a regime flip can be traded
    #: rather than just closed out.
    reenter_after_exit: bool = True
    #: Ceiling on entry *attempts* per session, so a spot level oscillating
    #: across the flip point cannot churn the book all day -- and so a leg
    #: that keeps failing cannot be retried indefinitely. Attempts rather
    #: than fills is the load-bearing part: a leg reported as unfilled that
    #: the exchange filled anyway would otherwise put the strategy straight
    #: back into an entry, at full size, every poll.
    max_entries_per_session: int = 3

    def tenor(self) -> "TenorPolicy":
        """The tenor rule these fields describe."""
        from .chain import TenorPolicy

        return TenorPolicy(
            min_days=self.min_days_to_expiry,
            max_days=self.max_days_to_expiry,
            prefer_days=(self.prefer_min_days_to_expiry, self.prefer_max_days_to_expiry),
            close_days=self.close_at_days_to_expiry,
            close_before_expiry_minutes=self.close_before_expiry_minutes,
            hold_over_weekends=self.hold_over_weekends,
        )

    def validate(self) -> None:
        if self.min_days_to_expiry < 0:
            raise ValueError("strategy.min_days_to_expiry must be >= 0")
        if self.min_days_to_expiry > self.max_days_to_expiry:
            raise ValueError("strategy.min_days_to_expiry > max_days_to_expiry")
        if self.close_before_expiry_minutes < 0:
            raise ValueError("strategy.close_before_expiry_minutes must be >= 0")
        self.tenor().validate()
        if self.entry_cutoff_time < self.entry_time:
            raise ValueError("strategy.entry_cutoff_time is before entry_time")
        if self.max_entries_per_session < 1:
            raise ValueError("strategy.max_entries_per_session must be >= 1")
        if self.event_blackout_minutes_before < 0 or self.event_blackout_minutes_after < 0:
            raise ValueError("strategy.event_blackout_minutes_* must be >= 0")
        if not isinstance(self.events, list):
            raise ValueError("strategy.events must be a list")


@dataclass
class VolConfig:
    """Volatility surface assumptions used to price strikes off ATM IV.

    IBKR gives us an at-the-money implied vol series for the future, not a
    full surface, so out-of-the-money strikes are priced by extrapolating
    along a log-moneyness skew:

        iv(K) = atm_iv + slope * ln(K/F) + curvature * ln(K/F)^2

    ``slope`` is negative so that lower strikes carry higher vol, which is
    the shape ES actually trades.  These are assumptions, not observations --
    override them from config if you have a fitted surface.
    """

    skew_slope: float = -1.5
    skew_curvature: float = 0.0
    min_iv: float = 0.02
    max_iv: float = 3.0
    #: Multiply the whole surface, e.g. 1.1 to stress-test 10% richer vol.
    iv_multiplier: float = 1.0
    #: Fallback ATM vol when the IV series has a gap.
    fallback_atm_iv: float = 0.15


@dataclass
class CostsConfig:
    """Commissions and slippage applied to every fill."""

    #: Slippage charged on option fills, in option ticks.
    option_slippage_ticks: float = 1.0
    #: Slippage charged on hedge fills, in hedge ticks.
    hedge_slippage_ticks: float = 0.5
    #: Exchange + clearing + regulatory fees per option contract, USD.
    option_fees_per_contract: float = 2.32
    hedge_fees_per_contract: float = 0.62
    #: Apply costs at all. Turn off to isolate strategy P&L.
    enabled: bool = True


@dataclass
class FlowConfig:
    """The trade tape, and how executions are turned into a dealer sign.

    This section is what replaces the static call/put sign convention with
    a measurement.  ``deltahedger.flow`` classifies each option execution as
    buyer- or seller-initiated -- from MDP 3.0's own aggressor flag where
    the feed carries it, from the MBO book-state change where it carries
    that, and from the Lee-Ready quote and tick rules otherwise -- and the
    dealer is the resting side of whatever it decides.  ``gex`` then blends
    the measured sign at each strike against ``gex.call_sign`` /
    ``gex.put_sign`` as the prior.

    ``source`` of ``"none"`` (the default) attaches no feed at all, and the
    system behaves exactly as it did before: every strike falls back to the
    prior.  That is deliberate -- the tape is an input a deployment either
    has or does not, and a missing one must degrade to the old behaviour
    loudly rather than silently synthesising flow.
    """

    #: ``"none"`` | ``"csv"`` | ``"synthetic"`` | ``"ibkr"`` |
    #: ``"databento"``. The last two need a live connection and are built
    #: by the live runner, not here.
    #:
    #: ``databento`` is the one that reads MDP 3.0's aggressor flag
    #: directly, so rule 1 of the classification chain resolves the tape
    #: and the Lee-Ready rules never have to run. It shares the session the
    #: Databento open-interest providers own, so it needs
    #: ``data.open_interest`` set to ``databento`` or ``databento_flow``
    #: too; ``ibkr`` relays no flag and falls back to Lee-Ready.
    source: str = "none"
    #: CSV replay: a tape with
    #: ``timestamp,expiry,strike,right,price,size`` and, where the export
    #: has them, ``bid,ask,aggressor,bid_size_delta,ask_size_delta``.
    #: ``aggressor`` takes MDP 3.0 tag 5797 verbatim (1 buy, 2 sell, 0 none).
    csv_path: str | None = None

    # -- the classification chain ----------------------------------------
    #: (1) MDP 3.0 tag 5797. Authoritative when present; nothing below runs.
    use_aggressor_flag: bool = True
    #: (2) MBO: the side whose resting size fell was the passive one.
    use_book_delta: bool = True
    #: (3) Lee-Ready quote rule, against the prevailing bid/ask.
    use_quote_rule: bool = True
    #: (4) Lee-Ready tick test, for midpoint trades and missing quotes.
    use_tick_rule: bool = True
    #: How far inside the touch still counts as trading "at" it, in option
    #: ticks. Capped at half the spread internally so the bid and the ask
    #: test can never both fire. Zero demands an exact touch, which throws
    #: away most of a wide options quote's tape.
    quote_tolerance_ticks: float = 1.0
    #: Minutes after which a classified contract counts half as much. 0
    #: disables the decay: for a 0DTE series the whole book was written in
    #: the window being watched and there is nothing stale to fade. Set it
    #: for longer-dated series, where last week's prints should not carry
    #: the same weight as this morning's.
    half_life_minutes: float = 0.0

    # -- the live feed ----------------------------------------------------
    #: Strikes either side of spot to subscribe to trades on, as a fraction
    #: of spot. Defaults to matching ``gex.strike_width_pct`` when null --
    #: there is no point classifying flow at a strike the profile excludes.
    strike_width_pct: float | None = None
    #: Seconds of tick-by-tick history to pull when a subscription first
    #: opens, so a restart mid-session does not start from an empty tape.
    backfill_seconds: float = 3600.0

    # -- synthetic (a harness, not a market model) ------------------------
    #: Synthetic tape: contracts traded per expiry per bar. The synthetic
    #: feed exists to exercise the classification path end to end; it says
    #: the machinery works, never that the signal does.
    synthetic_contracts_per_bar: float = 400.0
    #: Share of synthetic trades that carry an explicit aggressor flag; the
    #: rest are left to the quote and tick rules, so a generated run
    #: exercises the whole chain rather than only its first link.
    synthetic_flagged_share: float = 0.6
    synthetic_seed: int = 23

    def validate(self) -> None:
        if self.source.lower() not in (
            "none", "csv", "synthetic", "ibkr", "databento"
        ):
            raise ValueError(
                f"unknown flow.source {self.source!r}; use 'none', 'csv', "
                "'synthetic', 'ibkr' or 'databento'"
            )
        if self.source.lower() == "csv" and not self.csv_path:
            raise ValueError("flow.csv_path must be set when flow.source == 'csv'")
        if self.quote_tolerance_ticks < 0.0:
            raise ValueError("flow.quote_tolerance_ticks must be >= 0")
        if self.half_life_minutes < 0.0:
            raise ValueError("flow.half_life_minutes must be >= 0")
        if self.strike_width_pct is not None and self.strike_width_pct <= 0.0:
            raise ValueError("flow.strike_width_pct must be > 0 or null")
        if self.backfill_seconds < 0.0:
            raise ValueError("flow.backfill_seconds must be >= 0")
        if not 0.0 <= self.synthetic_flagged_share <= 1.0:
            raise ValueError("flow.synthetic_flagged_share must be in [0, 1]")

    def window_pct(self, gex: "GexConfig") -> float:
        """Strike window for the live trade subscription."""
        if self.strike_width_pct is not None:
            return float(self.strike_width_pct)
        return float(gex.strike_width_pct)


@dataclass
class DataConfig:
    """Where historical bars come from."""

    source: str = "ibkr"  # "ibkr" | "csv" | "synthetic"
    bar_size: str = "5 mins"
    #: Directory for cached IBKR downloads.
    cache_dir: str = "data_cache"
    #: CSV source: path to a file with timestamp,open,high,low,close[,iv].
    csv_path: str | None = None
    #: Synthetic source parameters, for testing the machinery without IBKR.
    synthetic_days: int = 20
    synthetic_start_price: float = 5000.0
    synthetic_annual_vol: float = 0.16
    synthetic_annual_drift: float = 0.0
    synthetic_seed: int = 7
    #: Dynamics of the generated implied-vol series. The defaults make IV
    #: wander and lean against returns, which is realistic and is also why a
    #: generated run is NOT zero-edge for a straddle: entry vol is marked
    #: away afterwards, and that vega P&L has nothing to do with the
    #: strategy. Set all three to 0 for a genuinely neutral control -- see
    #: configs/es_zero_edge.yaml.
    synthetic_vol_of_vol: float = 2.0
    synthetic_vol_mean_reversion: float = 0.08
    synthetic_vol_return_beta: float = -8.0
    #: Use this ATM IV when the source has no IV column.
    default_atm_iv: float = 0.15

    # -- open interest, which is what GEX is computed from ---------------
    #: "synthetic" | "csv" | "ibkr" | "databento" | "databento_flow". The
    #: bar source and the open-interest source are separate on purpose: real
    #: ES bars with modelled OI is a legitimate study, and pretending
    #: otherwise would hide which half of the result is assumed.
    #:
    #: "databento" reads exchange open interest directly off CME's MDP 3.0
    #: feed rather than through IBKR's relay -- see DatabentoConfig. It is
    #: still the exchange's once-a-session figure; "databento_flow" adds
    #: cumulative signed trade volume since that print as a same-day proxy.
    #: Both are live-only, like "ibkr".
    open_interest: str = "synthetic"
    #: CSV open interest: a file with date,strike,call_oi,put_oi.
    oi_csv_path: str | None = None
    #: Synthetic OI: total contracts spread across the surface.
    oi_total_contracts: float = 40_000.0
    #: Synthetic OI: Gaussian width of the strike distribution, as a
    #: fraction of the anchor price.
    oi_width_pct: float = 0.010
    #: Synthetic OI: where call and put mass sits relative to the anchor.
    #: Calls above and puts below is the shape a real index chain has, and
    #: it is what puts the gamma flip point between the two.
    oi_call_center_pct: float = 0.005
    oi_put_center_pct: float = -0.007
    #: Synthetic OI: mean share of open interest that is calls, and the
    #: day-to-day swing around it. The swing is what makes the generated
    #: sessions span both GEX regimes instead of only one.
    oi_call_share_mean: float = 0.50
    oi_call_share_swing: float = 0.16
    #: Synthetic OI: half-width, in expiry-days, of the window the call
    #: share is smoothed over. Non-zero makes neighbouring expiries lean the
    #: same way, which is what real positioning does and what keeps the
    #: front-expiry blend from averaging independent draws into a flat book.
    #: 0 restores independent per-expiry draws.
    oi_call_share_smoothing_days: int = 2
    oi_seed: int = 11


@dataclass
class IBKRConfig:
    """Connection settings for TWS / IB Gateway."""

    host: str = "127.0.0.1"
    #: 7497 paper TWS, 7496 live TWS, 4002 paper gateway, 4001 live gateway.
    port: int = 7497
    client_id: int = 17
    account: str | None = None
    #: Refuse to run against a live (non-paper) account unless set True.
    allow_live_trading: bool = False
    #: Seconds to wait for a connection.
    connect_timeout: float = 15.0
    #: Delayed data (15 min) when a realtime subscription is missing.
    use_delayed_data: bool = False
    #: Poll interval for the live loop, seconds.
    poll_seconds: float = 5.0
    #: Ask IBKR for real margin via whatIf orders instead of the heuristic.
    use_whatif_margin: bool = True
    #: Order type for every order this system sends -- straddle legs and
    #: hedges alike: "MKT" or "LMT". It was called ``hedge_order_type``,
    #: which was wrong: there has only ever been one setting and ``_send``
    #: has always read it for both instruments, so switching hedges to
    #: limit orders silently switched the option legs too.
    order_type: str = "MKT"
    #: The old name, still accepted so existing configs keep working. When
    #: set it wins, and ``validate`` says so once.
    hedge_order_type: str | None = None
    #: For LMT orders, cross the spread by this many ticks.
    limit_cross_ticks: float = 1.0

    def validate(self) -> None:
        if self.hedge_order_type is not None:
            log.warning(
                "ibkr.hedge_order_type is deprecated and has been renamed to "
                "ibkr.order_type -- it was never hedge-only, and %r is being "
                "applied to the straddle legs as well. Rename it in the config.",
                self.hedge_order_type,
            )
            self.order_type = self.hedge_order_type
            self.hedge_order_type = None
        if self.order_type.upper() not in ("MKT", "LMT"):
            raise ValueError('ibkr.order_type must be "MKT" or "LMT"')


@dataclass
class DatabentoConfig:
    """Live market data from CME's MDP 3.0 feed, via Databento.

    Read when ``data.open_interest`` is ``databento`` or ``databento_flow``.
    The API key is never written to a config file -- like the IBKR
    credentials, which live in the Gateway's own config rather than here,
    it is read from the environment at connect time.
    """

    #: Environment variable holding the Databento API key.
    api_key_env: str = "DATABENTO_API_KEY"
    dataset: str = "GLBX.MDP3"
    #: Override the CME root used for every expiry, e.g. "ES" (subscribed
    #: as "ES.OPT"). Leave unset for ES: its 0DTE weeklies are listed under
    #: a different Globex root per weekday (confirmed live: "ES.OPT" itself
    #: resolves to zero instruments), so the default instead asks IBKR to
    #: qualify each expiry and reads its root back -- see
    #: DatabentoSession.ensure_subscribed. Set this only for a risk source
    #: without that per-weekday split.
    parent_symbol: str | None = None
    #: Reconnect automatically if the live session drops.
    reconnect: bool = True


@dataclass
class LiveConfig:
    """Running unattended, for days at a time.

    A forward walk is not a long backtest: the process has to survive things
    a backtest never sees. IBKR force-restarts the gateway once a day, which
    drops the API connection; a VPS reboots; a network blips. None of those
    should end the test, and none of them should lose the record of what
    happened before them.
    """

    #: Write every decision to disk as it is made. Off means a crash takes
    #: the whole session's evidence with it, so it defaults on.
    journal: bool = True
    journal_dir: str = "runs/live"
    #: Reconnect after the connection drops, rather than ending the run.
    #: The daily gateway restart makes this mandatory for any walk longer
    #: than a day.
    reconnect: bool = True
    #: Backoff between reconnect attempts, seconds. Doubles up to the max.
    reconnect_backoff_seconds: float = 15.0
    max_reconnect_backoff_seconds: float = 300.0
    #: Give up after this many consecutive failures. ``None`` retries
    #: forever, which is usually what you want under a process supervisor.
    max_reconnect_attempts: int | None = None
    #: Log a heartbeat line this often even when nothing happens, so a
    #: silent log can be told apart from a stalled process.
    heartbeat_seconds: float = 300.0
    #: Re-check the broker's positions against the in-memory book this
    #: often, seconds. Between two connects the book is only this
    #: process's record of its own fills, and an order the broker filled
    #: after the runner gave up waiting on it is invisible to that record
    #: -- so a book that has drifted keeps trading on a position it does
    #: not know the size of. This is what notices. ``None`` checks only at
    #: connect, which is the old behaviour and is not recommended.
    reconcile_seconds: float | None = 300.0

    def validate(self) -> None:
        if self.reconcile_seconds is not None and self.reconcile_seconds <= 0:
            raise ValueError("live.reconcile_seconds must be positive, or null")
        if self.reconnect_backoff_seconds <= 0:
            raise ValueError("live.reconnect_backoff_seconds must be > 0")
        if self.max_reconnect_backoff_seconds < self.reconnect_backoff_seconds:
            raise ValueError(
                "live.max_reconnect_backoff_seconds < reconnect_backoff_seconds"
            )
        if self.max_reconnect_attempts is not None and self.max_reconnect_attempts < 1:
            raise ValueError("live.max_reconnect_attempts must be >= 1 or null")


@dataclass
class Config:
    """Top-level configuration."""

    risk_source: str = "ES"
    starting_equity: float = 100_000.0
    risk_free_rate: float = 0.04
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    gex: GexConfig = field(default_factory=GexConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    databento: DatabentoConfig = field(default_factory=DatabentoConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    #: Backtest window, ISO dates. ``None`` means "whatever the source has".
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        self.strategy.entry_time = _parse_time(self.strategy.entry_time)
        self.strategy.entry_cutoff_time = _parse_time(self.strategy.entry_cutoff_time)
        self.validate()

    @property
    def source(self) -> RiskSource:
        return get_risk_source(self.risk_source)

    def validate(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        get_risk_source(self.risk_source)  # raises on an unknown symbol
        for section in (
            self.hedge, self.sizing, self.gex, self.gates, self.strategy,
            self.flow, self.ibkr, self.live,
        ):
            section.validate()

    # -- serialisation -------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        def build(dc_type: type, values: Any) -> Any:
            if not is_dataclass(dc_type) or not isinstance(values, dict):
                return values
            known = {f.name: f for f in fields(dc_type)}
            unknown = set(values) - set(known)
            if unknown:
                raise ValueError(
                    f"unknown {dc_type.__name__} keys: {', '.join(sorted(unknown))}"
                )
            return dc_type(**{k: build(known[k].type, v) for k, v in values.items()})

        known = {f.name: f for f in fields(cls)}
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            target = {
                "hedge": HedgeConfig,
                "sizing": SizingConfig,
                "gex": GexConfig,
                "gates": GatesConfig,
                "strategy": StrategyConfig,
                "vol": VolConfig,
                "costs": CostsConfig,
                "data": DataConfig,
                "flow": FlowConfig,
                "ibkr": IBKRConfig,
                "live": LiveConfig,
            }.get(key)
            kwargs[key] = build(target, value) if target else value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    def event_calendar(self):
        """The parsed event blackout calendar, in the exchange's timezone."""
        from zoneinfo import ZoneInfo

        from .events import EventCalendar

        return EventCalendar.from_config(self.strategy, ZoneInfo(self.source.timezone))

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, time):
                return value.strftime("%H:%M")
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(dataclasses.asdict(self))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
