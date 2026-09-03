"""Configuration for the GEX-directed delta-hedged straddle system.

One ``Config`` object drives both the backtest and the live runner, so a
forward test routes the same parameters that were validated historically.
Load from YAML with ``Config.from_yaml`` or build in code.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import time
from pathlib import Path
from typing import Any

import yaml

from .instruments import RiskSource, get_risk_source


def _parse_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        parts = [int(p) for p in value.split(":")]
        while len(parts) < 3:
            parts.append(0)
        return time(*parts[:3])
    raise TypeError(f"cannot read a time-of-day from {value!r}")


@dataclass
class HedgeConfig:
    """The delta band -- a fixed, heuristic threshold.

    ``target`` and ``band`` are in delta units (1 unit == 1% of one ES
    contract).  The position is an ATM straddle, so the target is 0: hold
    the book delta-neutral and let the straddle express the gamma view.

    ``band`` is deliberately a *fixed heuristic* for the initial forward
    walk.  One MES contract moves net delta by 10 units, so a band narrower
    than 5 cannot bind (see ``hedger.py``); 10 is the smallest value that
    both binds and leaves room for a whole contract to land inside it.  A
    band that scales with gamma or realised vol is the obvious next step and
    is deliberately *not* in this version -- the paper test is meant to
    measure one threshold, not a fitted schedule.

    ``overnight_band_multiplier`` is the one exception, and it is a
    consequence of the tenor rather than a fitted schedule.  The position is
    now carried overnight, and the hours outside the regular session are not
    the same market: the book is quoted wider, a single MES fill costs
    proportionally more, and the delta that a hedge would chase is as likely
    to be reversed by the open as realised.  Outside RTH the band is
    multiplied by this factor, so only a *larger* breach is hedged; the
    hedge is not switched off, because an overnight gap is exactly when an
    unhedged straddle does the most damage.  Set it to 1.0 to hedge
    identically around the clock.

    Note that the band means opposite things in the two regimes.  Long the
    straddle (negative GEX) each hedge realises a gamma scalp, so a tighter
    band scalps more and pays more commission; short the straddle (positive
    GEX) each hedge locks in a loss against theta, so a tighter band bleeds
    faster.  One number for both is a simplification, and the results
    report is written to make its cost visible.
    """

    target: float = 0.0
    band: float = 10.0
    #: Widen the band by this factor outside the regular session. 1.0 hedges
    #: overnight exactly as it does intraday.
    overnight_band_multiplier: float = 2.5
    #: Don't send a hedge smaller than this many contracts.
    min_hedge_contracts: int = 1
    #: Cap on a single hedge order, as a guard against a data glitch.
    max_hedge_contracts: int = 200
    #: Seconds to wait between hedges; suppresses churn on noisy quotes.
    min_seconds_between_hedges: float = 0.0
    #: Flatten the hedge in the same order as the straddle, rather than
    #: leaving it for the band. Against a neutral target this is a smaller
    #: switch than it looks: an orphaned hedge is a naked directional
    #: position, so the band closes anything larger than itself on the very
    #: next pass either way. What the flag actually governs is whether a
    #: sub-band residual -- under one hedge contract -- is left behind.
    flatten_hedge_on_exit: bool = True

    def effective_band(self, in_session: bool = True) -> float:
        """The half-width that applies right now."""
        return self.band if in_session else self.band * self.overnight_band_multiplier

    def bounds(self, in_session: bool = True) -> tuple[float, float]:
        half = self.effective_band(in_session)
        return self.target - half, self.target + half

    @property
    def lower(self) -> float:
        return self.target - self.band

    @property
    def upper(self) -> float:
        return self.target + self.band

    def in_band(self, delta_units: float, in_session: bool = True) -> bool:
        low, high = self.bounds(in_session)
        return low <= delta_units <= high

    def validate(self) -> None:
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
    """How much of the account to commit to the straddle."""

    #: Fraction of portfolio equity to allocate as buying power. It covers
    #: margin for a short straddle and the debit for a long one.
    buying_power_pct: float = 0.15
    #: Hard cap on straddles regardless of buying power.
    max_straddles: int = 25
    #: Never open a position smaller than this.
    min_straddles: int = 1
    #: Fraction of the buying-power budget held back for hedge margin and
    #: variation margin. The straddle sizing sees the remainder.
    hedge_margin_reserve_pct: float = 0.30
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
        if not 0.0 <= self.hedge_margin_reserve_pct < 1.0:
            raise ValueError("sizing.hedge_margin_reserve_pct must be in [0, 1)")
        if self.margin_model not in ("span_scan", "reg_t", "fixed"):
            raise ValueError(
                "sizing.margin_model must be one of 'span_scan', 'reg_t', 'fixed'"
            )
        if self.max_straddles < self.min_straddles:
            raise ValueError("sizing.max_straddles < min_straddles")


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
    standard assumption and what every published GEX print uses.
    """

    enabled: bool = True
    #: Dealer inventory signs applied to open interest at each strike.
    call_sign: float = 1.0
    put_sign: float = -1.0
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
    3. **ensemble** -- recompute the regime over a small grid of skew and
       sign-convention perturbations and trade only if every member agrees.
       This is the only gate that tests the *model* rather than the data:
       both perturbed inputs are assumptions the README flags as
       load-bearing, and a regime that reverses under a plausible variation
       of either was never a reading of the market.
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

    def sign_conventions(self) -> list[tuple[float, float]]:
        return [(float(a), float(b)) for a, b in self.ensemble_sign_conventions]


@dataclass
class StrategyConfig:
    """Tenor, entry, strike selection and exit rules for the GEX straddle.

    The position is always an at-the-money straddle.  Its *direction* is not
    a parameter -- it is whatever the GEX regime says: long when dealers are
    short gamma, short when they are long it.

    The *tenor* is a parameter, and it is the one that changed.  The system
    traded 0DTE and now trades a listed expiry two to five sessions out,
    closing at ``close_at_days_to_expiry``.  All four numbers are trading
    days, not calendar days (see ``session.trading_days_between``).
    """

    #: Bounds on the expiry that may be entered, in trading days.
    min_days_to_expiry: int = 2
    max_days_to_expiry: int = 5
    #: Inside those bounds, prefer the expiry closest to this window. The
    #: middle of the range is where neither failure mode bites: entering at
    #: 2 DTE leaves one session before the close-out, and entering at 5
    #: carries the most vega for the longest.
    prefer_min_days_to_expiry: int = 3
    prefer_max_days_to_expiry: int = 4
    #: Close the position once it has decayed to this DTE, whatever it is
    #: worth. With the default the exit lands on the first bar of the
    #: session before expiry, so the book is never carried into the last two
    #: sessions where an ATM straddle's gamma, its pin risk and the staleness
    #: of the open-interest print all get worse together.
    close_at_days_to_expiry: int = 1
    #: Earliest time of day to open a position (exchange local time).
    #: Defaults to a morning window that starts after the exchange has
    #: published final open interest for the previous session -- the input
    #: GEX is computed from. Before it lands, the profile is built on a
    #: preliminary print. Gated by ``gates.entry_window``.
    entry_time: time = time(10, 0)
    #: Latest time of day to open a position.
    entry_cutoff_time: time = time(11, 30)
    #: Backstop close, this many minutes before expiry. With the DTE floor
    #: above this should never be what closes a position; it is here so that
    #: a config which disables the floor still cannot hold into settlement.
    close_before_expiry_minutes: int = 5
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
    #: Ceiling on entries per session, so a spot level oscillating across
    #: the flip point cannot churn the book all day.
    max_entries_per_session: int = 3

    def tenor(self) -> "TenorPolicy":
        """The tenor rule these fields describe."""
        from .chain import TenorPolicy

        return TenorPolicy(
            min_days=self.min_days_to_expiry,
            max_days=self.max_days_to_expiry,
            prefer_days=(self.prefer_min_days_to_expiry, self.prefer_max_days_to_expiry),
            close_days=self.close_at_days_to_expiry,
        )

    def validate(self) -> None:
        if self.min_days_to_expiry < 0:
            raise ValueError("strategy.min_days_to_expiry must be >= 0")
        if self.min_days_to_expiry > self.max_days_to_expiry:
            raise ValueError("strategy.min_days_to_expiry > max_days_to_expiry")
        self.tenor().validate()
        if self.entry_cutoff_time < self.entry_time:
            raise ValueError("strategy.entry_cutoff_time is before entry_time")
        if self.max_entries_per_session < 1:
            raise ValueError("strategy.max_entries_per_session must be >= 1")


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
    #: "synthetic" | "csv" | "ibkr". The bar source and the open-interest
    #: source are separate on purpose: real ES bars with modelled OI is a
    #: legitimate study, and pretending otherwise would hide which half of
    #: the result is assumed.
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
    #: Order type for hedges: "MKT" or "LMT".
    hedge_order_type: str = "MKT"
    #: For LMT hedges, cross the spread by this many ticks.
    limit_cross_ticks: float = 1.0


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

    def validate(self) -> None:
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
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
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
            self.hedge, self.sizing, self.gex, self.gates, self.strategy, self.live,
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
                "ibkr": IBKRConfig,
                "live": LiveConfig,
            }.get(key)
            kwargs[key] = build(target, value) if target else value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

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
