"""Configuration for the delta-hedged short-vol system.

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
    """The delta band.

    ``target`` and ``band`` are in delta units (1 unit == 1% of one ES
    contract), so the default below is the ``20 +/- 3`` the strategy is
    specified against: hold net delta at +20, and hedge whenever it leaves
    [17, 23].

    Note on granularity: one MES contract moves net delta by 10 units, which
    is wider than the 6-unit band.  The hedger therefore trades to the whole
    contract count that lands *closest* to ``target`` and stops -- it cannot
    always finish inside the band, and it deliberately does not keep trading
    trying to.  ``residual_delta`` in the results shows what was left over.
    """

    target: float = 20.0
    band: float = 3.0
    #: Don't send a hedge smaller than this many contracts.
    min_hedge_contracts: int = 1
    #: Cap on a single hedge order, as a guard against a data glitch.
    max_hedge_contracts: int = 200
    #: Seconds to wait between hedges; suppresses churn on noisy quotes.
    min_seconds_between_hedges: float = 0.0
    #: Flatten the hedge when the short option position is closed.
    flatten_hedge_on_exit: bool = True

    @property
    def lower(self) -> float:
        return self.target - self.band

    @property
    def upper(self) -> float:
        return self.target + self.band

    def in_band(self, delta_units: float) -> bool:
        return self.lower <= delta_units <= self.upper

    def validate(self) -> None:
        if self.band < 0:
            raise ValueError("hedge.band must be >= 0")
        if self.min_hedge_contracts < 1:
            raise ValueError("hedge.min_hedge_contracts must be >= 1")
        if self.max_hedge_contracts < self.min_hedge_contracts:
            raise ValueError("hedge.max_hedge_contracts < min_hedge_contracts")


@dataclass
class SizingConfig:
    """How much of the account to commit as margin."""

    #: Fraction of portfolio equity to allocate as buying power for margin.
    buying_power_pct: float = 0.15
    #: Hard cap on short option contracts regardless of buying power.
    max_short_contracts: int = 50
    #: Never open a position smaller than this.
    min_short_contracts: int = 1
    #: Fraction of the buying-power budget held back for hedge margin and
    #: variation margin. The short-put sizing sees the remainder.
    hedge_margin_reserve_pct: float = 0.30
    #: Margin model: "span_scan", "reg_t" or "fixed". See ``sizing.py`` --
    #: "span_scan" reproduces CME SPAN methodology and is the right default
    #: for futures options; "reg_t" is the equity-option rule and will
    #: badly overstate futures margin.
    margin_model: str = "span_scan"
    #: Used when margin_model == "fixed": USD initial margin per short put.
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
        if self.max_short_contracts < self.min_short_contracts:
            raise ValueError("sizing.max_short_contracts < min_short_contracts")


@dataclass
class StrategyConfig:
    """Entry, strike selection and exit rules for the short option book."""

    #: Prefer the shortest-dated expiry; 0 means "0DTE if one is listed".
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 1
    #: Target absolute delta of the put we sell (0.20 == the 20-delta put).
    short_put_delta: float = 0.20
    #: Accept a strike whose delta is within this of the target.
    short_put_delta_tolerance: float = 0.10
    #: Alternative selection: fixed % out of the money. Used when
    #: strike_mode == "moneyness".
    strike_mode: str = "delta"  # "delta" | "moneyness"
    short_put_otm_pct: float = 0.01
    #: Also sell a call against the same expiry, turning the put into a
    #: strangle (a literal same-strike straddle only coincidentally, since
    #: the two legs are selected independently by delta).  Net portfolio
    #: delta is still whatever ``hedge.target`` says -- adding a call does
    #: not change the directional bias, it changes the option book's shape:
    #: a symmetric call target (matching short_put_delta) roughly
    #: delta-neutralises the option legs *before* hedging, so the hedge
    #: still does all the work of holding the target bias, now against a
    #: book that collects premium on both sides.
    sell_call: bool = False
    #: Target absolute delta of the call we sell. Independent of
    #: short_put_delta so the strangle can be skewed deliberately; defaults
    #: to the same magnitude for a roughly symmetric structure.
    short_call_delta: float = 0.20
    short_call_delta_tolerance: float = 0.10
    #: Earliest time of day to open a position (exchange local time).
    entry_time: time = time(9, 35)
    #: Latest time of day to open a position.
    entry_cutoff_time: time = time(11, 0)
    #: Close the short option(s) this many minutes before they expire.
    close_before_expiry_minutes: int = 5
    #: Buy back the position once its *combined* mark (both legs, when
    #: sell_call is set) reaches this multiple of the *combined* entry
    #: credit. ``None`` disables the stop.
    stop_loss_premium_multiple: float | None = 3.0
    #: Buy back the position once this fraction of the combined credit has
    #: been captured. ``None`` holds to the timed exit.
    take_profit_pct: float | None = None
    #: Stop trading for the day after a loss this large, as a fraction of
    #: starting equity. ``None`` disables.
    daily_loss_limit_pct: float | None = 0.05
    #: Allow more than one entry per session.
    reenter_after_exit: bool = False

    def validate(self) -> None:
        if self.strike_mode not in ("delta", "moneyness"):
            raise ValueError("strategy.strike_mode must be 'delta' or 'moneyness'")
        if not 0.0 < self.short_put_delta < 1.0:
            raise ValueError("strategy.short_put_delta must be in (0, 1)")
        if not 0.0 < self.short_call_delta < 1.0:
            raise ValueError("strategy.short_call_delta must be in (0, 1)")
        if self.min_days_to_expiry > self.max_days_to_expiry:
            raise ValueError("strategy.min_days_to_expiry > max_days_to_expiry")
        if self.entry_cutoff_time < self.entry_time:
            raise ValueError("strategy.entry_cutoff_time is before entry_time")


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
    #: Use this ATM IV when the source has no IV column.
    default_atm_iv: float = 0.15
    #: Days before expiry at which the front-month future rolls. ES volume
    #: moves to the next quarterly about 8 days out (the Thursday before the
    #: third Friday). Used to stitch a continuous history from the concrete
    #: contracts that were front month at each point in time.
    roll_days_before_expiry: int = 8


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
class Config:
    """Top-level configuration."""

    risk_source: str = "ES"
    starting_equity: float = 100_000.0
    risk_free_rate: float = 0.04
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
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
        for section in (self.hedge, self.sizing, self.strategy):
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
                "strategy": StrategyConfig,
                "vol": VolConfig,
                "costs": CostsConfig,
                "data": DataConfig,
                "ibkr": IBKRConfig,
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
