"""Backtest metrics and reporting.

Two things get measured here that a generic backtest report would not
bother with, because they are the whole question for this strategy:

  * **hedge quality** -- what fraction of bars sat inside the delta band, and
    how far from target the position actually ran.  With MES granularity
    coarser than the band, "in band" is an outcome, not a given.
  * **P&L attribution** -- premium captured on the short put versus what the
    hedging cost.  A short-vol book that makes money on the option and gives
    it all back on the hedge is worth knowing about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..strategy import BarState, StrategyEvent

TRADING_DAYS = 252


@dataclass
class Metrics:
    starting_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    trading_days: int
    entries: int
    winning_days: int
    losing_days: int
    win_rate: float
    option_pnl: float
    hedge_pnl: float
    fees_paid: float
    hedges: int
    hedge_contracts_traded: int
    bars_in_band: int
    bars_with_position: int
    pct_bars_in_band: float
    mean_abs_delta_error: float
    max_abs_delta_error: float
    mean_net_delta: float
    # Band feasibility: is the target band even reachable at this size?
    median_gamma_units: float
    band_width_points: float
    pct_bars_band_below_tick: float
    hedge_tick_points: float
    hedge_quantum: float
    band_half_width: float

    def summary(self) -> str:
        pct = lambda x: f"{x * 100:,.2f}%"  # noqa: E731
        usd = lambda x: f"${x:,.0f}"  # noqa: E731
        return "\n".join(
            [
                "Performance",
                f"  starting equity      {usd(self.starting_equity)}",
                f"  final equity         {usd(self.final_equity)}",
                f"  total return         {pct(self.total_return)}",
                f"  max drawdown         {usd(-self.max_drawdown)} ({pct(-self.max_drawdown_pct)})",
                f"  Sharpe (daily, ann.) {self.sharpe:,.2f}",
                f"  Sortino              {self.sortino:,.2f}",
                f"  trading days         {self.trading_days}",
                "",
                "Attribution",
                f"  short put P&L        {usd(self.option_pnl)}",
                f"  hedge P&L            {usd(self.hedge_pnl)}",
                f"  fees & commissions   {usd(-self.fees_paid)}",
                "",
                "Trading",
                f"  entries              {self.entries}",
                f"  winning / losing days {self.winning_days} / {self.losing_days}"
                f"  ({pct(self.win_rate)})",
                f"  hedge trades         {self.hedges}"
                f" ({self.hedge_contracts_traded} contracts)",
                "",
                "Hedge quality",
                f"  bars inside band     {self.bars_in_band} / {self.bars_with_position}"
                f"  ({pct(self.pct_bars_in_band)})",
                f"  mean |delta - target| {self.mean_abs_delta_error:,.2f} delta units",
                f"  max  |delta - target| {self.max_abs_delta_error:,.2f} delta units",
                f"  mean net delta        {self.mean_net_delta:,.2f} delta units",
                "",
                "Band feasibility",
                f"  median position gamma {self.median_gamma_units:,.1f} delta units per point",
                f"  band width in points  {self.band_width_points:,.3f}"
                f"  (hedge tick {self.hedge_tick_points:g})",
                f"  band finer than a tick on {pct(self.pct_bars_band_below_tick)} of held bars",
                f"  hedge quantum         {self.hedge_quantum:,.0f} delta units"
                f" per contract",
                self._granularity_note(),
                self._feasibility_note(),
            ]
        )

    def _granularity_note(self) -> str:
        """Warn when the band is too narrow for the hedge instrument to see.

        A hedge is only issued when a whole contract moves net delta closer
        to target, which needs an error above half the quantum.  Any band
        narrower than that fires on exactly the same bars as a band of
        half-quantum: the parameter is inert.
        """
        dead_zone = self.hedge_quantum / 2.0
        if self.band_half_width >= dead_zone:
            return (
                f"  -> band +/-{self.band_half_width:g} is wider than half the "
                f"quantum ({dead_zone:g}); it binds."
            )
        return (
            f"  -> band +/-{self.band_half_width:g} is INERT: it is narrower than\n"
            f"     half a hedge contract ({dead_zone:g} delta units), so the system\n"
            f"     behaves exactly as if the band were +/-{dead_zone:g}. To make the\n"
            f"     band bind, hedge with a smaller instrument or widen it."
        )

    def _feasibility_note(self) -> str:
        """Plain-language read on whether the band can be held at this size."""
        if self.bars_with_position == 0:
            return "  (no position was ever held)"
        if self.pct_bars_band_below_tick > 0.5:
            return (
                "  -> the band is narrower than one tick of underlying movement\n"
                "     for most of the session: it will be breached on essentially\n"
                "     every bar. Widen hedge.band, cut size, or accept the churn."
            )
        if self.pct_bars_in_band < 0.8:
            return (
                "  -> the band is reachable but the hedge granularity leaves the\n"
                "     position outside it a material share of the time."
            )
        return "  -> the band is comfortably holdable at this position size."


@dataclass
class BacktestResult:
    metrics: Metrics
    bars: pd.DataFrame
    events: pd.DataFrame
    fills: pd.DataFrame
    daily: pd.DataFrame
    config: dict = field(default_factory=dict)

    def save(self, directory: str | "os.PathLike") -> None:  # noqa: F821
        from pathlib import Path

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.bars.to_csv(out / "bars.csv", index=False)
        self.events.to_csv(out / "events.csv", index=False)
        self.fills.to_csv(out / "fills.csv", index=False)
        self.daily.to_csv(out / "daily.csv", index=False)
        (out / "summary.txt").write_text(self.metrics.summary() + "\n")


def bars_to_frame(states: list[BarState], target: float) -> pd.DataFrame:
    rows = []
    for s in states:
        greeks = s.option_greeks
        rows.append(
            {
                "timestamp": s.timestamp,
                "future": s.future,
                "atm_iv": s.atm_iv,
                "hours_to_expiry": s.time_to_expiry * 365 * 24,
                "strike": s.strike,
                "option_contracts": s.option_contracts,
                "option_mark": s.option_mark,
                "option_delta": greeks.delta if greeks else None,
                "option_delta_units": s.option_delta_units,
                "hedge_contracts": s.hedge_contracts,
                "hedge_delta_units": s.hedge_delta_units,
                "net_delta_units": s.net_delta_units,
                "delta_error": s.net_delta_units - target,
                "gamma_units": s.gamma_units,
                "in_band": s.in_band,
                "equity": s.equity,
                "realised_pnl": s.realised_pnl,
                "fees_paid": s.fees_paid,
            }
        )
    return pd.DataFrame(rows)


def events_to_frame(events: list[StrategyEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": e.timestamp,
                "kind": e.kind,
                "detail": e.detail,
                "net_delta": e.net_delta,
                "equity": e.equity,
            }
            for e in events
        ]
    )


def fills_to_frame(fills) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": f.timestamp,
                "instrument": f.instrument,
                "quantity": f.quantity,
                "price": f.price,
                "fees": f.fees,
                "note": f.note,
            }
            for f in fills
        ]
    )


def daily_equity(bars: pd.DataFrame, starting_equity: float) -> pd.DataFrame:
    """End-of-day equity, P&L and return.

    Day one is measured against ``starting_equity`` rather than the first
    bar's equity, so a trade opened on the first bar is inside day one's
    return instead of being silently excluded from it.
    """
    if bars.empty:
        return pd.DataFrame(columns=["date", "equity", "pnl", "return"])
    frame = bars.copy()
    frame["date"] = frame["timestamp"].dt.date
    daily = frame.groupby("date", as_index=False)["equity"].last()
    previous = daily["equity"].shift(1)
    previous.iloc[0] = starting_equity
    daily["pnl"] = daily["equity"] - previous
    daily["return"] = daily["pnl"] / previous.replace(0.0, float("nan"))
    return daily


def _annualised(returns: pd.Series, downside_only: bool = False) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    mean = clean.mean()
    deviation = clean[clean < 0] if downside_only else clean
    std = deviation.std(ddof=1)
    if not std or not math.isfinite(std) or std == 0:
        return 0.0
    return float(mean / std * math.sqrt(TRADING_DAYS))


def compute_metrics(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    fills: pd.DataFrame,
    daily: pd.DataFrame,
    starting_equity: float,
    option_pnl: float,
    hedge_pnl: float,
    fees_paid: float,
    target: float,
    band_width: float = 0.0,
    hedge_tick: float = 0.25,
    hedge_quantum: float = 0.0,
) -> Metrics:
    if bars.empty:
        return Metrics(
            starting_equity, starting_equity, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0,
            0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    final_equity = float(bars["equity"].iloc[-1])
    peak = bars["equity"].cummax()
    drawdown = peak - bars["equity"]
    max_dd = float(drawdown.max())
    max_dd_pct = float((drawdown / peak).max()) if (peak > 0).all() else 0.0

    held = bars[bars["option_contracts"] != 0]
    hedge_fills = fills[fills["instrument"] == "hedge"] if not fills.empty else fills
    entries = int((events["kind"] == "entry").sum()) if not events.empty else 0
    hedge_events = int((events["kind"] == "hedge").sum()) if not events.empty else 0

    wins = int((daily["pnl"] > 0).sum()) if not daily.empty else 0
    losses = int((daily["pnl"] < 0).sum()) if not daily.empty else 0

    if held.empty:
        in_band = mean_err = max_err = mean_delta = 0.0
        median_gamma = band_points = below_tick = 0.0
        bars_in_band = bars_with_position = 0
    else:
        bars_with_position = len(held)
        bars_in_band = int(held["in_band"].sum())
        in_band = bars_in_band / bars_with_position
        errors = held["delta_error"].abs()
        mean_err = float(errors.mean())
        max_err = float(errors.max())
        mean_delta = float(held["net_delta_units"].mean())

        # How far the underlying must move to cross the band, given gamma.
        # Bars with ~zero gamma (deep out of the money, late in the session)
        # would divide to infinity, so they are treated as "band reachable"
        # rather than skewing the statistic.
        gamma = held["gamma_units"].abs()
        median_gamma = float(gamma.median())
        band_points = (
            band_width / median_gamma if median_gamma > 1e-9 else float("inf")
        )
        movable = gamma[gamma > 1e-9]
        if movable.empty:
            below_tick = 0.0
        else:
            below_tick = float(
                ((band_width / movable) < hedge_tick).sum() / len(held)
            )

    return Metrics(
        starting_equity=starting_equity,
        final_equity=final_equity,
        total_return=final_equity / starting_equity - 1.0,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe=_annualised(daily["return"]) if not daily.empty else 0.0,
        sortino=_annualised(daily["return"], downside_only=True) if not daily.empty else 0.0,
        trading_days=len(daily),
        entries=entries,
        winning_days=wins,
        losing_days=losses,
        win_rate=wins / (wins + losses) if (wins + losses) else 0.0,
        option_pnl=option_pnl,
        hedge_pnl=hedge_pnl,
        fees_paid=fees_paid,
        hedges=hedge_events,
        hedge_contracts_traded=(
            int(hedge_fills["quantity"].abs().sum()) if not hedge_fills.empty else 0
        ),
        bars_in_band=bars_in_band,
        bars_with_position=bars_with_position,
        pct_bars_in_band=in_band,
        mean_abs_delta_error=mean_err,
        max_abs_delta_error=max_err,
        mean_net_delta=mean_delta,
        median_gamma_units=median_gamma,
        band_width_points=band_points,
        pct_bars_band_below_tick=below_tick,
        hedge_tick_points=hedge_tick,
        hedge_quantum=hedge_quantum,
        band_half_width=band_width / 2.0,
    )
