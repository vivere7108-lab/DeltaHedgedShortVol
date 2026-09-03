"""Backtest metrics and reporting.

Three things get measured here that a generic backtest report would not
bother with, because they are the whole question for this strategy:

  * **regime attribution** -- what the long-gamma (negative GEX) trades made
    versus the short-gamma (positive GEX) ones.  A headline number that nets
    a working branch against a broken one says nothing; these two say
    whether reading GEX paid, and on which side.
  * **P&L attribution by leg** -- what the straddle did versus what hedging
    it did.  For a long straddle those two numbers *are* the strategy: the
    option leg is premium bled to theta, the hedge leg is the gamma scalped
    back, and the sum is the bet on realised vol.
  * **hedge quality** -- what fraction of bars sat inside the delta band, and
    how far from target the position actually ran.  With MES granularity
    coarser than the band, "in band" is an outcome, not a given.
  * **gate attribution** -- how many would-be entries and would-be flip
    exits each gate blocked.  Four gates that each look reasonable can
    between them leave a strategy that never trades, and a headline of zero
    trades does not say which one did it.  ``deltahedger sweep --gates``
    prices them one at a time against this count.

Band feasibility is reported per *branch* as well as in aggregate.  At a
multi-session tenor the two sides no longer carry comparable gamma: the
short branch is sized by SPAN margin, which barely moves with tenor, while
the long branch is sized by the debit, which roughly quadruples between
0DTE and 3DTE.  The same +/-10 band therefore covers a different number of
ES points on each side, and a regime comparison that did not know that
would be reading the sizing rule as if it were the signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..gex import NEGATIVE, NEUTRAL, POSITIVE
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
    # Regime attribution: position P&L (straddle + its hedge, net of fees)
    # booked against the GEX regime that opened the trade.
    long_gamma_pnl: float
    short_gamma_pnl: float
    long_gamma_trades: int
    short_gamma_trades: int
    neutral_skips: int
    pct_bars_negative_gex: float
    pct_bars_positive_gex: float
    pct_bars_neutral_gex: float
    pct_bars_above_flip: float
    mean_abs_gex: float
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
    # Per branch, because the two are sized by different constraints.
    median_gamma_units_long: float = 0.0
    median_gamma_units_short: float = 0.0
    band_width_points_long: float = 0.0
    band_width_points_short: float = 0.0
    # Tenor actually traded, in trading days, over bars holding a position.
    median_days_to_expiry: float = 0.0
    # Gate attribution: blocked entries and blocked flip exits by gate.
    gate_blocks: dict[str, int] = field(default_factory=dict)
    gate_blocked_exits: dict[str, int] = field(default_factory=dict)

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
                "Attribution by leg",
                f"  straddle P&L         {usd(self.option_pnl)}",
                f"  hedge P&L            {usd(self.hedge_pnl)}",
                f"  fees & commissions   {usd(-self.fees_paid)}",
                "",
                "GEX regime",
                f"  bars negative / positive / neutral  "
                f"{pct(self.pct_bars_negative_gex)} / "
                f"{pct(self.pct_bars_positive_gex)} / "
                f"{pct(self.pct_bars_neutral_gex)}",
                f"  bars above the gamma flip          {pct(self.pct_bars_above_flip)}",
                f"  mean |GEX|                         "
                f"${self.mean_abs_gex / 1e6:,.1f}M per 1% move",
                f"  long-gamma  (negative GEX) {self.long_gamma_trades:>3} trades"
                f"  {usd(self.long_gamma_pnl)}",
                f"  short-gamma (positive GEX) {self.short_gamma_trades:>3} trades"
                f"  {usd(self.short_gamma_pnl)}",
                f"  entries skipped as neutral {self.neutral_skips:>3}",
                self._regime_note(),
                "",
                "Gates",
                self._gate_lines(),
                "",
                "Trading",
                f"  entries              {self.entries}",
                f"  median tenor traded  {self.median_days_to_expiry:,.1f} DTE",
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
                f"    long-gamma branch   {self.band_width_points_long:,.3f} points"
                f"  (gamma {self.median_gamma_units_long:,.1f})",
                f"    short-gamma branch  {self.band_width_points_short:,.3f} points"
                f"  (gamma {self.median_gamma_units_short:,.1f})",
                f"  band finer than a tick on {pct(self.pct_bars_band_below_tick)} of held bars",
                f"  hedge quantum         {self.hedge_quantum:,.0f} delta units"
                f" per contract",
                self._granularity_note(),
                self._feasibility_note(),
                self._branch_note(),
            ]
        )

    def _gate_lines(self) -> str:
        """Which gate stopped what, or a plain statement that none did."""
        if not self.gate_blocks and not self.gate_blocked_exits:
            return "  no entry or flip was blocked by a gate"
        names = sorted(set(self.gate_blocks) | set(self.gate_blocked_exits))
        width = max(len(name) for name in names)
        lines = [f"  {'gate':<{width}}  {'entries blocked':>15}  {'flips held':>10}"]
        for name in names:
            lines.append(
                f"  {name:<{width}}  {self.gate_blocks.get(name, 0):>15}"
                f"  {self.gate_blocked_exits.get(name, 0):>10}"
            )
        return "\n".join(lines)

    def _branch_note(self) -> str:
        """Whether the two branches were held to comparable exposure.

        The band is one number, but what it costs to hold depends on the
        gamma underneath it, and the two branches are sized by different
        constraints. If they differ by much, a difference in their P&L is
        partly a difference in position size rather than in signal quality.
        """
        wide, narrow = (
            max(self.median_gamma_units_long, self.median_gamma_units_short),
            min(self.median_gamma_units_long, self.median_gamma_units_short),
        )
        if narrow <= 1e-9:
            return "  (only one branch held a position; no comparison to make)"
        ratio = wide / narrow
        if ratio < 1.5:
            return (
                f"  -> the two branches carried comparable gamma ({ratio:.1f}x "
                "apart), so\n     the regime comparison is about the signal."
            )
        heavier = (
            "long-gamma"
            if self.median_gamma_units_long > self.median_gamma_units_short
            else "short-gamma"
        )
        return (
            f"  -> the {heavier} branch carried {ratio:.1f}x the gamma of the "
            "other, because\n     the two are sized by different constraints "
            "(debit vs SPAN margin). Some\n     of any P&L difference between "
            "them is position size, not signal."
        )

    def _regime_note(self) -> str:
        """Say plainly whether the two branches were both exercised.

        A run that only ever saw one regime has not tested the strategy --
        it has tested half of it -- and the headline return would invite
        exactly the wrong conclusion.
        """
        if self.long_gamma_trades == 0 and self.short_gamma_trades == 0:
            return "  -> no trade was taken: every bar read as neutral."
        if self.long_gamma_trades == 0:
            return (
                "  -> only the SHORT-gamma branch traded. The long side is\n"
                "     untested here; do not read the headline as evidence for it."
            )
        if self.short_gamma_trades == 0:
            return (
                "  -> only the LONG-gamma branch traded. The short side is\n"
                "     untested here; do not read the headline as evidence for it."
            )
        return (
            "  -> both branches traded; the split above is the result that\n"
            "     matters, not the net."
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
        rows.append(
            {
                "timestamp": s.timestamp,
                "future": s.future,
                "atm_iv": s.atm_iv,
                "hours_to_expiry": s.time_to_expiry * 365 * 24,
                "gex_total": s.gex_total,
                "gex_flip": s.gex_flip,
                "gex_regime": s.gex_regime,
                "distance_to_flip": s.distance_to_flip,
                "gex_confidence": s.gex_confidence,
                "gex_gate": s.gex_gate,
                "confirmed_regime": s.confirmed_regime,
                "days_to_expiry": s.days_to_expiry,
                "in_session": s.in_session,
                "strike": s.strike,
                "straddle_contracts": s.straddle_contracts,
                "direction": s.direction,
                "straddle_mark": s.straddle_mark,
                "call_mark": s.call_mark,
                "put_mark": s.put_mark,
                "option_delta_units": s.option_delta_units,
                "hedge_contracts": s.hedge_contracts,
                "hedge_delta_units": s.hedge_delta_units,
                "net_delta_units": s.net_delta_units,
                "delta_error": s.net_delta_units - target,
                "gamma_units": s.gamma_units,
                "vega_dollars": s.vega_dollars,
                "theta_dollars": s.theta_dollars,
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
                "regime": e.regime,
                "gate": e.gate,
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
    regime_pnl: dict[str, float] | None = None,
    regime_trades: dict[str, int] | None = None,
    band_width: float = 0.0,
    hedge_tick: float = 0.25,
    hedge_quantum: float = 0.0,
) -> Metrics:
    regime_pnl = regime_pnl or {}
    regime_trades = regime_trades or {}
    if bars.empty:
        return Metrics(
            starting_equity=starting_equity, final_equity=starting_equity,
            total_return=0.0, max_drawdown=0.0, max_drawdown_pct=0.0, sharpe=0.0,
            sortino=0.0, trading_days=0, entries=0, winning_days=0, losing_days=0,
            win_rate=0.0, option_pnl=0.0, hedge_pnl=0.0, fees_paid=0.0,
            long_gamma_pnl=0.0, short_gamma_pnl=0.0, long_gamma_trades=0,
            short_gamma_trades=0, neutral_skips=0, pct_bars_negative_gex=0.0,
            pct_bars_positive_gex=0.0, pct_bars_neutral_gex=0.0,
            pct_bars_above_flip=0.0, mean_abs_gex=0.0, hedges=0,
            hedge_contracts_traded=0, bars_in_band=0, bars_with_position=0,
            pct_bars_in_band=0.0, mean_abs_delta_error=0.0, max_abs_delta_error=0.0,
            mean_net_delta=0.0, median_gamma_units=0.0, band_width_points=0.0,
            pct_bars_band_below_tick=0.0, hedge_tick_points=hedge_tick,
            hedge_quantum=hedge_quantum, band_half_width=band_width / 2.0,
        )

    final_equity = float(bars["equity"].iloc[-1])
    peak = bars["equity"].cummax()
    drawdown = peak - bars["equity"]
    max_dd = float(drawdown.max())
    max_dd_pct = float((drawdown / peak).max()) if (peak > 0).all() else 0.0

    held = bars[bars["straddle_contracts"] != 0]
    hedge_fills = fills[fills["instrument"] == "hedge"] if not fills.empty else fills
    entries = int((events["kind"] == "entry").sum()) if not events.empty else 0
    hedge_events = int((events["kind"] == "hedge").sum()) if not events.empty else 0

    wins = int((daily["pnl"] > 0).sum()) if not daily.empty else 0
    losses = int((daily["pnl"] < 0).sum()) if not daily.empty else 0

    # -- GEX regime -----------------------------------------------------
    # A bar with no profile (no 0DTE listed, or no open interest) is not a
    # neutral read; it is an absent one, and counting it as neutral would
    # overstate how often dealers were flat.
    scored = bars[bars["gex_total"].notna()]
    total_scored = len(scored)
    share = lambda name: (  # noqa: E731
        float((scored["gex_regime"] == name).sum() / total_scored)
        if total_scored
        else 0.0
    )
    with_flip = scored[scored["gex_flip"].notna()]
    pct_above_flip = (
        float((with_flip["future"] > with_flip["gex_flip"]).sum() / len(with_flip))
        if len(with_flip)
        else 0.0
    )
    mean_abs_gex = float(scored["gex_total"].abs().mean()) if total_scored else 0.0
    neutral_skips = (
        int(((events["kind"] == "entry_skipped") & (events["regime"] == NEUTRAL)).sum())
        if not events.empty
        else 0
    )
    gate_blocks = _gate_counts(events, "entry_skipped")
    gate_blocked_exits = _gate_counts(events, "exit_deferred")

    branch_gamma = {1: 0.0, -1: 0.0}
    branch_points = {1: 0.0, -1: 0.0}
    median_dte = 0.0
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

        # The same arithmetic per branch. They are sized by different
        # constraints -- a debit on the long side, SPAN margin on the short
        # one -- so at a multi-session tenor their gamma loads diverge and
        # one band means two different numbers of ES points.
        for direction in (1, -1):
            side = held[held["direction"] == direction]["gamma_units"].abs()
            if side.empty:
                continue
            value = float(side.median())
            branch_gamma[direction] = value
            branch_points[direction] = (
                band_width / value if value > 1e-9 else float("inf")
            )
        if "days_to_expiry" in held and held["days_to_expiry"].notna().any():
            median_dte = float(held["days_to_expiry"].dropna().median())

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
        long_gamma_pnl=regime_pnl.get(NEGATIVE, 0.0),
        short_gamma_pnl=regime_pnl.get(POSITIVE, 0.0),
        long_gamma_trades=regime_trades.get(NEGATIVE, 0),
        short_gamma_trades=regime_trades.get(POSITIVE, 0),
        neutral_skips=neutral_skips,
        pct_bars_negative_gex=share(NEGATIVE),
        pct_bars_positive_gex=share(POSITIVE),
        pct_bars_neutral_gex=share(NEUTRAL),
        pct_bars_above_flip=pct_above_flip,
        mean_abs_gex=mean_abs_gex,
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
        median_gamma_units_long=branch_gamma[1],
        median_gamma_units_short=branch_gamma[-1],
        band_width_points_long=branch_points[1],
        band_width_points_short=branch_points[-1],
        median_days_to_expiry=median_dte,
        gate_blocks=gate_blocks,
        gate_blocked_exits=gate_blocked_exits,
    )


def _gate_counts(events: pd.DataFrame, kind: str) -> dict[str, int]:
    """How many events of ``kind`` each named gate is responsible for.

    Blocks with no gate attached -- no expiry listed, no open interest, no
    buying power -- are counted under "ungated" rather than dropped, so the
    totals here add up to the events actually recorded.
    """
    if events.empty or "gate" not in events or "kind" not in events:
        return {}
    rows = events[events["kind"] == kind]
    if rows.empty:
        return {}
    named = rows["gate"].fillna("").replace("", "ungated")
    return {str(name): int(count) for name, count in named.value_counts().items()}
