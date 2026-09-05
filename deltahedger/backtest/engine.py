"""Backtest driver.

Walks the bar stream, hands each bar to ``GexStraddleStrategy`` with a
simulated execution handler and an open-interest provider, and turns what
comes out into a
``BacktestResult``.  All the decision logic lives in the strategy; this file
is a loop and a reporting step, which is deliberate -- it is what makes the
live runner a drop-in replacement rather than a parallel implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from ..broker.paper import SimulatedExecution
from ..config import Config
from ..data import build_open_interest_provider, build_source
from ..data.base import DataSource, ensure_sorted
from ..gex import OpenInterestProvider
from ..strategy import GexStraddleStrategy
from .results import (
    BacktestResult,
    bars_to_frame,
    compute_metrics,
    daily_equity,
    events_to_frame,
    fills_to_frame,
)

log = logging.getLogger(__name__)


def _within_window(cfg: Config, moment: datetime, tz: ZoneInfo) -> bool:
    if cfg.start_date:
        start = datetime.fromisoformat(cfg.start_date).replace(tzinfo=tz)
        if moment < start:
            return False
    if cfg.end_date:
        end = datetime.fromisoformat(cfg.end_date).replace(tzinfo=tz)
        if moment > end:
            return False
    return True


def run_backtest(
    cfg: Config,
    source: DataSource | None = None,
    open_interest: OpenInterestProvider | None = None,
) -> BacktestResult:
    risk_source = cfg.source
    data = source if source is not None else build_source(cfg, risk_source)
    oi = (
        open_interest
        if open_interest is not None
        else build_open_interest_provider(cfg, risk_source)
    )
    strategy = GexStraddleStrategy(cfg, risk_source, open_interest=oi)
    execution = SimulatedExecution(cfg.costs, risk_source)
    tz = strategy.clock.tz

    processed = 0
    for bar in ensure_sorted(data.bars()):
        moment = strategy.clock.localize(bar.timestamp)
        if not _within_window(cfg, moment, tz):
            continue
        strategy.on_bar(bar, execution)
        processed += 1

    if processed == 0:
        log.warning(
            "no bars fell inside the configured window "
            "(start_date=%s, end_date=%s)", cfg.start_date, cfg.end_date,
        )

    bars = bars_to_frame(strategy.bar_states, cfg.hedge.target)
    events = events_to_frame(strategy.events)
    fills = fills_to_frame(strategy.fills)
    daily = daily_equity(bars, cfg.starting_equity)
    metrics = compute_metrics(
        bars=bars,
        events=events,
        fills=fills,
        daily=daily,
        starting_equity=cfg.starting_equity,
        option_pnl=strategy.portfolio.option_realised,
        hedge_pnl=strategy.portfolio.hedge_realised,
        fees_paid=strategy.portfolio.fees_paid,
        regime_pnl=strategy.regime_pnl,
        regime_trades=strategy.regime_trades,
        target=cfg.hedge.target,
        band_model=cfg.hedge.band_model,
        risk_aversion=cfg.hedge.risk_aversion,
        fixed_band=cfg.hedge.band,
        hedge_tick=risk_source.hedge.tick_size,
        hedge_quantum=risk_source.hedge_quantum,
    )
    return BacktestResult(
        metrics=metrics,
        bars=bars,
        events=events,
        fills=fills,
        daily=daily,
        config=cfg.to_dict(),
    )
