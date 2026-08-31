"""Historical simulation of the short-vol delta hedger."""

from .engine import run_backtest
from .results import BacktestResult, Metrics

__all__ = ["run_backtest", "BacktestResult", "Metrics"]
