"""Historical and live market data sources."""

from .base import DataSource, MarketBar, ensure_sorted
from .csv_source import CsvSource
from .synthetic import SyntheticSource, bar_seconds

__all__ = [
    "DataSource",
    "MarketBar",
    "ensure_sorted",
    "CsvSource",
    "SyntheticSource",
    "bar_seconds",
    "build_source",
]


def build_source(cfg, source):
    """Construct the data source named by ``cfg.data.source``."""
    kind = cfg.data.source.lower()
    if kind == "synthetic":
        return SyntheticSource(cfg.data, source)
    if kind == "csv":
        return CsvSource(cfg.data, source)
    if kind == "ibkr":
        from .ibkr_history import IbkrHistorySource  # optional ib_async dependency

        return IbkrHistorySource(cfg, source)
    raise ValueError(
        f"unknown data source {cfg.data.source!r}; use 'ibkr', 'csv' or 'synthetic'"
    )
