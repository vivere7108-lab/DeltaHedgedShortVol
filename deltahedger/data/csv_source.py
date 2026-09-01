"""CSV bar source.

For replaying data you already have.  Expected columns (case-insensitive):

    timestamp, open, high, low, close [, atm_iv] [, volume]

``timestamp`` may be tz-aware ISO-8601, or naive -- naive timestamps are
interpreted as exchange-local time for the risk source, which is what
exports from most charting packages contain.  When ``atm_iv`` is absent,
``DataConfig.default_atm_iv`` is used for every bar and the backtest becomes
a constant-vol study; that is a real limitation, not a detail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from ..config import DataConfig
from ..instruments import RiskSource
from .base import MarketBar

REQUIRED = ("timestamp", "open", "high", "low", "close")


class CsvSource:
    def __init__(self, cfg: DataConfig, source: RiskSource):
        if not cfg.csv_path:
            raise ValueError("data.csv_path must be set when data.source == 'csv'")
        self.path = Path(cfg.csv_path)
        self.cfg = cfg
        self.tz = ZoneInfo(source.timezone)

    def bars(self) -> Iterator[MarketBar]:
        if not self.path.exists():
            raise FileNotFoundError(f"no CSV at {self.path}")
        frame = pd.read_csv(self.path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{self.path} is missing column(s): {', '.join(missing)}; "
                f"found {', '.join(frame.columns)}"
            )

        stamps = pd.to_datetime(frame["timestamp"], format="mixed", utc=False)
        if stamps.dt.tz is None:
            stamps = stamps.dt.tz_localize(self.tz)
        else:
            stamps = stamps.dt.tz_convert(self.tz)
        frame = frame.assign(timestamp=stamps).sort_values("timestamp")

        has_iv = "atm_iv" in frame.columns
        has_volume = "volume" in frame.columns
        for row in frame.itertuples(index=False):
            iv = float(getattr(row, "atm_iv")) if has_iv else self.cfg.default_atm_iv
            if not iv or iv != iv or iv <= 0:  # blank or NaN
                iv = self.cfg.default_atm_iv
            yield MarketBar(
                timestamp=row.timestamp.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                atm_iv=iv,
                volume=float(getattr(row, "volume")) if has_volume else 0.0,
            )
