"""Where open interest -- and therefore GEX -- comes from.

Three providers, all producing the same ``StrikeOpenInterest`` list so the
GEX calculator cannot tell them apart:

``SyntheticOpenInterest``
    Generates a plausible 0DTE chain for the backtest.  It is a test
    harness, not a market model -- it says the GEX machinery works, never
    that the GEX signal works.  The one property it is built to have is that
    generated sessions span *both* regimes, so a backtest exercises the long
    and the short branch rather than only whichever one a fixed shape
    happens to produce.

``CsvOpenInterest``
    Replays real open interest you already have, keyed by expiry date.

The IBKR provider lives in ``broker.ibkr`` with the rest of the live path.

A note on the anchor
--------------------
Real open interest sits where it was written and does not follow spot.  The
synthetic provider therefore fixes its strike distribution to the session's
first observed price and keeps it there for the day.  Re-centring it on spot
each bar would look harmless and would quietly destroy the whole exercise:
the flip point would track spot, the distance between them would never
change sign, and no regime would ever flip.
"""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import date, datetime
from pathlib import Path

from ..config import DataConfig
from ..gex import StrikeOpenInterest
from ..instruments import RiskSource

log = logging.getLogger(__name__)


def _unit_draw(*parts: object) -> float:
    """A deterministic pseudo-random number in [0, 1) from its arguments.

    Hashing rather than an RNG instance so the value for a given session is
    the same however the backtest is sliced -- a run windowed to one week
    must produce the same chain for those days as the full run does.
    """
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / float(1 << 64)


class SyntheticOpenInterest:
    """A generated 0DTE open-interest surface, deterministic per expiry."""

    def __init__(self, cfg: DataConfig, source: RiskSource):
        self.cfg = cfg
        self.source = source
        self._anchors: dict[date, float] = {}

    def anchor(self, expiry: date, future_price: float) -> float:
        """The strike level this expiry's OI is built around.

        Set from the first price seen for that expiry and then frozen, on
        the listed strike grid.
        """
        if expiry not in self._anchors:
            step = self.source.strike_increment
            self._anchors[expiry] = round(future_price / step) * step
        return self._anchors[expiry]

    def call_share(self, expiry: date) -> float:
        """Share of the day's open interest that is calls.

        This single number decides the regime: call-heavy chains give
        dealers long gamma (positive GEX), put-heavy chains give them short
        gamma (negative GEX).  It is drawn per expiry so a run covers both.
        """
        swing = self.cfg.oi_call_share_swing
        draw = _unit_draw(self.cfg.oi_seed, expiry.toordinal())
        share = self.cfg.oi_call_share_mean + (2.0 * draw - 1.0) * swing
        return min(max(share, 0.02), 0.98)

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        anchor = self.anchor(expiry, future_price)
        step = self.source.strike_increment
        width = max(anchor * self.cfg.oi_width_pct, step)
        share = self.call_share(expiry)

        call_center = anchor * (1.0 + self.cfg.oi_call_center_pct)
        put_center = anchor * (1.0 + self.cfg.oi_put_center_pct)

        # Cover +/- 4 standard deviations of the wider of the two humps,
        # which is well beyond where either carries usable gamma.
        span = 4.0 * width + abs(call_center - put_center)
        count = int(span / step)
        strikes = [round(anchor + i * step, 10) for i in range(-count, count + 1)]

        call_mass = [math.exp(-0.5 * ((k - call_center) / width) ** 2) for k in strikes]
        put_mass = [math.exp(-0.5 * ((k - put_center) / width) ** 2) for k in strikes]
        call_sum = sum(call_mass) or 1.0
        put_sum = sum(put_mass) or 1.0

        total = self.cfg.oi_total_contracts
        return [
            StrikeOpenInterest(
                strike=k,
                call_oi=round(total * share * cm / call_sum),
                put_oi=round(total * (1.0 - share) * pm / put_sum),
            )
            for k, cm, pm in zip(strikes, call_mass, put_mass)
        ]


class CsvOpenInterest:
    """Replays open interest from a CSV of ``date,strike,call_oi,put_oi``.

    ``date`` is the *expiry* the row belongs to.  A day with no rows yields
    an empty chain, which the calculator reports as "no open interest" and
    the strategy treats as a reason to stand aside -- rather than silently
    falling back to a generated surface, which would make a real-data run
    quietly part synthetic.
    """

    REQUIRED = ("date", "strike", "call_oi", "put_oi")

    def __init__(self, cfg: DataConfig, source: RiskSource):
        if not cfg.oi_csv_path:
            raise ValueError(
                "data.oi_csv_path must be set when data.open_interest == 'csv'"
            )
        self.path = Path(cfg.oi_csv_path)
        self.source = source
        self._by_expiry: dict[date, list[StrikeOpenInterest]] | None = None

    def _load(self) -> dict[date, list[StrikeOpenInterest]]:
        import pandas as pd

        if not self.path.exists():
            raise FileNotFoundError(f"no open-interest CSV at {self.path}")
        frame = pd.read_csv(self.path)
        frame.columns = [c.strip().lower() for c in frame.columns]
        missing = [c for c in self.REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{self.path} is missing column(s): {', '.join(missing)}; "
                f"found {', '.join(frame.columns)}"
            )
        frame["date"] = pd.to_datetime(frame["date"]).dt.date

        rows: dict[date, list[StrikeOpenInterest]] = {}
        for record in frame.itertuples(index=False):
            rows.setdefault(record.date, []).append(
                StrikeOpenInterest(
                    strike=float(record.strike),
                    call_oi=float(record.call_oi),
                    put_oi=float(record.put_oi),
                )
            )
        return rows

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        if self._by_expiry is None:
            self._by_expiry = self._load()
        rows = self._by_expiry.get(expiry)
        if rows is None:
            log.debug("no open interest in %s for expiry %s", self.path, expiry)
            return []
        return rows


def build_open_interest_provider(cfg, source: RiskSource):
    """Construct the provider named by ``cfg.data.open_interest``.

    ``ibkr`` is not constructible here -- it needs a live connection -- so
    the live runner builds it and the backtest refuses it loudly rather than
    substituting generated data for the real thing.
    """
    kind = cfg.data.open_interest.lower()
    if kind == "synthetic":
        return SyntheticOpenInterest(cfg.data, source)
    if kind == "csv":
        return CsvOpenInterest(cfg.data, source)
    if kind == "ibkr":
        raise ValueError(
            "data.open_interest == 'ibkr' needs a live IBKR connection; it is "
            "available in `deltahedger live`, not in a backtest. Use 'csv' to "
            "replay real open interest historically."
        )
    raise ValueError(
        f"unknown open-interest source {cfg.data.open_interest!r}; use "
        "'synthetic', 'csv' or 'ibkr'"
    )
