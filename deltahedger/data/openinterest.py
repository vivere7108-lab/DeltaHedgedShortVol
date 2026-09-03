"""Where open interest -- and therefore GEX -- comes from.

Three providers, all producing the same ``StrikeOpenInterest`` list so the
GEX calculator cannot tell them apart:

``SyntheticOpenInterest``
    Generates a plausible chain for the backtest, for any expiry it is
    asked about.  It is a test harness, not a market model -- it says the
    GEX machinery works, never that the GEX signal works.  Two properties
    it is built to have:

    * generated sessions span *both* regimes, so a backtest exercises the
      long and the short branch rather than only whichever one a fixed
      shape happens to produce;
    * neighbouring expiries lean the same way.  Real positioning is a
      property of the market, not of a date: if the public is put-heavy
      this week it is put-heavy across the whole front of the curve.
      Drawing each expiry's call share independently would make the
      front-expiry blend average four coin flips and read flat almost
      always -- an artefact of the generator that would look exactly like
      the confidence gate doing its job.  ``oi_call_share_smoothing_days``
      correlates adjacent expiries; set it to 0 for the old independent
      draws.

``CsvOpenInterest``
    Replays real open interest you already have, keyed by expiry date.

The IBKR provider lives in ``broker.ibkr`` with the rest of the live path.

A note on the anchor
--------------------
Real open interest sits where it was written and does not follow spot.  The
synthetic provider therefore fixes its strike distribution to the price
first observed for that expiry and keeps it there for the expiry's whole
life.  Re-centring it on spot each bar would look harmless and would
quietly destroy the whole exercise: the flip point would track spot, the
distance between them would never change sign, and no regime would ever
flip.

The *width* of the distribution is frozen at the same moment and for the
same reason.  A series listed several sessions out accumulates its open
interest while the market can still travel, so its strikes are spread wider
than a same-day series' -- but it does not re-spread as it ages, and a
generator that widened or narrowed an existing expiry's distribution over
time would be inventing open-interest flow rather than modelling it.
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
from ..session import trading_days_between

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
    """A generated open-interest surface, deterministic per expiry."""

    def __init__(self, cfg: DataConfig, source: RiskSource):
        self.cfg = cfg
        self.source = source
        self._anchors: dict[date, float] = {}
        self._spreads: dict[date, float] = {}

    def anchor(self, expiry: date, future_price: float) -> float:
        """The strike level this expiry's OI is built around.

        Set from the first price seen for that expiry and then frozen, on
        the listed strike grid.
        """
        if expiry not in self._anchors:
            step = self.source.strike_increment
            self._anchors[expiry] = round(future_price / step) * step
        return self._anchors[expiry]

    def spread(self, expiry: date, days_out: int) -> float:
        """How much wider this expiry's distribution is than a same-day one.

        ``sqrt(1 + DTE)`` at the moment the expiry is first seen, because a
        series listed several sessions out has had that much more room for
        the market to move while its open interest was written.  Frozen
        thereafter -- see the note on the anchor above.

        It scales the *whole* shape, the call and put centres included.  The
        width alone would be wrong: stretching the two humps while leaving
        their centres put makes them coincide, which turns every far expiry
        into a flat book by construction rather than by anything the
        generator was asked to say.
        """
        if expiry not in self._spreads:
            self._spreads[expiry] = math.sqrt(1.0 + max(days_out, 0))
        return self._spreads[expiry]

    def call_share(self, expiry: date) -> float:
        """Share of this expiry's open interest that is calls.

        This single number decides the regime: call-heavy chains give
        dealers long gamma (positive GEX), put-heavy chains give them short
        gamma (negative GEX).  It is drawn per expiry so a run covers both,
        and smoothed across neighbouring expiries so the front of the curve
        leans one way at a time rather than four independent ways at once.
        """
        window = max(int(self.cfg.oi_call_share_smoothing_days), 0)
        ordinal = expiry.toordinal()
        draws = [
            _unit_draw(self.cfg.oi_seed, ordinal + offset)
            for offset in range(-window, window + 1)
        ]
        draw = sum(draws) / len(draws)
        # Averaging n uniforms shrinks the spread by sqrt(n); undo that so
        # the swing parameter keeps meaning what it says.
        centred = (2.0 * draw - 1.0) * math.sqrt(len(draws))
        swing = self.cfg.oi_call_share_swing
        share = self.cfg.oi_call_share_mean + max(min(centred, 1.0), -1.0) * swing
        return min(max(share, 0.02), 0.98)

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        anchor = self.anchor(expiry, future_price)
        step = self.source.strike_increment
        spread = self.spread(expiry, trading_days_between(moment.date(), expiry))
        width = max(anchor * self.cfg.oi_width_pct * spread, step)
        share = self.call_share(expiry)

        call_center = anchor * (1.0 + self.cfg.oi_call_center_pct * spread)
        put_center = anchor * (1.0 + self.cfg.oi_put_center_pct * spread)

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
