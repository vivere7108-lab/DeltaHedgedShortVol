"""Dealer gamma exposure: the flip point, and the regime it implies.

What this computes
------------------
GEX is an estimate of the gamma the option dealer community is carrying,
inferred from open interest.  The standard assumption -- and the one every
published GEX print uses -- is that the public buys puts and sells calls, so
the dealer is *long the calls and short the puts*::

    gex(K) = mult * S^2 * 0.01 * gamma(K) * (call_sign*OI_call + put_sign*OI_put)

The ``S^2 * 0.01`` turns per-point gamma into dollars of delta the dealer
must trade for a 1% move, which is the unit the number is quoted in.

Why it matters is entirely mechanical.  A dealer who is **short gamma**
(negative GEX) has to sell as the market falls and buy as it rises: their
hedging *adds* to the move.  A dealer who is **long gamma** (positive GEX)
does the opposite and damps it.  So the sign of GEX is a statement about
whether hedging flow will amplify or suppress realised volatility -- which
is exactly the variable a delta-hedged straddle is a bet on.

The **gamma flip point** is the spot level at which total GEX crosses zero.
It is found by repricing the whole chain's gamma across a grid of
hypothetical spot levels, holding open interest fixed, and interpolating the
crossing.  Above it dealers are long gamma, below it they are short.

One book, several expiries
--------------------------
The regime is read off the **aggregate of the front expiries**, from the
one expiring today out to the one being traded, rather than off the traded
series alone.  A dealer does not hedge a series; they hedge a book, and
their delta is the sum over everything they are carrying.  At a 3-4 DTE
tenor the traded series is a minority of the gamma sitting in front of
them, so classifying on it alone would be reading a corner of the book and
calling it the book.

The blend needs no weights.  GEX is already gamma-weighted by
construction, and gamma per contract scales roughly as ``1/sqrt(T)``, so a
near-dated expiry contributes more than a far one *because it does* -- not
because a coefficient says so.  Summing the per-expiry contributions is the
whole of it.  What the sum is sensitive to is the ``min_hours_to_expiry``
floor, which stops the last hour of the 0DTE leg from swamping everything
else with a gamma that is about to stop existing.

The greeks the hedger acts on are **not** blended and never were: they come
from the traded straddle alone, marked at its own tenor.  This is the same
separation ``min_hours_to_expiry`` already draws -- what the profile is for
is classification, and what the position is for is exposure.

Standing aside
--------------
``GatesConfig`` describes four reasons to decline a read.  Two of them live
here, because they are properties of the profile rather than of the
strategy: the **confidence ratio** ``|total|/gross`` and the **distance to
the flip**.  A third, the **ensemble**, is computed here too --
``GexCalculator.ensemble`` reprices the regime over a grid of skew and
sign-convention perturbations -- but it is invoked by the strategy only
when a decision actually turns on it, because it costs a full profile per
member.  Persistence and the entry window are the strategy's, not the
calculator's.

What the strategy does with it
------------------------------
=================  ==================  ==============  ===================
GEX                dealer hedging      realised vol    the position
=================  ==================  ==============  ===================
negative           amplifies moves     runs above IV   LONG the straddle
positive           damps moves         runs below IV   SHORT the straddle
near zero / flip   about to change     unknown         stand aside
=================  ==================  ==============  ===================

Honest limits
-------------
1. **Open interest is not positioning.**  Who is long and who is short is
   not in the OI print; the call/put sign convention is an assumption, and
   it is the load-bearing one.  ``call_sign``/``put_sign`` are config so it
   can be varied rather than believed, and the ensemble gate turns that
   variation into a trading rule.
2. **OI is stale intraday.**  Exchange open interest is an end-of-previous-
   day figure.  This bites least on the series the blend weights most
   heavily by gamma and most on the ones that have been listed longest --
   which is the wrong way round, and is the single largest approximation
   in the GEX layer.  The entry window exists to at least ensure the
   figure being used is the exchange's *final* print rather than its
   preliminary one.
3. **Expiring gamma is a spike.**  As expiry approaches, gamma concentrates
   at the money and vanishes elsewhere, so the 0DTE leg of the blend
   becomes dominated by two or three strikes and the flip point gets noisy.
   ``min_hours_to_expiry`` floors the tenor used for classification so the
   shape stays legible; it never touches the greeks the hedger acts on.
4. **The flip point moves with vol.**  It is computed off the same modelled
   surface used to price the book, so an error in the skew moves the flip
   point as well as the credit.  The ensemble gate measures how much.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import NamedTuple, Protocol, Sequence

import numpy as np

from .config import GatesConfig, GexConfig
from .instruments import RiskSource
from .pricing import black76_gamma
from .volsurface import VolSurface

log = logging.getLogger(__name__)

#: The three regimes. Strings rather than an enum so they land in a CSV and
#: a log line unchanged.
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"

#: What each regime says to trade. The sign is the straddle's quantity sign.
LONG_STRADDLE = 1
SHORT_STRADDLE = -1
STAND_ASIDE = 0

#: Gate names, as they appear in the event log and the journal. Kept as
#: constants so the report, the sweep and the tests cannot drift from what
#: the strategy actually writes down.
GATE_CONFIDENCE = "confidence"
GATE_FLIP_DISTANCE = "flip_distance"
GATE_ENSEMBLE = "ensemble"
GATE_PERSISTENCE = "persistence"
GATE_ENTRY_WINDOW = "entry_window"
#: The nowcast veto/exit (config.NowcastConfig): a fifth gate, but one that
#: needs a Databento subscription and is off by default, so it is kept out
#: of GATE_NAMES -- the tuple every gate-sweep and default-config test
#: assumes is "the four gates that ship enabled".
GATE_NOWCAST = "nowcast"
GATE_NAMES = (
    GATE_CONFIDENCE,
    GATE_FLIP_DISTANCE,
    GATE_ENSEMBLE,
    GATE_PERSISTENCE,
    GATE_ENTRY_WINDOW,
)

HOURS_PER_YEAR = 365.0 * 24.0


@dataclass(frozen=True)
class StrikeOpenInterest:
    """Open interest at one strike of one expiry."""

    strike: float
    call_oi: float
    put_oi: float


@dataclass(frozen=True)
class ExpiryBook:
    """One expiry's open interest, with the tenor to price it at.

    The unit the blend is built from.  ``time_to_expiry`` is the real
    wall-clock tenor of that series; the ``min_hours_to_expiry`` floor is
    applied by the calculator rather than baked in here, so a caller cannot
    accidentally hand the hedger a floored tenor.
    """

    expiry: date
    time_to_expiry: float
    rows: tuple[StrikeOpenInterest, ...]
    days_to_expiry: int = 0

    @classmethod
    def of(
        cls,
        expiry: date,
        time_to_expiry: float,
        rows: Sequence[StrikeOpenInterest],
        days_to_expiry: int = 0,
    ) -> "ExpiryBook":
        return cls(expiry, time_to_expiry, tuple(rows), days_to_expiry)


class OpenInterestProvider(Protocol):
    """Anything that can say what open interest sits on a chain.

    The backtest generates it, a CSV replays it, and the live path reads it
    from IBKR -- but ``GexCalculator`` sees the same list either way, which
    is what lets the forward test exercise the classification code that was
    measured historically.
    """

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]: ...


@dataclass(frozen=True)
class StrikeFlow:
    """Aggressor-signed contract volume at one strike, since some reference
    time -- the flow analogue of ``StrikeOpenInterest``.  Unlike open
    interest these numbers are signed and are not a position count: see
    ``config.NowcastConfig`` and ``data/databento_flow.py`` for where they
    come from and what they mean.
    """

    strike: float
    call_signed_volume: float
    put_signed_volume: float


class NowcastProvider(Protocol):
    """Anything that can say what has traded at each strike since a moment.

    The live Databento feed and the historical replay source both
    implement this with the same signature, so ``GexCalculator.nowcast_profile``
    -- and the strategy above it -- cannot tell which one it is reading
    from, the same symmetry ``OpenInterestProvider`` already has.
    """

    def flow_since(
        self, moment: datetime, expiry: date, since: datetime
    ) -> tuple[StrikeFlow, ...]: ...


@dataclass(frozen=True)
class StrikeGex:
    """The dealer gamma one strike contributes, split by right.

    In a blended profile every field is the sum over the expiries in the
    blend, so ``gamma`` is per-contract gamma summed across tenors rather
    than any one series' gamma.  It is a display column; nothing decides
    anything on it.
    """

    strike: float
    call_oi: float
    put_oi: float
    gamma: float
    call_gex: float
    put_gex: float

    @property
    def net_gex(self) -> float:
        return self.call_gex + self.put_gex


@dataclass(frozen=True)
class ExpiryGex:
    """What one expiry contributed to the blend."""

    expiry: date
    days_to_expiry: int
    time_to_expiry: float
    total_gex: float
    gross_gex: float


@dataclass(frozen=True)
class EnsembleResult:
    """Whether the regime survives a plausible change of assumptions."""

    unanimous: bool
    regime: str
    regimes: tuple[str, ...]
    detail: str

    @property
    def members(self) -> int:
        return len(self.regimes)


@dataclass(frozen=True)
class GexProfile:
    """The whole picture at one spot level: the number, the flip, the call."""

    spot: float
    time_to_expiry: float
    total_gex: float
    #: Absolute gamma in the book, both rights summed unsigned. ``total_gex``
    #: measured against this is how *directional* dealer positioning is,
    #: which is what the confidence gate is written in terms of.
    gross_gex: float
    call_gex: float
    put_gex: float
    flip_point: float | None
    regime: str
    reason: str
    by_strike: tuple[StrikeGex, ...] = ()
    #: What each expiry in the blend contributed, nearest first. A single
    #: entry means the profile was read off one series.
    by_expiry: tuple[ExpiryGex, ...] = ()
    #: Which gate forced a NEUTRAL read, empty when the regime is a real
    #: one. This is what the journal records so a stand-aside can be
    #: attributed after the fact.
    gate: str = ""

    @property
    def direction(self) -> int:
        """The straddle quantity sign this profile implies.

        Negative GEX -> dealers amplify moves -> we want gamma -> long.
        Positive GEX -> dealers damp moves -> we want theta -> short.
        """
        if self.regime == NEGATIVE:
            return LONG_STRADDLE
        if self.regime == POSITIVE:
            return SHORT_STRADDLE
        return STAND_ASIDE

    @property
    def confidence(self) -> float:
        """``|total| / gross``: how directional dealer positioning is.

        Zero when the book's call and put gamma cancel exactly, one when it
        is all on one side.  This is the quantity the confidence gate
        thresholds, and it is scale-free -- a bigger book does not read as
        a more confident one.
        """
        if self.gross_gex <= 0.0:
            return 0.0
        return abs(self.total_gex) / self.gross_gex

    @property
    def above_flip(self) -> bool | None:
        if self.flip_point is None:
            return None
        return self.spot > self.flip_point

    @property
    def distance_to_flip(self) -> float | None:
        """Points from spot to the flip; positive means spot is above it."""
        if self.flip_point is None:
            return None
        return self.spot - self.flip_point

    @property
    def peak_strike(self) -> float | None:
        """The strike carrying the most absolute gamma -- the pin candidate."""
        if not self.by_strike:
            return None
        return max(self.by_strike, key=lambda s: abs(s.net_gex)).strike

    def describe(self) -> str:
        flip = f"{self.flip_point:,.1f}" if self.flip_point is not None else "none found"
        return (
            f"GEX {self.total_gex / 1e6:+,.1f}M/1% at {self.spot:,.2f}, "
            f"flip {flip}, confidence {self.confidence:.0%}, regime {self.regime}"
        )

    def table(self, limit: int = 15) -> str:
        """The strikes carrying the most gamma, for eyeballing a live read."""
        rows = sorted(self.by_strike, key=lambda s: -abs(s.net_gex))[:limit]
        rows.sort(key=lambda s: s.strike)
        lines = [f"{'strike':>9} {'call OI':>9} {'put OI':>9} {'net GEX ($M/1%)':>17}"]
        for row in rows:
            lines.append(
                f"{row.strike:>9,.0f} {row.call_oi:>9,.0f} {row.put_oi:>9,.0f} "
                f"{row.net_gex / 1e6:>17,.2f}"
            )
        return "\n".join(lines)

    def expiry_table(self) -> str:
        """What each expiry in the blend contributed, nearest first."""
        lines = [
            f"{'expiry':>12} {'DTE':>4} {'hours':>8} "
            f"{'net GEX ($M/1%)':>17} {'share of gross':>15}"
        ]
        gross = sum(row.gross_gex for row in self.by_expiry) or 1.0
        for row in self.by_expiry:
            lines.append(
                f"{row.expiry.isoformat():>12} {row.days_to_expiry:>4d} "
                f"{row.time_to_expiry * HOURS_PER_YEAR:>8.1f} "
                f"{row.total_gex / 1e6:>17,.2f} {row.gross_gex / gross:>15.1%}"
            )
        return "\n".join(lines)


def _apply_flow(
    rows: Sequence[StrikeOpenInterest],
    flow: Sequence[StrikeFlow],
    dealer_share: float,
) -> tuple[StrikeOpenInterest, ...]:
    """Base open interest, adjusted by dealer-share-scaled signed flow.

    Unioned by strike: a strike with flow but no listed open interest still
    gets a row (base zero), and a strike with open interest but no flow
    this session is passed through unchanged. The result can carry a
    negative call_oi/put_oi -- once flow is folded in these are no longer
    literal open interest counts, they are a signed dealer-inventory
    proxy, and the pricing pipeline downstream never assumed positivity.
    """
    by_strike: dict[float, list[float]] = {
        row.strike: [row.call_oi, row.put_oi] for row in rows
    }
    for row in flow:
        bucket = by_strike.setdefault(row.strike, [0.0, 0.0])
        bucket[0] += -dealer_share * row.call_signed_volume
        bucket[1] += -dealer_share * row.put_signed_volume
    return tuple(
        StrikeOpenInterest(strike=k, call_oi=v[0], put_oi=v[1])
        for k, v in sorted(by_strike.items())
    )


class PreparedBook(NamedTuple):
    """One expiry's open interest, windowed and arrayed, ready to price."""

    book: ExpiryBook
    strikes: np.ndarray
    calls: np.ndarray
    puts: np.ndarray
    tenor: float


class GexCalculator:
    """Turns open interest into a profile, a flip point and a regime."""

    def __init__(
        self,
        cfg: GexConfig,
        source: RiskSource,
        surface: VolSurface,
        risk_free_rate: float = 0.0,
        gates: GatesConfig | None = None,
    ):
        self.cfg = cfg
        self.source = source
        self.surface = surface
        self.risk_free_rate = risk_free_rate
        self.gates = gates if gates is not None else GatesConfig()

    # -- the profile -----------------------------------------------------

    def profile(
        self,
        spot: float,
        open_interest: Sequence[StrikeOpenInterest],
        time_to_expiry: float,
        atm_iv: float,
    ) -> GexProfile:
        """Total GEX at ``spot`` for a single expiry's open interest."""
        return self.blended_profile(
            spot,
            [ExpiryBook.of(date.min, time_to_expiry, open_interest)],
            atm_iv,
        )

    def blended_profile(
        self, spot: float, books: Sequence[ExpiryBook], atm_iv: float
    ) -> GexProfile:
        """Total GEX at ``spot`` across every expiry in ``books``.

        The books are summed, not averaged: each one contributes the dollars
        of delta that expiry forces dealers to trade for a 1% move, and what
        the strategy needs is the total across the book they are carrying.
        """
        prepared = self._prepare(spot, books)
        floored = [
            self._effective_tenor(book.time_to_expiry) for book in books
        ]
        blended_tenor = min(floored) if floored else 0.0

        if not prepared:
            return GexProfile(
                spot=spot, time_to_expiry=blended_tenor, total_gex=0.0,
                gross_gex=0.0, call_gex=0.0, put_gex=0.0, flip_point=None,
                regime=NEUTRAL, gate=GATE_CONFIDENCE,
                reason="no open interest inside the strike window",
            )

        scale = self.source.option.multiplier * spot * spot * 0.01
        per_strike: dict[float, list[float]] = {}
        by_expiry: list[ExpiryGex] = []
        total = gross = call_total = put_total = 0.0

        for book, strikes, calls, puts, tenor in prepared:
            gamma = black76_gamma(
                spot, strikes, tenor, self._vols(spot, strikes, atm_iv),
                self.risk_free_rate,
            )
            call_gex = scale * gamma * self.cfg.call_sign * calls
            put_gex = scale * gamma * self.cfg.put_sign * puts
            expiry_total = float((call_gex + put_gex).sum())
            # Gross is the gamma in the book, summed per *leg* rather than
            # per strike. Summing net-per-strike would collapse to zero for
            # a chain with matched call and put interest -- which is a
            # maximally gamma-laden book, not an empty one -- and the
            # confidence gate below divides by this.
            expiry_gross = float((np.abs(call_gex) + np.abs(put_gex)).sum())
            total += expiry_total
            gross += expiry_gross
            call_total += float(call_gex.sum())
            put_total += float(put_gex.sum())
            by_expiry.append(
                ExpiryGex(
                    expiry=book.expiry,
                    days_to_expiry=book.days_to_expiry,
                    time_to_expiry=tenor,
                    total_gex=expiry_total,
                    gross_gex=expiry_gross,
                )
            )
            for k, c, p, g, cg, pg in zip(strikes, calls, puts, gamma, call_gex, put_gex):
                row = per_strike.setdefault(float(k), [0.0, 0.0, 0.0, 0.0, 0.0])
                row[0] += float(c)
                row[1] += float(p)
                row[2] += float(g)
                row[3] += float(cg)
                row[4] += float(pg)

        flip = self._flip_point(spot, prepared, atm_iv)
        regime, reason, gate = self._classify(spot, total, gross, flip)

        return GexProfile(
            spot=spot,
            time_to_expiry=blended_tenor,
            total_gex=total,
            gross_gex=gross,
            call_gex=call_total,
            put_gex=put_total,
            flip_point=flip,
            regime=regime,
            reason=reason,
            gate=gate,
            by_strike=tuple(
                StrikeGex(
                    strike=strike, call_oi=row[0], put_oi=row[1], gamma=row[2],
                    call_gex=row[3], put_gex=row[4],
                )
                for strike, row in sorted(per_strike.items())
            ),
            by_expiry=tuple(by_expiry),
        )

    def nowcast_books(
        self,
        books: Sequence[ExpiryBook],
        flow_by_expiry: dict[date, Sequence[StrikeFlow]],
        dealer_share: float,
    ) -> list[ExpiryBook]:
        """``books``, with each strike's open interest corrected by signed
        flow since the print.

        A trade the aggressor bought (``+`` signed volume) is one the
        dealer sold, so it moves dealer inventory the *opposite* way::

            dealer_inventory_delta(strike, right) = -dealer_share * signed_volume(strike, right)

        That delta is added to the strike's base open interest -- for a
        strike with flow but no listed OI at all, the base is implicitly
        zero.  ``books`` is not mutated; a fresh list is returned, so a
        caller can safely reuse the same base ``books`` for both the
        entry-signal read and this one.

        Split out from ``nowcast_profile`` because the reconciliation check
        (``NowcastConfig.reconciliation_enabled``) wants the adjusted
        per-strike table itself -- what the flow implied the book looked
        like -- not the profile computed from it.
        """
        return [
            ExpiryBook.of(
                book.expiry,
                book.time_to_expiry,
                _apply_flow(
                    book.rows, flow_by_expiry.get(book.expiry, ()), dealer_share
                ),
                book.days_to_expiry,
            )
            for book in books
        ]

    def nowcast_profile(
        self,
        spot: float,
        books: Sequence[ExpiryBook],
        flow_by_expiry: dict[date, Sequence[StrikeFlow]],
        atm_iv: float,
        dealer_share: float,
    ) -> GexProfile:
        """The daily-OI blend, corrected by signed flow since the print --
        see ``nowcast_books``.  Repriced through exactly the same
        gamma-weighted pipeline ``blended_profile`` uses, so everything
        downstream of the strike table (sign convention, the confidence and
        flip-distance gates, the flip search) is identical between a base
        read and a nowcast one; the only difference is what went into it.
        """
        adjusted = self.nowcast_books(books, flow_by_expiry, dealer_share)
        return self.blended_profile(spot, adjusted, atm_iv)

    def total_at(
        self,
        hypothetical_spot: float,
        spot: float,
        open_interest: Sequence[StrikeOpenInterest],
        time_to_expiry: float,
        atm_iv: float,
    ) -> float:
        """Total GEX the current book would carry if spot were elsewhere."""
        return self.blended_total_at(
            hypothetical_spot,
            spot,
            [ExpiryBook.of(date.min, time_to_expiry, open_interest)],
            atm_iv,
        )

    def blended_total_at(
        self,
        hypothetical_spot: float,
        spot: float,
        books: Sequence[ExpiryBook],
        atm_iv: float,
    ) -> float:
        prepared = self._prepare(spot, books)
        if not prepared:
            return 0.0
        return float(
            self._curve(np.array([hypothetical_spot]), prepared, atm_iv)[0]
        )

    # -- the ensemble gate ------------------------------------------------

    def ensemble(
        self, spot: float, books: Sequence[ExpiryBook], atm_iv: float
    ) -> EnsembleResult:
        """Recompute the regime over perturbed assumptions and check agreement.

        Two inputs are varied, and they are exactly the two the README
        flags as load-bearing: the skew slope, which prices every gamma in
        the profile and therefore moves the flip point, and the dealer sign
        convention, which decides what open interest *means*.  A regime
        that reverses under a plausible change to either was a property of
        the model rather than a reading of the market, and the strategy has
        no business acting on it.

        Unanimity is over the regime, including NEUTRAL: a member that
        cannot make up its mind counts as dissent.  That is deliberate --
        the gate answers "would every version of me take this trade?", and
        "no, one of them would stand aside" is a no.
        """
        gates = self.gates
        regimes: list[str] = []
        for delta in gates.ensemble_skew_slope_deltas:
            surface = self._perturbed_surface(float(delta))
            for call_sign, put_sign in gates.sign_conventions():
                member = GexCalculator(
                    dataclasses.replace(
                        self.cfg, call_sign=call_sign, put_sign=put_sign
                    ),
                    self.source,
                    surface,
                    self.risk_free_rate,
                    gates,
                )
                regimes.append(member.blended_profile(spot, books, atm_iv).regime)

        distinct = sorted(set(regimes))
        unanimous = len(distinct) == 1
        regime = distinct[0] if unanimous else NEUTRAL
        if unanimous:
            detail = (
                f"all {len(regimes)} ensemble members read {regime}"
            )
        else:
            counts = ", ".join(
                f"{name} x{regimes.count(name)}" for name in distinct
            )
            detail = (
                f"the ensemble does not agree ({counts} across {len(regimes)} "
                "members): the regime is a property of the assumed skew or "
                "sign convention rather than of the chain"
            )
        return EnsembleResult(unanimous, regime, tuple(regimes), detail)

    def _perturbed_surface(self, slope_delta: float) -> VolSurface:
        """The vol surface with its skew slope shifted, same type as ours."""
        if slope_delta == 0.0:
            return self.surface
        cfg = dataclasses.replace(
            self.surface.cfg, skew_slope=self.surface.cfg.skew_slope + slope_delta
        )
        return type(self.surface)(cfg)

    # -- internals -------------------------------------------------------

    def _effective_tenor(self, time_to_expiry: float) -> float:
        """Tenor used for classification, floored (see the module docstring)."""
        return max(time_to_expiry, self.cfg.min_hours_to_expiry / HOURS_PER_YEAR)

    def _prepare(
        self, spot: float, books: Sequence[ExpiryBook]
    ) -> list["PreparedBook"]:
        """Each book as sorted arrays, plus the tenor to price it at.

        Books with nothing inside the strike window are dropped rather than
        carried as empty arrays -- an expiry with no listed open interest
        near the money contributes no gamma, and keeping it would only put
        a zero row in the per-expiry attribution.
        """
        prepared: list[PreparedBook] = []
        for book in books:
            strikes, calls, puts = self._arrays(spot, book.rows)
            if not strikes.size:
                continue
            prepared.append(
                PreparedBook(
                    book, strikes, calls, puts,
                    self._effective_tenor(book.time_to_expiry),
                )
            )
        return prepared

    def _arrays(
        self, spot: float, open_interest: Sequence[StrikeOpenInterest]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Strikes inside the window, with their OI, as sorted arrays."""
        half = spot * self.cfg.strike_width_pct
        rows = sorted(
            (r for r in open_interest if abs(r.strike - spot) <= half and r.strike > 0),
            key=lambda r: r.strike,
        )
        if not rows:
            empty = np.zeros(0)
            return empty, empty, empty.copy()
        return (
            np.array([r.strike for r in rows], dtype=float),
            np.array([r.call_oi for r in rows], dtype=float),
            np.array([r.put_oi for r in rows], dtype=float),
        )

    def _vols(self, spot, strikes, atm_iv: float) -> np.ndarray:
        return self.surface.iv_array(spot, strikes, atm_iv)

    def _curve(
        self, spots: np.ndarray, prepared: Sequence["PreparedBook"], atm_iv: float
    ) -> np.ndarray:
        """Total GEX at each of ``spots``, holding open interest fixed.

        One vectorised block per expiry rather than a loop over strikes: the
        flip search reprices every strike at every grid point on every bar,
        and doing that a scalar at a time dominates the whole backtest.
        """
        column = spots[:, None]
        scale = self.source.option.multiplier * column * column * 0.01
        out = np.zeros(spots.shape, dtype=float)
        for entry in prepared:
            strikes = entry.strikes
            vols = self.surface.iv_array(column, strikes[None, :], atm_iv)
            gamma = black76_gamma(
                column, strikes[None, :], entry.tenor, vols, self.risk_free_rate
            )
            weight = self.cfg.call_sign * entry.calls + self.cfg.put_sign * entry.puts
            out = out + (scale * gamma * weight[None, :]).sum(axis=1)
        return out

    def _flip_point(
        self, spot: float, prepared: Sequence["PreparedBook"], atm_iv: float
    ) -> float | None:
        """The spot level where total GEX crosses zero, nearest to ``spot``.

        Returns ``None`` when the curve holds one sign across the whole
        search range -- a real answer ("there is no flip nearby"), not a
        failure, and the caller must not fabricate one from the endpoints.
        """
        half = spot * self.cfg.flip_search_pct
        grid = np.linspace(spot - half, spot + half, self.cfg.flip_search_steps)
        grid = grid[grid > 0.0]
        if grid.size < 2:
            return None

        curve = self._curve(grid, prepared, atm_iv)
        crossings: list[float] = []
        for i in range(len(grid) - 1):
            lo, hi = curve[i], curve[i + 1]
            if lo == 0.0:
                crossings.append(float(grid[i]))
            elif (lo < 0.0) != (hi < 0.0):
                # Linear interpolation between the bracketing grid points.
                crossings.append(float(grid[i] + (grid[i + 1] - grid[i]) * lo / (lo - hi)))
        if curve[-1] == 0.0:
            crossings.append(float(grid[-1]))
        if not crossings:
            return None
        return min(crossings, key=lambda level: abs(level - spot))

    def _classify(
        self, spot: float, total: float, gross: float, flip: float | None
    ) -> tuple[str, str, str]:
        """Regime, the sentence explaining it, and the gate that forced it.

        The two gates are checked in the order they can each be *right*
        about: a book with no directional gamma has no sign to be near the
        flip of, so confidence comes first.
        """
        gates = self.gates
        if gross <= 0.0:
            return NEUTRAL, "no gamma in the chain", GATE_CONFIDENCE

        share = abs(total) / gross
        if gates.confidence and share < gates.min_confidence_ratio:
            return NEUTRAL, (
                f"net GEX is only {share:.1%} of gross "
                f"(threshold {gates.min_confidence_ratio:.0%}); dealers are "
                "close to flat and the sign is noise in the open-interest "
                "print rather than positioning"
            ), GATE_CONFIDENCE

        if (
            gates.flip_distance
            and flip is not None
            and abs(spot - flip) <= spot * self.cfg.flip_proximity_pct
        ):
            return NEUTRAL, (
                f"spot {spot:,.2f} is within {self.cfg.flip_proximity_pct:.2%} of the "
                f"gamma flip at {flip:,.2f}; the sign is about to change"
            ), GATE_FLIP_DISTANCE

        flip_text = f", flip {flip:,.1f}" if flip is not None else ""
        if total > 0.0:
            return POSITIVE, (
                f"GEX {total / 1e6:+,.1f}M/1% at {spot:,.2f}{flip_text} "
                f"({share:.0%} of gross): dealers are long gamma and hedge "
                "against the move, damping realised vol"
            ), ""
        return NEGATIVE, (
            f"GEX {total / 1e6:+,.1f}M/1% at {spot:,.2f}{flip_text} "
            f"({share:.0%} of gross): dealers are short gamma and hedge with "
            "the move, amplifying realised vol"
        ), ""
