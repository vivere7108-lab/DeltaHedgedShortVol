"""Dealer gamma exposure: the flip point, and the regime it implies.

What this computes
------------------
GEX is an estimate of the gamma the option dealer community is carrying,
inferred from open interest and a *sign* at each strike::

    gex(K) = mult * S^2 * 0.01 * gamma(K) * (s_call(K)*OI_call + s_put(K)*OI_put)

The ``S^2 * 0.01`` turns per-point gamma into dollars of delta the dealer
must trade for a 1% move, which is the unit the number is quoted in.

Where the sign comes from
-------------------------
Every published GEX print takes ``s_call = +1`` and ``s_put = -1`` at every
strike, all day: the public buys puts and sells calls, so the dealer is long
the calls and short the puts.  That is an assumption, it is the load-bearing
one in the whole system, and it is not in the open-interest print.  It is,
however, in the **tape**.  Each execution has an aggressor and a resting
side, and the dealer is the resting side: a customer lifting the offer
leaves a dealer short the option and short its gamma; a customer hitting the
bid leaves one long it.

``flow.py`` classifies executions -- from CME MDP 3.0's own aggressor flag
(tag 5797) where the feed carries it, from the market-by-order book-state
change where it carries that, and from the Lee-Ready quote and tick rules
otherwise -- and accumulates the signed dealer position per strike and
right.  ``StrikeDealerFlow.sign`` is that position over the classified
volume: a number in ``[-1, +1]`` on exactly the scale ``call_sign`` and
``put_sign`` live on.

A measured sign is only as good as the tape behind it, so this module
blends rather than substitutes::

    w      = n / (n + gex.flow_confidence_contracts)
    s(K)   = w * measured(K) + (1 - w) * prior

with ``n`` the classified contracts at that strike and right.  A strike that
has not traded keeps the prior exactly -- which is the behaviour this module
had before the tape was wired in -- a heavily traded one is essentially all
measurement, and everything between is a weighted admission of how much is
actually known.  ``GexProfile.flow_coverage`` reports where on that scale a
given read sits, so a number driven by the assumption can be told from one
driven by the evidence.  ``gex.use_flow_signs`` turns the whole thing off
and restores the static convention, which is the control a measured run
should be compared against.

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
their delta is the sum over everything they are carrying.  With today's
series traded that is one book for most of the day; in the roll window,
when tomorrow's series is the one being entered, it is today's and
tomorrow's together, and today's expiring gamma is still the larger part
of what dealers are hedging.

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
``GexCalculator.ensemble`` reprices the regime over a grid of skew,
sign-prior and flow-trust perturbations -- but it is invoked by the strategy
only when a decision actually turns on it, because it costs a full profile
per member.  Persistence and the entry window are the strategy's, not the
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
1. **Open interest is not positioning -- the tape is, as far as it goes.**
   Who is long and who is short is not in the OI print.  With a trade feed
   attached it is measured from classified executions at each strike, which
   turns the old assumption into an estimate with a known amount of
   evidence behind it; without one, ``call_sign``/``put_sign`` are still an
   assumption and still the load-bearing one.  Either way the ensemble gate
   prices the remaining uncertainty rather than believing the number.
   Note what the measurement does *not* fix: classified flow is a flow and
   open interest is a stock, so applying a strike's measured sign to all of
   its open interest extrapolates from the contracts that traded while the
   process was watching to the ones that did not.  For a 0DTE series, where
   the book is written the same session, that gap is small.  For a series
   listed a week ago it is not, and ``flow.half_life_minutes`` and
   ``gex.flow_confidence_contracts`` are the two dials that decide how
   loudly the part that *was* seen speaks for the rest.
2. **OI is only as fresh as the feed.**  With intraday open interest
   from the exchange's MDP 3.0 feed the same-day series' print describes
   the book that is actually there, which is what makes a 0DTE read
   usable; on a feed that only carries the previous session's close the
   same-day series is stalest exactly where most of the flow is, and the
   GEX layer has no way to tell.  ``gex.refresh_seconds`` is how often
   the print is re-read.
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
from .flow import CALL, PUT, StrikeDealerFlow
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
    """One expiry's open interest and measured flow, with its tenor.

    The unit the blend is built from.  ``time_to_expiry`` is the real
    wall-clock tenor of that series; the ``min_hours_to_expiry`` floor is
    applied by the calculator rather than baked in here, so a caller cannot
    accidentally hand the hedger a floored tenor.

    ``flow`` is what ``DealerFlowBook`` measured at each strike, and it is
    carried *raw* -- signed dealer contracts and the classified volume
    behind them, not a finished sign.  The blend against the prior happens
    in the calculator, because the ensemble gate re-derives it under
    perturbed priors and perturbed trust in the tape and would have nothing
    to perturb if the answer arrived already computed.  Empty means no feed,
    and every strike falls back to the prior.
    """

    expiry: date
    time_to_expiry: float
    rows: tuple[StrikeOpenInterest, ...]
    days_to_expiry: int = 0
    flow: tuple[StrikeDealerFlow, ...] = ()

    @classmethod
    def of(
        cls,
        expiry: date,
        time_to_expiry: float,
        rows: Sequence[StrikeOpenInterest],
        days_to_expiry: int = 0,
        flow: Sequence[StrikeDealerFlow] = (),
    ) -> "ExpiryBook":
        return cls(expiry, time_to_expiry, tuple(rows), days_to_expiry, tuple(flow))

    @property
    def has_flow(self) -> bool:
        return bool(self.flow)


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
    #: The signs this strike's GEX was actually computed with, after the
    #: measured flow was blended against the prior. Printed in the strike
    #: table so a read can be checked against the assumption it started
    #: from: ``+1.00``/``-1.00`` is an untraded strike carrying the prior
    #: unchanged, anything else is the tape having moved it. In a blended
    #: profile these are the OI-weighted mean across the expiries summed.
    call_sign: float = 0.0
    put_sign: float = 0.0
    #: Classified contracts behind those signs, summed across the blend.
    call_flow: float = 0.0
    put_flow: float = 0.0

    @property
    def net_gex(self) -> float:
        return self.call_gex + self.put_gex

    @property
    def flow_volume(self) -> float:
        return self.call_flow + self.put_flow


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
    #: Open-interest-weighted share of this profile's signs that came from
    #: classified trades rather than from ``call_sign``/``put_sign``. 0.0 is
    #: the pre-tape system -- every sign assumed; 1.0 would be every sign
    #: measured. It is a description of the *evidence*, not of the regime,
    #: and nothing gates on it directly: a low-coverage read is the old read
    #: and the old read is still the honest fallback. What it is for is
    #: telling a number driven by the market apart from one driven by the
    #: convention, in a log line and after the fact in the journal.
    flow_coverage: float = 0.0
    #: Classified contracts standing behind the measured part of the signs.
    flow_volume: float = 0.0

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
        thresholds, and it is scale-free in the *book* -- a bigger book does
        not read as a more confident one -- but deliberately not scale-free
        in the *measurement*: ``gross_gex`` weights the chain's gamma by the
        sign prior rather than by the blended sign, so a book the tape has
        measured as dealer-flat reports a small number against a large one
        and reads unconfident, rather than reporting the direction of
        whatever prior survived the shrinkage as though it were positioning.
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
        measured = (
            f", signs {self.flow_coverage:.0%} measured"
            if self.flow_coverage > 0.0 else ""
        )
        return (
            f"GEX {self.total_gex / 1e6:+,.1f}M/1% at {self.spot:,.2f}, "
            f"flip {flip}, confidence {self.confidence:.0%}, regime {self.regime}"
            f"{measured}"
        )

    def table(self, limit: int = 15) -> str:
        """The strikes carrying the most gamma, for eyeballing a live read."""
        rows = sorted(self.by_strike, key=lambda s: -abs(s.net_gex))[:limit]
        rows.sort(key=lambda s: s.strike)
        lines = [
            f"{'strike':>9} {'call OI':>9} {'put OI':>9} {'net GEX ($M/1%)':>17} "
            f"{'call sgn':>9} {'put sgn':>9} {'flow':>9}"
        ]
        for row in rows:
            lines.append(
                f"{row.strike:>9,.0f} {row.call_oi:>9,.0f} {row.put_oi:>9,.0f} "
                f"{row.net_gex / 1e6:>17,.2f} "
                f"{row.call_sign:>9,.2f} {row.put_sign:>9,.2f} "
                f"{row.flow_volume:>9,.0f}"
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


class PreparedBook(NamedTuple):
    """One expiry's book, windowed and arrayed, ready to price.

    ``call_signs``/``put_signs`` are per-strike and already blended: the
    measured dealer sign where the tape has classified enough of it, the
    configured prior where it has not, and a weighted mix in between.  They
    are computed once here rather than inside ``_curve`` because the flip
    search evaluates the same book at sixty-one hypothetical spots and the
    signs do not depend on spot.
    """

    book: ExpiryBook
    strikes: np.ndarray
    calls: np.ndarray
    puts: np.ndarray
    tenor: float
    call_signs: np.ndarray
    put_signs: np.ndarray
    call_flow: np.ndarray
    put_flow: np.ndarray


class GexCalculator:
    """Turns open interest into a profile, a flip point and a regime."""

    def __init__(
        self,
        cfg: GexConfig,
        source: RiskSource,
        surface: VolSurface,
        risk_free_rate: float = 0.0,
        gates: GatesConfig | None = None,
        flow_confidence_scale: float = 1.0,
    ):
        self.cfg = cfg
        self.source = source
        self.surface = surface
        self.risk_free_rate = risk_free_rate
        self.gates = gates if gates is not None else GatesConfig()
        #: Multiplier on ``cfg.flow_confidence_contracts``, so an ensemble
        #: member can be built that trusts the tape more or less readily
        #: without a second copy of the config. 1.0 is the traded setting.
        self.flow_confidence_scale = float(flow_confidence_scale)

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

        measured_oi = flow_volume = total_oi = 0.0
        for entry in prepared:
            book, strikes, calls, puts, tenor = (
                entry.book, entry.strikes, entry.calls, entry.puts, entry.tenor
            )
            gamma = black76_gamma(
                spot, strikes, tenor, self._vols(spot, strikes, atm_iv),
                self.risk_free_rate,
            )
            call_gex = scale * gamma * entry.call_signs * calls
            put_gex = scale * gamma * entry.put_signs * puts
            # Coverage is open-interest weighted: what is being reported is
            # the share of the *positioning in the profile* whose sign was
            # measured, not the share of strikes -- a measured sign on a
            # strike carrying no open interest changes nothing and should
            # not read as evidence.
            call_weight = self._flow_weight(entry.call_flow)
            put_weight = self._flow_weight(entry.put_flow)
            measured_oi += float((call_weight * calls + put_weight * puts).sum())
            total_oi += float(calls.sum() + puts.sum())
            flow_volume += float(entry.call_flow.sum() + entry.put_flow.sum())
            expiry_total = float((call_gex + put_gex).sum())
            # Gross is the gamma in the book, summed per *leg* rather than
            # per strike, and weighted by the magnitude of the *prior* --
            # not of the blended sign.
            #
            # Per leg, because summing net-per-strike would collapse to zero
            # for a chain with matched call and put interest, which is a
            # maximally gamma-laden book rather than an empty one.
            #
            # At the prior, because dividing by a gross that carried the
            # blended sign would make the confidence ratio scale-invariant
            # in the measurement itself: a book the tape has measured as
            # dealer-flat has both a tiny numerator and a tiny denominator,
            # and their ratio would report whatever residual prior survived
            # the shrinkage as a confident read on an essentially empty
            # book. Against the gamma the prior says is there, a flat book
            # reads flat -- which is what the classification measured and
            # what the gate exists to catch. Where nothing has been
            # measured the blended sign *is* the prior, so this is exactly
            # the old quantity and a run without a feed is unchanged.
            expiry_gross = float(
                (
                    abs(self.cfg.call_sign) * np.abs(scale * gamma * calls)
                    + abs(self.cfg.put_sign) * np.abs(scale * gamma * puts)
                ).sum()
            )
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
            for k, c, p, g, cg, pg, cs, ps, cf, pf in zip(
                strikes, calls, puts, gamma, call_gex, put_gex,
                entry.call_signs, entry.put_signs, entry.call_flow, entry.put_flow,
            ):
                row = per_strike.setdefault(float(k), [0.0] * 9)
                row[0] += float(c)
                row[1] += float(p)
                row[2] += float(g)
                row[3] += float(cg)
                row[4] += float(pg)
                # The displayed sign is OI-weighted across the expiries so a
                # strike whose open interest sits almost entirely in one
                # series reports that series' sign rather than an unweighted
                # average of it with an empty one.
                row[5] += float(cs) * float(c)
                row[6] += float(ps) * float(p)
                row[7] += float(cf)
                row[8] += float(pf)

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
            flow_coverage=(measured_oi / total_oi) if total_oi > 0.0 else 0.0,
            flow_volume=flow_volume,
            by_strike=tuple(
                StrikeGex(
                    strike=strike, call_oi=row[0], put_oi=row[1], gamma=row[2],
                    call_gex=row[3], put_gex=row[4],
                    call_sign=(row[5] / row[0]) if row[0] > 0.0 else self.cfg.call_sign,
                    put_sign=(row[6] / row[1]) if row[1] > 0.0 else self.cfg.put_sign,
                    call_flow=row[7], put_flow=row[8],
                )
                for strike, row in sorted(per_strike.items())
            ),
            by_expiry=tuple(by_expiry),
        )

    def _flow_weight(self, volume: np.ndarray) -> np.ndarray:
        """``n / (n + confidence)`` -- the share of a sign that is measured."""
        confidence = self.cfg.flow_confidence_contracts * self.flow_confidence_scale
        if not self.cfg.use_flow_signs or confidence <= 0.0:
            return np.zeros(volume.shape, dtype=float)
        return volume / (volume + confidence)

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

        Three inputs are varied, and they are exactly the ones the README
        flags as load-bearing:

        * the **skew slope**, which prices every gamma in the profile and
          therefore moves the flip point;
        * the **sign prior**, which decides what open interest means at a
          strike the tape has said nothing about;
        * how readily a measured sign **overrules that prior** --
          ``gex.flow_confidence_contracts``, scaled by
          ``gates.ensemble_flow_confidence_scales``.

        The third axis is there because measuring the sign did not remove
        the assumption, it moved it.  "The flow I classified at this strike
        represents the book standing at it" is the new load-bearing claim,
        and it is strongest where a strike is heavily traded and weakest
        where the read rests on a handful of prints -- which is precisely
        the difference the scales expose.  Where the tape is thick every
        scale agrees and the axis is free; where it is thin they diverge and
        the gate blocks, which is the behaviour wanted in both cases.

        With no flow at all the axis is a no-op -- every scale gives the
        same profile -- so it collapses to a single member and a run without
        a trade feed pays nothing for it.

        A regime that reverses under a plausible change to any of the three
        was a property of the model rather than a reading of the market, and
        the strategy has no business acting on it.

        Unanimity is over the regime, including NEUTRAL: a member that
        cannot make up its mind counts as dissent.  That is deliberate --
        the gate answers "would every version of me take this trade?", and
        "no, one of them would stand aside" is a no.
        """
        gates = self.gates
        measured = self.cfg.use_flow_signs and any(book.has_flow for book in books)
        scales = gates.flow_confidence_scales() if measured else [1.0]
        regimes: list[str] = []
        for delta in gates.ensemble_skew_slope_deltas:
            surface = self._perturbed_surface(float(delta))
            for call_sign, put_sign in gates.sign_conventions():
                for scale in scales:
                    member = GexCalculator(
                        dataclasses.replace(
                            self.cfg, call_sign=call_sign, put_sign=put_sign
                        ),
                        self.source,
                        surface,
                        self.risk_free_rate,
                        gates,
                        flow_confidence_scale=scale,
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
            axis = (
                "assumed skew, the sign prior or how far the classified tape "
                "is trusted"
                if measured else "assumed skew or sign convention"
            )
            detail = (
                f"the ensemble does not agree ({counts} across {len(regimes)} "
                f"members): the regime is a property of the {axis} rather "
                "than of the chain"
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
            call_signs, call_flow = self._signs(book, strikes, CALL)
            put_signs, put_flow = self._signs(book, strikes, PUT)
            prepared.append(
                PreparedBook(
                    book, strikes, calls, puts,
                    self._effective_tenor(book.time_to_expiry),
                    call_signs, put_signs, call_flow, put_flow,
                )
            )
        return prepared

    def _signs(
        self, book: ExpiryBook, strikes: np.ndarray, right: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """The dealer sign at each strike, and the tape behind it.

        The shrinkage in one place: ``w = n / (n + confidence)`` with ``n``
        the classified contracts at that strike and right, and the answer
        ``w * measured + (1 - w) * prior``.  Three properties make it the
        right functional form here rather than a threshold:

        * a strike with no flow returns the prior *exactly*, so attaching a
          feed can never change a read it has no evidence about;
        * a strike with overwhelming flow returns the measurement, so the
          assumption stops mattering where it stopped being needed;
        * nothing between them is a cliff, so the profile does not jump the
          moment one more contract prints.

        ``strike_flow`` is indexed rather than zipped because the flow rows
        and the open-interest rows are different sets: an option can trade
        at a strike carrying no listed open interest, and open interest sits
        at strikes that have not traded all day.
        """
        prior = self.cfg.call_sign if right == CALL else self.cfg.put_sign
        signs = np.full(strikes.shape, float(prior), dtype=float)
        volume = np.zeros(strikes.shape, dtype=float)
        if not self.cfg.use_flow_signs or not book.flow:
            return signs, volume

        by_strike = {float(row.strike): row for row in book.flow}
        measured = np.zeros(strikes.shape, dtype=float)
        for i, strike in enumerate(strikes):
            row = by_strike.get(float(strike))
            if row is None:
                continue
            sign = row.sign(right)
            n = row.volume(right)
            if sign is None or n <= 0.0:
                continue
            measured[i] = sign
            volume[i] = n
        # One definition of the weight, shared with the coverage figure: two
        # copies of ``n / (n + confidence)`` would be two things to keep in
        # step, and a profile whose reported coverage disagreed with the
        # signs it actually used would be worse than one reporting nothing.
        # A strike with no flow has weight 0 and keeps the prior exactly.
        weights = self._flow_weight(volume)
        return weights * measured + (1.0 - weights) * signs, volume

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
            weight = entry.call_signs * entry.calls + entry.put_signs * entry.puts
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
