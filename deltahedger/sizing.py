"""Capital requirements and buying-power based position sizing.

The number of straddles is the smallest count three constraints allow, and
the delta band then absorbs whatever delta that position happens to carry::

    contracts = min(risk budget      / loss per straddle at its branch stop,
                    gamma ceiling    / gamma per straddle,
                    capital cap      / margin or debit per straddle,
                    max_straddles)

**Why three.**  Sizing to the capital cap alone -- what this did until now,
80% of equity less a hedge reserve -- makes the exit ladder decoration.  At
$250k, spot 5000, 15 vol, a 0DTE book sized that way is 173 long straddles
($140k of debit, 56% of equity) or 83 short ($139k of margin), and against
a 5% daily loss limit:

===========================  ==================  ========================
branch                       its stop needs      the daily limit is
===========================  ==================  ========================
long, 50% of the debit       5.6x the limit      1.3 vol points of IV
short, 2.5x the credit       8.1x the limit      2.8 vol points of IV
short 1DTE at the roll       19.4x the limit     1.2 vol points of IV
===========================  ==================  ========================

So the daily loss limit fired first every time and halted the session --
on generated data, 27 of 56 positions ended that way against 7 on a branch
rule.  A book carrying $9,300 of vega per point against a $12,500 daily
limit is sized so that ordinary intraday noise ends the day.

The **risk budget** fixes that by inverting the question: rather than "how
much capital may we commit", it asks "how much may a stop-out cost", and
divides.  What one straddle loses at its stop is ``long_stop_loss_pct`` of
the debit, or ``short_stop_loss_premium_multiple - 1`` times the credit;
with the branch stop disabled it falls back to the requirement itself,
which is the right number either way -- for a long straddle the debit *is*
the maximum loss, and for a short one the SPAN scan is a one-day adverse
move, which is exactly the loss being bounded.

The **gamma ceiling** is there because the risk budget alone holds premium
flat and lets gamma run: the same budget buys 30 straddles carrying 118
delta units per point at 09:35 and 63 carrying 512 at 14:30, since a cheap
late straddle risks less per contract and carries more gamma.  Gamma is
what the strategy is a bet on, so the bet would silently grow into the end
of the day.  Targeting gamma *alone* fails the other way -- at the 1DTE
roll, where gamma per straddle is small, a flat gamma target buys 46% of
equity in premium -- which is why the capital cap and the risk budget stay.

The hedge leg has no reserve carved out for it any more.  With the risk
budget binding, the straddle takes about a tenth of equity, and the unused
part of the capital cap covers the MES margin even for a fully in-the-money
book at the pre-settlement buffer.

The requirement means different things in the two regimes, and conflating
them would misstate the risk in both directions:

* **short straddle** (positive GEX) -- the requirement is *margin*.  Loss is
  unbounded, the broker holds collateral against it, and SPAN is what
  decides how much.
* **long straddle** (negative GEX) -- the requirement is the *debit*.  There
  is no margin: the premium is paid in full and is also the entire maximum
  loss on the option leg.  Charging a scenario margin on top would size the
  long branch smaller than the risk justifies, and charging nothing would
  ignore that the cash actually leaves the account.

Three models ship here.

``SpanScanMarginModel`` is the default and the only one appropriate for
futures options.  It reproduces CME SPAN's risk-array method: reprice the
position across a grid of price and volatility scenarios and charge the
worst loss.  Because we already have Black-76, the scenarios can be priced
exactly rather than approximated, so the model captures the thing that
matters most for a short option -- margin exploding as the strike comes
into range of the scan.

What the tenor does to the requirement
--------------------------------------
The traded series is today's, rolled into tomorrow's at the end of the
day, so the book is sized at 0DTE in the morning and at 1DTE at the roll.
The two branches respond to tenor in opposite directions, and the reason
is worth stating because it is not what most people expect.

The **scan range does not lengthen with the option**.  SPAN scans a
one-day move -- about 49 ES points at a 2455 outright margin -- whatever
the tenor of what you are holding.  What changes is how much the straddle
is worth *after* that move relative to what it is worth now, and a
longer-dated straddle has already collected most of the value that move
would create.  So the short branch's margin per straddle is close to flat
across the range, while the debit roughly doubles from the morning's 0DTE
entry to the afternoon's 1DTE roll.

Under the three constraints neither of those is what usually decides the
count.  Measured at 5000, 15 vol, a $250k account, on the shipped
settings::

    moment                premium   SPAN    debit   short n         long n
    0DTE at 09:35 (6.4h)    16.17  $1,679    $809   10  risk    30  risk
    0DTE at 12:00 (4.0h)    12.79  $1,822    $639   13  risk    30  gamma
    0DTE at 14:30 (1.5h)     7.83  $2,063    $392   18  gamma   18  gamma
    1DTE at the roll        31.32  $1,352  $1,566    5  risk    15  risk

The short branch is smaller than the long one at the same risk budget, and
that is the rule working rather than a bias: its stop sits 2.5x the credit
away, so a stop-out costs 1.5x the premium against the long side's 0.5x,
and a wider stop earns fewer contracts.  If that reads as too small, the
lever is ``short_stop_loss_premium_multiple`` -- now that it is reachable,
it is a real parameter rather than decoration -- not the budget.

By the afternoon the gamma ceiling takes over from the risk budget on both
sides, which is what it is for: a cheap late straddle risks little per
contract and carries a lot of gamma, so the risk budget alone would let the
size of the bet grow into the end of the session.  The backtest's band
section still reports median gamma and band per branch, so a regime
comparison can see what size difference remains.

A one-day scan is a conservative charge against a 0DTE position that will
be flat by the bell, and exactly the horizon a rolled 1DTE position is
carried over.

Under the risk budget the long branch spends about a tenth of equity on a
same-day straddle's debit rather than more than half of it.  That debit is
still the maximum loss on the option leg and can still be reached in a
single session -- the daily loss limit in ``StrategyConfig`` remains the
backstop -- but it is now a backstop rather than the only stop that ever
fires.

``RegTMarginModel`` is the 15%-of-notional equity-option rule.  It is
included because it is what most people reach for, and it overstates ES
option margin by roughly an order of magnitude; use it only to compare.

None of these are IBKR's number.  For live trading use
``broker.ibkr.WhatIfMarginModel``, which asks IBKR to price the margin
impact of the actual order before it is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .chain import OptionQuote, StraddleQuote
from .config import SizingConfig
from .instruments import RiskSource
from .pricing import black76

#: SPAN's 16-scenario risk array: (fraction of the price scan range,
#: fraction of the volatility scan range, weight applied to the loss).
#: The final two rows are the "extreme move" scenarios, covered at 35%.
SPAN_SCENARIOS: tuple[tuple[float, float, float], ...] = (
    (0.0, +1.0, 1.0),
    (0.0, -1.0, 1.0),
    (+1 / 3, +1.0, 1.0),
    (+1 / 3, -1.0, 1.0),
    (-1 / 3, +1.0, 1.0),
    (-1 / 3, -1.0, 1.0),
    (+2 / 3, +1.0, 1.0),
    (+2 / 3, -1.0, 1.0),
    (-2 / 3, +1.0, 1.0),
    (-2 / 3, -1.0, 1.0),
    (+1.0, +1.0, 1.0),
    (+1.0, -1.0, 1.0),
    (-1.0, +1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (+2.0, 0.0, 0.35),
    (-2.0, 0.0, 0.35),
)


class MarginModel(Protocol):
    """Capital required to carry one straddle, in USD."""

    def straddle_requirement(
        self, quote: StraddleQuote, future_price: float, source: RiskSource,
        direction: int,
    ) -> float: ...

    def hedge_margin(self, source: RiskSource) -> float: ...


def straddle_debit(quote: StraddleQuote, source: RiskSource) -> float:
    """Cash paid for one long straddle -- and its maximum loss."""
    return quote.price * source.option.multiplier


@dataclass
class FixedMarginModel:
    """Flat margin per short leg. Simple, and easy to stress."""

    per_option_contract: float
    per_hedge_contract: float

    def straddle_requirement(
        self, quote: StraddleQuote, future_price: float, source: RiskSource,
        direction: int,
    ) -> float:
        if direction > 0:
            return straddle_debit(quote, source)
        return 2.0 * self.per_option_contract

    def hedge_margin(self, source: RiskSource) -> float:
        return self.per_hedge_contract


@dataclass
class SpanScanMarginModel:
    """CME SPAN risk-array margin for a short option on a future.

    The price scan range is inferred from the risk source's outright future
    margin -- CME sets that margin *to* the scan range, so
    ``future_initial_margin / multiplier`` recovers the point move being
    scanned (about 49 ES points, ~1%, at a 2455 margin).  Volatility is
    scanned as a relative bump.

    The scan is a *one-day* move and does not stretch with the option's
    tenor, which is SPAN's design and not an approximation here.  See the
    module docstring for what that does to the two branches.
    """

    scan_multiplier: float = 1.0
    vol_scan_pct: float = 0.30
    short_option_minimum: float = 250.0
    risk_free_rate: float = 0.0

    def price_scan_range(self, source: RiskSource) -> float:
        """The price move SPAN scans, in underlying points."""
        return (
            source.future_initial_margin / source.future.multiplier
        ) * self.scan_multiplier

    def straddle_requirement(
        self, quote: StraddleQuote, future_price: float, source: RiskSource,
        direction: int,
    ) -> float:
        if direction > 0:
            return straddle_debit(quote, source)

        scan = self.price_scan_range(source)
        mult = source.option.multiplier
        # We are short, so a scenario that raises the pair's value is a loss.
        # Both legs are repriced in the same scenario and netted before the
        # worst case is taken: a straddle is one position, and charging each
        # leg its own worst case would double-count a move that cannot hurt
        # both at once.
        entry_value = quote.price
        worst_loss = 0.0
        for price_frac, vol_frac, weight in SPAN_SCENARIOS:
            scenario_future = max(future_price + price_frac * scan, 1e-9)
            scenario_value = 0.0
            for leg in quote.legs():
                scenario_vol = max(leg.iv * (1.0 + vol_frac * self.vol_scan_pct), 1e-6)
                scenario_value += black76(
                    scenario_future,
                    leg.strike,
                    quote.time_to_expiry,
                    scenario_vol,
                    self.risk_free_rate,
                    leg.right,
                ).price
            loss = (scenario_value - entry_value) * mult * weight
            worst_loss = max(worst_loss, loss)
        return max(worst_loss, 2.0 * self.short_option_minimum)

    def hedge_margin(self, source: RiskSource) -> float:
        return source.hedge_initial_margin


@dataclass
class RegTMarginModel:
    """Equity-option style margin. Wrong for futures; kept for comparison.

        margin = premium + max(a*futures_notional - otm, b*strike_notional)
    """

    a: float = 0.15
    b: float = 0.10

    def leg_margin(
        self, quote: OptionQuote, future_price: float, source: RiskSource
    ) -> float:
        mult = source.option.multiplier
        premium = quote.price * mult
        if quote.right.upper() == "P":
            out_of_the_money = max(future_price - quote.strike, 0.0) * mult
        else:
            out_of_the_money = max(quote.strike - future_price, 0.0) * mult
        return premium + max(
            self.a * future_price * mult - out_of_the_money,
            self.b * quote.strike * mult,
        )

    def straddle_requirement(
        self, quote: StraddleQuote, future_price: float, source: RiskSource,
        direction: int,
    ) -> float:
        if direction > 0:
            return straddle_debit(quote, source)
        # The Reg-T short-straddle rule: margin the losing side in full and
        # add the other side's premium. Only one leg can finish in the money.
        mult = source.option.multiplier
        legs = quote.legs()
        margins = [self.leg_margin(leg, future_price, source) for leg in legs]
        worst = max(range(len(legs)), key=lambda i: margins[i])
        other = 1 - worst
        return margins[worst] + legs[other].price * mult

    def hedge_margin(self, source: RiskSource) -> float:
        return source.hedge_initial_margin


def build_margin_model(
    cfg: SizingConfig, source: RiskSource, risk_free_rate: float = 0.0
) -> MarginModel:
    if cfg.margin_model == "fixed":
        return FixedMarginModel(
            per_option_contract=cfg.fixed_margin_per_contract,
            per_hedge_contract=source.hedge_initial_margin,
        )
    if cfg.margin_model == "reg_t":
        return RegTMarginModel(a=cfg.reg_t_a, b=cfg.reg_t_b)
    return SpanScanMarginModel(
        scan_multiplier=cfg.span_scan_multiplier,
        vol_scan_pct=cfg.span_vol_scan_pct,
        short_option_minimum=cfg.span_short_option_minimum,
        risk_free_rate=risk_free_rate,
    )


#: The four things that can decide the count, as they appear in the entry
#: event and in ``deltahedger sweep --sizing``.
BIND_RISK = "risk"
BIND_GAMMA = "gamma"
BIND_CAPITAL = "capital"
BIND_MAX = "max_straddles"
BINDING_NAMES = (BIND_RISK, BIND_GAMMA, BIND_CAPITAL, BIND_MAX)


def describe_limits(limits: dict[str, int]) -> str:
    """``risk 30, gamma 38, capital 173`` -- what each constraint allowed."""
    return ", ".join(
        f"{name} {limits[name]}" for name in BINDING_NAMES if name in limits
    )


@dataclass(frozen=True)
class SizingResult:
    contracts: int
    margin_per_contract: float
    total_margin: float
    budget: float
    direction: int = 0
    reason: str = ""
    #: Which constraint decided the count -- one of ``BINDING_NAMES``.
    binding: str = ""
    #: What each constraint would have allowed on its own, so a size can be
    #: explained rather than inferred.
    limits: dict[str, int] = field(default_factory=dict)
    #: Dollars one straddle loses if its branch stop fires.
    risk_per_straddle: float = 0.0
    #: Delta units per point one straddle carries.
    gamma_per_straddle: float = 0.0

    @property
    def ok(self) -> bool:
        return self.contracts > 0

    @property
    def requirement_kind(self) -> str:
        return "debit" if self.direction > 0 else "margin"

    def describe_limits(self) -> str:
        """``risk 30, gamma 38, capital 173`` -- the count each one allowed."""
        return describe_limits(self.limits)


def size_straddles(
    equity: float,
    quote: StraddleQuote,
    future_price: float,
    direction: int,
    cfg: SizingConfig,
    source: RiskSource,
    model: MarginModel,
    stop_fraction: float | None = None,
) -> SizingResult:
    """How many straddles to trade: the smallest count any constraint allows.

    ``direction`` is the sign the GEX regime asked for: +1 buys the
    straddle, -1 sells it.  It changes what is being budgeted -- a debit
    against cash or margin against collateral -- and what a stop-out costs,
    but not the shape of the rule.

    ``stop_fraction`` is the fraction of the entry premium one straddle
    loses when its branch stop fires: ``long_stop_loss_pct`` on the long
    side, ``short_stop_loss_premium_multiple - 1`` on the short.  ``None``
    (or a non-positive value, meaning the stop is switched off) falls back
    to the per-straddle requirement, which bounds the same loss -- the
    debit is a long straddle's maximum loss, and the SPAN scan is the
    short's one-day adverse move.  See the module docstring for why the
    risk budget rather than the capital cap is what should normally bind.
    """
    if direction == 0:
        return SizingResult(0, 0.0, 0.0, 0.0, 0, "no direction to size")

    equity = max(equity, 0.0)
    budget = equity * cfg.buying_power_pct
    per_contract = model.straddle_requirement(quote, future_price, source, direction)
    kind = "debit" if direction > 0 else "margin"

    if per_contract <= 0.0:
        return SizingResult(
            0, per_contract, 0.0, budget, direction,
            f"the {kind} model returned a non-positive requirement",
        )

    premium = quote.price * source.option.multiplier
    risk_per_straddle = (
        stop_fraction * premium
        if stop_fraction is not None and stop_fraction > 0.0 and premium > 0.0
        else per_contract
    )
    gamma_per_straddle = quote.gamma * source.delta_units_per_contract(source.option)

    limits: dict[str, int] = {BIND_CAPITAL: int(budget // per_contract)}
    if cfg.risk_budget_pct is not None and risk_per_straddle > 0.0:
        limits[BIND_RISK] = int(equity * cfg.risk_budget_pct // risk_per_straddle)
    if cfg.gamma_ceiling_units_per_100k is not None and gamma_per_straddle > 0.0:
        ceiling = cfg.gamma_ceiling_units_per_100k * equity / 100_000.0
        limits[BIND_GAMMA] = int(ceiling // gamma_per_straddle)
    limits[BIND_MAX] = cfg.max_straddles

    # A disabled constraint is absent from ``limits`` rather than infinite,
    # so only the ones actually in play are compared. ``min`` keeps the
    # first of a tie, and ``BINDING_NAMES`` is the order they are worth
    # naming in: what a bad day costs, then how big the bet is, then what
    # the account can carry.
    binding = min(
        (name for name in BINDING_NAMES if name in limits), key=limits.__getitem__
    )
    contracts = limits[binding]

    common = dict(
        budget=budget, direction=direction, limits=limits,
        risk_per_straddle=risk_per_straddle, gamma_per_straddle=gamma_per_straddle,
    )
    if contracts < cfg.min_straddles:
        return SizingResult(
            0, per_contract, 0.0, reason=(
                f"{binding} allows {contracts} straddles, minimum is "
                f"{cfg.min_straddles} (${per_contract:,.0f} {kind}, "
                f"${risk_per_straddle:,.0f} at risk and "
                f"{gamma_per_straddle:.1f} delta units of gamma each; "
                f"{describe_limits(limits)})"
            ),
            binding=binding, **common,
        )
    return SizingResult(
        contracts=contracts,
        margin_per_contract=per_contract,
        total_margin=contracts * per_contract,
        reason=f"bound by {binding}",
        binding=binding,
        **common,
    )
