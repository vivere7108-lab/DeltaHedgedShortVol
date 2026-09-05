"""Capital requirements and buying-power based position sizing.

The number of straddles is driven by buying power, not by the delta target:
``buying_power_pct`` (80% by default -- the margin limit less a 20% buffer)
of portfolio equity is the budget, part of it is reserved for the hedge
leg, and the remainder divided by the per-straddle requirement gives the
count.  The delta band then absorbs whatever delta that position happens
to carry.

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
across the range (measured at 5000, 15 vol, a 250k account)::

    tenor   premium   SPAN margin   debit    straddles short / long
    0DTE      15.66        $1,699    $783                 15 / 25*
    2DTE      46.97        $1,324   $2,349                 19 / 11
    3DTE      56.45        $1,373   $2,822                 19 /  9
    5DTE      71.73        $1,502   $3,586                 17 /  7

    (* capped by max_straddles rather than by the budget)

The **long branch is a different story**, because its requirement is the
debit and the debit roughly doubles between the morning's 0DTE entry and
the afternoon's 1DTE roll.  So the same ``buying_power_pct`` buys a short
book of roughly the same size at either moment and a long book about half
as big at the roll, and the two branches do not carry comparable gamma.
That is a real asymmetry, it is a property of the requirement rather than
of the signal, and the backtest's band section reports median gamma and
band per branch so a regime comparison cannot mistake it for one.

A one-day scan is a conservative charge against a 0DTE position that will
be flat by the bell, and exactly the horizon a rolled 1DTE position is
carried over.

At the default 80% allocation the long branch spends more than half of
equity on a same-day straddle's debit.  That is the maximum loss on the
option leg, and it can be reached in a single session; the daily loss
limit in ``StrategyConfig`` is the rule that stops it getting there.

``RegTMarginModel`` is the 15%-of-notional equity-option rule.  It is
included because it is what most people reach for, and it overstates ES
option margin by roughly an order of magnitude; use it only to compare.

None of these are IBKR's number.  For live trading use
``broker.ibkr.WhatIfMarginModel``, which asks IBKR to price the margin
impact of the actual order before it is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SizingResult:
    contracts: int
    margin_per_contract: float
    total_margin: float
    budget: float
    option_budget: float
    direction: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.contracts > 0

    @property
    def requirement_kind(self) -> str:
        return "debit" if self.direction > 0 else "margin"


def size_straddles(
    equity: float,
    quote: StraddleQuote,
    future_price: float,
    direction: int,
    cfg: SizingConfig,
    source: RiskSource,
    model: MarginModel,
) -> SizingResult:
    """How many straddles the buying-power allocation supports.

    ``direction`` is the sign the GEX regime asked for: +1 buys the
    straddle, -1 sells it.  It changes what is being budgeted -- a debit
    against cash or margin against collateral -- but not how the budget is
    carved up, so the same reserve still stands behind the hedge leg in
    both cases.
    """
    if direction == 0:
        return SizingResult(0, 0.0, 0.0, 0.0, 0.0, 0, "no direction to size")

    budget = max(equity, 0.0) * cfg.buying_power_pct
    option_budget = budget * (1.0 - cfg.hedge_margin_reserve_pct)
    per_contract = model.straddle_requirement(quote, future_price, source, direction)
    kind = "debit" if direction > 0 else "margin"

    if per_contract <= 0.0:
        return SizingResult(
            0, per_contract, 0.0, budget, option_budget, direction,
            f"the {kind} model returned a non-positive requirement",
        )

    raw = int(option_budget // per_contract)
    contracts = min(raw, cfg.max_straddles)
    if contracts < cfg.min_straddles:
        return SizingResult(
            0, per_contract, 0.0, budget, option_budget, direction,
            f"buying power supports {raw} straddles, minimum is "
            f"{cfg.min_straddles} (${per_contract:,.0f} {kind} each vs "
            f"${option_budget:,.0f} available)",
        )
    return SizingResult(
        contracts=contracts,
        margin_per_contract=per_contract,
        total_margin=contracts * per_contract,
        budget=budget,
        option_budget=option_budget,
        direction=direction,
        reason="capped by max_straddles" if raw > contracts else "",
    )
