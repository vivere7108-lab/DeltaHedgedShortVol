"""Margin models and buying-power based position sizing.

The number of puts we sell is driven by buying power, not by the delta
target: ``buying_power_pct`` (15% by default) of portfolio equity is the
margin budget, part of it is reserved for the hedge leg, and the remainder
divided by per-contract margin gives the contract count.  The delta band
then absorbs whatever delta that position happens to carry.

Three margin models ship here.

``SpanScanMarginModel`` is the default and the only one appropriate for
futures options.  It reproduces CME SPAN's risk-array method: reprice the
position across a grid of price and volatility scenarios and charge the
worst loss.  Because we already have Black-76, the scenarios can be priced
exactly rather than approximated, so the model captures the thing that
matters most for a short 0DTE put -- margin exploding as the strike comes
into range of the scan.

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

from .chain import OptionQuote
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
    """Initial margin required to carry a position, in USD."""

    def short_option_margin(
        self, quote: OptionQuote, future_price: float, source: RiskSource
    ) -> float: ...

    def combined_margin(
        self, quotes: list[OptionQuote], future_price: float, source: RiskSource
    ) -> float:
        """Margin for several short legs held together (e.g. a strangle).

        Must degenerate to ``short_option_margin`` for a single quote.
        """
        ...

    def hedge_margin(self, source: RiskSource) -> float: ...


@dataclass
class FixedMarginModel:
    """Flat margin per contract. Simple, and easy to stress."""

    per_option_contract: float
    per_hedge_contract: float

    def short_option_margin(
        self, quote: OptionQuote, future_price: float, source: RiskSource
    ) -> float:
        return self.per_option_contract

    def combined_margin(
        self, quotes: list[OptionQuote], future_price: float, source: RiskSource
    ) -> float:
        """No offsetting benefit modelled -- flat per-leg cost summed."""
        return self.per_option_contract * len(quotes)

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

    def _scenario_losses(
        self, quote: OptionQuote, future_price: float, source: RiskSource
    ) -> list[float]:
        """Loss on one short leg at each of the 16 SPAN scenarios, in order."""
        scan = self.price_scan_range(source)
        mult = source.option.multiplier
        entry_value = quote.price
        losses = []
        for price_frac, vol_frac, weight in SPAN_SCENARIOS:
            scenario_future = max(future_price + price_frac * scan, 1e-9)
            scenario_vol = max(quote.iv * (1.0 + vol_frac * self.vol_scan_pct), 1e-6)
            scenario_value = black76(
                scenario_future,
                quote.strike,
                quote.time_to_expiry,
                scenario_vol,
                self.risk_free_rate,
                quote.right,
            ).price
            losses.append((scenario_value - entry_value) * mult * weight)
        return losses

    def short_option_margin(
        self, quote: OptionQuote, future_price: float, source: RiskSource
    ) -> float:
        # We are short, so a scenario that raises the option's value is a loss.
        worst_loss = max(self._scenario_losses(quote, future_price, source))
        return max(worst_loss, self.short_option_minimum)

    def combined_margin(
        self, quotes: list[OptionQuote], future_price: float, source: RiskSource
    ) -> float:
        """SPAN margin for several short legs scanned together.

        Sums each leg's loss *within* a scenario before taking the worst
        across scenarios, rather than summing each leg's own worst-case
        margin.  A put and a call cannot both be maximally hurt by the same
        price move, so scanning them jointly recovers the capital
        efficiency real portfolio margining gives a strangle over two
        independently-margined naked legs -- summing single-leg margins
        would double-count a scenario that can only happen once.
        """
        if not quotes:
            return 0.0
        per_leg = [self._scenario_losses(q, future_price, source) for q in quotes]
        worst_combined = max(sum(scenario) for scenario in zip(*per_leg))
        return max(worst_combined, self.short_option_minimum)

    def hedge_margin(self, source: RiskSource) -> float:
        return source.hedge_initial_margin


@dataclass
class RegTMarginModel:
    """Equity-option style margin. Wrong for futures; kept for comparison.

        margin = premium + max(a*futures_notional - otm, b*strike_notional)
    """

    a: float = 0.15
    b: float = 0.10

    def short_option_margin(
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

    def combined_margin(
        self, quotes: list[OptionQuote], future_price: float, source: RiskSource
    ) -> float:
        """The standard equity-style combination rule: margin the worse
        single leg, then add the other leg's premium on top (only one side
        of a strangle can be assigned against at once, but both premiums
        were collected)."""
        if not quotes:
            return 0.0
        if len(quotes) == 1:
            return self.short_option_margin(quotes[0], future_price, source)
        margins = [self.short_option_margin(q, future_price, source) for q in quotes]
        premiums = [q.price * source.option.multiplier for q in quotes]
        worst = max(range(len(quotes)), key=lambda i: margins[i])
        other_premiums = sum(p for i, p in enumerate(premiums) if i != worst)
        return margins[worst] + other_premiums

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
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.contracts > 0


def size_short_option_position(
    equity: float,
    quotes: list[OptionQuote],
    future_price: float,
    cfg: SizingConfig,
    source: RiskSource,
    model: MarginModel,
) -> SizingResult:
    """How many units the buying-power allocation supports.

    A "unit" is one contract of *every* quote in ``quotes`` together --
    for a single put that's the familiar one-contract-per-unit case; for a
    strangle (put + call) it's one of each, sized off their combined
    margin so the strangle's real capital efficiency (see
    ``MarginModel.combined_margin``) is what drives the contract count,
    not each leg's margin summed independently.
    """
    budget = max(equity, 0.0) * cfg.buying_power_pct
    option_budget = budget * (1.0 - cfg.hedge_margin_reserve_pct)
    per_contract = model.combined_margin(quotes, future_price, source)

    if per_contract <= 0.0:
        return SizingResult(
            0, per_contract, 0.0, budget, option_budget,
            "margin model returned a non-positive requirement",
        )

    raw = int(option_budget // per_contract)
    contracts = min(raw, cfg.max_short_contracts)
    if contracts < cfg.min_short_contracts:
        return SizingResult(
            0, per_contract, 0.0, budget, option_budget,
            f"buying power supports {raw} contracts, minimum is "
            f"{cfg.min_short_contracts} (${per_contract:,.0f} margin each vs "
            f"${option_budget:,.0f} available)",
        )
    return SizingResult(
        contracts=contracts,
        margin_per_contract=per_contract,
        total_margin=contracts * per_contract,
        budget=budget,
        option_budget=option_budget,
        reason="capped by max_short_contracts" if raw > contracts else "",
    )


def size_short_puts(
    equity: float,
    quote: OptionQuote,
    future_price: float,
    cfg: SizingConfig,
    source: RiskSource,
    model: MarginModel,
) -> SizingResult:
    """Single-put alias of ``size_short_option_position``, kept for existing callers."""
    return size_short_option_position(equity, [quote], future_price, cfg, source, model)
