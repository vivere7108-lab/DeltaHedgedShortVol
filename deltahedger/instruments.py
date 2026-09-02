"""Instrument and risk-source definitions.

A *risk source* bundles everything the system needs to trade one underlying:
the future we get exposure from, the option series we trade volatility in,
and the instrument we delta-hedge with.  ES is the only one wired up today;
adding NQ, CL or anything else is a matter of appending a ``RiskSource`` to
``REGISTRY`` -- no strategy or engine code changes.

Delta units
-----------
Every delta in this codebase is expressed in *delta units*, where

    1 delta unit == 1% of one reference future contract

For ES the reference future is ES itself, so:

    long 1 ES future             -> +100 delta units
    long 1 MES future            -> + 10 delta units   (multiplier 5 vs 50)
    long 1 straddle at delta .38 -> + 38 delta units

Working in delta units (rather than raw contract deltas or dollar deltas)
keeps the hedging band -- ``0 +/- 10`` by default -- readable and makes the
micro/mini contract sizes fall out arithmetically.  It also makes the band
mean the same thing on any risk source: ``reference_multiplier`` rescales
NQ/MNQ to the same 100/10 relationship, so the band does not have to be
re-tuned per underlying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# One delta unit is 1% of the reference future.
DELTA_UNITS_PER_FUTURE = 100.0


@dataclass(frozen=True)
class ContractSpec:
    """A tradeable contract type (a future, or an option series)."""

    symbol: str
    sec_type: str  # "FUT" | "FOP"
    exchange: str
    currency: str = "USD"
    multiplier: float = 50.0  # dollars of P&L per 1.00 move in the underlying
    tick_size: float = 0.25  # minimum price increment, in underlying points
    commission_per_contract: float = 2.25  # USD, round turn charged per side
    trading_class: str | None = None

    @property
    def tick_value(self) -> float:
        """Dollar value of one minimum price increment."""
        return self.tick_size * self.multiplier


@dataclass(frozen=True)
class RiskSource:
    """One underlying we can run the short-vol strategy on."""

    name: str
    future: ContractSpec
    option: ContractSpec
    hedge: ContractSpec
    #: The future whose size defines a delta unit for this risk source.
    #: Normally the same as ``future``.
    reference_multiplier: float = 50.0
    #: Exchange calendar hints, all in the exchange's local timezone.
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    session_close: str = "16:00"
    #: Settlement time of the shortest-dated (daily) option series.
    option_expiry_time: str = "16:00"
    #: Typical strike spacing of the daily series, in underlying points.
    strike_increment: float = 5.0
    #: Approximate initial margin per short future contract, USD. Used by the
    #: heuristic margin model; the live path can query IBKR instead.
    future_initial_margin: float = 2455.0
    hedge_initial_margin: float = 245.5
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def delta_units_per_contract(self, spec: ContractSpec) -> float:
        """Delta units carried by one long contract of ``spec`` at delta 1.0."""
        return DELTA_UNITS_PER_FUTURE * (spec.multiplier / self.reference_multiplier)

    @property
    def hedge_quantum(self) -> float:
        """Delta units moved by one hedge contract. 10.0 for MES against ES."""
        return self.delta_units_per_contract(self.hedge)


ES = RiskSource(
    name="ES",
    future=ContractSpec(
        symbol="ES",
        sec_type="FUT",
        exchange="CME",
        multiplier=50.0,
        tick_size=0.25,
        commission_per_contract=2.25,
    ),
    option=ContractSpec(
        symbol="ES",
        sec_type="FOP",
        exchange="CME",
        multiplier=50.0,
        tick_size=0.05,
        commission_per_contract=2.32,
    ),
    hedge=ContractSpec(
        symbol="MES",
        sec_type="FUT",
        exchange="CME",
        multiplier=5.0,
        tick_size=0.25,
        commission_per_contract=0.62,
    ),
    reference_multiplier=50.0,
    strike_increment=5.0,
    future_initial_margin=2455.0,
    hedge_initial_margin=245.5,
    aliases=("ES", "SPX-ES", "EMINI"),
)

#: Registry of supported risk sources, keyed by uppercase name.
REGISTRY: dict[str, RiskSource] = {ES.name: ES}


def register(source: RiskSource) -> None:
    """Add a risk source so it can be selected by name from config."""
    REGISTRY[source.name.upper()] = source


def get_risk_source(name: str) -> RiskSource:
    key = name.strip().upper()
    if key in REGISTRY:
        return REGISTRY[key]
    for source in REGISTRY.values():
        if key in (alias.upper() for alias in source.aliases):
            return source
    known = ", ".join(sorted(REGISTRY)) or "<none>"
    raise KeyError(f"unknown risk source {name!r}; registered: {known}")
