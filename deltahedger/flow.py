"""Who was the aggressor, and therefore which side the dealer took.

The problem this module exists to solve
---------------------------------------
``gex.py`` computes dealer gamma from open interest and a *sign convention*:
dealers are assumed long the calls and short the puts, so ``call_sign`` is
``+1`` and ``put_sign`` is ``-1`` at every strike, all day, forever.  That
assumption is the single load-bearing input in the whole system -- get it
backwards and the strategy is confidently wrong in exactly the wrong
direction -- and it is not observable in the open-interest print.  Open
interest says how many contracts exist at a strike.  It says nothing about
who is long them.

The tape does say.  Every execution has an aggressor and a resting side, and
the dealer is on the resting side by construction: a customer who lifts the
offer bought the option *from* a dealer, who is now short it and short its
gamma; a customer who hits the bid sold the option *to* a dealer, who is now
long it.  So the sign at a strike is not a convention to be assumed but a
quantity to be measured, one execution at a time.

That is what this module does.  It classifies executions, accumulates the
signed dealer position implied by them per strike and per right, and hands
``GexCalculator`` a *measured* sign in place of the assumed one.

Four rules, in order of how much they actually know
---------------------------------------------------
Classification is a precedence chain, not a vote.  Each rule is tried only
when every rule above it declined, and every classification records which
rule produced it so a live read can be audited rather than trusted:

1. ``RULE_AGGRESSOR`` -- **the MDP 3.0 aggressor flag.**  CME's Market Data
   Platform 3.0 states the aggressor side outright in the trade summary
   message (tag 5797 ``AggressorSide``: 1 buy, 2 sell, 0 none).  This is not
   an inference and there is nothing to improve on it; when the feed carries
   the flag, the other three rules never run.  ``aggressor_from_mdp``
   converts the raw tag.
2. ``RULE_BOOK_DELTA`` -- **the MBO book-state change.**  MDP 3.0 is a
   market-by-order feed: it publishes individual orders in the queue rather
   than aggregated depth, so the incremental refresh that accompanies an
   execution says which side's resting liquidity was *removed*.  Liquidity
   consumed on the offer means an inbound order swept the ask, which means a
   customer bought and a dealer is short that strike's gamma.  This rule is
   what recovers a trade that printed at the midpoint of a stale top-of-book
   quote, where the Lee-Ready rules below have nothing to work with.
3. ``RULE_QUOTE`` -- **the Lee-Ready quote rule.**  Compare the execution
   price with the prevailing bid and ask: at or near the ask is
   buyer-initiated, at or near the bid is seller-initiated, and anything
   strictly off the midpoint takes the side of the quote it is nearer to.
   ``quote_tolerance_ticks`` is the "or near": an option that trades a tick
   inside a wide quote is still overwhelmingly an aggressor paying up, and
   requiring an exact touch would discard most of the tape.
4. ``RULE_TICK`` -- **the Lee-Ready tick test.**  For a trade exactly at the
   midpoint, or with no usable quote at all, compare against the previous
   trade in the same option: an uptick is buyer-initiated, a downtick is
   seller-initiated, and a zero-tick inherits the last non-zero direction.

A trade no rule can resolve is ``UNKNOWN`` and is counted but never signed.
Guessing on it would put fabricated positioning into the number that decides
which side the strategy takes, which is the failure this module exists to
remove rather than relocate.

From classified trades to a sign
--------------------------------
``DealerFlowBook`` accumulates, per ``(expiry, strike, right)``, the signed
contracts the dealer community took::

    dealer_position = seller_initiated_volume - buyer_initiated_volume

positive meaning dealers are long that option.  Dividing by the classified
volume gives a number in ``[-1, +1]`` on exactly the scale the old
``call_sign``/``put_sign`` lived on -- ``+1`` is "every classified contract
here was sold to a dealer", ``-1`` is "every one was bought from one", and
``0`` is two-way flow that left dealers flat.  ``StrikeDealerFlow.sign``
is that ratio.

The shrinkage, and why it is not here
-------------------------------------
A strike with four classified contracts has measured a sign, but it has not
measured it *well*, and the honest thing to do with a thin measurement is to
fall back towards the prior.  ``GexCalculator`` does that -- it blends
``StrikeDealerFlow.sign`` against ``gex.call_sign``/``gex.put_sign`` with a
weight that grows with classified volume -- and it does it there rather than
here on purpose: the ensemble gate re-prices the whole profile under
perturbed priors *and* perturbed trust in the tape, so the blend has to be
recomputed per ensemble member.  What this module owns is the measurement.
What the calculator owns is how much of the answer the measurement is
allowed to be.

Honest limits
-------------
1. **Flow is a flow; open interest is a stock.**  What is measured here is
   the sign of the contracts that traded while the process was watching, and
   it is applied to *all* the open interest at that strike.  For a 0DTE
   series -- where essentially the whole book is written the same session --
   that is very nearly exact.  For a series listed days ago it is an
   extrapolation from the part of the book that has traded to the part that
   has not, and the shrinkage weight is the only thing standing between a
   handful of prints and a whole strike's assumed positioning.
2. **The resting side is not always a dealer.**  A customer resting a limit
   order is passive too, and this module will book them as one.  What is
   measured is aggressor-versus-resting, which is a good proxy for
   customer-versus-dealer in listed options and is not the same thing.
3. **A classification is not a certainty.**  Rules 3 and 4 are inferences
   with a known error rate, worse for wide quotes and thin books, and the
   error is not symmetric across the two sides.  ``rule_counts`` is on the
   book so a read can be judged by *how* it was classified rather than only
   by what it concluded.
4. **Unseen flow is unmeasured, not absent.**  A process that started at
   noon has no morning tape, and nothing here can tell that apart from a
   quiet morning.  ``half_life_minutes`` decays what has been seen; it
   cannot conjure what was not.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Protocol, Sequence

log = logging.getLogger(__name__)

#: The three answers to "who initiated this trade?". Strings rather than an
#: enum so they land in a CSV, a log line and a journal record unchanged.
BUY = "buy"  # buyer-initiated: the customer lifted the offer, the dealer is SHORT
SELL = "sell"  # seller-initiated: the customer hit the bid, the dealer is LONG
UNKNOWN = "unknown"

#: Which rule produced a classification. Recorded per trade and counted per
#: book, because a profile classified entirely off the tick test deserves
#: less trust than one classified off MDP 3.0's own aggressor flag, and
#: nothing downstream can tell the difference unless it is written down.
RULE_AGGRESSOR = "aggressor_flag"
RULE_BOOK_DELTA = "book_delta"
RULE_QUOTE = "quote"
RULE_TICK = "tick"
RULE_NONE = "unclassified"
RULES = (RULE_AGGRESSOR, RULE_BOOK_DELTA, RULE_QUOTE, RULE_TICK, RULE_NONE)

#: MDP 3.0 tag 5797 (AggressorSide) in the trade summary message.
MDP_AGGRESSOR_NONE = 0
MDP_AGGRESSOR_BUY = 1
MDP_AGGRESSOR_SELL = 2

CALL = "C"
PUT = "P"

SECONDS_PER_MINUTE = 60.0


def aggressor_from_mdp(value: object) -> str:
    """Read MDP 3.0's ``AggressorSide`` (tag 5797) into ``BUY``/``SELL``.

    Accepts the raw numeric tag, the strings a CSV export of it tends to
    carry, and the values this module already uses, so a replay file can be
    written by whatever produced it rather than normalised first.  Anything
    unrecognised -- including the feed's own "no aggressor" (0), which is
    what an implied or administrative match reports -- is ``UNKNOWN``, and
    an unknown flag falls through to the next rule rather than being
    guessed at.
    """
    if value is None:
        return UNKNOWN
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "b", "buy", "bid", "buyer", "buyer_initiated", BUY):
            return BUY
        if text in ("2", "s", "sell", "ask", "offer", "seller", "seller_initiated", SELL):
            return SELL
        return UNKNOWN
    if isinstance(value, bool):  # bool is an int; refuse the ambiguity
        return UNKNOWN
    if isinstance(value, (int, float)):
        # NaN is what a CSV reader puts in an absent cell, and it is a
        # missing flag rather than a malformed one -- so it falls through to
        # the next rule exactly as a 0 does, and does not raise on the way.
        if not math.isfinite(value):
            return UNKNOWN
        if int(value) == MDP_AGGRESSOR_BUY:
            return BUY
        if int(value) == MDP_AGGRESSOR_SELL:
            return SELL
    return UNKNOWN


@dataclass(frozen=True)
class OptionTrade:
    """One execution in one option, with whatever the feed knew about it.

    Everything past ``size`` is optional because the four classification
    rules are a fallback chain: a feed that carries MDP 3.0's aggressor flag
    needs nothing else, one that carries the MBO book deltas needs no quote,
    one that carries only a quote gets Lee-Ready, and one that carries only
    prints gets the tick test.  The classifier uses the best it is given.
    """

    timestamp: datetime
    expiry: date
    strike: float
    right: str  # "C" or "P"
    price: float
    size: float
    #: The prevailing quote at the moment of the execution, for the Lee-Ready
    #: quote rule. ``None`` on either side disables it.
    bid: float | None = None
    ask: float | None = None
    #: MDP 3.0 tag 5797, already normalised to BUY/SELL/UNKNOWN.
    aggressor: str = UNKNOWN
    #: MBO: the change in resting size at each side of the book across this
    #: execution. Negative means liquidity was removed from that side, which
    #: is the side that was resting -- so the aggressor was the other one.
    bid_size_delta: float | None = None
    ask_size_delta: float | None = None

    def __post_init__(self) -> None:
        if self.right not in (CALL, PUT):
            raise ValueError(f"OptionTrade.right must be 'C' or 'P', got {self.right!r}")
        if self.size < 0:
            raise ValueError("OptionTrade.size must be >= 0")

    @property
    def key(self) -> tuple[date, float, str]:
        """What makes this a distinct option: expiry, strike, right."""
        return (self.expiry, float(self.strike), self.right)


@dataclass(frozen=True)
class Classification:
    """Which side initiated a trade, which rule said so, and why."""

    side: str
    rule: str
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.side in (BUY, SELL)

    @property
    def dealer_sign(self) -> float:
        """``-1`` when the dealer sold this trade, ``+1`` when they bought it.

        The whole point of the module in one property.  A customer buy is a
        dealer sale, and a dealer who has sold an option is short its gamma.
        """
        if self.side == BUY:
            return -1.0
        if self.side == SELL:
            return 1.0
        return 0.0


class TradeClassifier:
    """The four-rule precedence chain, with the tick test's memory.

    Stateful only because the tick test has to be: it compares an execution
    against the previous one *in the same option*, so the classifier keeps
    the last price and the last non-zero direction per contract.  Everything
    else is a pure function of the trade.
    """

    def __init__(
        self,
        tick_size: float,
        quote_tolerance_ticks: float = 1.0,
        use_aggressor_flag: bool = True,
        use_book_delta: bool = True,
        use_quote_rule: bool = True,
        use_tick_rule: bool = True,
    ):
        self.tick_size = float(tick_size)
        self.quote_tolerance_ticks = float(quote_tolerance_ticks)
        self.use_aggressor_flag = use_aggressor_flag
        self.use_book_delta = use_book_delta
        self.use_quote_rule = use_quote_rule
        self.use_tick_rule = use_tick_rule
        self._last_price: dict[tuple[date, float, str], float] = {}
        self._last_direction: dict[tuple[date, float, str], str] = {}

    def reset(self) -> None:
        self._last_price.clear()
        self._last_direction.clear()

    def classify(self, trade: OptionTrade) -> Classification:
        """Classify one trade, then record it for the next tick test.

        The tick-test memory is updated for *every* trade, including ones
        resolved by an earlier rule: the previous print is the previous
        print regardless of how it was classified, and skipping the update
        would compare the next midpoint trade against a stale price.
        """
        result = self._classify(trade)
        self._remember(trade, result)
        return result

    def classify_all(self, trades: Iterable[OptionTrade]) -> list[
        tuple[OptionTrade, Classification]
    ]:
        return [(trade, self.classify(trade)) for trade in trades]

    # -- the chain --------------------------------------------------------

    def _classify(self, trade: OptionTrade) -> Classification:
        if self.use_aggressor_flag:
            side = aggressor_from_mdp(trade.aggressor)
            if side != UNKNOWN:
                return Classification(
                    side, RULE_AGGRESSOR,
                    f"MDP 3.0 aggressor side is {side}",
                )

        if self.use_book_delta:
            book = self._book_delta_rule(trade)
            if book is not None:
                return book

        if self.use_quote_rule:
            quote = self._quote_rule(trade)
            if quote is not None:
                return quote

        if self.use_tick_rule:
            tick = self._tick_rule(trade)
            if tick is not None:
                return tick

        return Classification(
            UNKNOWN, RULE_NONE,
            "no aggressor flag, no book change, no usable quote and no prior "
            "print in this option",
        )

    def _book_delta_rule(self, trade: OptionTrade) -> Classification | None:
        """MBO: whichever side lost resting size was the passive one.

        Both deltas are required, and one of them has to have gone down
        while the other did not: a book where liquidity left *both* sides
        across the execution is a book that also cancelled or re-priced, and
        it no longer identifies the aggressor.
        """
        bid_delta, ask_delta = trade.bid_size_delta, trade.ask_size_delta
        if bid_delta is None or ask_delta is None:
            return None
        if not (math.isfinite(bid_delta) and math.isfinite(ask_delta)):
            return None
        if ask_delta < 0.0 <= bid_delta:
            return Classification(
                BUY, RULE_BOOK_DELTA,
                f"{-ask_delta:,.0f} lots of resting offer removed: the inbound "
                "order swept the ask",
            )
        if bid_delta < 0.0 <= ask_delta:
            return Classification(
                SELL, RULE_BOOK_DELTA,
                f"{-bid_delta:,.0f} lots of resting bid removed: the inbound "
                "order hit the bid",
            )
        return None

    def _quote_rule(self, trade: OptionTrade) -> Classification | None:
        """Lee-Ready: the execution price against the prevailing quote."""
        bid, ask = trade.bid, trade.ask
        if bid is None or ask is None:
            return None
        if not (math.isfinite(bid) and math.isfinite(ask)) or ask <= bid or bid < 0.0:
            return None

        # "At or near" the touch: a tolerance, capped at half the spread so
        # a quote narrower than the tolerance cannot classify a trade as
        # both. Inside that cap the two tests are mutually exclusive.
        tolerance = min(
            self.quote_tolerance_ticks * self.tick_size, (ask - bid) / 2.0
        )
        if trade.price >= ask - tolerance:
            return Classification(
                BUY, RULE_QUOTE,
                f"traded {trade.price:g} at the offer {ask:g}",
            )
        if trade.price <= bid + tolerance:
            return Classification(
                SELL, RULE_QUOTE,
                f"traded {trade.price:g} at the bid {bid:g}",
            )

        mid = 0.5 * (bid + ask)
        if trade.price > mid:
            return Classification(
                BUY, RULE_QUOTE,
                f"traded {trade.price:g} above the mid {mid:g}",
            )
        if trade.price < mid:
            return Classification(
                SELL, RULE_QUOTE,
                f"traded {trade.price:g} below the mid {mid:g}",
            )
        return None  # exactly at the mid: the tick test's job

    def _tick_rule(self, trade: OptionTrade) -> Classification | None:
        """Lee-Ready's fallback: this print against the last one."""
        key = trade.key
        previous = self._last_price.get(key)
        if previous is None:
            return None
        if trade.price > previous:
            return Classification(
                BUY, RULE_TICK, f"uptick from {previous:g} to {trade.price:g}"
            )
        if trade.price < previous:
            return Classification(
                SELL, RULE_TICK, f"downtick from {previous:g} to {trade.price:g}"
            )
        carried = self._last_direction.get(key)
        if carried in (BUY, SELL):
            return Classification(
                carried, RULE_TICK,
                f"zero tick at {trade.price:g}, carrying the last {carried} tick",
            )
        return None

    def _remember(self, trade: OptionTrade, result: Classification) -> None:
        key = trade.key
        previous = self._last_price.get(key)
        if previous is not None and trade.price != previous:
            self._last_direction[key] = BUY if trade.price > previous else SELL
        self._last_price[key] = trade.price


@dataclass
class StrikeDealerFlow:
    """Measured dealer positioning in the two options at one strike.

    ``call_dealer``/``put_dealer`` are signed contracts -- positive when
    dealers bought (customers sold), negative when dealers sold -- and the
    volumes are the classified contracts behind them.  ``unclassified`` is
    counted separately and deliberately excluded from both: it is the part
    of the tape the rules could not resolve, and folding it into either
    would make an unresolved trade look like a resolved one.
    """

    strike: float
    call_dealer: float = 0.0
    put_dealer: float = 0.0
    call_volume: float = 0.0
    put_volume: float = 0.0
    call_unclassified: float = 0.0
    put_unclassified: float = 0.0

    def sign(self, right: str) -> float | None:
        """Measured dealer sign in ``[-1, +1]``, or ``None`` if unmeasured.

        On exactly the scale ``gex.call_sign``/``gex.put_sign`` uses, so it
        can be blended against them without a conversion: ``+1`` is a dealer
        who bought every classified contract at this strike, ``-1`` one who
        sold every one, and ``0`` two-way flow that left them flat.
        """
        volume = self.volume(right)
        if volume <= 0.0:
            return None
        dealer = self.call_dealer if right == CALL else self.put_dealer
        return max(-1.0, min(1.0, dealer / volume))

    def volume(self, right: str) -> float:
        """Classified contracts behind this strike's sign for one right."""
        return self.call_volume if right == CALL else self.put_volume

    def unclassified(self, right: str) -> float:
        return self.call_unclassified if right == CALL else self.put_unclassified

    def scaled(self, factor: float) -> "StrikeDealerFlow":
        """This row with every quantity scaled -- how the decay is applied."""
        return StrikeDealerFlow(
            strike=self.strike,
            call_dealer=self.call_dealer * factor,
            put_dealer=self.put_dealer * factor,
            call_volume=self.call_volume * factor,
            put_volume=self.put_volume * factor,
            call_unclassified=self.call_unclassified * factor,
            put_unclassified=self.put_unclassified * factor,
        )

    def add(self, right: str, dealer: float, volume: float, unclassified: float) -> None:
        if right == CALL:
            self.call_dealer += dealer
            self.call_volume += volume
            self.call_unclassified += unclassified
        else:
            self.put_dealer += dealer
            self.put_volume += volume
            self.put_unclassified += unclassified


class OptionTradeFeed(Protocol):
    """Anything that can produce the executions in one expiry's options.

    Pulled per bar with the window since the last pull, so the backtest
    replaying a tape and the live runner draining a tick-by-tick buffer
    present the same interface -- which is what lets the classification code
    a forward walk runs be the code a historical run measured.
    """

    def trades(
        self, start: datetime, end: datetime, expiry: date
    ) -> Sequence[OptionTrade]: ...


@dataclass
class DealerFlowBook:
    """Classified flow, accumulated per expiry and strike.

    The book is the bridge between the tape and the GEX profile: trades go
    in through ``observe``, and ``rows(expiry)`` comes out in the shape
    ``ExpiryBook.flow`` wants.  It holds no opinion about how much the
    measurement should count for -- that is the calculator's shrinkage --
    only about what was measured.
    """

    classifier: TradeClassifier
    #: Contracts of classified flow whose weight halves. ``0`` disables the
    #: decay, which is the right default for a same-day series where the
    #: whole book was written in the window being watched; a longer-dated
    #: series is better served by letting last week's prints fade.
    half_life_minutes: float = 0.0
    _rows: dict[date, dict[float, StrikeDealerFlow]] = field(default_factory=dict)
    _decayed_at: dict[date, datetime] = field(default_factory=dict)
    _rule_counts: dict[str, float] = field(default_factory=dict)

    # -- writing ----------------------------------------------------------

    def observe(self, trade: OptionTrade) -> Classification:
        """Classify one trade and fold it into the expiry's row."""
        result = self.classifier.classify(trade)
        self._decay(trade.expiry, trade.timestamp)
        rows = self._rows.setdefault(trade.expiry, {})
        row = rows.setdefault(float(trade.strike), StrikeDealerFlow(float(trade.strike)))
        if result.known:
            row.add(trade.right, result.dealer_sign * trade.size, trade.size, 0.0)
        else:
            row.add(trade.right, 0.0, 0.0, trade.size)
        self._rule_counts[result.rule] = (
            self._rule_counts.get(result.rule, 0.0) + trade.size
        )
        return result

    def observe_all(self, trades: Iterable[OptionTrade]) -> list[Classification]:
        return [self.observe(trade) for trade in trades]

    # -- reading ----------------------------------------------------------

    def rows(
        self, expiry: date, moment: datetime | None = None
    ) -> tuple[StrikeDealerFlow, ...]:
        """This expiry's measured flow, decayed to ``moment``, by strike."""
        if moment is not None:
            self._decay(expiry, moment)
        rows = self._rows.get(expiry)
        if not rows:
            return ()
        return tuple(row for _, row in sorted(rows.items()))

    def classified_volume(self, expiry: date | None = None) -> float:
        """Contracts a rule actually resolved, for one expiry or all of them."""
        expiries = [expiry] if expiry is not None else list(self._rows)
        return sum(
            row.call_volume + row.put_volume
            for e in expiries
            for row in self._rows.get(e, {}).values()
        )

    def unclassified_volume(self, expiry: date | None = None) -> float:
        expiries = [expiry] if expiry is not None else list(self._rows)
        return sum(
            row.call_unclassified + row.put_unclassified
            for e in expiries
            for row in self._rows.get(e, {}).values()
        )

    def rule_counts(self) -> dict[str, float]:
        """Contracts classified by each rule, for auditing a live read."""
        return dict(self._rule_counts)

    def describe(self) -> str:
        classified = self.classified_volume()
        total = classified + self.unclassified_volume()
        if total <= 0.0:
            return "no option trades classified yet"
        parts = ", ".join(
            f"{rule} {self._rule_counts.get(rule, 0.0) / total:.0%}"
            for rule in RULES
            if self._rule_counts.get(rule, 0.0) > 0.0
        )
        return (
            f"{classified:,.0f} of {total:,.0f} contracts classified "
            f"({classified / total:.0%}): {parts}"
        )

    # -- housekeeping -----------------------------------------------------

    def prune(self, before: date) -> None:
        """Forget expiries that have passed; they hedge nothing now."""
        for expiry in [e for e in self._rows if e < before]:
            self._rows.pop(expiry, None)
            self._decayed_at.pop(expiry, None)

    def reset(self) -> None:
        self._rows.clear()
        self._decayed_at.clear()
        self._rule_counts.clear()
        self.classifier.reset()

    def _decay(self, expiry: date, moment: datetime) -> None:
        """Age this expiry's accumulator forward to ``moment``.

        Exponential decay applied to the accumulator itself rather than
        re-weighted per trade: halving every stored quantity is exactly
        equivalent and costs one multiplication per strike per update
        instead of one per trade ever seen.
        """
        if self.half_life_minutes <= 0.0:
            return
        last = self._decayed_at.get(expiry)
        self._decayed_at[expiry] = moment
        if last is None:
            return
        elapsed = (moment - last).total_seconds() / SECONDS_PER_MINUTE
        if elapsed <= 0.0:
            return
        factor = 0.5 ** (elapsed / self.half_life_minutes)
        rows = self._rows.get(expiry)
        if not rows:
            return
        self._rows[expiry] = {
            strike: row.scaled(factor) for strike, row in rows.items()
        }


def build_flow_book(cfg, source) -> DealerFlowBook:
    """A book wired from ``flow:`` config, for the option's tick size."""
    flow = cfg.flow
    return DealerFlowBook(
        classifier=TradeClassifier(
            tick_size=source.option.tick_size,
            quote_tolerance_ticks=flow.quote_tolerance_ticks,
            use_aggressor_flag=flow.use_aggressor_flag,
            use_book_delta=flow.use_book_delta,
            use_quote_rule=flow.use_quote_rule,
            use_tick_rule=flow.use_tick_rule,
        ),
        half_life_minutes=flow.half_life_minutes,
    )
