"""A fake IBKR gateway, shared by the live-runner and broker tests.

``deltahedger.broker.ibkr`` is the only part of the system that a backtest
cannot exercise, and it is where the order path, the margin probe and the
position reconciliation all live.  Testing it needs something that behaves
like TWS without being TWS.

What is faked and what is not
-----------------------------
Only the ``IB`` *handle* is faked.  Everything it hands back -- ``Trade``,
``Order``, ``OrderStatus``, ``Contract``, ``Fill``, ``CommissionReport`` --
is the real ``ib_async`` type, so a test cannot pass by agreeing with a
guess about the API.  That matters here more than usual: the behaviour
under test is precisely how the broker code reads those objects, and the
subtleties that bite (``waitOnUpdate`` returning True for *any* network
update, ``initMarginChange`` arriving as an empty string, ``filled``
being a float) are properties of the real classes.

``ib_async`` is an optional dependency, so importing this module raises if
it is missing; test modules that use it call ``pytest.importorskip``.

Scripting an order
------------------
``FakeIb`` is a scriptable exchange rather than a fixed one.  Each order is
resolved by an ``OrderOutcome``, which says how much fills, at what price,
after how many update ticks, and -- the case the live path exists to
survive -- how much leaks through *after* a cancel has been sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ib_async import (  # noqa: F401 - re-exported for tests
    CommissionReport,
    Contract,
    Execution,
    Fill,
    Order,
    OrderState,
    OrderStatus,
    Trade,
    TradeLogEntry,
)

from deltahedger.gex import StrikeOpenInterest


@dataclass
class OrderOutcome:
    """What the fake exchange does with one order.

    ``filled`` is an absolute contract count, or ``None`` for "all of it".
    ``updates_before_done`` is how many ``waitOnUpdate`` calls pass before
    the trade reaches a done state -- ``None`` means it never does on its
    own, which is the order still working when the caller gives up waiting.
    ``fills_on_cancel`` is what the exchange fills *after* ``cancelOrder``
    has been sent: the race the live path has to survive, and the one that
    turns a "nothing happened" into a position nobody recorded.
    """

    filled: float | None = None
    price: float = 100.0
    status: str = "Filled"
    commission: float = 0.0
    updates_before_done: int | None = 0
    fills_on_cancel: float = 0.0
    cancel_confirms: bool = True
    #: An error code the gateway sends back on placement. ib_async turns
    #: any code it does not list as a warning into a local ``Cancelled``
    #: before the order is acknowledged -- 10349 ("Order TIF was set to
    #: DAY") is the informational one that did it in production; 201 is a
    #: genuine rejection. The fake reproduces the library's behaviour: the
    #: trade reads Cancelled with the code in its log, and, unless the code
    #: is a real rejection, the order goes on to fill as scripted.
    error_on_placement: int = 0
    #: How many ``waitOnUpdate`` calls pass before the commission report
    #: lands. The real gateway sends it after the ``Filled`` status, so a
    #: fill read the instant it is done carries no commission. ``None``
    #: means the report never comes.
    commission_after: int | None = 0


#: Fills everything immediately, which is what a healthy market looks like.
FILLS_IN_FULL = OrderOutcome()


@dataclass
class _Pending:
    trade: Trade
    outcome: OrderOutcome
    updates_left: int | None


class FakeIb:
    """Just enough ``ib_async.IB`` to drive the runner and the broker.

    ``outcome_for`` decides what happens to each order; it is handed the
    contract and the order so a test can treat the two straddle legs
    differently.  The default fills everything.
    """

    def __init__(
        self,
        drop_after: int | None = None,
        outcome_for: Callable[[Any, Order], OrderOutcome] | None = None,
        account_values: dict[str, float] | None = None,
        margin_change: str | float | None = None,
        idle_updates: int = 0,
    ):
        self.drop_after = drop_after
        self.sleeps = 0
        self._connected = False
        self.outcome_for = outcome_for or (lambda contract, order: FILLS_IN_FULL)
        self.account_values = account_values or {}
        self.margin_change = margin_change
        #: Updates arriving from *unrelated* subscriptions before the socket
        #: goes quiet. The real ``waitOnUpdate`` wakes on any of them, so
        #: this is what a busy market does to a loop written as "wait until
        #: this trade is done".
        self.idle_updates = idle_updates
        #: Ceiling on how long a quiet ``waitOnUpdate`` actually sleeps, so
        #: a test can use a production-sized timeout without paying for it.
        self.max_quiet_sleep = 0.05

        #: Every order the fake was asked to place, in order, and every
        #: cancel it was asked to send. What the broker *did*, as opposed
        #: to what it reported doing.
        self.placed: list[tuple[Any, Order]] = []
        self.trades: list[Trade] = []
        self.cancelled: list[Order] = []
        self.whatif_orders: list[tuple[Any, Order]] = []
        self.market_data: list[Any] = []
        self._pending: list[_Pending] = []
        #: Fills whose commission report is still on its way.
        self._commissions_due: list[tuple[Trade, int | None]] = []
        self._rejected: list[_Pending] = []
        self._next_order_id = 1
        self._positions: list[Any] = []
        self.tickers_for: Callable[[Any], Any] | None = None

    # -- connection ------------------------------------------------------

    def connect(self, *_, **__):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def isConnected(self):
        return self._connected

    def managedAccounts(self):
        return ["DU1234567"]

    def reqMarketDataType(self, *_):
        pass

    def sleep(self, _seconds):
        self.sleeps += 1
        if self.drop_after is not None and self.sleeps >= self.drop_after:
            self._connected = False  # the daily gateway restart

    # -- positions and account -------------------------------------------

    def positions(self, *_):
        return list(self._positions)

    def set_positions(self, positions: list[Any]) -> None:
        self._positions = list(positions)

    def accountValues(self, *_):
        rows = [
            _AccountValue(tag, str(value), "USD", "DU1234567")
            for tag, value in self.account_values.items()
        ]
        # The real one mixes currencies and non-numeric tags into the same
        # list, and a reader that does not filter picks up whichever came
        # last. Both are represented so a test can catch that.
        rows.append(_AccountValue("AccountType", "INDIVIDUAL", "", "DU1234567"))
        rows.append(_AccountValue("NetLiquidation", "1", "EUR", "DU1234567"))
        return rows

    def accountSummary(self, *_):
        return self.accountValues()

    # -- contracts and market data ---------------------------------------

    def qualifyContracts(self, *contracts):
        qualified = []
        for index, contract in enumerate(contracts, start=1):
            contract.conId = contract.conId or 1000 + index
            if not contract.lastTradeDateOrContractMonth:
                contract.lastTradeDateOrContractMonth = "20250620"
            contract.localSymbol = contract.localSymbol or (
                f"{contract.symbol}{contract.lastTradeDateOrContractMonth}"
            )
            qualified.append(contract)
        return qualified

    def reqTickers(self, *contracts):
        maker = self.tickers_for or (lambda c: fake_ticker())
        return [maker(contract) for contract in contracts]

    def reqMktData(self, contract, *_, **__):
        ticker = fake_ticker()
        ticker.contract = contract
        self.market_data.append(ticker)
        return ticker

    def cancelMktData(self, contract):
        pass

    # -- orders ----------------------------------------------------------

    def placeOrder(self, contract, order: Order) -> Trade:
        order.orderId = order.orderId or self._next_order_id
        self._next_order_id += 1
        self.placed.append((contract, order))

        outcome = self.outcome_for(contract, order)
        trade = Trade(
            contract=contract,
            order=order,
            orderStatus=OrderStatus(
                orderId=order.orderId, status="Submitted", remaining=order.totalQuantity
            ),
        )
        self.trades.append(trade)
        pending = _Pending(trade, outcome, outcome.updates_before_done)
        if outcome.error_on_placement:
            # What ib_async's wrapper.error does with a non-warning code on
            # a not-yet-done trade: status Cancelled, the code in the log,
            # nothing yet acknowledged by the gateway (permId 0).
            trade.orderStatus.status = "Cancelled"
            trade.log.append(TradeLogEntry(
                datetime(2025, 6, 10, 10, 0), "Cancelled",
                f"Error {outcome.error_on_placement}", outcome.error_on_placement,
            ))
            if outcome.error_on_placement == 201:
                self._rejected.append(pending)
                return trade
        if pending.updates_left == 0:
            # Filled on arrival, so it is never pending: leaving it in the
            # queue would let the next waitOnUpdate settle it a second time.
            self._settle(pending)
        else:
            self._pending.append(pending)
        return trade

    def cancelOrder(self, order: Order) -> None:
        self.cancelled.append(order)
        for pending in list(self._pending):
            if pending.trade.order is not order:
                continue
            outcome = pending.outcome
            if outcome.fills_on_cancel:
                # The order the broker gave up on, filling anyway.
                self._apply_fill(pending, outcome.fills_on_cancel, "Filled")
            elif outcome.cancel_confirms:
                self._gateway_status(pending.trade, "Cancelled")
            self._pending.remove(pending)
            return
        # Not working: held Inactive, or already settled. The gateway still
        # answers a cancel on it.
        for trade in self.trades:
            if trade.order is order and trade.orderStatus.status != "Filled":
                self._gateway_status(trade, "Cancelled")

    def _gateway_status(self, trade: Trade, status: str) -> None:
        """An ``orderStatus`` message from the gateway: the status, and a
        log entry with no error code -- which is how ib_async's own record
        tells a gateway-set status from one the library set itself."""
        trade.orderStatus.status = status
        trade.log.append(TradeLogEntry(datetime(2025, 6, 10, 10, 0), status, ""))

    def waitOnUpdate(self, timeout: float = 0) -> bool:
        """True if an update arrived, False on timeout -- as the real one.

        The distinction the broker code leans on is that False means "the
        socket stayed quiet for the whole timeout", not "this order is
        finished". The real ``waitOnUpdate`` wakes on any network traffic,
        a tick on an unrelated subscription included, so ``idle_updates``
        models a market busy enough to keep a wait-until-done loop awake
        while the order itself goes nowhere.
        """
        progressed = False
        for pending in list(self._pending):
            if pending.updates_left is None:
                continue  # still working; it will not settle by itself
            pending.updates_left -= 1
            progressed = True
            if pending.updates_left <= 0:
                self._settle(pending)
                self._pending.remove(pending)
        for due in list(self._commissions_due):
            trade, left = due
            if left is None:
                continue  # never arrives
            progressed = True
            if left <= 1:
                self._report_commissions(trade)
                self._commissions_due.remove(due)
            else:
                self._commissions_due[self._commissions_due.index(due)] = (trade, left - 1)
        if progressed:
            return True
        if self.idle_updates > 0:
            self.idle_updates -= 1
            return True
        # Quiet socket: the real one blocks for the whole timeout before
        # reporting the timeout. Sleeping here rather than returning at
        # once is what makes a wall-clock deadline observable in a test --
        # and what stops a caller that loops on this from spinning.
        time.sleep(min(max(timeout, 0.0), self.max_quiet_sleep))
        return False

    def whatIfOrder(self, contract, order: Order):
        self.whatif_orders.append((contract, order))
        if not order.tif:
            # A what-if with no TIF is answered with error 10349 rather than
            # an OrderState; ib_async then resolves the request to an empty
            # list. This is what every live probe got until the TIF was set.
            return []
        state = OrderState(status="PreSubmitted")
        if self.margin_change is not None:
            state.initMarginChange = str(self.margin_change)
        return state

    def filled_quantity(self) -> float:
        """Net contracts the exchange actually filled, signed by side.

        The broker's truth, as opposed to what the caller was told. A book
        that disagrees with this number is holding a position nobody
        recorded, which is the failure this whole module exists to catch.
        """
        total = 0.0
        for trade in self.trades:
            side = 1.0 if trade.order.action == "BUY" else -1.0
            total += side * float(trade.orderStatus.filled)
        return total

    # -- internals -------------------------------------------------------

    def _settle(self, pending: _Pending) -> None:
        outcome = pending.outcome
        wanted = float(pending.trade.order.totalQuantity)
        filled = wanted if outcome.filled is None else float(outcome.filled)
        self._apply_fill(pending, filled, outcome.status)

    def _apply_fill(self, pending: _Pending, filled: float, status: str) -> None:
        trade, outcome = pending.trade, pending.outcome
        wanted = float(trade.order.totalQuantity)
        trade.orderStatus.filled = filled
        trade.orderStatus.remaining = max(wanted - filled, 0.0)
        trade.orderStatus.avgFillPrice = outcome.price if filled else 0.0
        trade.orderStatus.permId = trade.orderStatus.permId or 900_000 + trade.order.orderId
        self._gateway_status(trade, status)
        if not filled:
            return
        trade.fills.append(
            Fill(
                contract=trade.contract,
                execution=Execution(
                    execId=f"exec-{trade.order.orderId}",
                    shares=filled,
                    price=outcome.price,
                ),
                commissionReport=CommissionReport(),
                time=datetime(2025, 6, 10, 10, 0),
            )
        )
        if outcome.commission_after == 0:
            self._report_commissions(trade)
        else:
            self._commissions_due.append((trade, outcome.commission_after))

    def _report_commissions(self, trade: Trade) -> None:
        """The commission report landing: ``dataclassUpdate`` in place, as
        the real wrapper does, so the fill's report gains an execId."""
        for fill in trade.fills:
            commission = self._commission_for(trade)
            fill.commissionReport.execId = fill.execution.execId
            fill.commissionReport.commission = commission

    def _commission_for(self, trade: Trade) -> float:
        for contract, order in self.placed:
            if order is trade.order:
                return self.outcome_for(contract, order).commission
        return 0.0


@dataclass
class _AccountValue:
    tag: str
    value: str
    currency: str
    account: str


class FakeConnection:
    """Stands in for IbkrConnection: connects, prices, never talks to TWS."""

    def __init__(self, cfg, source, drop_after=None, price=5000.0):
        self.cfg = cfg
        self.source = source
        self.account = "DU1234567"
        self.price = price
        self.ib = FakeIb(drop_after)
        self.connects = 0

    def __enter__(self):
        self.ib.connect()
        self.connects += 1
        return self

    def __exit__(self, *_):
        self.ib.disconnect()

    def future_price(self):
        if not self.ib.isConnected():
            raise ConnectionError("not connected")
        return self.price


class FakeOpenInterest:
    def open_interest(self, moment, future_price, expiry):
        return [
            StrikeOpenInterest(future_price + 5.0 * i, 4000.0, 200.0)
            for i in range(-20, 21)
        ]


class FakeTicker:
    """A ticker with only the fields the broker code reads."""

    def __init__(self, bid=None, ask=None, last=None, close=None, **extra):
        self.bid, self.ask, self.last, self.close = bid, ask, last, close
        self.contract = None
        self.modelGreeks = None
        for name, value in extra.items():
            setattr(self, name, value)

    def marketPrice(self):
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.last


def fake_ticker(bid=99.5, ask=100.5, **extra) -> FakeTicker:
    return FakeTicker(bid=bid, ask=ask, **extra)


def fake_position(sec_type: str, symbol: str, quantity: float, avg_cost: float = 0.0,
                  expiry: str = "20250610", strike: float = 5000.0, right: str = "P"):
    """One row of ``ib.positions()``, shaped the way the runner reads it."""
    contract = Contract(
        secType=sec_type,
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        strike=strike,
        right=right,
    )
    return _Position(contract, float(quantity), float(avg_cost))


@dataclass
class _Position:
    """One row of ``ib.positions()``: the fields the runner actually reads."""

    contract: Any
    position: float
    avgCost: float
    account: str = "DU1234567"
