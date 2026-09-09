"""What ``broker/ibkr.py`` actually does, pinned against a fake gateway.

This module had no tests at all: 600 lines carrying the order path, the
margin probe and the position reconciliation, none of it reachable from a
backtest and none of it exercised in CI.  Every remaining known defect in
the system lives in here, so it is characterised *before* it is changed --
what follows is a record of today's behaviour, not an endorsement of it.

Tests that pin a defect say so and name the phase that fixes it, so the
change shows up as an edited assertion with a reason rather than as a
silent flip.  See the plan in the commit that added this file.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from deltahedger.broker.base import ExecutionError
from deltahedger.chain import OptionQuote, StraddleQuote
from deltahedger.config import Config
from deltahedger.pricing import black76

pytest.importorskip("ib_async", reason="the live path is an optional extra")

from deltahedger.broker.ibkr import (  # noqa: E402
    MAX_ORDER_CONTRACTS,
    IbkrConnection,
    IbkrExecution,
    WhatIfMarginModel,
    _is_paper_account,
    _pick_price,
    _round_to_tick,
)
from fakes import FakeIb, FakeTicker, OrderOutcome, fake_ticker  # noqa: E402

NY = ZoneInfo("America/New_York")
NOW = datetime(2025, 6, 10, 10, 0, tzinfo=NY)
EXPIRY = date(2025, 6, 10)
T = 6.4 / 24 / 365


@pytest.fixture
def cfg():
    config = Config()
    config.starting_equity = 250_000.0
    return config


@pytest.fixture
def conn(cfg):
    """A real ``IbkrConnection`` holding a fake handle.

    Real, so that contract qualification and the FOP cache are the ones
    production uses; ``connect()`` is skipped because it is the only part
    that needs a socket.
    """
    connection = IbkrConnection(cfg, cfg.source, ib=FakeIb(), account="DU1234567")
    (connection.future_contract,) = connection.ib.qualifyContracts(
        _future(cfg.source.future.symbol)
    )
    (connection.hedge_contract,) = connection.ib.qualifyContracts(
        _future(cfg.source.hedge.symbol)
    )
    return connection


def _future(symbol: str):
    from ib_async import Future

    return Future(symbol=symbol, exchange="CME", currency="USD")


def option(right: str = "C", strike: float = 5000.0, price: float = 8.0):
    greeks = black76(5000.0, strike, T, 0.15, 0.04, right)
    return OptionQuote(
        strike=strike, right=right, expiry=EXPIRY, price=price, iv=0.15,
        greeks=greeks, time_to_expiry=T,
    )


def straddle(price: float = 8.0) -> StraddleQuote:
    return StraddleQuote(
        strike=5000.0, expiry=EXPIRY, call=option("C", price=price),
        put=option("P", price=price), time_to_expiry=T,
    )


class TestPickPrice:
    """Which of a ticker's fields is trusted, and in what order."""

    def test_the_mid_is_preferred(self):
        assert _pick_price(fake_ticker(bid=99.0, ask=101.0)) == 100.0

    def test_a_crossed_book_falls_through_to_the_last(self):
        ticker = FakeTicker(bid=101.0, ask=99.0, last=50.0)
        assert _pick_price(ticker) == 50.0

    def test_a_missing_side_falls_through_to_the_last(self):
        assert _pick_price(FakeTicker(bid=99.0, ask=None, last=50.0)) == 50.0

    def test_the_close_is_the_last_resort(self):
        assert _pick_price(FakeTicker(last=None, close=42.0)) == 42.0

    def test_nothing_usable_reads_as_no_price(self):
        assert _pick_price(FakeTicker()) is None

    def test_a_non_positive_price_is_not_usable(self):
        """Zero and negative marks are rejected rather than traded on."""
        assert _pick_price(FakeTicker(last=0.0, close=-1.0)) is None

    def test_a_nan_is_not_usable(self):
        assert _pick_price(FakeTicker(last=float("nan"), close=7.0)) == 7.0


class TestAccountGate:
    def test_paper_accounts_begin_with_d(self):
        assert _is_paper_account("DU1234567")
        assert _is_paper_account("df123")
        assert not _is_paper_account("U1234567")


class TestOrderRouting:
    """Fills, sides, prices and fees, as the strategy receives them."""

    def test_a_filled_buy_comes_back_signed_and_priced(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: OrderOutcome(price=8.25, commission=2.32)
        execution = IbkrExecution(conn, cfg)
        fill = execution.execute_option(option(), 4, NOW)
        assert (fill.quantity, fill.price, fill.fees) == (4, 8.25, 2.32)
        assert fill.instrument == "option"

    def test_a_sale_comes_back_negative(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: OrderOutcome(price=8.25)
        fill = IbkrExecution(conn, cfg).execute_option(option(), -4, NOW)
        assert fill.quantity == -4
        assert conn.ib.placed[0][1].action == "SELL"

    def test_a_partial_fill_reports_what_filled(self, conn, cfg):
        """The strategy sizes its book off this number, so it has to be
        the exchange's count and not the order's."""
        conn.ib.outcome_for = lambda c, o: OrderOutcome(filled=3, price=8.0)
        fill = IbkrExecution(conn, cfg).execute_option(option(), -10, NOW)
        assert fill.quantity == -3

    def test_fees_are_summed_across_fills(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: OrderOutcome(commission=1.16)
        fill = IbkrExecution(conn, cfg).execute_option(option(), 2, NOW)
        assert fill.fees == pytest.approx(1.16)

    def test_a_zero_quantity_places_nothing(self, conn, cfg):
        assert IbkrExecution(conn, cfg).execute_option(option(), 0, NOW) is None
        assert conn.ib.placed == []

    def test_a_hedge_goes_to_the_hedge_contract(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: OrderOutcome(price=5000.25)
        fill = IbkrExecution(conn, cfg).execute_hedge(-7, 5000.0, NOW)
        assert fill.instrument == "hedge"
        assert conn.ib.placed[0][0] is conn.hedge_contract
        assert conn.ib.placed[0][1].totalQuantity == 7

    def test_dry_run_places_nothing_and_reports_the_reference(self, conn, cfg):
        fill = IbkrExecution(conn, cfg, dry_run=True).execute_option(option(), 3, NOW)
        assert conn.ib.placed == []
        assert (fill.quantity, fill.price, fill.note) == (3, 8.0, "dry-run")


class TestOrderType:
    def test_mkt_is_the_default(self, conn, cfg):
        cfg.ibkr.hedge_order_type = "MKT"
        IbkrExecution(conn, cfg).execute_hedge(1, 5000.0, NOW)
        assert conn.ib.placed[0][1].orderType == "MKT"

    def test_a_limit_crosses_the_spread_in_the_direction_that_fills(self, conn, cfg):
        cfg.ibkr.hedge_order_type = "LMT"
        cfg.ibkr.limit_cross_ticks = 2.0
        execution = IbkrExecution(conn, cfg)
        execution.execute_hedge(1, 5000.0, NOW)     # a buy pays up
        execution.execute_hedge(-1, 5000.0, NOW)    # a sale gives up
        buy, sell = (order for _, order in conn.ib.placed)
        tick = cfg.source.hedge.tick_size
        assert buy.lmtPrice == pytest.approx(5000.0 + 2 * tick)
        assert sell.lmtPrice == pytest.approx(5000.0 - 2 * tick)

    def test_the_limit_is_rounded_to_the_instrument_tick(self, conn, cfg):
        cfg.ibkr.hedge_order_type = "LMT"
        cfg.ibkr.limit_cross_ticks = 0.3  # lands off-tick on purpose
        IbkrExecution(conn, cfg).execute_hedge(1, 5000.0, NOW)
        limit = conn.ib.placed[0][1].lmtPrice
        assert limit == _round_to_tick(limit, cfg.source.hedge.tick_size)

    def test_the_hedge_order_type_also_governs_option_orders(self, conn, cfg):
        """Pinned because the name says otherwise.

        ``ibkr.hedge_order_type`` is the only order-type setting there is,
        and ``_send`` reads it for both instruments -- so switching hedges
        to limit orders silently switches the straddle legs too. Phase 3
        renames it; this records that today they cannot be set apart.
        """
        cfg.ibkr.hedge_order_type = "LMT"
        IbkrExecution(conn, cfg).execute_option(option(), 1, NOW)
        assert conn.ib.placed[0][1].orderType == "LMT"


class TestOrderSizeCheck:
    def test_an_oversized_option_order_is_refused_before_it_is_sent(self, conn, cfg):
        cfg.sizing.max_straddles = 10
        with pytest.raises(ExecutionError, match="refusing to send"):
            IbkrExecution(conn, cfg).execute_option(option(), 11, NOW)
        assert conn.ib.placed == []

    def test_an_oversized_hedge_order_is_refused(self, conn, cfg):
        cfg.hedge.max_hedge_contracts = 5
        with pytest.raises(ExecutionError, match="refusing to send"):
            IbkrExecution(conn, cfg).execute_hedge(-6, 5000.0, NOW)

    def test_the_hard_backstop_never_binds_at_the_shipped_config(self, conn, cfg):
        """The defect behind issue 5, stated as arithmetic.

        ``MAX_ORDER_CONTRACTS`` is described as a backstop against a sizing
        bug turning into a position nobody intended. It is applied as
        ``min(MAX_ORDER_CONTRACTS, max_straddles)`` and both are 500, so
        the config limit is always the binding one and the backstop can
        never fire. Phase 3 gives it something to catch.
        """
        assert cfg.sizing.max_straddles == MAX_ORDER_CONTRACTS
        conn.ib.outcome_for = lambda c, o: OrderOutcome(price=8.0)
        fill = IbkrExecution(conn, cfg).execute_option(option(), MAX_ORDER_CONTRACTS, NOW)
        assert fill.quantity == MAX_ORDER_CONTRACTS, "the backstop did not fire"


class TestAnOrderThatDoesNotFill:
    """The timeout path -- and the race that makes it dangerous."""

    STILL_WORKING = OrderOutcome(updates_before_done=None, filled=0, status="Submitted")

    def test_an_unfilled_order_is_cancelled_and_read_as_no_fill(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: self.STILL_WORKING
        execution = IbkrExecution(conn, cfg, fill_timeout=0.01)
        assert execution.execute_option(option(), 5, NOW) is None
        assert conn.ib.cancelled, "the working order was left on the book"
        assert conn.ib.filled_quantity() == 0, "nothing traded, correctly reported"

    def test_an_order_filling_after_the_cancel_is_still_reported_as_nothing(
        self, conn, cfg
    ):
        """The race behind issue 2, and the reason a book can drift.

        ``_send`` cancels and returns ``None`` without waiting for the
        cancel to be confirmed. IBKR is free to fill in the meantime, and
        it does. The strategy reads ``None`` as "the leg did not fill",
        believes it is flat, and tries again on the next poll -- five
        seconds later, at full size, with no limit on the retries, because
        ``_entries_this_session`` only counts entries that succeeded.

        Phase 3 makes this return the fill. Until then, this is what the
        exchange holds that the book does not.
        """
        conn.ib.outcome_for = lambda c, o: OrderOutcome(
            updates_before_done=None, filled=0, status="Submitted",
            fills_on_cancel=5, price=8.0,
        )
        execution = IbkrExecution(conn, cfg, fill_timeout=0.01)
        reported = execution.execute_option(option(), -5, NOW)

        assert reported is None, "today the strategy is told nothing happened"
        assert conn.ib.cancelled, "the order was cancelled"
        # ... and the exchange filled it anyway. Five short options the
        # strategy has no record of, and it will size a fresh entry on the
        # next poll as though the book were flat.
        assert conn.ib.filled_quantity() == -5

    def test_the_fill_timeout_does_not_bound_how_long_send_blocks(self, conn, cfg):
        """``fill_timeout`` is a gap between updates, not a deadline.

        The wait loop is written as "keep waiting while this trade is not
        done", and ``waitOnUpdate`` returns True for *any* traffic on the
        socket -- a tick on one of the option-chain subscriptions counts.
        So on a busy market a working order keeps the loop awake and
        ``_send`` blocks for as long as the market is noisy, whatever
        ``fill_timeout`` is set to. The whole poll loop is stalled behind
        it, which means no hedging while it lasts.

        Pinned rather than fixed: Phase 3 owns the wait loop.
        """
        conn.ib.idle_updates = 40
        conn.ib.outcome_for = lambda c, o: self.STILL_WORKING
        IbkrExecution(conn, cfg, fill_timeout=0.01).execute_option(option(), 5, NOW)
        assert conn.ib.idle_updates == 0, (
            "the loop should have consumed every unrelated update before "
            "giving up -- a single fill_timeout did not bound it"
        )

    def test_an_unconfirmed_cancel_is_also_read_as_no_fill(self, conn, cfg):
        """An unknown order state is not the same as a flat book.

        Nothing here distinguishes "cancelled, nothing traded" from "we do
        not know" -- both return ``None``, and the caller treats ``None``
        as flat.
        """
        conn.ib.outcome_for = lambda c, o: OrderOutcome(
            updates_before_done=None, filled=0, status="Submitted",
            cancel_confirms=False,
        )
        execution = IbkrExecution(conn, cfg, fill_timeout=0.01)
        assert execution.execute_option(option(), -5, NOW) is None


class TestWhatIfMargin:
    """The margin the account is actually charged, or a quiet guess."""

    class Heuristic:
        """A stand-in for the fallback model, loud about being used."""

        def __init__(self):
            self.short_calls = 0

        def straddle_requirement(self, quote, future_price, source, direction):
            if direction < 0:
                self.short_calls += 1
            return 1234.0

        def hedge_margin(self, source):
            return 99.0

    def test_a_short_straddle_is_probed_as_one_combo_not_two_orders(self, conn, cfg):
        """SPAN nets the legs, so probing them separately overstates the
        requirement and undersizes the book."""
        conn.ib.margin_change = 16_800.0
        model = WhatIfMarginModel(conn, self.Heuristic())
        model.straddle_requirement(straddle(), 5000.0, cfg.source, -1)

        assert len(conn.ib.whatif_orders) == 1
        contract, order = conn.ib.whatif_orders[0]
        assert contract.secType == "BAG"
        assert [leg.action for leg in contract.comboLegs] == ["SELL", "SELL"]
        assert order.action == "BUY", "the legs carry the sell side"

    def test_the_margin_returned_is_per_straddle(self, conn, cfg):
        conn.ib.margin_change = 33_600.0
        model = WhatIfMarginModel(conn, self.Heuristic(), probe_quantity=2)
        assert model.straddle_requirement(
            straddle(), 5000.0, cfg.source, -1
        ) == pytest.approx(16_800.0)

    def test_a_long_straddle_is_never_probed(self, conn, cfg):
        """A purchase has no margin change; reading zero as "free" would
        size the long branch without limit."""
        model = WhatIfMarginModel(conn, self.Heuristic())
        model.straddle_requirement(straddle(), 5000.0, cfg.source, +1)
        assert conn.ib.whatif_orders == []

    def test_a_silent_fallback_when_ibkr_returns_no_margin(self, conn, cfg):
        """The gap behind issue 4, and how the original bug stayed hidden.

        ``use_whatif_margin: true`` reads as "size against the real
        number". When the probe comes back empty -- routine for a CME FOP
        combo -- the model drops to the heuristic and logs a warning, and
        sizing proceeds as if nothing had changed. For as long as that
        heuristic was wrong the account was sized off it with no signal
        anywhere but one log line. Phase 4 makes it loud.
        """
        conn.ib.margin_change = None  # OrderState.initMarginChange stays ''
        fallback = self.Heuristic()
        model = WhatIfMarginModel(conn, fallback)
        assert model.straddle_requirement(
            straddle(), 5000.0, cfg.source, -1
        ) == 1234.0
        assert fallback.short_calls == 1

    def test_a_zero_or_negative_margin_change_is_not_trusted(self, conn, cfg):
        conn.ib.margin_change = 0.0
        fallback = self.Heuristic()
        assert WhatIfMarginModel(conn, fallback).straddle_requirement(
            straddle(), 5000.0, cfg.source, -1
        ) == 1234.0

    def test_a_probe_that_raises_falls_back_rather_than_blocking_sizing(
        self, conn, cfg
    ):
        def boom(*_):
            raise RuntimeError("no market data permissions")

        conn.ib.whatIfOrder = boom
        fallback = self.Heuristic()
        assert WhatIfMarginModel(conn, fallback).straddle_requirement(
            straddle(), 5000.0, cfg.source, -1
        ) == 1234.0

    def test_the_hedge_margin_is_delegated(self, conn, cfg):
        assert WhatIfMarginModel(conn, self.Heuristic()).hedge_margin(
            cfg.source
        ) == 99.0


class TestOptionContracts:
    def test_a_qualified_contract_is_cached(self, conn):
        first = conn.option_contract(EXPIRY, 5000.0, "C")
        second = conn.option_contract(EXPIRY, 5000.0, "C")
        assert first is second

    def test_the_two_rights_are_separate_contracts(self, conn):
        call = conn.option_contract(EXPIRY, 5000.0, "C")
        put = conn.option_contract(EXPIRY, 5000.0, "P")
        assert call is not put
        assert (call.right, put.right) == ("C", "P")

    def test_an_unlistable_strike_raises_rather_than_returning_none(self, conn):
        conn.ib.qualifyContracts = lambda *_: []
        with pytest.raises(ExecutionError, match="could not qualify"):
            conn.option_contract(EXPIRY, 4321.0, "C")
