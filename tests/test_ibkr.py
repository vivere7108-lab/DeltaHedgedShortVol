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

import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from deltahedger.broker.base import ExecutionError, OrderStateUnknown
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
        cfg.ibkr.order_type = "MKT"
        IbkrExecution(conn, cfg).execute_hedge(1, 5000.0, NOW)
        assert conn.ib.placed[0][1].orderType == "MKT"

    def test_a_limit_crosses_the_spread_in_the_direction_that_fills(self, conn, cfg):
        cfg.ibkr.order_type = "LMT"
        cfg.ibkr.limit_cross_ticks = 2.0
        execution = IbkrExecution(conn, cfg)
        execution.execute_hedge(1, 5000.0, NOW)     # a buy pays up
        execution.execute_hedge(-1, 5000.0, NOW)    # a sale gives up
        buy, sell = (order for _, order in conn.ib.placed)
        tick = cfg.source.hedge.tick_size
        assert buy.lmtPrice == pytest.approx(5000.0 + 2 * tick)
        assert sell.lmtPrice == pytest.approx(5000.0 - 2 * tick)

    def test_the_limit_is_rounded_to_the_instrument_tick(self, conn, cfg):
        cfg.ibkr.order_type = "LMT"
        cfg.ibkr.limit_cross_ticks = 0.3  # lands off-tick on purpose
        IbkrExecution(conn, cfg).execute_hedge(1, 5000.0, NOW)
        limit = conn.ib.placed[0][1].lmtPrice
        assert limit == _round_to_tick(limit, cfg.source.hedge.tick_size)

    def test_one_setting_governs_both_instruments(self, conn, cfg):
        """And is now named for that.

        ``_send`` has always read a single setting for straddle legs and
        hedges alike, so the old ``hedge_order_type`` was misleading:
        switching hedges to limit orders silently switched the option legs
        too. The behaviour is unchanged; the name is honest.
        """
        cfg.ibkr.order_type = "LMT"
        IbkrExecution(conn, cfg).execute_option(option(), 1, NOW)
        assert conn.ib.placed[0][1].orderType == "LMT"

    def test_the_old_name_still_works_and_says_it_is_deprecated(self, caplog):
        """Renaming a config key that ships in every example file has to
        keep the old one working, or the rename is a breaking change."""
        with caplog.at_level(logging.WARNING, logger="deltahedger.config"):
            cfg = Config.from_dict({"ibkr": {"hedge_order_type": "LMT"}})
        assert cfg.ibkr.order_type == "LMT"
        assert "deprecated" in caplog.text
        assert "ibkr.order_type" in caplog.text

    def test_an_unknown_order_type_is_refused_at_load(self):
        with pytest.raises(ValueError, match="order_type"):
            Config.from_dict({"ibkr": {"order_type": "STP"}})


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

    def test_the_hard_backstop_is_inert_while_it_equals_the_config_limit(
        self, conn, cfg
    ):
        """Issue 5, stated as arithmetic and left as a config decision.

        ``MAX_ORDER_CONTRACTS`` is described as a backstop against a sizing
        bug turning into a position nobody intended. It is applied as
        ``min(MAX_ORDER_CONTRACTS, max_straddles)`` and both default to
        500, so the config limit is always the binding one and the
        backstop cannot fire.

        No number is invented here to fix that. The original margin bug
        sized 83 straddles, and no order-count ceiling anywhere near a
        plausible setting would have caught it -- the margin cross-check
        does, and that is phase 4. What phase 3 adds is that the condition
        is now said out loud at startup instead of being invisible; see
        ``deltahedger.config``. Lowering ``sizing.max_straddles`` is the
        owner's call.
        """
        assert cfg.sizing.max_straddles == MAX_ORDER_CONTRACTS
        conn.ib.outcome_for = lambda c, o: OrderOutcome(price=8.0)
        fill = IbkrExecution(conn, cfg).execute_option(option(), MAX_ORDER_CONTRACTS, NOW)
        assert fill.quantity == MAX_ORDER_CONTRACTS


class TestAnOrderThatDoesNotFill:
    """The timeout path: cancel, then find out what the order actually did."""

    STILL_WORKING = OrderOutcome(updates_before_done=None, filled=0, status="Submitted")

    def _execution(self, conn, cfg):
        return IbkrExecution(conn, cfg, fill_timeout=0.01, cancel_timeout=0.01)

    def test_an_unfilled_order_is_cancelled_and_read_as_no_fill(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: self.STILL_WORKING
        assert self._execution(conn, cfg).execute_option(option(), 5, NOW) is None
        assert conn.ib.cancelled, "the working order was left on the book"
        assert conn.ib.filled_quantity() == 0, "nothing traded, correctly reported"

    def test_an_order_filling_after_the_cancel_is_reported_as_the_fill_it_was(
        self, conn, cfg
    ):
        """The race behind issue 2, closed.

        ``_send`` used to cancel and return ``None`` without waiting for
        the cancel to be confirmed. IBKR is free to fill in the meantime,
        and it does. The strategy read ``None`` as "the leg did not fill",
        believed it was flat, and opened again on the next poll -- five
        seconds later, at full size, with nothing bounding the retries.

        The cancel is now awaited, and whatever filled before it landed is
        reported as the fill it was. The book and the account agree.
        """
        conn.ib.outcome_for = lambda c, o: OrderOutcome(
            updates_before_done=None, filled=0, status="Submitted",
            fills_on_cancel=5, price=8.0,
        )
        fill = self._execution(conn, cfg).execute_option(option(), -5, NOW)

        assert conn.ib.cancelled
        assert fill is not None, "the strategy was told nothing happened"
        assert fill.quantity == -5
        assert fill.quantity == conn.ib.filled_quantity(), (
            "the book and the exchange disagree about what was traded"
        )

    def test_a_partial_fill_before_the_cancel_is_reported(self, conn, cfg):
        conn.ib.outcome_for = lambda c, o: OrderOutcome(
            updates_before_done=None, filled=0, status="Submitted",
            fills_on_cancel=2, price=8.0,
        )
        fill = self._execution(conn, cfg).execute_option(option(), -5, NOW)
        assert fill.quantity == -2
        assert conn.ib.filled_quantity() == -2

    def test_an_unconfirmed_cancel_raises_rather_than_reporting_a_flat_book(
        self, conn, cfg
    ):
        """"We do not know" is not "nothing happened".

        Those two call for opposite responses: one can be retried, the
        other means the account may hold a position this process has no
        record of, and retrying is how one such position becomes several.
        Returning ``None`` made them indistinguishable to the caller, which
        then assumed the safe-looking one.
        """
        conn.ib.outcome_for = lambda c, o: OrderOutcome(
            updates_before_done=None, filled=0, status="Submitted",
            cancel_confirms=False,
        )
        with pytest.raises(OrderStateUnknown, match="do not assume it is flat"):
            self._execution(conn, cfg).execute_option(option(), -5, NOW)

    def test_the_unknown_state_is_distinguishable_from_a_refusal(self, conn, cfg):
        """A refusal happens before anything is sent, so the book is still
        correct and a retry is safe. They must not share a type."""
        assert issubclass(OrderStateUnknown, ExecutionError)
        cfg.sizing.max_straddles = 1
        with pytest.raises(ExecutionError) as refused:
            IbkrExecution(conn, cfg).execute_option(option(), 5, NOW)
        assert not isinstance(refused.value, OrderStateUnknown)

    def test_the_wait_is_bounded_by_the_clock_not_by_gaps_between_updates(
        self, conn, cfg
    ):
        """``fill_timeout`` is now a deadline.

        It used to be a gap between updates, and ``waitOnUpdate`` returns
        True for *any* traffic on the socket -- a tick on one of the
        option-chain subscriptions counts. So a working order on a busy
        market kept the loop awake for as long as the market was noisy,
        with the whole poll loop stalled behind it and nothing being
        hedged meanwhile.
        """
        conn.ib.outcome_for = lambda c, o: self.STILL_WORKING
        conn.ib.idle_updates = 10**9  # a market that never goes quiet
        execution = IbkrExecution(conn, cfg, fill_timeout=0.2, cancel_timeout=0.2)

        started = time.monotonic()
        assert execution.execute_option(option(), 5, NOW) is None
        elapsed = time.monotonic() - started
        assert elapsed < 2.0, f"blocked for {elapsed:.1f}s on a 0.2s timeout"
        assert conn.ib.idle_updates > 0, "it drained the market instead of the clock"


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

    def test_the_fallback_is_used_but_said_out_loud(self, conn, cfg, caplog):
        """How the original bug stayed hidden.

        ``use_whatif_margin: true`` reads as "size against the real
        number". When the probe comes back empty -- routine for a CME FOP
        combo -- the model drops to the heuristic, and for as long as that
        heuristic was wrong the account was sized off it with no signal
        anywhere but one WARNING line among the ordinary ones.

        Still falling back, because refusing to trade on a failed probe
        would mean never trading on some accounts. But at ERROR, naming
        the estimate, and pointing at the account figure that can check it.
        """
        conn.ib.margin_change = None  # OrderState.initMarginChange stays ''
        fallback = self.Heuristic()
        model = WhatIfMarginModel(conn, fallback)
        with caplog.at_level(logging.ERROR, logger="deltahedger.broker.ibkr"):
            assert model.straddle_requirement(
                straddle(), 5000.0, cfg.source, -1
            ) == 1234.0
        assert fallback.short_calls >= 1
        assert "not on IBKR" in caplog.text
        assert "FullInitMarginReq" in caplog.text

    def test_the_loud_fallback_does_not_repeat_forever(self, conn, cfg, caplog):
        """A probe that never works would otherwise print the same line on
        every entry, which is how a real warning stops being read."""
        conn.ib.margin_change = None
        model = WhatIfMarginModel(conn, self.Heuristic())
        with caplog.at_level(logging.ERROR, logger="deltahedger.broker.ibkr"):
            for _ in range(10):
                model.straddle_requirement(straddle(), 5000.0, cfg.source, -1)
        assert caplog.text.count("not on IBKR") <= 3

    def test_a_model_far_from_the_brokers_number_is_reported(
        self, conn, cfg, caplog
    ):
        """IBKR's figure is used either way; the comparison is a check on
        the *model*, which is what the backtest sized against and what the
        live path falls back to when a probe fails."""
        conn.ib.margin_change = 16_800.0  # against a fallback that says 1234
        model = WhatIfMarginModel(conn, self.Heuristic())
        with caplog.at_level(logging.WARNING, logger="deltahedger.broker.ibkr"):
            used = model.straddle_requirement(straddle(), 5000.0, cfg.source, -1)
        assert used == pytest.approx(16_800.0), "IBKR's number is the one used"
        assert "margin model is off by" in caplog.text

    def test_a_model_close_to_the_brokers_number_is_not_reported(
        self, conn, cfg, caplog
    ):
        conn.ib.margin_change = 1_500.0  # within a factor of two of 1234
        model = WhatIfMarginModel(conn, self.Heuristic())
        with caplog.at_level(logging.WARNING, logger="deltahedger.broker.ibkr"):
            model.straddle_requirement(straddle(), 5000.0, cfg.source, -1)
        assert "margin model is off by" not in caplog.text

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
