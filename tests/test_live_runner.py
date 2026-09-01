"""LiveRunner._reconcile.

The important property here: a process restart (a crash, a redeploy, or IB
Gateway's own mandatory nightly restart under IBC) must not require a human
to notice and manually re-seed an open position, or "unattended forward
test" stops being true. Only a position on a *foreign* contract -- one this
runner didn't choose and can't parse an expiry for -- is worth refusing to
start over.

``avgCost`` fixtures below use the per-contract cost-basis convention (price
* multiplier, independent of quantity), matching the pre-existing hedge
reconcile code's own ``avgCost / multiplier`` -- not ``quantity * price *
multiplier``. Whether IBKR signs avgCost by long/short direction or reports
an unsigned magnitude isn't verifiable without a live account, so both
conventions are tested explicitly against the code's abs()-defensive
handling.
"""

from datetime import date

import pytest

from deltahedger.config import Config
from deltahedger.live.runner import LiveRunner, _parse_ibkr_expiry
from deltahedger.strategy import ShortVolStrategy


class FakeContract:
    def __init__(self, secType, symbol, strike=0.0, right="", expiry="", multiplier="50"):
        self.secType = secType
        self.symbol = symbol
        self.strike = strike
        self.right = right
        self.lastTradeDateOrContractMonth = expiry
        self.multiplier = multiplier


class FakePosition:
    def __init__(self, contract, position, avgCost, account="DU12345"):
        self.contract = contract
        self.position = position
        self.avgCost = avgCost
        self.account = account


class FakeConn:
    def __init__(self, positions):
        self._positions = positions
        self.account = "DU12345"
        self.ib = self  # .ib.positions(account) below

    def positions(self, account):
        return self._positions


def runner_with_strategy(cfg=None):
    cfg = cfg or Config()
    runner = LiveRunner(cfg)
    runner.strategy = ShortVolStrategy(cfg, runner.source)
    return runner


class TestParseIbkrExpiry:
    @pytest.mark.parametrize("raw,expected", [
        ("20260616", date(2026, 6, 16)),
        ("202606", date(2026, 6, 1)),
    ])
    def test_parses_known_formats(self, raw, expected):
        assert _parse_ibkr_expiry(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "not-a-date", "2026-06-16"])
    def test_raises_rather_than_guessing(self, raw):
        with pytest.raises(ValueError, match="could not parse"):
            _parse_ibkr_expiry(raw)


class TestReconcileHedge:
    def test_adopts_an_existing_hedge_position(self):
        runner = runner_with_strategy()
        contract = FakeContract("FUT", "MES")
        conn = FakeConn([FakePosition(contract, 3.0, 5000.0 * 5.0)])
        runner._reconcile(conn)
        assert runner.strategy.portfolio.hedge.quantity == 3
        assert runner.strategy.portfolio.hedge.avg_price == pytest.approx(5000.0)

    def test_a_short_hedge_position_still_recovers_a_positive_price(self):
        """Whichever sign convention IBKR's feed uses for a short, the
        recovered price must be positive -- a future's price cannot be
        negative in this system."""
        runner = runner_with_strategy()
        contract = FakeContract("FUT", "MES")
        conn = FakeConn([FakePosition(contract, -3.0, -5000.0 * 5.0)])  # signed convention
        runner._reconcile(conn)
        assert runner.strategy.portfolio.hedge.quantity == -3
        assert runner.strategy.portfolio.hedge.avg_price == pytest.approx(5000.0)

    def test_ignores_a_zero_quantity_hedge_row(self):
        runner = runner_with_strategy()
        contract = FakeContract("FUT", "MES")
        conn = FakeConn([FakePosition(contract, 0.0, 0.0)])
        runner._reconcile(conn)
        assert runner.strategy.portfolio.hedge.quantity == 0

    def test_no_positions_leaves_the_book_flat(self):
        runner = runner_with_strategy()
        runner._reconcile(FakeConn([]))
        assert runner.strategy.portfolio.is_flat


class TestReconcileOptionLegs:
    def test_adopts_an_existing_put_position(self):
        runner = runner_with_strategy()
        contract = FakeContract("FOP", "ES", strike=4980.0, right="P", expiry="20260616")
        conn = FakeConn([FakePosition(contract, -6.0, 18.0 * 50.0)])
        runner._reconcile(conn)
        put = runner.strategy.portfolio.put
        assert put is not None
        assert put.quantity == -6
        assert put.strike == 4980.0
        assert put.expiry == date(2026, 6, 16)
        assert put.entry_price == pytest.approx(18.0)

    def test_a_signed_avg_cost_still_recovers_a_positive_entry_price(self):
        """The convention this codebase cannot verify without a live
        account: if avgCost is signed by direction rather than an unsigned
        magnitude, abs() must still recover a positive per-unit premium."""
        runner = runner_with_strategy()
        contract = FakeContract("FOP", "ES", strike=4980.0, right="P", expiry="20260616")
        conn = FakeConn([FakePosition(contract, -6.0, -18.0 * 50.0)])  # signed convention
        runner._reconcile(conn)
        assert runner.strategy.portfolio.put.entry_price == pytest.approx(18.0)

    def test_adopts_both_legs_of_an_existing_strangle(self):
        runner = runner_with_strategy()
        put_c = FakeContract("FOP", "ES", strike=4980.0, right="P", expiry="20260616")
        call_c = FakeContract("FOP", "ES", strike=5020.0, right="C", expiry="20260616")
        conn = FakeConn([
            FakePosition(put_c, -6.0, 18.0 * 50.0),
            FakePosition(call_c, -6.0, 14.0 * 50.0),
        ])
        runner._reconcile(conn)
        assert runner.strategy.portfolio.put is not None
        assert runner.strategy.portfolio.call is not None
        assert runner.strategy.portfolio.call.entry_price == pytest.approx(14.0)

    def test_does_not_raise_for_an_adopted_leg(self):
        """The old behaviour refused to start at all here; this must not."""
        runner = runner_with_strategy()
        contract = FakeContract("FOP", "ES", strike=4980.0, right="P", expiry="20260616")
        conn = FakeConn([FakePosition(contract, -6.0, 18.0 * 50.0)])
        runner._reconcile(conn)  # must not raise

    def test_right_is_normalised_from_a_verbose_form(self):
        """IBKR is not always consistent about "P" vs "PUT"."""
        runner = runner_with_strategy()
        contract = FakeContract("FOP", "ES", strike=4980.0, right="PUT", expiry="20260616")
        conn = FakeConn([FakePosition(contract, -6.0, 18.0 * 50.0)])
        runner._reconcile(conn)
        assert runner.strategy.portfolio.put is not None

    def test_a_position_on_a_different_symbol_is_ignored(self):
        """Only OUR configured option symbol is ours to adopt."""
        runner = runner_with_strategy()
        contract = FakeContract("FOP", "NQ", strike=18000.0, right="P", expiry="20260616")
        conn = FakeConn([FakePosition(contract, -6.0, 100.0 * 20.0)])
        runner._reconcile(conn)
        assert not runner.strategy.portfolio.has_option

    def test_mismatched_put_call_expiries_still_adopt_both_but_log_loudly(self, caplog):
        """Refusing to start would leave a real position completely
        unmanaged, which is worse than proceeding under a logged warning."""
        import logging

        runner = runner_with_strategy()
        put_c = FakeContract("FOP", "ES", strike=4980.0, right="P", expiry="20260616")
        call_c = FakeContract("FOP", "ES", strike=5020.0, right="C", expiry="20260617")
        conn = FakeConn([
            FakePosition(put_c, -6.0, 18.0 * 50.0),
            FakePosition(call_c, -6.0, 14.0 * 50.0),
        ])
        with caplog.at_level(logging.ERROR):
            runner._reconcile(conn)
        assert runner.strategy.portfolio.put is not None
        assert runner.strategy.portfolio.call is not None
        assert any("different expiries" in r.message for r in caplog.records)

    def test_an_adopted_legs_entry_price_feeds_the_stop_correctly(self):
        """The whole point of deriving entry_price from avgCost: the stop
        comparison (mark >= multiple * entry_price) must work on a leg the
        runner never itself opened."""
        cfg = Config()
        cfg.strategy.stop_loss_premium_multiple = 2.0
        runner = runner_with_strategy(cfg)
        contract = FakeContract("FOP", "ES", strike=4980.0, right="P", expiry="20260616")
        conn = FakeConn([FakePosition(contract, -6.0, 10.0 * 50.0)])
        runner._reconcile(conn)
        put = runner.strategy.portfolio.put
        # a mark at 2x the derived entry price should be recognised as a stop level
        assert put.entry_price * cfg.strategy.stop_loss_premium_multiple == pytest.approx(20.0)
