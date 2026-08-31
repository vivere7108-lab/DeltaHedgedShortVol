"""Position accounting, especially the fill bookkeeping that a hedger
exercises hardest: repeatedly adding to, reducing, and flipping a position."""

from datetime import date, datetime, timezone

import pytest

from deltahedger.portfolio import HedgePosition, OptionPosition, Portfolio
from deltahedger.pricing import Greeks

NOW = datetime(2025, 6, 10, 9, 35, tzinfo=timezone.utc)


@pytest.fixture
def book(es):
    return Portfolio(starting_equity=100_000.0, source=es)


def short_put(quantity=-10, entry=5.0):
    return OptionPosition(
        strike=4980.0, expiry=date(2025, 6, 10), right="P",
        quantity=quantity, entry_price=entry, entry_time=NOW,
    )


class TestDeltaUnits:
    def test_a_short_put_carries_positive_delta(self, book):
        book.open_option(short_put(quantity=-1))
        greeks = Greeks(price=5.0, delta=-0.20, gamma=0.01, vega=0.4, theta=-2.0)
        assert book.option_delta_units(greeks) == pytest.approx(20.0)

    def test_scales_with_contract_count(self, book):
        book.open_option(short_put(quantity=-7))
        greeks = Greeks(5.0, -0.20, 0.01, 0.4, -2.0)
        assert book.option_delta_units(greeks) == pytest.approx(140.0)

    def test_one_mes_is_ten_delta_units(self, book):
        book.hedge.quantity = 1
        assert book.hedge_delta_units() == pytest.approx(10.0)
        book.hedge.quantity = -10
        assert book.hedge_delta_units() == pytest.approx(-100.0)

    def test_net_delta_combines_both_legs(self, book):
        book.open_option(short_put(quantity=-7))
        book.hedge.quantity = -12
        greeks = Greeks(5.0, -0.20, 0.01, 0.4, -2.0)
        assert book.net_delta_units(greeks) == pytest.approx(140.0 - 120.0)

    def test_gamma_is_reported_in_delta_units(self, book):
        book.open_option(short_put(quantity=-10))
        greeks = Greeks(5.0, -0.20, 0.02, 0.4, -2.0)
        assert book.option_gamma_units(greeks) == pytest.approx(-20.0)

    def test_no_position_is_no_delta(self, book):
        assert book.net_delta_units(None) == 0.0


class TestHedgeFills:
    def test_opening_sets_the_average_price(self):
        position = HedgePosition()
        assert position.apply_fill(5, 5000.0) == 0.0
        assert position.quantity == 5 and position.avg_price == 5000.0

    def test_adding_weights_the_average(self):
        position = HedgePosition()
        position.apply_fill(5, 5000.0)
        position.apply_fill(5, 5010.0)
        assert position.quantity == 10
        assert position.avg_price == pytest.approx(5005.0)

    def test_reducing_realises_the_closed_portion(self):
        position = HedgePosition()
        position.apply_fill(10, 5000.0)
        realised = position.apply_fill(-4, 5010.0)
        assert realised == pytest.approx(40.0)  # 4 contracts * 10 points
        assert position.quantity == 6
        assert position.avg_price == pytest.approx(5000.0)  # unchanged on the rest

    def test_closing_flat_realises_everything(self):
        position = HedgePosition()
        position.apply_fill(-8, 5000.0)  # short
        realised = position.apply_fill(8, 4990.0)
        assert realised == pytest.approx(80.0)  # short, market fell
        assert position.quantity == 0 and position.avg_price == 0.0

    def test_flipping_through_zero_rebases_the_average(self):
        position = HedgePosition()
        position.apply_fill(5, 5000.0)
        realised = position.apply_fill(-12, 5020.0)
        assert realised == pytest.approx(100.0)  # 5 closed at +20 points
        assert position.quantity == -7
        assert position.avg_price == pytest.approx(5020.0)

    def test_a_long_loses_when_the_market_falls(self):
        position = HedgePosition()
        position.apply_fill(3, 5000.0)
        assert position.apply_fill(-3, 4990.0) == pytest.approx(-30.0)


class TestValuation:
    def test_short_option_profits_when_the_mark_falls(self, book):
        book.open_option(short_put(quantity=-10, entry=5.0))
        assert book.unrealised_pnl(option_mark=2.0, hedge_mark=5000.0) == pytest.approx(
            1500.0  # 10 contracts * 3.00 * $50
        )

    def test_closing_moves_unrealised_into_realised(self, book):
        book.open_option(short_put(quantity=-10, entry=5.0))
        assert book.close_option(2.0) == pytest.approx(1500.0)
        assert book.option_realised == pytest.approx(1500.0)
        assert book.option is None

    def test_pnl_is_attributed_by_leg(self, book):
        book.open_option(short_put(quantity=-10, entry=5.0))
        book.close_option(2.0)
        book.apply_hedge_fill(2, 5000.0)
        book.apply_hedge_fill(-2, 4990.0)
        assert book.option_realised == pytest.approx(1500.0)
        assert book.hedge_realised == pytest.approx(-100.0)
        assert book.realised_pnl == pytest.approx(1400.0)

    def test_fees_reduce_equity(self, book):
        book.charge_fees(250.0)
        assert book.equity(None, 5000.0) == pytest.approx(99_750.0)

    def test_refuses_to_double_open(self, book):
        book.open_option(short_put())
        with pytest.raises(RuntimeError, match="already open"):
            book.open_option(short_put())

    def test_is_flat_reports_both_legs(self, book):
        assert book.is_flat
        book.hedge.quantity = 1
        assert not book.is_flat
