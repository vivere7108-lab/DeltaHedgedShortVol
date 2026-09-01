"""Position accounting, especially the fill bookkeeping that a hedger
exercises hardest: repeatedly adding to, reducing, and flipping a position,
and now two option legs (put + call) held and closed together."""

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


def short_call(quantity=-10, entry=4.0):
    return OptionPosition(
        strike=5020.0, expiry=date(2025, 6, 10), right="C",
        quantity=quantity, entry_price=entry, entry_time=NOW,
    )


PUT_GREEKS = Greeks(price=5.0, delta=-0.20, gamma=0.01, vega=0.4, theta=-2.0)
CALL_GREEKS = Greeks(price=4.0, delta=0.22, gamma=0.012, vega=0.35, theta=-1.8)


class TestDeltaUnits:
    def test_a_short_put_carries_positive_delta(self, book):
        book.open_leg(short_put(quantity=-1))
        assert book.option_delta_units({"P": PUT_GREEKS}) == pytest.approx(20.0)

    def test_scales_with_contract_count(self, book):
        book.open_leg(short_put(quantity=-7))
        assert book.option_delta_units({"P": PUT_GREEKS}) == pytest.approx(140.0)

    def test_one_mes_is_ten_delta_units(self, book):
        book.hedge.quantity = 1
        assert book.hedge_delta_units() == pytest.approx(10.0)
        book.hedge.quantity = -10
        assert book.hedge_delta_units() == pytest.approx(-100.0)

    def test_net_delta_combines_the_option_and_the_hedge(self, book):
        book.open_leg(short_put(quantity=-7))
        book.hedge.quantity = -12
        assert book.net_delta_units({"P": PUT_GREEKS}) == pytest.approx(140.0 - 120.0)

    def test_gamma_is_reported_in_delta_units(self, book):
        book.open_leg(short_put(quantity=-10))
        greeks = Greeks(5.0, -0.20, 0.02, 0.4, -2.0)
        assert book.option_gamma_units({"P": greeks}) == pytest.approx(-20.0)

    def test_no_position_is_no_delta(self, book):
        assert book.net_delta_units({}) == 0.0


class TestTwoLegs:
    """A put and a call held together (a strangle)."""

    def test_both_legs_carry_delta_at_once(self, book):
        book.open_leg(short_put(quantity=-10))
        book.open_leg(short_call(quantity=-10))
        greeks = {"P": PUT_GREEKS, "C": CALL_GREEKS}
        expected = -10 * PUT_GREEKS.delta * 100 + -10 * CALL_GREEKS.delta * 100
        assert book.option_delta_units(greeks) == pytest.approx(expected)

    def test_a_roughly_symmetric_strangle_is_close_to_delta_neutral(self, book):
        """The put and call deltas partly cancel before hedging -- the
        hedger, not the strangle's own shape, is what holds net delta at
        the target."""
        book.open_leg(short_put(quantity=-10))
        book.open_leg(short_call(quantity=-10))
        greeks = {"P": PUT_GREEKS, "C": CALL_GREEKS}
        naked_put_only = -10 * PUT_GREEKS.delta * 100
        assert abs(book.option_delta_units(greeks)) < abs(naked_put_only)

    def test_put_and_call_properties(self, book):
        put, call = short_put(), short_call()
        book.open_leg(put)
        book.open_leg(call)
        assert book.put is put
        assert book.call is call

    def test_a_second_put_cannot_open_while_one_is_already_open(self, book):
        book.open_leg(short_put())
        with pytest.raises(RuntimeError, match="already open"):
            book.open_leg(short_put())

    def test_combined_credit_sums_both_legs(self, book):
        book.open_leg(short_put(quantity=-6, entry=18.0))
        book.open_leg(short_call(quantity=-6, entry=20.0))
        assert book.combined_credit_received() == pytest.approx(6 * (18.0 + 20.0) * 50)

    def test_combined_close_value_sums_both_legs(self, book):
        book.open_leg(short_put(quantity=-6, entry=18.0))
        book.open_leg(short_call(quantity=-6, entry=20.0))
        value = book.combined_close_value({"P": 30.0, "C": 5.0})
        assert value == pytest.approx(6 * (30.0 + 5.0) * 50)

    def test_combined_close_value_is_none_if_a_mark_is_missing(self, book):
        book.open_leg(short_put())
        book.open_leg(short_call())
        assert book.combined_close_value({"P": 3.0}) is None  # call mark missing

    def test_close_all_legs_realises_both_and_flattens(self, book):
        book.open_leg(short_put(quantity=-10, entry=5.0))
        book.open_leg(short_call(quantity=-10, entry=4.0))
        pnl = book.close_all_legs({"P": 2.0, "C": 1.0})
        assert pnl == pytest.approx(10 * (5.0 - 2.0) * 50 + 10 * (4.0 - 1.0) * 50)
        assert not book.has_option
        assert book.put is None and book.call is None

    def test_close_all_legs_leaves_an_unpriced_leg_open(self, book):
        book.open_leg(short_put(quantity=-10, entry=5.0))
        book.open_leg(short_call(quantity=-10, entry=4.0))
        book.close_all_legs({"P": 2.0})  # call has no exit price
        assert book.put is None
        assert book.call is not None

    def test_close_one_leg_independently(self, book):
        book.open_leg(short_put(quantity=-10, entry=5.0))
        book.open_leg(short_call(quantity=-10, entry=4.0))
        pnl = book.close_leg("P", 2.0)
        assert pnl == pytest.approx(10 * (5.0 - 2.0) * 50)
        assert book.put is None
        assert book.call is not None

    def test_has_option_is_true_with_either_leg_alone(self, book):
        assert not book.has_option
        book.open_leg(short_put())
        assert book.has_option


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
        book.open_leg(short_put(quantity=-10, entry=5.0))
        assert book.unrealised_pnl({"P": 2.0}, hedge_mark=5000.0) == pytest.approx(
            1500.0  # 10 contracts * 3.00 * $50
        )

    def test_closing_moves_unrealised_into_realised(self, book):
        book.open_leg(short_put(quantity=-10, entry=5.0))
        assert book.close_leg("P", 2.0) == pytest.approx(1500.0)
        assert book.option_realised == pytest.approx(1500.0)
        assert book.put is None

    def test_pnl_is_attributed_by_family(self, book):
        book.open_leg(short_put(quantity=-10, entry=5.0))
        book.close_leg("P", 2.0)
        book.apply_hedge_fill(2, 5000.0)
        book.apply_hedge_fill(-2, 4990.0)
        assert book.option_realised == pytest.approx(1500.0)
        assert book.hedge_realised == pytest.approx(-100.0)
        assert book.realised_pnl == pytest.approx(1400.0)

    def test_option_realised_combines_both_legs(self, book):
        book.open_leg(short_put(quantity=-10, entry=5.0))
        book.open_leg(short_call(quantity=-10, entry=4.0))
        book.close_all_legs({"P": 2.0, "C": 1.0})
        assert book.option_realised == pytest.approx(
            10 * (5.0 - 2.0) * 50 + 10 * (4.0 - 1.0) * 50
        )

    def test_fees_reduce_equity(self, book):
        book.charge_fees(250.0)
        assert book.equity({}, 5000.0) == pytest.approx(99_750.0)

    def test_refuses_to_double_open_the_same_right(self, book):
        book.open_leg(short_put())
        with pytest.raises(RuntimeError, match="already open"):
            book.open_leg(short_put())

    def test_is_flat_reports_option_legs_and_the_hedge(self, book):
        assert book.is_flat
        book.hedge.quantity = 1
        assert not book.is_flat
        book.hedge.quantity = 0
        book.open_leg(short_put())
        assert not book.is_flat
