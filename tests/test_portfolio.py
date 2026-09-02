"""Position accounting, especially the fill bookkeeping that a hedger
exercises hardest: repeatedly adding to, reducing, and flipping a position."""

from datetime import date, datetime, timezone

import pytest

from deltahedger.chain import OptionQuote, StraddleQuote
from deltahedger.portfolio import HedgePosition, Portfolio, StraddlePosition
from deltahedger.pricing import Greeks

NOW = datetime(2025, 6, 10, 9, 35, tzinfo=timezone.utc)
EXPIRY = date(2025, 6, 10)


@pytest.fixture
def book(es):
    return Portfolio(starting_equity=100_000.0, source=es)


def straddle(quantity=-10, call_entry=8.0, put_entry=8.0):
    return StraddlePosition(
        strike=5000.0, expiry=EXPIRY, quantity=quantity,
        call_entry=call_entry, put_entry=put_entry, entry_time=NOW,
    )


def quote(call_delta=0.5, put_delta=-0.5, gamma=0.01, vega=0.4, theta=-2.0,
          call_price=8.0, put_price=8.0):
    """A straddle quote with the greeks stated directly, so the delta-unit
    arithmetic is tested rather than Black-76."""
    leg = lambda right, price, delta: OptionQuote(  # noqa: E731
        5000.0, right, EXPIRY, price, 0.15,
        Greeks(price, delta, gamma, vega, theta), 0.001,
    )
    return StraddleQuote(
        strike=5000.0, expiry=EXPIRY,
        call=leg("C", call_price, call_delta),
        put=leg("P", put_price, put_delta),
        time_to_expiry=0.001,
    )


class TestDeltaUnits:
    def test_an_atm_straddle_is_close_to_delta_neutral(self, book):
        book.open_straddle(straddle(quantity=-1))
        assert book.option_delta_units(quote()) == pytest.approx(0.0)

    def test_a_skewed_straddle_carries_the_net_of_its_legs(self, book):
        book.open_straddle(straddle(quantity=-1))
        # +0.60 call and -0.30 put nets +0.30 per straddle; short 1 is -30.
        assert book.option_delta_units(
            quote(call_delta=0.60, put_delta=-0.30)
        ) == pytest.approx(-30.0)

    def test_scales_with_contract_count(self, book):
        book.open_straddle(straddle(quantity=-7))
        assert book.option_delta_units(
            quote(call_delta=0.60, put_delta=-0.30)
        ) == pytest.approx(-210.0)

    def test_one_mes_is_ten_delta_units(self, book):
        book.hedge.quantity = 1
        assert book.hedge_delta_units() == pytest.approx(10.0)
        book.hedge.quantity = -10
        assert book.hedge_delta_units() == pytest.approx(-100.0)

    def test_net_delta_combines_both_legs(self, book):
        book.open_straddle(straddle(quantity=-7))
        book.hedge.quantity = -12
        assert book.net_delta_units(
            quote(call_delta=0.60, put_delta=-0.30)
        ) == pytest.approx(-210.0 - 120.0)

    def test_a_long_straddle_carries_positive_gamma(self, book):
        book.open_straddle(straddle(quantity=10))
        # Both legs carry gamma 0.02, so 0.04 per straddle; 10 long = +40.
        assert book.option_gamma_units(quote(gamma=0.02)) == pytest.approx(40.0)

    def test_a_short_straddle_carries_negative_gamma(self, book):
        book.open_straddle(straddle(quantity=-10))
        assert book.option_gamma_units(quote(gamma=0.02)) == pytest.approx(-40.0)

    def test_a_long_straddle_pays_theta_and_a_short_one_collects_it(self, book):
        book.open_straddle(straddle(quantity=1))
        assert book.option_theta(quote(theta=-2.0)) == pytest.approx(-200.0)
        book.close_straddle(8.0, 8.0)
        book.open_straddle(straddle(quantity=-1))
        assert book.option_theta(quote(theta=-2.0)) == pytest.approx(200.0)

    def test_vega_follows_the_same_sign(self, book):
        book.open_straddle(straddle(quantity=2))
        assert book.option_vega(quote(vega=0.4)) == pytest.approx(80.0)

    def test_no_position_is_no_delta(self, book):
        assert book.net_delta_units(None) == 0.0

    def test_no_quote_is_no_delta(self, book):
        book.open_straddle(straddle())
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
    def test_a_short_straddle_profits_when_the_premium_falls(self, book):
        book.open_straddle(straddle(quantity=-10, call_entry=8.0, put_entry=8.0))
        marks = quote(call_price=5.0, put_price=5.0)
        assert book.unrealised_pnl(marks, hedge_mark=5000.0) == pytest.approx(
            3000.0  # 10 straddles * 6.00 of decay * $50
        )

    def test_a_long_straddle_loses_on_the_same_decay(self, book):
        book.open_straddle(straddle(quantity=10, call_entry=8.0, put_entry=8.0))
        marks = quote(call_price=5.0, put_price=5.0)
        assert book.unrealised_pnl(marks, hedge_mark=5000.0) == pytest.approx(-3000.0)

    def test_only_the_pair_matters_not_which_leg_moved(self, book):
        """The two legs are one instrument: 16.00 of premium is 16.00 of
        premium however it is split between the call and the put."""
        book.open_straddle(straddle(quantity=-10, call_entry=8.0, put_entry=8.0))
        split = book.unrealised_pnl(quote(call_price=12.0, put_price=1.0), 5000.0)
        even = book.unrealised_pnl(quote(call_price=6.5, put_price=6.5), 5000.0)
        assert split == pytest.approx(even)

    def test_closing_moves_unrealised_into_realised(self, book):
        book.open_straddle(straddle(quantity=-10, call_entry=8.0, put_entry=8.0))
        assert book.close_straddle(5.0, 5.0) == pytest.approx(3000.0)
        assert book.option_realised == pytest.approx(3000.0)
        assert book.straddle is None

    def test_pnl_is_attributed_by_leg(self, book):
        book.open_straddle(straddle(quantity=-10, call_entry=8.0, put_entry=8.0))
        book.close_straddle(5.0, 5.0)
        book.apply_hedge_fill(2, 5000.0)
        book.apply_hedge_fill(-2, 4990.0)
        assert book.option_realised == pytest.approx(3000.0)
        assert book.hedge_realised == pytest.approx(-100.0)
        assert book.realised_pnl == pytest.approx(2900.0)

    def test_the_debit_is_signed_and_the_premium_at_risk_is_not(self, es):
        mult = es.option.multiplier
        long = straddle(quantity=10)
        short = straddle(quantity=-10)
        assert long.debit_paid(mult) == pytest.approx(8000.0)
        assert short.debit_paid(mult) == pytest.approx(-8000.0)
        assert long.premium_at_risk(mult) == short.premium_at_risk(mult) == 8000.0

    def test_fees_reduce_equity(self, book):
        book.charge_fees(250.0)
        assert book.equity(None, 5000.0) == pytest.approx(99_750.0)

    def test_refuses_to_double_open(self, book):
        book.open_straddle(straddle())
        with pytest.raises(RuntimeError, match="already open"):
            book.open_straddle(straddle())

    def test_is_flat_reports_both_legs(self, book):
        assert book.is_flat
        book.hedge.quantity = 1
        assert not book.is_flat
