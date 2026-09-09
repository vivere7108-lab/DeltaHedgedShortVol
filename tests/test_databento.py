"""DatabentoSession's aggregation and root-resolution logic, driven without
a live connection.

The record-parsing handlers (``_on_definition``/``_on_stat``/``_on_trade``)
are thin, mechanical translations off real ``databento_dbn`` record types
and are exercised against the live feed directly rather than against faked
Rust objects here -- see the module docstring in ``databento_source.py`` for
what is and is not verified that way. What *is* worth pinning down without a
live connection is: the aggregation ``rows()`` does over whatever state the
handlers would have produced (grouping by strike, the flow adjustment, the
floor at zero, which strikes get reported at all); and ``ensure_subscribed``'s
root resolution and caching, which is the part that replaced a hardcoded
parent symbol after "ES.OPT" turned out to resolve to nothing live -- see
the module docstring for what CME actually lists ES's weeklies under.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from deltahedger.config import Config
from deltahedger.data.databento_source import (
    DatabentoFlowAdjustedOpenInterestProvider,
    DatabentoOpenInterestProvider,
    DatabentoSession,
    DatabentoTradeFeed,
    _aggressor_side,
    _Definition,
)
from deltahedger.flow import BUY, SELL, UNKNOWN, OptionTrade
from deltahedger.instruments import get_risk_source

NOW = datetime(2026, 9, 8, 10, 0)
EXPIRY = date(2026, 9, 8)
OTHER_EXPIRY = date(2026, 9, 9)
UTC = timezone.utc
TAPE_START = datetime(2026, 9, 8, 14, 0, tzinfo=UTC)


def buffered(minutes: float, side: str = SELL, size: float = 10.0) -> OptionTrade:
    """One execution in the session's buffer, ``minutes`` after TAPE_START."""
    return OptionTrade(
        timestamp=TAPE_START + timedelta(minutes=minutes),
        expiry=EXPIRY, strike=5000.0, right="C",
        price=10.0, size=size, aggressor=side,
    )


class FakeContract:
    def __init__(self, trading_class):
        self.tradingClass = trading_class


class FakeConnection:
    """Just enough of IbkrConnection for root resolution: a price, the
    risk source (for its strike increment) and option_contract()."""

    def __init__(self, source, price=5000.0, trading_class="E2B"):
        self.source = source
        self._price = price
        self._trading_class = trading_class
        self.qualify_calls: list[tuple] = []

    def future_price(self):
        return self._price

    def option_contract(self, expiry, strike, right):
        self.qualify_calls.append((expiry, strike, right))
        return FakeContract(self._trading_class)


class FakeLive:
    """Just enough of databento.Live to verify subscribe/start sequencing."""

    def __init__(self):
        self.subscriptions: list[tuple] = []
        self.started = False
        self.start_call_count = 0

    def subscribe(self, dataset, schema, stype_in, symbols):
        self.subscriptions.append((dataset, schema, stype_in, symbols))

    def add_callback(self, *_a, **_kw):
        pass

    def start(self):
        self.started = True
        self.start_call_count += 1

    def stop(self):
        self.started = False


@pytest.fixture
def es():
    return get_risk_source("ES")


@pytest.fixture
def session(es):
    return DatabentoSession(Config(), es)


class TestRows:
    def test_a_strike_with_both_legs_reports_both(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._definitions[2] = _Definition(5000.0, "P", EXPIRY)
        session._oi[1] = (120.0, 100)
        session._oi[2] = (80.0, 100)

        rows = session.rows(EXPIRY, adjusted=False)

        assert len(rows) == 1
        assert rows[0].strike == 5000.0
        assert rows[0].call_oi == 120.0
        assert rows[0].put_oi == 80.0

    def test_a_leg_with_no_oi_print_yet_is_left_out_not_zeroed(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        # no self._oi entry for instrument 1 -- no print has arrived

        rows = session.rows(EXPIRY, adjusted=False)

        assert rows == []

    def test_other_expiries_are_excluded(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (120.0, 100)
        session._definitions[2] = _Definition(5010.0, "C", OTHER_EXPIRY)
        session._oi[2] = (50.0, 100)

        rows = session.rows(EXPIRY, adjusted=False)

        assert len(rows) == 1
        assert rows[0].strike == 5000.0

    def test_rows_are_sorted_by_strike(self, session):
        session._definitions[1] = _Definition(5010.0, "C", EXPIRY)
        session._oi[1] = (10.0, 100)
        session._definitions[2] = _Definition(4990.0, "C", EXPIRY)
        session._oi[2] = (20.0, 100)

        rows = session.rows(EXPIRY, adjusted=False)

        assert [r.strike for r in rows] == [4990.0, 5010.0]

    def test_unadjusted_read_ignores_accumulated_flow(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (120.0, 100)
        session._flow[1] = 40.0

        rows = session.rows(EXPIRY, adjusted=False)

        assert rows[0].call_oi == 120.0

    def test_adjusted_read_adds_positive_flow(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (120.0, 100)
        session._flow[1] = 40.0

        rows = session.rows(EXPIRY, adjusted=True)

        assert rows[0].call_oi == 160.0

    def test_adjusted_read_subtracts_negative_flow(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (120.0, 100)
        session._flow[1] = -50.0

        rows = session.rows(EXPIRY, adjusted=True)

        assert rows[0].call_oi == 70.0

    def test_adjusted_read_floors_at_zero(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (120.0, 100)
        session._flow[1] = -500.0

        rows = session.rows(EXPIRY, adjusted=True)

        assert rows[0].call_oi == 0.0

    def test_a_leg_missing_entirely_reports_zero_on_that_side(self, session):
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (120.0, 100)
        # no put definition at this strike at all

        rows = session.rows(EXPIRY, adjusted=False)

        assert rows[0].call_oi == 120.0
        assert rows[0].put_oi == 0.0

    def test_no_rows_for_an_expiry_logs_once_not_every_poll(self, session, caplog):
        with caplog.at_level("WARNING"):
            session.rows(EXPIRY, adjusted=False)
            session.rows(EXPIRY, adjusted=False)

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1


class TestProviders:
    """Providers delegate to ensure_subscribed + rows. Pre-seeding
    _root_by_expiry makes ensure_subscribed a no-op (its first check),
    so these exercise the delegation without needing a real connection or
    live session -- that resolution/subscription path is TestEnsureSubscribed's
    job."""

    def test_raw_provider_reads_unadjusted(self, session):
        session._root_by_expiry[EXPIRY] = "E2B"
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (100.0, 1)
        session._flow[1] = 999.0

        provider = DatabentoOpenInterestProvider(session, connection=None)
        rows = provider.open_interest(NOW, 5000.0, EXPIRY)

        assert rows[0].call_oi == 100.0

    def test_flow_adjusted_provider_reads_adjusted(self, session):
        session._root_by_expiry[EXPIRY] = "E2B"
        session._definitions[1] = _Definition(5000.0, "C", EXPIRY)
        session._oi[1] = (100.0, 1)
        session._flow[1] = 25.0

        provider = DatabentoFlowAdjustedOpenInterestProvider(session, connection=None)
        rows = provider.open_interest(NOW, 5000.0, EXPIRY)

        assert rows[0].call_oi == 125.0


class TestEnsureSubscribed:
    def test_the_root_is_resolved_via_ibkrs_own_contract_qualification(self, session):
        session._live = FakeLive()
        conn = FakeConnection(session.source, price=5000.0, trading_class="E2B")

        session.ensure_subscribed(EXPIRY, conn)

        assert session._root_by_expiry[EXPIRY] == "E2B"
        assert conn.qualify_calls == [(EXPIRY, 5000.0, "C")]

    def test_it_subscribes_definition_statistics_and_trades_for_the_root(self, session):
        session._live = FakeLive()
        conn = FakeConnection(session.source, trading_class="E2B")

        session.ensure_subscribed(EXPIRY, conn)

        schemas = {s[1] for s in session._live.subscriptions}
        assert schemas == {"definition", "statistics", "trades"}
        for sub in session._live.subscriptions:
            assert sub[3] == "E2B.OPT"  # symbols
        assert session._live.started

    def test_a_second_call_for_the_same_expiry_does_nothing(self, session):
        session._live = FakeLive()
        conn = FakeConnection(session.source, trading_class="E2B")

        session.ensure_subscribed(EXPIRY, conn)
        session.ensure_subscribed(EXPIRY, conn)

        assert len(conn.qualify_calls) == 1
        assert len(session._live.subscriptions) == 3  # not 6

    def test_a_different_expiry_sharing_a_root_subscribes_once_but_resolves_twice(
        self, session
    ):
        session._live = FakeLive()
        conn = FakeConnection(session.source, trading_class="E2B")

        session.ensure_subscribed(EXPIRY, conn)
        session.ensure_subscribed(OTHER_EXPIRY, conn)

        assert len(conn.qualify_calls) == 2  # each expiry resolved once
        assert len(session._live.subscriptions) == 3  # same root, one subscribe
        assert session._live.start_call_count == 1  # start() called only once

    def test_a_different_root_gets_its_own_subscription(self, session):
        session._live = FakeLive()
        conn = FakeConnection(session.source, trading_class="E2B")
        session.ensure_subscribed(EXPIRY, conn)

        conn2 = FakeConnection(session.source, trading_class="E2C")
        session.ensure_subscribed(OTHER_EXPIRY, conn2)

        assert session._root_by_expiry == {EXPIRY: "E2B", OTHER_EXPIRY: "E2C"}
        assert len(session._live.subscriptions) == 6
        assert session._live.start_call_count == 1  # still only once

    def test_a_missing_trading_class_from_ibkr_is_a_loud_error(self, session):
        session._live = FakeLive()
        conn = FakeConnection(session.source, trading_class=None)

        with pytest.raises(RuntimeError, match="tradingClass"):
            session.ensure_subscribed(EXPIRY, conn)

    def test_an_explicit_parent_symbol_override_skips_ibkr_entirely(self, session):
        session.cfg.parent_symbol = "ES.OPT"
        session._live = FakeLive()
        conn = FakeConnection(session.source, trading_class="E2B")

        session.ensure_subscribed(EXPIRY, conn)

        assert conn.qualify_calls == []
        assert session._root_by_expiry[EXPIRY] == "ES"
        assert session._live.subscriptions[0][3] == "ES.OPT"


class TestStart:
    def test_a_missing_api_key_is_rejected_before_connecting(self, session, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
            session.start()


class TestAggressorSide:
    """The one direction that decides which side the whole system takes.

    MDP 3.0 reports the side of the *aggressor* on a trade, and Databento's
    own enum documents it that way. ``BID`` is therefore a buy aggressor --
    a customer who bought, leaving the dealer short the option and short
    its gamma. Inverting this does not degrade the strategy, it reverses
    it, which is why the mapping has one definition and this test.
    """

    def test_bid_is_the_buy_aggressor(self):
        dbn = pytest.importorskip("databento_dbn")
        assert _aggressor_side(dbn.Side.BID) == BUY

    def test_ask_is_the_sell_aggressor(self):
        dbn = pytest.importorskip("databento_dbn")
        assert _aggressor_side(dbn.Side.ASK) == SELL

    def test_no_named_aggressor_is_unknown_rather_than_a_guess(self):
        # An implied or administrative match names no aggressor. It falls
        # through to the rest of the classification chain instead of being
        # booked as one side or the other.
        dbn = pytest.importorskip("databento_dbn")
        assert _aggressor_side(dbn.Side.NONE) == UNKNOWN

    def test_the_open_interest_adjustment_uses_the_same_mapping(self):
        """The session signs its OI adjustment off this same function.

        Two copies of the direction would be two chances to invert one, and
        an inverted copy in either place is invisible in a log.
        """
        import inspect

        from deltahedger.data import databento_source

        body = inspect.getsource(databento_source.DatabentoSession._on_trade)
        assert "_aggressor_side" in body
        assert "dbn.Side" not in body


class TestTakeTrades:
    """Buffering and windowing, driven without a live connection."""

    def test_a_window_returns_only_the_trades_inside_it(self, session):
        session._trades[EXPIRY] = [buffered(0), buffered(5), buffered(10)]
        rows = session.take_trades(
            EXPIRY, TAPE_START, TAPE_START + timedelta(minutes=5)
        )
        assert [t.timestamp for t in rows] == [TAPE_START + timedelta(minutes=5)]

    def test_windows_are_half_open_so_nothing_is_counted_twice(self, session):
        session._trades[EXPIRY] = [buffered(0), buffered(5), buffered(10)]
        first = session.take_trades(
            EXPIRY, TAPE_START - timedelta(minutes=1), TAPE_START + timedelta(minutes=5)
        )
        second = session.take_trades(
            EXPIRY, TAPE_START + timedelta(minutes=5), TAPE_START + timedelta(minutes=10)
        )
        assert len(first) == 2 and len(second) == 1
        assert not (set(id(t) for t in first) & set(id(t) for t in second))

    def test_consumed_trades_are_dropped_but_later_ones_are_kept(self, session):
        session._trades[EXPIRY] = [buffered(0), buffered(10)]
        session.take_trades(EXPIRY, TAPE_START - timedelta(minutes=1), TAPE_START)
        # The 10-minute trade has not been asked for yet and must survive.
        assert len(session._trades[EXPIRY]) == 1

    def test_expiries_are_kept_apart(self, session):
        session._trades[EXPIRY] = [buffered(0)]
        assert session.take_trades(
            OTHER_EXPIRY, TAPE_START - timedelta(minutes=1),
            TAPE_START + timedelta(minutes=1),
        ) == []

    def test_an_expiry_with_no_tape_yet_returns_nothing(self, session):
        assert session.take_trades(EXPIRY, TAPE_START, NOW.replace(tzinfo=UTC)) == []

    def test_an_overflowing_buffer_is_reported_not_silently_truncated(
        self, session, caplog
    ):
        session._dropped_trades = 250
        session._trades[EXPIRY] = [buffered(0)]
        with caplog.at_level("WARNING"):
            session.take_trades(
                EXPIRY, TAPE_START - timedelta(minutes=1),
                TAPE_START + timedelta(minutes=1),
            )
        assert "dropped 250" in caplog.text
        # Reported once, not on every subsequent poll.
        assert session._dropped_trades == 0


class TestTradeFeedWiring:
    def test_constructing_the_feed_turns_capture_on(self, session, es):
        assert session._capture_trades is False
        DatabentoTradeFeed(session, FakeConnection(es, trading_class="E2B"))
        assert session._capture_trades is True

    def test_capture_is_off_until_something_asks_for_it(self, session):
        # The session decodes every trade anyway for the flow-adjusted
        # provider; keeping them costs memory a walk that is not measuring
        # the dealer sign has no use for.
        assert session._capture_trades is False
