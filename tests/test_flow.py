"""Trade classification, and the dealer sign it produces.

These tests sit next to ``test_gex.py`` in importance and for the same
reason.  GEX decides which side the strategy takes; this module decides what
GEX believes about who is long.  An inverted classification here does not
produce a worse read -- it produces a confidently reversed one, with the
system buying straddles precisely when it should be selling them.

So the direction is asserted in words at every layer rather than only by
arithmetic: **a customer buy leaves the dealer short.**
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import Config, FlowConfig, GexConfig, VolConfig
from deltahedger.data.tradeflow import (
    CsvTradeFeed,
    NullTradeFeed,
    SyntheticTradeFeed,
    build_trade_feed,
)
from deltahedger.flow import (
    BUY,
    CALL,
    PUT,
    RULE_AGGRESSOR,
    RULE_BOOK_DELTA,
    RULE_NONE,
    RULE_QUOTE,
    RULE_TICK,
    SELL,
    UNKNOWN,
    DealerFlowBook,
    OptionTrade,
    TradeClassifier,
    aggressor_from_mdp,
    build_flow_book,
)

NY = ZoneInfo("America/New_York")
NOW = datetime(2025, 6, 10, 10, 0, tzinfo=NY)
EXPIRY = date(2025, 6, 10)
TICK = 0.25


def trade(**kwargs) -> OptionTrade:
    """A trade with everything defaulted except what a test is about."""
    fields = dict(
        timestamp=NOW, expiry=EXPIRY, strike=5000.0, right=CALL,
        price=10.0, size=1.0,
    )
    fields.update(kwargs)
    return OptionTrade(**fields)


@pytest.fixture
def classifier():
    return TradeClassifier(tick_size=TICK, quote_tolerance_ticks=1.0)


class TestAggressorTag:
    """MDP 3.0 tag 5797, which is the one rule that is not an inference."""

    @pytest.mark.parametrize("raw", [1, "1", "buy", "B", "Buyer", BUY])
    def test_buy_side_aggressor(self, raw):
        assert aggressor_from_mdp(raw) == BUY

    @pytest.mark.parametrize("raw", [2, "2", "sell", "S", "Seller", SELL])
    def test_sell_side_aggressor(self, raw):
        assert aggressor_from_mdp(raw) == SELL

    @pytest.mark.parametrize("raw", [0, None, "", "x", 7, float("nan")])
    def test_no_aggressor_is_unknown_not_a_guess(self, raw):
        assert aggressor_from_mdp(raw) == UNKNOWN

    def test_a_bool_is_refused_rather_than_read_as_its_int(self):
        # True == 1 in Python, and 1 is "buy side aggressor". A feed handing
        # us a bool is a feed we have misunderstood, not a buy.
        assert aggressor_from_mdp(True) == UNKNOWN
        assert aggressor_from_mdp(False) == UNKNOWN


class TestPrecedence:
    """Each rule runs only when every rule above it declined."""

    def test_the_aggressor_flag_wins_over_a_contradicting_quote(self, classifier):
        # Priced at the bid, which the quote rule would call a customer
        # sell -- but the feed says the buy side aggressed, and the feed is
        # not inferring.
        result = classifier.classify(
            trade(price=9.0, bid=9.0, ask=11.0, aggressor=BUY)
        )
        assert (result.side, result.rule) == (BUY, RULE_AGGRESSOR)

    def test_the_book_delta_wins_over_the_quote(self, classifier):
        result = classifier.classify(
            trade(price=10.0, bid=9.0, ask=11.0,
                  bid_size_delta=-40.0, ask_size_delta=0.0)
        )
        assert (result.side, result.rule) == (SELL, RULE_BOOK_DELTA)

    def test_the_quote_rule_runs_when_there_is_no_flag_or_book(self, classifier):
        result = classifier.classify(trade(price=11.0, bid=9.0, ask=11.0))
        assert (result.side, result.rule) == (BUY, RULE_QUOTE)

    def test_a_disabled_rule_falls_through_to_the_next(self):
        classifier = TradeClassifier(TICK, use_aggressor_flag=False)
        result = classifier.classify(
            trade(price=9.0, bid=9.0, ask=11.0, aggressor=BUY)
        )
        assert (result.side, result.rule) == (SELL, RULE_QUOTE)


class TestBookDeltaRule:
    """MBO: whichever side lost resting size was the passive one."""

    def test_liquidity_swept_from_the_ask_is_a_customer_buy(self, classifier):
        result = classifier.classify(
            trade(ask_size_delta=-25.0, bid_size_delta=0.0)
        )
        assert (result.side, result.rule) == (BUY, RULE_BOOK_DELTA)
        assert result.dealer_sign == -1.0  # the dealer sold it

    def test_liquidity_taken_from_the_bid_is_a_customer_sell(self, classifier):
        result = classifier.classify(
            trade(bid_size_delta=-25.0, ask_size_delta=0.0)
        )
        assert (result.side, result.rule) == (SELL, RULE_BOOK_DELTA)
        assert result.dealer_sign == 1.0  # the dealer bought it

    def test_a_side_that_gained_liquidity_does_not_block_the_rule(self, classifier):
        result = classifier.classify(
            trade(ask_size_delta=-10.0, bid_size_delta=+30.0)
        )
        assert result.side == BUY

    def test_both_sides_losing_liquidity_identifies_nothing(self, classifier):
        # A book that also cancelled or re-priced across the execution no
        # longer says who aggressed, so the rule declines rather than
        # picking whichever side fell further.
        result = classifier.classify(
            trade(price=10.0, bid=9.0, ask=11.0,
                  bid_size_delta=-10.0, ask_size_delta=-40.0)
        )
        assert result.rule != RULE_BOOK_DELTA

    def test_one_missing_delta_disables_the_rule(self, classifier):
        result = classifier.classify(trade(ask_size_delta=-25.0))
        assert result.rule != RULE_BOOK_DELTA


class TestQuoteRule:
    """Lee-Ready: the execution price against the prevailing quote."""

    def test_a_trade_at_the_ask_is_a_customer_buy(self, classifier):
        assert classifier.classify(trade(price=11.0, bid=9.0, ask=11.0)).side == BUY

    def test_a_trade_at_the_bid_is_a_customer_sell(self, classifier):
        assert classifier.classify(trade(price=9.0, bid=9.0, ask=11.0)).side == SELL

    def test_near_the_ask_counts_as_at_it(self, classifier):
        # One tick inside a wide quote is still an aggressor paying up.
        assert classifier.classify(
            trade(price=11.0 - TICK, bid=9.0, ask=11.0)
        ).side == BUY

    def test_between_the_touch_and_the_mid_takes_the_nearer_side(self, classifier):
        assert classifier.classify(trade(price=10.4, bid=9.0, ask=11.0)).side == BUY
        assert classifier.classify(trade(price=9.6, bid=9.0, ask=11.0)).side == SELL

    def test_a_quote_narrower_than_the_tolerance_cannot_classify_both_ways(self):
        # With a 4-tick tolerance and a 1-tick spread, both tests would fire
        # and the first one written would win. The tolerance is capped at
        # half the spread so the two stay mutually exclusive.
        classifier = TradeClassifier(TICK, quote_tolerance_ticks=4.0)
        bid, ask = 10.0, 10.0 + TICK
        assert classifier.classify(trade(price=ask, bid=bid, ask=ask)).side == BUY
        classifier.reset()
        assert classifier.classify(trade(price=bid, bid=bid, ask=ask)).side == SELL

    def test_a_midpoint_trade_falls_through_to_the_tick_test(self, classifier):
        result = classifier.classify(trade(price=10.0, bid=9.0, ask=11.0))
        # No prior print in this option, so nothing can classify it -- and
        # nothing guesses.
        assert (result.side, result.rule) == (UNKNOWN, RULE_NONE)

    @pytest.mark.parametrize(
        "bid,ask", [(11.0, 9.0), (10.0, 10.0), (None, 11.0), (-1.0, 11.0)]
    )
    def test_an_unusable_quote_declines_rather_than_inverting(self, classifier, bid, ask):
        result = classifier.classify(trade(price=10.0, bid=bid, ask=ask))
        assert result.rule != RULE_QUOTE


class TestTickRule:
    """Lee-Ready's fallback, for midpoint trades and missing quotes."""

    def test_an_uptick_is_a_customer_buy(self, classifier):
        classifier.classify(trade(price=10.0))
        result = classifier.classify(trade(price=10.5))
        assert (result.side, result.rule) == (BUY, RULE_TICK)

    def test_a_downtick_is_a_customer_sell(self, classifier):
        classifier.classify(trade(price=10.0))
        result = classifier.classify(trade(price=9.5))
        assert (result.side, result.rule) == (SELL, RULE_TICK)

    def test_a_zero_tick_carries_the_last_non_zero_direction(self, classifier):
        classifier.classify(trade(price=10.0))
        classifier.classify(trade(price=10.5))  # uptick
        result = classifier.classify(trade(price=10.5))
        assert (result.side, result.rule) == (BUY, RULE_TICK)

    def test_the_first_print_in_an_option_classifies_nothing(self, classifier):
        assert classifier.classify(trade(price=10.0)).side == UNKNOWN

    def test_the_memory_is_per_option_not_per_chain(self, classifier):
        classifier.classify(trade(strike=5000.0, price=10.0))
        # A different strike is a different option: its first print has no
        # predecessor, whatever the neighbouring strike did.
        assert classifier.classify(trade(strike=5010.0, price=10.5)).side == UNKNOWN
        # And a different right at the same strike likewise.
        assert classifier.classify(
            trade(strike=5000.0, right=PUT, price=10.5)
        ).side == UNKNOWN

    def test_a_trade_resolved_by_an_earlier_rule_still_feeds_the_memory(
        self, classifier
    ):
        # The previous print is the previous print however it was classified;
        # skipping the update would compare the next midpoint trade against
        # a stale price.
        classifier.classify(trade(price=10.0, aggressor=SELL))
        result = classifier.classify(trade(price=10.5))
        assert (result.side, result.rule) == (BUY, RULE_TICK)


class TestDealerSign:
    """The direction, stated in words: a customer buy leaves the dealer short."""

    def test_a_customer_buy_makes_the_dealer_short(self, classifier):
        assert classifier.classify(trade(aggressor=BUY)).dealer_sign == -1.0

    def test_a_customer_sell_makes_the_dealer_long(self, classifier):
        assert classifier.classify(trade(aggressor=SELL)).dealer_sign == 1.0

    def test_an_unclassified_trade_signs_nothing(self, classifier):
        assert classifier.classify(trade()).dealer_sign == 0.0


class TestDealerFlowBook:
    def book(self, half_life: float = 0.0) -> DealerFlowBook:
        return DealerFlowBook(TradeClassifier(TICK), half_life_minutes=half_life)

    def test_customers_buying_calls_leaves_dealers_short_them(self):
        book = self.book()
        for _ in range(5):
            book.observe(trade(right=CALL, size=10.0, aggressor=BUY))
        (row,) = book.rows(EXPIRY)
        assert row.call_dealer == -50.0
        assert row.sign(CALL) == pytest.approx(-1.0)

    def test_customers_selling_puts_leaves_dealers_long_them(self):
        book = self.book()
        book.observe(trade(right=PUT, size=40.0, aggressor=SELL))
        (row,) = book.rows(EXPIRY)
        assert row.sign(PUT) == pytest.approx(1.0)

    def test_two_way_flow_nets_towards_flat(self):
        book = self.book()
        book.observe(trade(size=30.0, aggressor=BUY))
        book.observe(trade(size=30.0, aggressor=SELL))
        (row,) = book.rows(EXPIRY)
        assert row.sign(CALL) == pytest.approx(0.0)
        assert row.volume(CALL) == 60.0  # both trades are evidence

    def test_a_lopsided_book_reads_between_the_extremes(self):
        book = self.book()
        book.observe(trade(size=75.0, aggressor=SELL))
        book.observe(trade(size=25.0, aggressor=BUY))
        (row,) = book.rows(EXPIRY)
        assert row.sign(CALL) == pytest.approx(0.5)

    def test_unclassified_volume_is_counted_but_never_signed(self):
        book = self.book()
        book.observe(trade(size=100.0))  # no flag, no quote, no prior print
        (row,) = book.rows(EXPIRY)
        assert row.sign(CALL) is None
        assert row.volume(CALL) == 0.0
        assert row.unclassified(CALL) == 100.0
        assert book.unclassified_volume() == 100.0

    def test_the_two_rights_are_measured_separately(self):
        book = self.book()
        book.observe(trade(right=CALL, size=10.0, aggressor=SELL))
        book.observe(trade(right=PUT, size=10.0, aggressor=BUY))
        (row,) = book.rows(EXPIRY)
        assert row.sign(CALL) == pytest.approx(1.0)
        assert row.sign(PUT) == pytest.approx(-1.0)

    def test_strikes_are_kept_apart_and_returned_in_order(self):
        book = self.book()
        for strike in (5010.0, 4990.0, 5000.0):
            book.observe(trade(strike=strike, size=10.0, aggressor=BUY))
        assert [r.strike for r in book.rows(EXPIRY)] == [4990.0, 5000.0, 5010.0]

    def test_expiries_are_kept_apart(self):
        book = self.book()
        other = date(2025, 6, 11)
        book.observe(trade(expiry=EXPIRY, size=10.0, aggressor=BUY))
        book.observe(trade(expiry=other, size=10.0, aggressor=SELL))
        assert book.rows(EXPIRY)[0].sign(CALL) == pytest.approx(-1.0)
        assert book.rows(other)[0].sign(CALL) == pytest.approx(1.0)

    def test_rule_counts_say_how_a_read_was_classified(self):
        book = self.book()
        book.observe(trade(size=10.0, aggressor=BUY))
        book.observe(trade(size=20.0, bid=9.0, ask=11.0, price=11.0))
        counts = book.rule_counts()
        assert counts[RULE_AGGRESSOR] == 10.0
        assert counts[RULE_QUOTE] == 20.0
        assert "aggressor_flag" in book.describe()

    def test_decay_fades_old_flow_without_changing_a_one_sided_sign(self):
        book = self.book(half_life=30.0)
        book.observe(trade(timestamp=NOW, size=100.0, aggressor=BUY))
        (row,) = book.rows(EXPIRY, NOW + timedelta(minutes=30))
        # Half the evidence, so half the weight the calculator gives it --
        # but the direction it measured has not changed.
        assert row.volume(CALL) == pytest.approx(50.0)
        assert row.sign(CALL) == pytest.approx(-1.0)

    def test_decay_lets_new_flow_outweigh_old_flow(self):
        book = self.book(half_life=30.0)
        book.observe(trade(timestamp=NOW, size=100.0, aggressor=BUY))
        later = NOW + timedelta(minutes=60)  # two half-lives: 100 -> 25
        book.observe(trade(timestamp=later, size=50.0, aggressor=SELL))
        (row,) = book.rows(EXPIRY, later)
        assert row.sign(CALL) > 0.0  # the recent selling now dominates

    def test_no_decay_by_default_keeps_the_whole_session(self):
        book = self.book()
        book.observe(trade(timestamp=NOW, size=100.0, aggressor=BUY))
        (row,) = book.rows(EXPIRY, NOW + timedelta(hours=6))
        assert row.volume(CALL) == 100.0

    def test_pruning_forgets_expiries_that_have_passed(self):
        book = self.book()
        book.observe(trade(expiry=EXPIRY, size=10.0, aggressor=BUY))
        book.prune(date(2025, 6, 11))
        assert book.rows(EXPIRY) == ()

    def test_build_flow_book_reads_the_config(self):
        cfg = Config()
        cfg.flow = FlowConfig(quote_tolerance_ticks=3.0, half_life_minutes=45.0)
        book = build_flow_book(cfg, cfg.source)
        assert book.half_life_minutes == 45.0
        assert book.classifier.quote_tolerance_ticks == 3.0
        assert book.classifier.tick_size == cfg.source.option.tick_size


class TestFeeds:
    def test_the_null_feed_delivers_nothing(self):
        assert NullTradeFeed().trades(NOW, NOW + timedelta(minutes=5), EXPIRY) == ()

    def test_the_factory_refuses_ibkr_outside_a_live_run(self):
        cfg = Config()
        cfg.flow.source = "ibkr"
        with pytest.raises(ValueError, match="live IBKR connection"):
            build_trade_feed(cfg, cfg.source)

    def test_the_factory_rejects_an_unknown_source(self):
        cfg = Config()
        cfg.flow.source = "nasdaq"
        with pytest.raises(ValueError, match="unknown flow source"):
            build_trade_feed(cfg, cfg.source)

    def test_csv_replay_reads_the_mdp_aggressor_column(self, tmp_path, es):
        path = tmp_path / "tape.csv"
        path.write_text(
            "timestamp,expiry,strike,right,price,size,bid,ask,aggressor\n"
            "2025-06-10T10:00:00-04:00,2025-06-10,5000,C,11.0,10,9.0,11.0,2\n"
            "2025-06-10T10:01:00-04:00,2025-06-10,5000,P,9.0,20,9.0,11.0,1\n"
        )
        feed = CsvTradeFeed(FlowConfig(source="csv", csv_path=str(path)), es)
        rows = feed.trades(NOW - timedelta(minutes=1), NOW + timedelta(minutes=5), EXPIRY)
        assert [t.right for t in rows] == [CALL, PUT]
        # Tag 5797: 2 is a sell-side aggressor, 1 a buy-side one -- and the
        # quotes deliberately contradict them, so a file read wrongly would
        # come back with the signs the quote rule would have given.
        assert [t.aggressor for t in rows] == [SELL, BUY]

    def test_csv_replay_windows_are_half_open(self, tmp_path, es):
        path = tmp_path / "tape.csv"
        path.write_text(
            "timestamp,expiry,strike,right,price,size\n"
            "2025-06-10T10:00:00-04:00,2025-06-10,5000,C,11.0,10\n"
            "2025-06-10T10:05:00-04:00,2025-06-10,5000,C,11.0,10\n"
        )
        feed = CsvTradeFeed(FlowConfig(source="csv", csv_path=str(path)), es)
        first = feed.trades(NOW - timedelta(minutes=5), NOW, EXPIRY)
        second = feed.trades(NOW, NOW + timedelta(minutes=5), EXPIRY)
        # Each trade lands in exactly one window: a boundary trade counted
        # twice would double into the dealer position with nothing
        # downstream able to see that it had.
        assert len(first) == 1 and len(second) == 1
        assert first[0].timestamp != second[0].timestamp

    def test_csv_replay_without_the_optional_columns_still_classifies(
        self, tmp_path, es
    ):
        path = tmp_path / "tape.csv"
        path.write_text(
            "timestamp,expiry,strike,right,price,size\n"
            "2025-06-10T10:00:00-04:00,2025-06-10,5000,C,10.0,10\n"
            "2025-06-10T10:01:00-04:00,2025-06-10,5000,C,10.5,10\n"
        )
        feed = CsvTradeFeed(FlowConfig(source="csv", csv_path=str(path)), es)
        book = DealerFlowBook(TradeClassifier(es.option.tick_size))
        book.observe_all(feed.trades(NOW - timedelta(minutes=1), NOW + timedelta(hours=1), EXPIRY))
        # The tick test alone, and the rule counts say so.
        assert book.rule_counts().get(RULE_TICK) == 10.0

    def test_csv_replay_rejects_a_file_missing_required_columns(self, tmp_path, es):
        path = tmp_path / "tape.csv"
        path.write_text("timestamp,strike,price\n2025-06-10T10:00:00,5000,10\n")
        feed = CsvTradeFeed(FlowConfig(source="csv", csv_path=str(path)), es)
        with pytest.raises(ValueError, match="missing column"):
            feed.trades(NOW, NOW + timedelta(hours=1), EXPIRY)

    def test_the_synthetic_feed_agrees_with_the_open_interest_it_is_built_on(self, es):
        """The harness's one hard requirement: it must not contradict itself.

        A generated tape that said the opposite of its own generated book
        would make the measured sign fight the prior at every strike, and
        the resulting backtest would be measuring the disagreement between
        two generators rather than any part of the system.
        """
        from deltahedger.data.openinterest import SyntheticOpenInterest

        cfg = Config()
        oi = SyntheticOpenInterest(cfg.data, es)
        feed = SyntheticTradeFeed(cfg.flow, es, oi)
        book = DealerFlowBook(TradeClassifier(es.option.tick_size))

        oi.open_interest(NOW, 5000.0, EXPIRY)  # freezes the strike anchor
        book.observe_all(feed.trades(NOW, NOW + timedelta(minutes=5), EXPIRY))
        rows = book.rows(EXPIRY)
        assert rows, "the synthetic feed produced no tape"

        call_heavy = oi.call_share(EXPIRY) > 0.5
        signs = [r.sign(CALL) for r in rows if r.sign(CALL) is not None]
        # A call-heavy chain is one where customers sold calls to dealers,
        # so the measured call sign leans positive -- the same direction the
        # +1 prior points, rather than against it.
        assert (sum(signs) / len(signs) > 0) == call_heavy

    def test_the_synthetic_feed_says_nothing_before_its_anchor_exists(self, es):
        from deltahedger.data.openinterest import SyntheticOpenInterest

        cfg = Config()
        feed = SyntheticTradeFeed(cfg.flow, es, SyntheticOpenInterest(cfg.data, es))
        assert feed.trades(NOW, NOW + timedelta(minutes=5), EXPIRY) == ()

    def test_the_synthetic_feed_exercises_more_than_one_rule(self, es):
        from deltahedger.data.openinterest import SyntheticOpenInterest

        cfg = Config()
        oi = SyntheticOpenInterest(cfg.data, es)
        oi.open_interest(NOW, 5000.0, EXPIRY)
        feed = SyntheticTradeFeed(cfg.flow, es, oi)
        book = DealerFlowBook(TradeClassifier(es.option.tick_size))
        book.observe_all(feed.trades(NOW, NOW + timedelta(minutes=5), EXPIRY))
        used = {rule for rule, n in book.rule_counts().items() if n > 0}
        assert {RULE_AGGRESSOR, RULE_QUOTE} <= used
