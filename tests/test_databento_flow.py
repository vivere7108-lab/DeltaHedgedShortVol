"""The intraday flow nowcast's data layer: signing, mapping, accumulation,
and the historical backfill/replay round trip.

Built against genuine ``databento``/``databento_dbn`` record types rather
than hand-rolled fakes wherever the package makes that possible -- the
Rust-backed record classes are directly constructible from Python, so a
test that builds a real ``TradeMsg`` or ``InstrumentDefMsg`` and pushes it
through the same code path production uses is exercising the actual field
names and enum values Databento ships, not a guess at them.  The historical
replay tests go one step further and round-trip through a real on-disk DBN
file (``Metadata`` + record bytes, written and read back), which is what
makes ``DatabentoHistoricalFlowSource`` trustworthy without ever touching
the network.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

db = pytest.importorskip("databento")
dbn = pytest.importorskip("databento_dbn")

from deltahedger.config import NowcastConfig
from deltahedger.data.databento_flow import (
    DatabentoHistoricalFlowSource,
    DatabentoLiveFlowFeed,
    FlowAccumulator,
    InstrumentInfo,
    InstrumentMap,
    _aggregate,
    _signed_size,
)
from deltahedger.gex import StrikeFlow

UTC = timezone.utc


def _ns(moment: datetime) -> int:
    return int(moment.timestamp() * 1e9)


def make_definition(
    instrument_id: int,
    strike: float,
    right: str,
    expiry: date,
    ts_event: datetime,
    instrument_class=None,
) -> "db.InstrumentDefMsg":
    ts = _ns(ts_event)
    exp = _ns(datetime(expiry.year, expiry.month, expiry.day, 21, 0, tzinfo=UTC))
    return db.InstrumentDefMsg(
        publisher_id=1, instrument_id=instrument_id, ts_event=ts, ts_recv=ts,
        min_price_increment=25_000_000, display_factor=1_000_000_000,
        raw_symbol=f"ESM5 {right}{int(strike)}", asset="ES", security_type="OOF",
        instrument_class=(
            instrument_class
            or (db.InstrumentClass.CALL if right == "C" else db.InstrumentClass.PUT)
        ),
        security_update_action=db.SecurityUpdateAction.ADD,
        expiration=exp, strike_price=int(strike * 1e9),
        underlying="ES", exchange="XCME", currency="USD",
    )


def make_trade(
    instrument_id: int, ts_event: datetime, side, size: float, price: float = 5000.0
) -> "db.TradeMsg":
    ts = _ns(ts_event)
    return db.TradeMsg(
        publisher_id=1, instrument_id=instrument_id, ts_event=ts, ts_recv=ts,
        price=int(price * 1e9), size=int(size), side=side,
        action=db.Action.TRADE, flags=0, depth=0, sequence=0, ts_in_delta=0,
    )


def dbn_bytes(schema, records) -> bytes:
    """Real DBN wire bytes for ``records`` -- a ``Metadata`` header plus the
    concatenated record bytes, exactly what ``Historical.timeseries.get_range``
    would hand back and what ``to_file``/``from_file`` round-trip."""
    start = min((int(r.ts_event) for r in records), default=0)
    meta = dbn.Metadata(
        dataset="GLBX.MDP3", start=start, stype_in=db.SType.PARENT,
        stype_out=db.SType.INSTRUMENT_ID, schema=schema, symbols=["ES.OPT"],
        partial=[], not_found=[], mappings=[],
    )
    return bytes(meta) + b"".join(bytes(r) for r in records)


def dbn_store(schema, records):
    return db.DBNStore.from_bytes(dbn_bytes(schema, records))


EXPIRY = date(2025, 6, 16)
T10 = datetime(2025, 6, 10, 10, 0, tzinfo=UTC)


class TestInstrumentInfo:
    def test_a_call_definition_resolves(self):
        record = make_definition(100, 5000.0, "C", EXPIRY, T10)
        info = InstrumentInfo.from_definition(record)
        assert info == InstrumentInfo(100, EXPIRY, 5000.0, "C")

    def test_a_put_definition_resolves(self):
        record = make_definition(101, 5000.0, "P", EXPIRY, T10)
        info = InstrumentInfo.from_definition(record)
        assert info.right == "P"

    def test_the_underlying_future_is_not_an_option(self):
        record = db.InstrumentDefMsg(
            publisher_id=1, instrument_id=999, ts_event=_ns(T10), ts_recv=_ns(T10),
            min_price_increment=25_000_000, display_factor=1_000_000_000,
            raw_symbol="ESM5", asset="ES", security_type="FUT",
            instrument_class=db.InstrumentClass.FUTURE,
            security_update_action=db.SecurityUpdateAction.ADD,
            underlying="ES", exchange="XCME", currency="USD",
        )
        assert InstrumentInfo.from_definition(record) is None

    def test_a_duck_typed_stand_in_also_resolves(self):
        """The resolver reads attributes, not the real DBN class -- a
        record from a source that is not the live Databento SDK (a test
        double, or a future alternate data vendor) works the same way."""
        class Fake:
            instrument_id = 42
            instrument_class = "C"
            pretty_strike_price = 4950.0
            pretty_expiration = datetime(2025, 6, 16, 21, 0, tzinfo=UTC)

        info = InstrumentInfo.from_definition(Fake())
        assert info == InstrumentInfo(42, EXPIRY, 4950.0, "C")


class TestInstrumentMap:
    def test_it_maps_calls_and_puts_and_skips_the_future(self):
        im = InstrumentMap()
        im.add_definition(make_definition(100, 5000.0, "C", EXPIRY, T10))
        im.add_definition(make_definition(101, 5000.0, "P", EXPIRY, T10))
        im.add_definition(
            make_definition(999, 0.0, "C", EXPIRY, T10, db.InstrumentClass.FUTURE)
        )
        assert len(im) == 2
        assert im.get(100).right == "C"
        assert im.get(999) is None

    def test_an_unknown_instrument_id_is_none(self):
        assert InstrumentMap().get(12345) is None


class TestSignedSize:
    def test_ask_side_is_a_buy(self):
        assert _signed_size(db.Side.ASK, 10) == 10.0

    def test_bid_side_is_a_sell(self):
        assert _signed_size(db.Side.BID, 10) == -10.0

    def test_none_side_carries_no_information(self):
        assert _signed_size(db.Side.NONE, 10) == 0.0

    def test_a_plain_character_works_the_same_as_the_enum(self):
        """Duck-typed side values (a bare 'A'/'B' string) sign the same way
        as the real enum -- useful for building test records without the
        SDK, and exercised for real via the historical replay tests."""
        assert _signed_size("A", 10) == 10.0
        assert _signed_size("B", 10) == -10.0


class TestAggregate:
    def test_it_sums_per_strike_and_right(self):
        rows = [
            (5000.0, "C", 10.0), (5000.0, "C", -3.0),
            (5000.0, "P", 4.0), (5005.0, "C", 2.0),
        ]
        result = _aggregate(rows)
        by_strike = {r.strike: r for r in result}
        assert by_strike[5000.0].call_signed_volume == pytest.approx(7.0)
        assert by_strike[5000.0].put_signed_volume == pytest.approx(4.0)
        assert by_strike[5005.0].call_signed_volume == pytest.approx(2.0)

    def test_an_empty_input_is_an_empty_tuple(self):
        assert _aggregate([]) == ()

    def test_results_are_sorted_by_strike(self):
        rows = [(5010.0, "C", 1.0), (4990.0, "C", 1.0), (5000.0, "P", 1.0)]
        result = _aggregate(rows)
        assert [r.strike for r in result] == [4990.0, 5000.0, 5010.0]


class TestFlowAccumulator:
    def test_a_buy_and_a_sell_at_the_same_strike_net(self):
        acc = FlowAccumulator()
        info = InstrumentInfo(100, EXPIRY, 5000.0, "C")
        acc.record(info, T10, db.Side.ASK, 10.0)
        acc.record(info, T10 + timedelta(minutes=1), db.Side.BID, 4.0)
        snap = acc.since(EXPIRY, T10, T10 + timedelta(hours=1))
        assert snap == (StrikeFlow(5000.0, 6.0, 0.0),)

    def test_window_excludes_trades_outside_it(self):
        acc = FlowAccumulator()
        info = InstrumentInfo(100, EXPIRY, 5000.0, "C")
        acc.record(info, T10, db.Side.ASK, 10.0)
        acc.record(info, T10 + timedelta(hours=2), db.Side.ASK, 10.0)
        snap = acc.since(EXPIRY, T10, T10 + timedelta(minutes=30))
        assert snap == (StrikeFlow(5000.0, 10.0, 0.0),)

    def test_a_different_expiry_is_kept_separate(self):
        acc = FlowAccumulator()
        other_expiry = date(2025, 6, 17)
        acc.record(InstrumentInfo(100, EXPIRY, 5000.0, "C"), T10, db.Side.ASK, 10.0)
        acc.record(InstrumentInfo(101, other_expiry, 5000.0, "C"), T10, db.Side.ASK, 5.0)
        assert acc.since(EXPIRY, T10, T10 + timedelta(hours=1)) == (
            StrikeFlow(5000.0, 10.0, 0.0),
        )
        assert acc.since(other_expiry, T10, T10 + timedelta(hours=1)) == (
            StrikeFlow(5000.0, 5.0, 0.0),
        )

    def test_an_unsigned_side_is_dropped(self):
        acc = FlowAccumulator()
        info = InstrumentInfo(100, EXPIRY, 5000.0, "C")
        acc.record(info, T10, db.Side.NONE, 10.0)
        assert acc.since(EXPIRY, T10, T10 + timedelta(hours=1)) == ()

    def test_prune_drops_trades_older_than_retain(self):
        acc = FlowAccumulator(retain=timedelta(hours=1))
        info = InstrumentInfo(100, EXPIRY, 5000.0, "C")
        acc.record(info, T10, db.Side.ASK, 10.0)
        acc.prune(T10 + timedelta(hours=3))
        assert acc.since(EXPIRY, T10 - timedelta(hours=1), T10 + timedelta(hours=4)) == ()


class TestLiveFlowFeedRecordDispatch:
    """``_on_record`` without ever calling ``start()`` -- no network, no
    background thread, just the routing logic that decides whether a
    record is a definition or a trade."""

    def test_a_definition_record_is_mapped(self):
        feed = DatabentoLiveFlowFeed(NowcastConfig())
        feed._on_record(make_definition(100, 5000.0, "C", EXPIRY, T10))
        assert feed.instruments.get(100) is not None

    def test_a_trade_for_a_mapped_instrument_is_accumulated(self):
        feed = DatabentoLiveFlowFeed(NowcastConfig())
        feed._on_record(make_definition(100, 5000.0, "C", EXPIRY, T10))
        feed._on_record(make_trade(100, T10 + timedelta(minutes=1), db.Side.ASK, 7.0))
        snap = feed.flow_since(T10 + timedelta(hours=1), EXPIRY, T10)
        assert snap == (StrikeFlow(5000.0, 7.0, 0.0),)

    def test_a_trade_for_an_unmapped_instrument_is_dropped_not_fatal(self):
        feed = DatabentoLiveFlowFeed(NowcastConfig())
        feed._on_record(make_trade(999, T10, db.Side.ASK, 7.0))  # no definition seen
        snap = feed.flow_since(T10 + timedelta(hours=1), EXPIRY, T10)
        assert snap == ()


class TestHistoricalFlowSource:
    """The backfill + replay round trip, through real on-disk DBN files."""

    class FakeTimeseries:
        def __init__(self, definitions, trades):
            self.definitions = definitions
            self.trades = trades
            self.requested_schemas: list[str] = []

        def get_range(self, dataset, schema, symbols, stype_in, start, end):
            self.requested_schemas.append(str(schema))
            records = (
                self.definitions if str(schema) == "definition" else self.trades
            )
            return dbn_store(schema, records)

    class FakeHistorical:
        def __init__(self, key=None, *, timeseries):
            self.timeseries = timeseries

    def _source(self, tmp_path, definitions, trades, monkeypatch):
        fake_ts = self.FakeTimeseries(definitions, trades)
        monkeypatch.setattr(
            db, "Historical", lambda key=None: self.FakeHistorical(timeseries=fake_ts),
        )
        source = DatabentoHistoricalFlowSource(
            NowcastConfig(), cache_dir=tmp_path, api_key="test"
        )
        return source, fake_ts

    def test_backfill_writes_one_day_of_cache_files(self, tmp_path, monkeypatch):
        day = date(2025, 6, 10)
        defs = [make_definition(100, 5000.0, "C", EXPIRY, T10)]
        trades = [make_trade(100, T10 + timedelta(minutes=5), db.Side.ASK, 12.0)]
        source, _ = self._source(tmp_path, defs, trades, monkeypatch)

        fetched = source.backfill(day, day)
        assert fetched == [day]
        assert source._definitions_path(day).exists()
        assert source._trades_path(day).exists()

    def test_backfill_skips_an_already_cached_day(self, tmp_path, monkeypatch):
        day = date(2025, 6, 10)
        source, _ = self._source(tmp_path, [], [], monkeypatch)
        source.backfill(day, day)
        assert source.backfill(day, day) == []  # nothing re-fetched

    def test_backfill_force_re_fetches(self, tmp_path, monkeypatch):
        day = date(2025, 6, 10)
        source, _ = self._source(tmp_path, [], [], monkeypatch)
        source.backfill(day, day)
        assert source.backfill(day, day, force=True) == [day]

    def test_backfill_skips_weekends(self, tmp_path, monkeypatch):
        friday = date(2025, 6, 13)
        monday = date(2025, 6, 16)
        source, _ = self._source(tmp_path, [], [], monkeypatch)
        fetched = source.backfill(friday, monday)
        assert fetched == [friday, monday]

    def test_flow_since_replays_the_cached_day(self, tmp_path, monkeypatch):
        day = date(2025, 6, 10)
        moment = datetime(day.year, day.month, day.day, 10, 0, tzinfo=UTC)
        defs = [
            make_definition(100, 5000.0, "C", EXPIRY, moment),
            make_definition(101, 5000.0, "P", EXPIRY, moment),
        ]
        trades = [
            make_trade(100, moment + timedelta(minutes=5), db.Side.ASK, 12.0),
            make_trade(101, moment + timedelta(minutes=10), db.Side.BID, 3.0),
        ]
        source, _ = self._source(tmp_path, defs, trades, monkeypatch)
        source.backfill(day, day)

        result = source.flow_since(
            moment + timedelta(hours=1), EXPIRY, moment,
        )
        by_strike = {r.strike: r for r in result}
        assert by_strike[5000.0].call_signed_volume == pytest.approx(12.0)
        # a sell (Side.BID) is a negative signed put flow
        assert by_strike[5000.0].put_signed_volume == pytest.approx(-3.0)

    def test_flow_since_raises_a_clear_error_when_not_backfilled(self, tmp_path, monkeypatch):
        source, _ = self._source(tmp_path, [], [], monkeypatch)
        with pytest.raises(FileNotFoundError, match="nowcast-backfill"):
            source.flow_since(
                datetime(2025, 6, 10, 12, tzinfo=UTC), EXPIRY,
                datetime(2025, 6, 10, 10, tzinfo=UTC),
            )

    def test_flow_since_spans_multiple_cached_days(self, tmp_path, monkeypatch):
        day1 = date(2025, 6, 10)
        day2 = date(2025, 6, 11)
        moment1 = datetime(2025, 6, 10, 15, 0, tzinfo=UTC)
        moment2 = datetime(2025, 6, 11, 10, 0, tzinfo=UTC)
        defs = [make_definition(100, 5000.0, "C", EXPIRY, moment1)]
        trades_day1 = [make_trade(100, moment1, db.Side.ASK, 5.0)]
        trades_day2 = [make_trade(100, moment2, db.Side.ASK, 7.0)]

        class TwoDayTimeseries:
            def get_range(self, dataset, schema, symbols, stype_in, start, end):
                if str(schema) == "definition":
                    return dbn_store(schema, defs)
                day = start.date()
                trades = trades_day1 if day == day1 else trades_day2
                return dbn_store(schema, trades)

        fake_ts = TwoDayTimeseries()
        monkeypatch.setattr(
            db, "Historical", lambda key=None: self.FakeHistorical(timeseries=fake_ts),
        )
        source = DatabentoHistoricalFlowSource(
            NowcastConfig(), cache_dir=tmp_path, api_key="test"
        )
        source.backfill(day1, day2)

        result = source.flow_since(moment2 + timedelta(hours=1), EXPIRY, moment1)
        assert result == (StrikeFlow(5000.0, 12.0, 0.0),)  # 5 + 7 across both days

    def test_a_second_query_reuses_the_replayed_day(self, tmp_path, monkeypatch):
        """Performance property, not just correctness: a day is replayed
        once and memoised, since a backtest re-derives the same window
        repeatedly on the nowcast's refresh timer."""
        day = date(2025, 6, 10)
        moment = datetime(day.year, day.month, day.day, 10, 0, tzinfo=UTC)
        defs = [make_definition(100, 5000.0, "C", EXPIRY, moment)]
        trades = [make_trade(100, moment, db.Side.ASK, 1.0)]
        source, _ = self._source(tmp_path, defs, trades, monkeypatch)
        source.backfill(day, day)

        source.flow_since(moment + timedelta(hours=1), EXPIRY, moment)
        assert day in source._accumulators
        cached = source._accumulators[day]
        source.flow_since(moment + timedelta(hours=2), EXPIRY, moment)
        assert source._accumulators[day] is cached
