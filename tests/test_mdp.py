"""The MDP 3.0 open-interest path: the snapshot, its staleness rule, and the
record handling that builds it. The Databento client itself is not driven
here; ``OpenInterestBook`` and ``apply_record`` are, with synthetic records
shaped like the library's."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import Config
from deltahedger.data.mdp import (
    STAT_OPEN_INTEREST,
    DatabentoOpenInterestFeed,
    MdpOpenInterest,
    OpenInterestBook,
    apply_record,
    describe_snapshot,
    read_snapshot,
    write_snapshot,
)
from deltahedger.data.openinterest import (
    build_live_open_interest_provider,
    build_open_interest_provider,
)

NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 9, 8, 11, 0, tzinfo=NY)
TODAY = NOW.date()


def provider(tmp_path, max_age=1800.0):
    cfg = Config()
    cfg.data.open_interest = "mdp"
    cfg.data.oi_mdp_path = str(tmp_path / "oi.json")
    cfg.data.oi_max_age_seconds = max_age
    return MdpOpenInterest(cfg.data, cfg.source), cfg


class TestSnapshot:
    def test_round_trips_per_expiry_per_strike(self, tmp_path):
        p, _ = provider(tmp_path)
        write_snapshot(
            p.path, NOW,
            {TODAY: {5000.0: (1200.0, 800.0), 5005.0: (300.0, 0.0)},
             TODAY + timedelta(days=1): {5000.0: (10.0, 20.0)}},
        )
        rows = p.open_interest(NOW + timedelta(minutes=5), 5000.0, TODAY)
        assert [(r.strike, r.call_oi, r.put_oi) for r in rows] == [
            (5000.0, 1200.0, 800.0), (5005.0, 300.0, 0.0),
        ]
        tomorrow = p.open_interest(NOW, 5000.0, TODAY + timedelta(days=1))
        assert len(tomorrow) == 1 and tomorrow[0].put_oi == 20.0

    def test_an_expiry_not_in_the_snapshot_is_empty(self, tmp_path):
        p, _ = provider(tmp_path)
        write_snapshot(p.path, NOW, {TODAY: {5000.0: (1.0, 1.0)}})
        assert p.open_interest(NOW, 5000.0, TODAY + timedelta(days=7)) == []

    def test_a_stale_snapshot_is_not_served(self, tmp_path):
        """The whole point of the age limit: a dead feed must read as no
        open interest, never as yesterday's book."""
        p, _ = provider(tmp_path, max_age=600.0)
        write_snapshot(p.path, NOW - timedelta(minutes=11), {TODAY: {5000.0: (1.0, 1.0)}})
        assert p.open_interest(NOW, 5000.0, TODAY) == []
        write_snapshot(p.path, NOW - timedelta(minutes=9), {TODAY: {5000.0: (1.0, 1.0)}})
        assert len(p.open_interest(NOW, 5000.0, TODAY)) == 1

    def test_a_missing_snapshot_is_empty_not_fatal(self, tmp_path):
        p, _ = provider(tmp_path)
        assert p.open_interest(NOW, 5000.0, TODAY) == []

    def test_the_write_is_atomic(self, tmp_path):
        p, _ = provider(tmp_path)
        write_snapshot(p.path, NOW, {TODAY: {5000.0: (1.0, 1.0)}})
        assert not p.path.with_suffix(".json.tmp").exists()
        assert read_snapshot(p.path)["as_of"] == NOW.isoformat()

    def test_the_path_is_required(self):
        cfg = Config()
        cfg.data.open_interest = "mdp"
        with pytest.raises(ValueError, match="oi_mdp_path"):
            MdpOpenInterest(cfg.data, cfg.source)

    def test_describe_reports_age_and_coverage(self, tmp_path):
        p, _ = provider(tmp_path)
        write_snapshot(p.path, NOW - timedelta(minutes=2), {TODAY: {5000.0: (100.0, 50.0)}})
        text = describe_snapshot(p.path, NOW, TODAY)
        assert "1 expiries" in text and "0:02:00 old" in text and "150 contracts" in text
        assert "mdp-feed" in describe_snapshot(tmp_path / "nothing.json", NOW)


class _Def:
    def __init__(self, instrument_id, expiry: date, strike, right, underlying="ES"):
        self.instrument_id = instrument_id
        # settlement 16:00 New York on the expiry date, in UTC nanoseconds
        self.expiration = int(
            datetime(expiry.year, expiry.month, expiry.day, 16, 0, tzinfo=NY).timestamp() * 1e9
        )
        self.strike_price = int(strike * 1e9)
        self.instrument_class = right
        self.underlying = underlying


_Def.__name__ = "InstrumentDefMsg"


class _Stat:
    def __init__(self, instrument_id, quantity, stat_type=STAT_OPEN_INTEREST, when=NOW):
        self.instrument_id = instrument_id
        self.stat_type = stat_type
        self.quantity = quantity
        self.ts_event = int(when.timestamp() * 1e9)


_Stat.__name__ = "StatMsg"


class TestRecordHandling:
    def test_definitions_and_statistics_fold_into_the_snapshot_shape(self):
        book = OpenInterestBook(NY)
        for record in (
            _Def(1, TODAY, 5000.0, "C"), _Def(2, TODAY, 5000.0, "P"),
            _Def(3, TODAY + timedelta(days=1), 5010.0, "P"),
            _Def(9, TODAY, 5000.0, "F"),  # the future itself: not a leg
            _Stat(1, 1500), _Stat(2, 700), _Stat(3, 40), _Stat(9, 999_999),
            _Stat(1, 12, stat_type=1),  # some other statistic: ignored
        ):
            apply_record(book, record)
        assert book.expiries() == {
            TODAY: {5000.0: (1500.0, 700.0)},
            TODAY + timedelta(days=1): {5010.0: (0.0, 40.0)},
        }
        assert book.last_update == NOW

    def test_a_statistic_before_its_definition_waits(self):
        book = OpenInterestBook(NY)
        apply_record(book, _Stat(5, 100))
        assert book.expiries() == {}
        apply_record(book, _Def(5, TODAY, 4990.0, "C"))
        assert book.expiries() == {TODAY: {4990.0: (100.0, 0.0)}}

    def test_other_underlyings_are_ignored(self):
        book = OpenInterestBook(NY, "ES")
        apply_record(book, _Def(7, TODAY, 20000.0, "C", underlying="NQ"))
        apply_record(book, _Stat(7, 55))
        assert book.expiries() == {}

    def test_the_feed_writes_what_it_was_handed(self, tmp_path):
        feed = DatabentoOpenInterestFeed(Config().source, tmp_path / "oi.json", api_key="x")
        feed.run(records=[_Def(1, TODAY, 5000.0, "C"), _Def(2, TODAY, 5000.0, "P"),
                          _Stat(1, 10), _Stat(2, 20)])
        document = read_snapshot(tmp_path / "oi.json")
        assert document["expiries"][TODAY.isoformat()] == {"5000": [10.0, 20.0]}
        assert document["as_of"] == NOW.isoformat()


class TestFactories:
    def test_a_backtest_refuses_the_live_snapshot(self):
        cfg = Config()
        cfg.data.open_interest = "mdp"
        cfg.data.oi_mdp_path = "x.json"
        with pytest.raises(ValueError, match="deltahedger live"):
            build_open_interest_provider(cfg, cfg.source)

    def test_the_live_factory_builds_the_snapshot_provider(self, tmp_path):
        cfg = Config()
        cfg.data.open_interest = "mdp"
        cfg.data.oi_mdp_path = str(tmp_path / "oi.json")
        assert isinstance(build_live_open_interest_provider(cfg, cfg.source), MdpOpenInterest)

    def test_the_live_factory_refuses_a_generated_surface(self):
        cfg = Config()
        cfg.data.open_interest = "synthetic"
        with pytest.raises(ValueError, match="not a live source"):
            build_live_open_interest_provider(cfg, cfg.source)
