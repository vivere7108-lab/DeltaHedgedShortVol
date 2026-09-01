"""Front-month stitching.

IBKR rejects ``endDateTime`` on a continuous future (Error 10339), so
history has to be assembled from the concrete quarterly contracts that were
front month at each point in time.  Getting the slicing wrong does not
raise -- it silently stitches the wrong contracts and the backtest runs on
prices that were never front month -- so it is tested directly.

The expiries below are the real ES quarterlies (third Friday of March,
June, September and December).
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import Config
from deltahedger.data.ibkr_history import (
    IbkrHistorySource, _parse_expiry, front_month_windows,
)

NY = ZoneInfo("America/New_York")

EXPIRIES = [
    date(2024, 12, 20), date(2025, 3, 21), date(2025, 6, 20),
    date(2025, 9, 19), date(2025, 12, 19), date(2026, 3, 20),
    date(2026, 6, 19), date(2026, 9, 18),
]


def moment(y, m, d):
    return datetime(y, m, d, tzinfo=NY)


def windows(start, end, roll_days=8):
    return front_month_windows(EXPIRIES, start, end, roll_days=roll_days, tz=NY)


class TestFrontMonthWindows:
    def test_a_half_year_spans_three_contracts(self):
        result = windows(moment(2025, 1, 2), moment(2025, 6, 30))
        assert [w.expiry for w in result] == [
            date(2025, 3, 21), date(2025, 6, 20), date(2025, 9, 19)
        ]

    def test_the_roll_happens_eight_days_before_expiry(self):
        result = windows(moment(2025, 1, 2), moment(2025, 6, 30))
        assert result[0].end.date() == date(2025, 3, 13)  # 2025-03-21 minus 8
        assert result[1].start.date() == date(2025, 3, 13)

    def test_the_roll_offset_is_configurable(self):
        result = windows(moment(2025, 1, 2), moment(2025, 6, 30), roll_days=3)
        assert result[0].end.date() == date(2025, 3, 18)

    def test_a_window_inside_one_contract_returns_one_span(self):
        result = windows(moment(2025, 2, 1), moment(2025, 2, 28))
        assert len(result) == 1
        assert result[0].expiry == date(2025, 3, 21)

    def test_spans_are_contiguous_and_cover_the_window(self):
        start, end = moment(2025, 1, 2), moment(2025, 6, 30)
        result = windows(start, end)
        assert result[0].start == start
        assert result[-1].end == end
        for earlier, later in zip(result, result[1:]):
            assert earlier.end == later.start, "a gap or overlap between contracts"

    def test_spans_never_leave_the_requested_window(self):
        start, end = moment(2025, 5, 1), moment(2025, 7, 15)
        for window in windows(start, end):
            assert start <= window.start < window.end <= end

    def test_expired_contracts_before_the_window_are_dropped(self):
        result = windows(moment(2025, 1, 2), moment(2025, 6, 30))
        assert date(2024, 12, 20) not in [w.expiry for w in result]

    def test_the_newest_contract_covers_a_window_past_the_last_roll(self):
        result = windows(moment(2026, 8, 1), moment(2026, 12, 1))
        assert result[-1].expiry == date(2026, 9, 18)
        assert result[-1].end == moment(2026, 12, 1)

    def test_the_newest_contract_is_not_duplicated(self):
        result = windows(moment(2026, 8, 1), moment(2026, 12, 1))
        assert len(result) == len({w.expiry for w in result})

    def test_every_span_is_non_empty(self):
        for start, end in [
            (moment(2025, 1, 2), moment(2025, 12, 31)),
            (moment(2025, 3, 13), moment(2025, 3, 14)),
            (moment(2025, 6, 12), moment(2025, 9, 11)),
        ]:
            assert all(w.start < w.end for w in front_month_windows(
                EXPIRIES, start, end, tz=NY))

    def test_an_inverted_window_is_rejected(self):
        with pytest.raises(ValueError, match="not before"):
            windows(moment(2025, 6, 30), moment(2025, 1, 2))

    def test_no_listed_contracts_gives_no_spans(self):
        assert front_month_windows([], moment(2025, 1, 2), moment(2025, 6, 30)) == []

    def test_unsorted_and_duplicated_expiries_are_tolerated(self):
        shuffled = [EXPIRIES[3], EXPIRIES[1], EXPIRIES[1], EXPIRIES[2], EXPIRIES[0]]
        result = front_month_windows(
            shuffled, moment(2025, 1, 2), moment(2025, 6, 30), tz=NY
        )
        assert [w.expiry for w in result] == [
            date(2025, 3, 21), date(2025, 6, 20), date(2025, 9, 19)
        ]


class TestVerifiedStart:
    """The regression: IBKR's expired-contract lookup silently omits
    anything older than roughly a year back, with no error -- just an
    absence.  A live fetch against a real account returned only ESU5
    (2025-09-19) and later for a query that also needed ESH5 (March) and
    ESM5 (June); the stitcher extended ESU5's window to cover the entire
    Jan-Jun 2025 request, which would have silently fetched a forward
    quarter's prices and reported them as the front month's.
    """

    def test_a_fully_known_chain_is_verified(self):
        """The normal case: every window is bounded by a real predecessor."""
        result = windows(moment(2025, 1, 2), moment(2025, 6, 30))
        assert all(w.verified_start for w in result)

    def test_a_missing_predecessor_is_unverified(self):
        """Reproduces the live failure: only the recent contracts resolve."""
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        result = front_month_windows(
            recent_only, moment(2025, 1, 2), moment(2025, 6, 30), tz=NY
        )
        assert len(result) == 1
        assert result[0].expiry == date(2025, 9, 19)
        assert not result[0].verified_start
        # this is exactly the bug: the unverified window silently claims the
        # whole requested range, not just the portion near its own roll.
        assert result[0].start == moment(2025, 1, 2)
        assert result[0].end == moment(2025, 6, 30)

    def test_only_the_earliest_window_can_be_unverified(self):
        """Every later contract in a stitch is bounded by the one before it,
        even when the very first is not."""
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        result = front_month_windows(
            recent_only, moment(2025, 1, 2), moment(2026, 3, 25), tz=NY
        )
        assert not result[0].verified_start
        assert all(w.verified_start for w in result[1:])

    def test_a_lone_contract_past_its_own_roll_is_unverified(self):
        """The trailing-extension branch must carry verified_start through,
        not default it back to True by dropping the flag on rebuild."""
        result = front_month_windows(
            [date(2026, 9, 18)], moment(2026, 8, 1), moment(2026, 12, 1), tz=NY
        )
        assert len(result) == 1
        assert not result[0].verified_start

    def test_resolve_windows_propagates_the_flag(self, tmp_path):
        """IbkrHistorySource._resolve_windows rebuilds ContractWindow with a
        real contract attached -- that rebuild must not silently drop
        verified_start back to its True default."""
        cfg = Config()
        cfg.start_date, cfg.end_date = "2025-01-02", "2025-06-30"
        cfg.data.cache_dir = str(tmp_path)
        source = IbkrHistorySource(cfg, cfg.source)
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        ib = FakeIB(recent_only, [])
        found = source._resolve_windows(ib, *source._window())
        assert not found[0].verified_start


class TestCoverageCheck:
    @pytest.fixture
    def source(self, tmp_path):
        cfg = Config()
        cfg.start_date, cfg.end_date = "2025-01-02", "2025-06-30"
        cfg.data.cache_dir = str(tmp_path)
        return IbkrHistorySource(cfg, cfg.source)

    def test_an_unverified_earliest_window_is_refused(self, source):
        start, end = source._window()
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        bad = front_month_windows(recent_only, start, end, tz=NY)
        with pytest.raises(RuntimeError, match="cannot verify"):
            source._check_coverage(bad, start)

    def test_the_error_names_the_resolvable_contract_and_retention_cause(self, source):
        start, end = source._window()
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        bad = front_month_windows(recent_only, start, end, tz=NY)
        with pytest.raises(RuntimeError, match=r"ESU5|2025-09-19") as excinfo:
            source._check_coverage(bad, start)
        assert "retention" in str(excinfo.value)

    def test_a_fully_verified_stitch_passes(self, source):
        start, end = source._window()
        good = front_month_windows(EXPIRIES, start, end, tz=NY)
        source._check_coverage(good, start)  # must not raise

    def test_suggests_a_narrower_start_when_part_of_the_range_recovers(self, source):
        """The recoverable case: the unresolved contract's own roll-in date
        falls inside the requested window, so trimming --start is real advice."""
        start = moment(2025, 1, 2)
        end = moment(2026, 1, 15)  # extends past ESU5's own roll-in (2025-09-11)
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        bad = front_month_windows(recent_only, start, end, tz=NY)
        with pytest.raises(RuntimeError, match=r"2025-09-11"):
            source._check_coverage(bad, start)

    def test_says_nothing_is_recoverable_when_the_roll_in_is_past_the_end(self, source):
        """The unrecoverable case from the live failure: the whole requested
        window ends before the earliest resolvable contract even starts."""
        start, end = source._window()  # 2025-01-02 .. 2025-06-30
        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        bad = front_month_windows(recent_only, start, end, tz=NY)
        with pytest.raises(RuntimeError, match="No part of the requested range"):
            source._check_coverage(bad, start)


class TestExpiryParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("20250321", date(2025, 3, 21)),
        ("202503", date(2025, 3, 1)),
        (" 20250321 ", date(2025, 3, 21)),
    ])
    def test_parses_ibkr_formats(self, raw, expected):
        assert _parse_expiry(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "not-a-date", "2025-03-21"])
    def test_returns_none_rather_than_raising(self, raw):
        assert _parse_expiry(raw) is None


class TestCachePath:
    def test_encodes_the_window_and_bar_size(self, tmp_path):
        cfg = Config()
        cfg.start_date, cfg.end_date = "2025-01-02", "2025-06-30"
        cfg.data.cache_dir = str(tmp_path)
        path = IbkrHistorySource(cfg, cfg.source).cache_path()
        assert path.name == "ES_5mins_20250102_20250630.csv"

    def test_differs_by_bar_size(self, tmp_path):
        cfg = Config()
        cfg.start_date, cfg.end_date = "2025-01-02", "2025-06-30"
        cfg.data.cache_dir = str(tmp_path)
        first = IbkrHistorySource(cfg, cfg.source).cache_path()
        cfg.data.bar_size = "1 min"
        assert IbkrHistorySource(cfg, cfg.source).cache_path() != first

    def test_an_inverted_range_is_rejected(self, tmp_path):
        cfg = Config()
        cfg.start_date, cfg.end_date = "2025-06-30", "2025-01-02"
        cfg.data.cache_dir = str(tmp_path)
        with pytest.raises(ValueError, match="not before"):
            IbkrHistorySource(cfg, cfg.source).cache_path()


class FakeIB:
    """A stand-in for ib_async.IB covering just what the fetcher calls.

    ``reqHistoricalData`` raises IBKR's real Error 10339 if handed a
    continuous future, so the test fails loudly if the fetcher ever goes
    back to paging a ContFuture.
    """

    def __init__(self, expiries, session_days, bar_minutes=5):
        self.expiries = expiries
        self.session_days = session_days
        self.bar_minutes = bar_minutes
        self.requests = []

    def reqContractDetails(self, query):
        from ib_async import ContractDetails, Future

        return [
            ContractDetails(contract=Future(
                symbol=query.symbol, exchange=query.exchange,
                currency=query.currency,
                lastTradeDateOrContractMonth=expiry.strftime("%Y%m%d"),
                localSymbol=f"ES{expiry:%y%m}",
            ))
            for expiry in self.expiries
        ]

    def reqHistoricalData(self, contract, endDateTime, durationStr,
                          barSizeSetting, whatToShow, useRTH, formatDate):
        from datetime import timedelta

        from ib_async import BarData, ContFuture

        if isinstance(contract, ContFuture):
            raise RuntimeError(
                "Error 10339: Setting end date/time for continuous future "
                "security type is not allowed."
            )
        self.requests.append((contract.localSymbol, endDateTime, whatToShow))

        span = int(durationStr.split()[0])
        window_start = endDateTime - timedelta(days=span)
        bars = []
        for day in self.session_days:
            open_bar = datetime(day.year, day.month, day.day, 9, 35, tzinfo=NY)
            if not (window_start <= open_bar <= endDateTime):
                continue
            for i in range(3):
                stamp = open_bar + timedelta(minutes=self.bar_minutes * i)
                value = 0.15 if whatToShow == "OPTION_IMPLIED_VOLATILITY" else 5000.0
                bars.append(BarData(
                    date=stamp, open=value, high=value, low=value,
                    close=value, volume=100.0, average=value, barCount=1,
                ))
        return bars

    def connect(self, *a, **kw):
        pass

    def disconnect(self):
        pass


class TestDownload:
    @pytest.fixture
    def source(self, tmp_path, monkeypatch):
        import deltahedger.data.ibkr_history as module

        monkeypatch.setattr(module, "PACING_SECONDS", 0.0)
        cfg = Config()
        cfg.start_date, cfg.end_date = "2025-01-02", "2025-06-30"
        cfg.data.cache_dir = str(tmp_path)
        return module.IbkrHistorySource(cfg, cfg.source)

    def sessions(self):
        from datetime import timedelta

        from deltahedger.session import is_trading_day

        day, days = date(2025, 1, 2), []
        while day <= date(2025, 6, 30):
            if is_trading_day(day):
                days.append(day)
            day += timedelta(days=1)
        return days

    def test_resolves_the_expected_front_month_contracts(self, source):
        ib = FakeIB(EXPIRIES, self.sessions())
        found = source._resolve_windows(ib, *source._window())
        assert [w.expiry for w in found] == [
            date(2025, 3, 21), date(2025, 6, 20), date(2025, 9, 19)
        ]
        assert all(w.contract is not None for w in found)

    def test_fetches_and_stitches_every_contract(self, source):
        ib = FakeIB(EXPIRIES, self.sessions())
        found = source._resolve_windows(ib, *source._window())
        frame = source._fetch_windows(ib, found, "TRADES")
        assert not frame.empty
        assert frame["timestamp"].is_monotonic_increasing
        assert not frame["timestamp"].duplicated().any()
        requested = {symbol for symbol, _, _ in ib.requests}
        assert len(requested) == 3, f"expected 3 contracts, queried {requested}"

    def test_never_pages_a_continuous_future(self, source):
        """The regression: ContFuture + endDateTime is IBKR Error 10339."""
        ib = FakeIB(EXPIRIES, self.sessions())
        source._fetch_windows(ib, source._resolve_windows(ib, *source._window()), "TRADES")
        assert ib.requests, "no historical requests were made at all"

    def test_bars_stay_inside_the_requested_window(self, source):
        ib = FakeIB(EXPIRIES, self.sessions())
        start, end = source._window()
        found = source._resolve_windows(ib, start, end)
        frame = source._fetch_windows(ib, found, "TRADES")
        assert frame["timestamp"].min() >= start
        assert frame["timestamp"].max() <= end

    def test_merge_carries_implied_vol_onto_price_bars(self, source):
        ib = FakeIB(EXPIRIES, self.sessions())
        found = source._resolve_windows(ib, *source._window())
        merged = source._merge(
            source._fetch_windows(ib, found, "TRADES"),
            source._fetch_windows(ib, found, "OPTION_IMPLIED_VOLATILITY"),
        )
        assert merged["atm_iv"].to_list() == pytest.approx([0.15] * len(merged))
        assert set(merged.columns) >= {
            "timestamp", "open", "high", "low", "close", "volume", "atm_iv"
        }

    def test_a_missing_vol_series_falls_back_rather_than_failing(self, source):
        import pandas as pd

        ib = FakeIB(EXPIRIES, self.sessions())
        found = source._resolve_windows(ib, *source._window())
        prices = source._fetch_windows(ib, found, "TRADES")
        merged = source._merge(prices, pd.DataFrame())
        assert merged["atm_iv"].to_list() == pytest.approx(
            [source.cfg.data.default_atm_iv] * len(merged)
        )
        assert len(merged) == len(prices)

    def test_the_result_drives_a_backtest(self, source, monkeypatch):
        """The fetched frame is consumable by the engine unchanged."""
        from deltahedger.backtest import run_backtest

        ib = FakeIB(EXPIRIES, self.sessions())
        found = source._resolve_windows(ib, *source._window())
        frame = source._merge(
            source._fetch_windows(ib, found, "TRADES"),
            source._fetch_windows(ib, found, "OPTION_IMPLIED_VOLATILITY"),
        )
        monkeypatch.setattr(type(source), "load", lambda self: frame)
        result = run_backtest(source.cfg, source)
        assert len(result.bars) == len(frame)

    def test_download_refuses_an_unverifiable_stitch_before_fetching_anything(
        self, source, monkeypatch
    ):
        """End-to-end reproduction of the live failure: an account whose
        expired-contract lookup only reaches back to ESU5 must not have
        download() silently return ESU5's prices labelled as the front
        month for January through May 2025. It must refuse, and it must do
        so before spending any paced historical-data request on data it is
        about to throw away."""
        import ib_async

        recent_only = [e for e in EXPIRIES if e >= date(2025, 9, 19)]
        fake = FakeIB(recent_only, self.sessions())
        monkeypatch.setattr(ib_async, "IB", lambda: fake)

        with pytest.raises(RuntimeError, match="cannot verify"):
            source.download()
        assert not fake.requests, (
            "a historical-data request was sent before the coverage check ran"
        )
