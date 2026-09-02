"""The live runner's unattended behaviour, driven against a fake IBKR.

None of this is reachable from the backtest, and all of it is what decides
whether a multi-day forward walk produces evidence or a silent dead process.
The gateway restart is not an edge case -- IBKR forces one every day -- so
"survives a dropped connection" is a functional requirement, not hardening.
"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import Config
from deltahedger.gex import StrikeOpenInterest
from deltahedger.live.journal import (
    JournallingStrategy,
    SessionJournal,
    read_journal,
)
from deltahedger.live.runner import LiveRunner

NY = ZoneInfo("America/New_York")
OPEN = datetime(2025, 6, 10, 10, 0, tzinfo=NY)


class FakeIb:
    """Just enough ib_async to drive the runner's loop."""

    def __init__(self, drop_after: int | None = None):
        self.drop_after = drop_after
        self.sleeps = 0
        self._connected = False

    def connect(self, *_, **__):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def isConnected(self):
        return self._connected

    def managedAccounts(self):
        return ["DU1234567"]

    def reqMarketDataType(self, *_):
        pass

    def positions(self, *_):
        return []

    def sleep(self, _seconds):
        self.sleeps += 1
        if self.drop_after is not None and self.sleeps >= self.drop_after:
            self._connected = False  # the daily gateway restart


class FakeConnection:
    """Stands in for IbkrConnection: connects, prices, never talks to TWS."""

    def __init__(self, cfg, source, drop_after=None, price=5000.0):
        self.cfg = cfg
        self.source = source
        self.account = "DU1234567"
        self.price = price
        self.ib = FakeIb(drop_after)
        self.connects = 0

    def __enter__(self):
        self.ib.connect()
        self.connects += 1
        return self

    def __exit__(self, *_):
        self.ib.disconnect()

    def future_price(self):
        if not self.ib.isConnected():
            raise ConnectionError("not connected")
        return self.price


class FakeOpenInterest:
    def open_interest(self, moment, future_price, expiry):
        return [
            StrikeOpenInterest(future_price + 5.0 * i, 4000.0, 200.0)
            for i in range(-20, 21)
        ]


def build_runner(tmp_path, drop_after=None, **live):
    cfg = Config()
    cfg.starting_equity = 250_000.0
    cfg.live.journal_dir = str(tmp_path)
    cfg.live.reconnect_backoff_seconds = 0.01
    cfg.live.max_reconnect_backoff_seconds = 0.02
    for key, value in live.items():
        setattr(cfg.live, key, value)

    runner = LiveRunner(cfg, dry_run=True)
    runner.connection = FakeConnection(cfg, cfg.source, drop_after)
    return runner


def patch_session(monkeypatch, runner, moment=OPEN):
    """Pin the clock inside the session and stub the IBKR-only pieces."""
    import deltahedger.live.runner as module

    monkeypatch.setattr(module, "IbkrOpenInterestProvider",
                        lambda conn, cfg: FakeOpenInterest())
    monkeypatch.setattr(module, "IbkrExecution",
                        lambda conn, cfg, dry_run=False: _NoExecution())
    monkeypatch.setattr(module, "IbkrChainProvider", lambda conn, cfg: _NoChain())
    monkeypatch.setattr(module, "WhatIfMarginModel", lambda conn, fallback: fallback)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    monkeypatch.setattr(module, "datetime", Frozen)


class _NoExecution:
    def execute_option(self, quote, quantity, moment):
        return None

    def execute_hedge(self, quantity, price, moment):
        return None


class _NoChain:
    def straddle(self, future_price, expiry, t):
        return None


class TestReconnection:
    def test_a_dropped_connection_is_reconnected_not_fatal(self, tmp_path, monkeypatch):
        """The daily gateway restart must not end the walk."""
        runner = build_runner(tmp_path, drop_after=2)
        patch_session(monkeypatch, runner)
        runner.run(max_cycles=6)
        assert runner.connection.connects > 1, "the runner never reconnected"

    def test_reconnection_can_be_turned_off(self, tmp_path, monkeypatch):
        runner = build_runner(tmp_path, drop_after=1, reconnect=False)
        patch_session(monkeypatch, runner)
        with pytest.raises(Exception):
            runner.run(max_cycles=6)

    def test_it_gives_up_after_the_configured_attempts(self, tmp_path, monkeypatch):
        """Retrying forever is right under a supervisor, but a bounded run
        must be able to fail rather than spin silently."""
        runner = build_runner(tmp_path, drop_after=1, max_reconnect_attempts=2)
        patch_session(monkeypatch, runner)

        # Every reconnection drops again immediately.
        original = runner.connection
        connects = {"n": 0}

        class AlwaysDrops(FakeConnection):
            def __enter__(self):
                connects["n"] += 1
                super().__enter__()
                self.ib._connected = False  # dead on arrival
                return self

        runner.connection = AlwaysDrops(original.cfg, original.source)
        with pytest.raises(Exception):
            runner.run(max_cycles=50)
        assert connects["n"] <= 4  # bounded, not spinning

    def test_a_stop_signal_beats_a_backoff(self, tmp_path, monkeypatch):
        """SIGTERM during a reconnect backoff must not wait it out."""
        runner = build_runner(tmp_path, reconnect_backoff_seconds=30.0,
                              max_reconnect_backoff_seconds=30.0)
        runner._stop = True
        import time as time_module

        start = time_module.monotonic()
        runner._sleep(30.0)
        assert time_module.monotonic() - start < 1.0


class TestJournal:
    def test_it_writes_records_as_they_happen(self, tmp_path):
        """Flushed per record: a crash keeps everything up to the crash."""
        journal = SessionJournal(tmp_path)
        strategy = _StubStrategy()
        driver = JournallingStrategy(strategy, journal)

        driver.on_bar(object(), object())
        path = tmp_path / f"bars-{OPEN.date().isoformat()}.jsonl"
        assert path.exists()
        assert len(path.read_text().strip().splitlines()) == 1

        driver.on_bar(object(), object())
        assert len(path.read_text().strip().splitlines()) == 2

    def test_events_and_fills_land_in_their_own_files(self, tmp_path):
        journal = SessionJournal(tmp_path)
        driver = JournallingStrategy(_StubStrategy(emit=True), journal)
        driver.on_bar(object(), object())
        for kind in ("events", "fills", "bars"):
            assert (tmp_path / f"{kind}-{OPEN.date().isoformat()}.jsonl").exists()

    def test_each_record_is_one_json_object(self, tmp_path):
        journal = SessionJournal(tmp_path)
        driver = JournallingStrategy(_StubStrategy(emit=True), journal)
        driver.on_bar(object(), object())
        text = (tmp_path / f"events-{OPEN.date().isoformat()}.jsonl").read_text()
        for line in text.strip().splitlines():
            assert isinstance(json.loads(line), dict)

    def test_a_restart_appends_rather_than_truncating(self, tmp_path):
        """An interrupted walk loses the position, never the history."""
        for _ in range(2):
            driver = JournallingStrategy(_StubStrategy(), SessionJournal(tmp_path))
            driver.on_bar(object(), object())
        path = tmp_path / f"bars-{OPEN.date().isoformat()}.jsonl"
        assert len(path.read_text().strip().splitlines()) == 2

    def test_it_reads_back_into_a_frame(self, tmp_path):
        driver = JournallingStrategy(_StubStrategy(emit=True), SessionJournal(tmp_path))
        driver.on_bar(object(), object())
        frame = read_journal(tmp_path, "events")
        assert len(frame) == 1
        assert "kind" in frame.columns

    def test_a_half_written_line_is_skipped_not_fatal(self, tmp_path):
        """What a hard kill leaves behind."""
        path = tmp_path / "events-2025-06-10.jsonl"
        path.write_text('{"kind":"entry","timestamp":"2025-06-10T10:00:00"}\n{"kind":"ex')
        frame = read_journal(tmp_path, "events")
        assert len(frame) == 1

    def test_an_empty_directory_reads_as_empty(self, tmp_path):
        assert read_journal(tmp_path, "events").empty

    def test_a_failed_write_does_not_stop_the_strategy(self, tmp_path):
        """Losing the log must never take the trading with it."""
        journal = SessionJournal(tmp_path)
        journal.directory = tmp_path / "deleted"  # never created
        driver = JournallingStrategy(_StubStrategy(), journal)
        assert driver.on_bar(object(), object()) is not None

    def test_the_wrapper_passes_attributes_through(self, tmp_path):
        strategy = _StubStrategy()
        driver = JournallingStrategy(strategy, SessionJournal(tmp_path))
        assert driver.portfolio is strategy.portfolio


class _StubStrategy:
    """A strategy-shaped object that appends one of each record per bar."""

    def __init__(self, emit: bool = False):
        from deltahedger.broker.base import Fill
        from deltahedger.strategy import BarState, StrategyEvent

        self.emit = emit
        self.portfolio = object()
        self.events: list = []
        self.fills: list = []
        self._Fill, self._Event, self._State = Fill, StrategyEvent, BarState
        self._n = 0

    def on_bar(self, bar, execution):
        self._n += 1
        moment = OPEN + timedelta(minutes=self._n)
        if self.emit:
            self.events.append(
                self._Event(moment, "entry", "stub", 0.0, 250_000.0, "positive")
            )
            self.fills.append(self._Fill(1, 10.0, 2.32, moment, "option", ""))
        return self._State(
            timestamp=moment, future=5000.0, atm_iv=0.15, time_to_expiry=0.0007,
            straddle_mark=16.0, call_mark=8.0, put_mark=8.0,
            option_delta_units=0.0, hedge_delta_units=0.0, net_delta_units=0.0,
            gamma_units=0.0, vega_dollars=0.0, theta_dollars=0.0,
            hedge_contracts=0, straddle_contracts=0, direction=0, strike=5000.0,
            equity=250_000.0, realised_pnl=0.0, fees_paid=0.0, in_band=True,
            gex_total=1.0e9, gex_flip=4990.0, gex_regime="positive",
            distance_to_flip=10.0,
        )


class TestFailureBudget:
    """A reconnect budget must count *consecutive* failures.

    The daily gateway restart means a healthy multi-week walk reconnects
    every night. If those counted against a bounded budget, the walk would
    die after `max_reconnect_attempts` days for the crime of working.
    """

    def test_a_healthy_session_resets_the_budget(self, tmp_path, monkeypatch):
        runner = build_runner(tmp_path, drop_after=2, max_reconnect_attempts=3)
        patch_session(monkeypatch, runner)
        # Ten cycles with a drop every two: eight reconnects, budget of 3.
        runner.run(max_cycles=10)
        assert runner.connection.connects > 3, (
            "the runner exhausted its budget despite every session polling "
            "successfully before the drop"
        )

    def test_repeated_immediate_failures_still_exhaust_it(self, tmp_path, monkeypatch):
        """The reset must key on progress, not merely on having tried."""
        runner = build_runner(tmp_path, max_reconnect_attempts=3)
        patch_session(monkeypatch, runner)

        connects = {"n": 0}

        class DeadOnArrival(FakeConnection):
            def __enter__(self):
                connects["n"] += 1
                super().__enter__()
                self.ib._connected = False
                return self

        original = runner.connection
        runner.connection = DeadOnArrival(original.cfg, original.source)
        with pytest.raises(Exception):
            runner.run(max_cycles=100)
        assert connects["n"] <= 5
