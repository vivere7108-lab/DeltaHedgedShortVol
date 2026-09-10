"""The live runner's unattended behaviour, driven against a fake IBKR.

None of this is reachable from the backtest, and all of it is what decides
whether a multi-day forward walk produces evidence or a silent dead process.
The gateway restart is not an edge case -- IBKR forces one every day -- so
"survives a dropped connection" is a functional requirement, not hardening.
"""

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import Config
from deltahedger.live.journal import (
    JournallingStrategy,
    SessionJournal,
    read_journal,
)
from deltahedger.live.runner import LiveRunner, ReconciliationError

pytest.importorskip("ib_async", reason="the live path is an optional extra")

from fakes import FakeConnection, FakeOpenInterest, fake_position  # noqa: E402


def _straddle_rows(quantity, call=8.0, put=8.0, strike=5000.0, expiry="20250610"):
    """The two option rows IBKR reports for one straddle.

    ``avgCost`` on an option is the fill price times the multiplier, which
    is how each leg's entry premium survives a restart.
    """
    multiplier = Config().source.option.multiplier
    return [
        fake_position("FOP", "ES", quantity, avg_cost=call * multiplier,
                      strike=strike, right="C", expiry=expiry),
        fake_position("FOP", "ES", quantity, avg_cost=put * multiplier,
                      strike=strike, right="P", expiry=expiry),
    ]

NY = ZoneInfo("America/New_York")
OPEN = datetime(2025, 6, 10, 10, 0, tzinfo=NY)


def build_runner(tmp_path, drop_after=None, **live):
    cfg = Config()
    cfg.starting_equity = 250_000.0
    cfg.data.open_interest = "ibkr"
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


class TestReconciliation:
    """What the runner does about positions it finds already open.

    The runner trusts the broker rather than its own memory, because the
    in-memory book is only this process's record of its own fills and a
    restart throws it away.
    """

    def _reconciled(self, tmp_path, monkeypatch, positions, **live):
        runner = build_runner(tmp_path, **live)
        patch_session(monkeypatch, runner)
        runner.connection.ib.set_positions(positions)
        return runner

    def test_an_existing_hedge_is_adopted_at_its_average_price(
        self, tmp_path, monkeypatch
    ):
        """Starting with a stale in-memory book is how a hedger doubles a
        position, so the broker's hedge leg is taken as the truth."""
        runner = self._reconciled(
            tmp_path, monkeypatch,
            # IBKR reports avgCost for a future as price x multiplier.
            [fake_position("FUT", "MES", -12, avg_cost=5000.0 * 5.0)],
        )
        runner.run(max_cycles=1)
        hedge = runner.strategy.portfolio.hedge
        assert hedge.quantity == -12
        assert hedge.avg_price == pytest.approx(5000.0)

    def test_an_unrelated_symbol_is_not_adopted_and_halts_entries(
        self, tmp_path, monkeypatch
    ):
        """Not the hedge, so not the hedge -- and not nothing either. See
        ``TestPositionsTheBookCannotRepresent``."""
        runner = self._reconciled(
            tmp_path, monkeypatch, [fake_position("FUT", "NQ", 3, avg_cost=100.0)]
        )
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.hedge.quantity == 0
        assert runner.strategy.halted

    def test_a_zero_row_is_not_a_position(self, tmp_path, monkeypatch):
        """IBKR reports closed positions as zero rows rather than dropping
        them, and a zero-size straddle is not a shape to reason about."""
        runner = self._reconciled(
            tmp_path, monkeypatch, [
                fake_position("FOP", "ES", 0, right="C"),
                fake_position("FOP", "ES", 0, right="P"),
            ],
        )
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.straddle is None

    def test_a_matched_straddle_is_adopted(self, tmp_path, monkeypatch):
        """The position the strategy carries overnight is its own.

        IBKR force-restarts the gateway nightly and the shipped tenor rolls
        into tomorrow's series before the bell, so on the ordinary path the
        runner reconnects holding a straddle it placed itself. Refusing to
        pick it up meant refusing to hedge it.
        """
        runner = self._reconciled(tmp_path, monkeypatch, _straddle_rows(-8))
        runner.run(max_cycles=1)

        position = runner.strategy.portfolio.straddle
        assert position is not None
        assert position.quantity == -8
        assert position.strike == 5000.0
        assert position.expiry == date(2025, 6, 10)

    def test_the_entry_premium_is_recovered_from_the_average_cost(
        self, tmp_path, monkeypatch
    ):
        """The number both exit rules are written against, so it has to
        come across intact: avgCost / multiplier is each leg's fill."""
        runner = self._reconciled(
            tmp_path, monkeypatch, _straddle_rows(-8, call=9.0, put=7.0)
        )
        runner.run(max_cycles=1)
        position = runner.strategy.portfolio.straddle
        assert position.call_entry == pytest.approx(9.0)
        assert position.put_entry == pytest.approx(7.0)
        assert position.entry_premium == pytest.approx(16.0)

    def test_an_adopted_position_is_not_attributed_to_a_regime(
        self, tmp_path, monkeypatch
    ):
        """This process never made a GEX read for it, and putting its P&L
        in a regime bucket would corrupt the one number that answers
        whether reading GEX paid."""
        from deltahedger.live.runner import ADOPTED

        runner = self._reconciled(tmp_path, monkeypatch, _straddle_rows(-8))
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.straddle.regime == ADOPTED

    @pytest.mark.parametrize("rows, why", [
        ([fake_position("FOP", "ES", -8, right="C")], "a lone leg"),
        (_straddle_rows(-8)[:1] + [fake_position("FOP", "ES", -4, right="P")],
         "legs of different sizes"),
        (_straddle_rows(-8) + [fake_position("FOP", "ES", -8, strike=5100.0, right="C")],
         "more than one strike"),
        ([fake_position("FOP", "ES", -8, right="C", strike=5000.0),
          fake_position("FOP", "ES", -8, right="P", strike=5010.0)],
         "a strangle, not a straddle"),
        ([fake_position("FOP", "ES", -8, right="C", expiry="20250610"),
          fake_position("FOP", "ES", -8, right="P", expiry="20250611")],
         "two expiries"),
    ])
    def test_anything_that_is_not_one_matched_straddle_is_refused(
        self, tmp_path, monkeypatch, rows, why
    ):
        """Adoption is strict on purpose.

        The strategy has exactly one shape of option position and no way to
        represent %s, so adopting one would leave the hedger working from a
        delta it computed for a book it does not hold. That is the failure
        the original blanket refusal was reaching for, and it still stands
        -- what changed is only that a matched straddle no longer gets
        refused along with everything else.
        """
        runner = self._reconciled(tmp_path, monkeypatch, rows,
                                  max_reconnect_attempts=3)
        with pytest.raises(ReconciliationError):
            runner.run(max_cycles=5)

    def test_a_refusal_is_not_retried_as_though_it_were_a_dropped_socket(
        self, tmp_path, monkeypatch
    ):
        """The deadlock this phase exists to fix.

        The refusal raises out of ``_run_connected``, where the reconnect
        loop used to catch it as a connection failure and retry. It is not
        one -- reconnecting cannot change what the account holds, so every
        attempt failed identically, and at the shipped
        ``max_reconnect_attempts: null`` the runner spun forever at a
        300-second backoff having polled zero times.
        """
        runner = self._reconciled(
            tmp_path, monkeypatch, [fake_position("FOP", "ES", -8, right="C")],
            max_reconnect_attempts=4,
        )
        with pytest.raises(ReconciliationError):
            runner.run(max_cycles=50)
        assert runner.connection.connects == 1, "it retried an answer that cannot change"

    def test_a_contract_that_cannot_be_read_is_a_reconciliation_failure(
        self, tmp_path, monkeypatch
    ):
        """And not a ValueError.

        Anything that escapes as an ordinary exception lands back in the
        reconnect loop, which retries it -- against an account that has not
        changed, forever. The category of the error is what keeps it out of
        that path, so it is worth pinning.
        """
        runner = self._reconciled(
            tmp_path, monkeypatch,
            [fake_position("FOP", "ES", -8, right="C", expiry="not-a-date")],
            max_reconnect_attempts=3,
        )
        with pytest.raises(ReconciliationError, match="contract month"):
            runner.run(max_cycles=5)
        assert runner.connection.connects == 1

    def test_the_refusal_names_what_it_found(self, tmp_path, monkeypatch):
        """A halt at 3am is only actionable if it says what to go and look
        at."""
        runner = self._reconciled(
            tmp_path, monkeypatch, [fake_position("FOP", "ES", -8, right="C")],
        )
        with pytest.raises(ReconciliationError, match=r"-8 2025-06-10 5000C"):
            runner.run(max_cycles=5)


class TestPeriodicReconciliation:
    """The book is re-checked against the broker while the session runs.

    Between two connects the book is only this process's record of its own
    fills. An order the runner gave up waiting on, cancelled, and reported
    as unfilled can still be filled by the exchange -- so the record can be
    wrong in exactly the direction that stacks positions.
    """

    def _drifting(self, tmp_path, monkeypatch, appears_after=1, **live):
        """A broker that is flat at connect and grows a position later."""
        live.setdefault("reconcile_seconds", 0.001)
        runner = build_runner(tmp_path, **live)
        patch_session(monkeypatch, runner)
        calls = {"n": 0}

        def positions(*_):
            calls["n"] += 1
            return [] if calls["n"] <= appears_after else _straddle_rows(-8)

        runner.connection.ib.positions = positions
        return runner

    def test_a_position_the_book_does_not_know_about_halts_entries(
        self, tmp_path, monkeypatch
    ):
        runner = self._drifting(tmp_path, monkeypatch)
        runner.run(max_cycles=4)
        assert runner.strategy.halted, (
            "the book kept trading against a position it did not know the size of"
        )

    def test_the_halt_says_what_disagreed(self, tmp_path, monkeypatch):
        runner = self._drifting(tmp_path, monkeypatch)
        runner.run(max_cycles=4)
        assert "-8" in runner.strategy._halt_reason
        assert "+0" in runner.strategy._halt_reason

    def test_the_halt_stops_entries_without_stopping_the_runner(
        self, tmp_path, monkeypatch
    ):
        """Whatever is open still has to be hedged. A halt that stopped the
        poll loop would leave a straddle unmanaged, which is worse than the
        drift it was reacting to."""
        runner = self._drifting(tmp_path, monkeypatch)
        runner.run(max_cycles=6)
        assert runner.strategy.halted
        assert runner._cycles == 6, "the runner stopped polling"

    def test_a_book_that_agrees_with_the_broker_is_left_alone(
        self, tmp_path, monkeypatch
    ):
        runner = build_runner(tmp_path, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        runner.run(max_cycles=4)
        assert not runner.strategy.halted

    def test_checking_can_be_switched_off(self, tmp_path, monkeypatch):
        """``null`` is the old connect-only behaviour, kept reachable."""
        runner = self._drifting(tmp_path, monkeypatch, reconcile_seconds=None)
        runner.run(max_cycles=4)
        assert not runner.strategy.halted

    def test_hedge_drift_is_adopted_rather_than_halted_on(
        self, tmp_path, monkeypatch
    ):
        """The hedge is one signed number with no shape to get wrong, and
        the band puts it right on the next pass -- so it is corrected
        rather than treated as a reason to stop trading."""
        runner = build_runner(tmp_path, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        calls = {"n": 0}

        def positions(*_):
            calls["n"] += 1
            if calls["n"] <= 1:
                return []
            return [fake_position("FUT", "MES", -3, avg_cost=5000.0 * 5.0)]

        runner.connection.ib.positions = positions
        runner.run(max_cycles=4)
        assert runner.strategy.portfolio.hedge.quantity == -3
        assert not runner.strategy.halted


class TestEquityFromTheBroker:
    """Sizing runs off the account's own value, not off a YAML file."""

    def _with_account(self, tmp_path, monkeypatch, values, positions=(), **live):
        runner = build_runner(tmp_path, **live)
        patch_session(monkeypatch, runner)
        runner.connection.ib.account_values = dict(values)
        runner.connection.ib.set_positions(list(positions))
        runner.connection.net_liquidation = (
            lambda: values.get("NetLiquidation")
        )
        runner.connection.account_values = lambda: dict(values)
        runner.connection.hedge_price = lambda: 5000.0
        return runner

    def test_the_book_is_sized_against_net_liquidation(self, tmp_path, monkeypatch):
        """``starting_equity`` is a number in a config file, and the book's
        equity is that number plus whatever *this process* has realised. A
        reconnect rebuilds the strategy, so the realised part resets to
        zero -- after a drawdown and the nightly gateway restart, the
        strategy would size its next entry as though the loss had not
        happened."""
        runner = self._with_account(
            tmp_path, monkeypatch, {"NetLiquidation": 180_000.0}
        )
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.starting_equity == pytest.approx(180_000.0)

    def test_an_adopted_hedge_is_not_counted_twice(self, tmp_path, monkeypatch):
        """NetLiquidation already values open positions at market, so the
        hedge's open P&L has to come back out or it is counted once inside
        the broker's figure and again in ``unrealised``."""
        runner = self._with_account(
            tmp_path, monkeypatch, {"NetLiquidation": 180_000.0},
            positions=[fake_position("FUT", "MES", 10, avg_cost=4900.0 * 5.0)],
        )
        runner.run(max_cycles=1)
        book = runner.strategy.portfolio
        # 10 MES bought at 4900, marked at 5000: $5,000 of open profit.
        assert book.hedge.unrealised(5000.0, 5.0) == pytest.approx(5_000.0)
        assert book.equity(None, 5000.0) == pytest.approx(180_000.0)

    def test_a_missing_account_value_keeps_the_configured_equity(
        self, tmp_path, monkeypatch
    ):
        runner = self._with_account(tmp_path, monkeypatch, {})
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.starting_equity == pytest.approx(250_000.0)

    def test_an_account_read_that_raises_is_not_fatal(self, tmp_path, monkeypatch):
        runner = build_runner(tmp_path)
        patch_session(monkeypatch, runner)

        def boom():
            raise RuntimeError("no subscription")

        runner.connection.net_liquidation = boom
        runner.run(max_cycles=2)
        assert runner._cycles == 2

    def test_equity_is_not_rebased_while_a_straddle_is_open(
        self, tmp_path, monkeypatch
    ):
        """With a position open the subtraction cannot be done -- the option
        legs have no mark here -- so it waits until the book is flat. Entries
        only happen when flat, so sizing always sees the broker's figure."""
        runner = self._with_account(
            tmp_path, monkeypatch, {"NetLiquidation": 180_000.0},
            positions=_straddle_rows(-2),
        )
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.straddle is not None
        assert runner.strategy.portfolio.starting_equity == pytest.approx(250_000.0)


class TestBrokerMarginGuard:
    """The one check that does not go through the margin model."""

    def _with_margin(self, tmp_path, monkeypatch, held, nav=250_000.0):
        runner = build_runner(tmp_path, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        values = {"NetLiquidation": nav, "FullInitMarginReq": held}
        runner.connection.account_values = lambda: dict(values)
        runner.connection.net_liquidation = lambda: nav
        runner.connection.hedge_price = lambda: 5000.0
        return runner

    def test_margin_past_the_buying_power_limit_halts_entries(
        self, tmp_path, monkeypatch
    ):
        """What would have caught the original bug.

        A ``future_initial_margin`` carrying the micro contract's figure
        made every short straddle look a tenth as expensive as CME charges.
        The book was sized several times too large with the sizing
        arithmetic, the entry log and the equity curve all internally
        consistent and all wrong -- nothing derived from the model could
        have noticed. This asks the account instead.
        """
        runner = self._with_margin(tmp_path, monkeypatch, held=220_000.0)
        runner.run(max_cycles=4)
        assert runner.strategy.halted
        assert "88%" in runner.strategy._halt_reason

    def test_margin_inside_the_limit_is_left_alone(self, tmp_path, monkeypatch):
        runner = self._with_margin(tmp_path, monkeypatch, held=120_000.0)
        runner.run(max_cycles=4)
        assert not runner.strategy.halted

    def test_a_missing_margin_figure_is_not_treated_as_a_breach(
        self, tmp_path, monkeypatch
    ):
        """Absent data is not evidence of anything; halting on it would
        stop the walk whenever a subscription lapsed."""
        runner = build_runner(tmp_path, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        runner.connection.account_values = lambda: {}
        runner.connection.net_liquidation = lambda: None
        runner.run(max_cycles=4)
        assert not runner.strategy.halted


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


class TestTradeFeedSelection:
    """Which live tape the runner builds, and when it refuses to."""

    def runner(self, tmp_path, flow_source: str, oi_source: str = "ibkr"):
        cfg = Config()
        cfg.live.journal = False
        cfg.live.journal_dir = str(tmp_path)
        cfg.flow.source = flow_source
        cfg.data.open_interest = oi_source
        return LiveRunner(cfg)

    def test_no_feed_is_configured_by_default(self, tmp_path):
        assert self.runner(tmp_path, "none")._trade_feed(None) is None

    def test_databento_without_its_session_is_refused_not_left_empty(self, tmp_path):
        # The Databento tape rides the session the open-interest providers
        # own. Without one it would deliver nothing -- which looks exactly
        # like a quiet market and leaves every dealer sign on the prior, so
        # it is refused loudly instead.
        runner = self.runner(tmp_path, "databento", oi_source="ibkr")
        with pytest.raises(ValueError, match="data.open_interest must be"):
            runner._trade_feed(None)

    def test_the_error_names_the_ibkr_fallback_and_what_it_costs(self, tmp_path):
        runner = self.runner(tmp_path, "databento", oi_source="ibkr")
        with pytest.raises(ValueError, match="Lee-Ready"):
            runner._trade_feed(None)

    def test_databento_rides_the_session_when_one_is_running(self, tmp_path):
        from deltahedger.data.databento_source import (
            DatabentoSession,
            DatabentoTradeFeed,
        )

        runner = self.runner(tmp_path, "databento", oi_source="databento_flow")
        runner._databento = DatabentoSession(runner.cfg, runner.source)
        feed = runner._trade_feed(None)
        assert isinstance(feed, DatabentoTradeFeed)
        assert runner._databento._capture_trades is True


class TestPositionsTheBookCannotRepresent:
    """A future that is not the hedge, an option on another root.

    On 2026-09-09 a naked 0DTE call expired in the money and IBKR's paper
    processing left 11 ES in the account. The runner only looked for MES,
    so the reconnect logged "no existing positions to adopt" and the book
    reported flat and net delta zero against eleven outright contracts.
    """

    def _holding(self, tmp_path, monkeypatch, positions, **live):
        runner = build_runner(tmp_path, **live)
        patch_session(monkeypatch, runner)
        runner.connection.ib.set_positions(positions)
        return runner

    def test_an_outright_future_at_connect_halts_entries(self, tmp_path, monkeypatch):
        runner = self._holding(
            tmp_path, monkeypatch, [fake_position("FUT", "ES", 11, avg_cost=382_500.0)]
        )
        runner.run(max_cycles=1)
        assert runner.strategy.halted
        assert runner.strategy.portfolio.hedge.quantity == 0, "it is not the hedge"

    def test_the_halt_names_the_position(self, tmp_path, monkeypatch):
        runner = self._holding(
            tmp_path, monkeypatch, [fake_position("FUT", "ES", 11, avg_cost=382_500.0)]
        )
        runner.run(max_cycles=1)
        reason = runner.strategy.halt_reason
        assert "+11" in reason and "ES" in reason
        assert "cannot be marked, hedged or closed" in reason

    def test_an_option_on_another_root_halts_too(self, tmp_path, monkeypatch):
        runner = self._holding(
            tmp_path, monkeypatch, [fake_position("FOP", "NQ", -2, right="P")]
        )
        runner.run(max_cycles=1)
        assert runner.strategy.halted

    def test_one_appearing_mid_session_halts_entries(self, tmp_path, monkeypatch):
        runner = build_runner(tmp_path, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        calls = {"n": 0}

        def positions(*_):
            calls["n"] += 1
            return [] if calls["n"] <= 2 else [fake_position("FUT", "ES", 13)]

        runner.connection.ib.positions = positions
        runner.run(max_cycles=4)
        assert runner.strategy.halted
        assert "+13" in runner.strategy.halt_reason

    def test_equity_is_not_rebased_over_it(self, tmp_path, monkeypatch):
        """NetLiquidation carries that position's open P&L and the book has
        no mark to take it back out with, so the configured equity stands."""
        runner = self._holding(
            tmp_path, monkeypatch, [fake_position("FUT", "ES", 11, avg_cost=382_500.0)]
        )
        runner.connection.net_liquidation = lambda: 318_000.0
        runner.connection.account_values = lambda: {"NetLiquidation": 318_000.0}
        runner.connection.hedge_price = lambda: 5000.0
        runner.run(max_cycles=1)
        assert runner.strategy.portfolio.starting_equity == pytest.approx(250_000.0)


class TestTheHaltOutlivesAReconnect:
    """A halt is a finding about the account, and the account does not
    change because the gateway restarted at 02:00.

    The strategy that carries the flag is rebuilt on every reconnect. On
    2026-09-10 the runner came back from the nightly restart un-halted,
    against the same account, and only the margin timer -- five minutes
    later -- stopped it trading.
    """

    def test_the_reconnected_strategy_starts_halted(self, tmp_path, monkeypatch):
        runner = build_runner(tmp_path, drop_after=3, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        connection = runner.connection
        calls = {"n": 0}

        def positions(*_):
            # First session: flat at connect, then a straddle the book did
            # not place. Second session: flat again -- nothing to halt on.
            calls["n"] += 1
            if connection.connects == 1 and calls["n"] > 2:
                return _straddle_rows(-8)
            return []

        connection.ib.positions = positions
        runner.run(max_cycles=6)
        assert connection.connects >= 2, "the drop did not reconnect"
        assert runner.strategy.halted
        assert "found earlier in this run" in runner.strategy.halt_reason
        assert "-8" in runner.strategy.halt_reason


class TestReconciliationAfterAFailedLeg:
    """The strategy just said a leg did not fill. That is exactly when the
    book and the account are most likely to disagree, and the next poll
    is five seconds away -- the timer is for drift nobody saw coming."""

    def test_a_failed_leg_triggers_a_check_before_the_next_poll(
        self, tmp_path, monkeypatch
    ):
        import deltahedger.live.runner as module
        from deltahedger.strategy import GexStraddleStrategy

        runner = build_runner(tmp_path, reconcile_seconds=10**6)
        patch_session(monkeypatch, runner)
        original = GexStraddleStrategy.on_bar

        def on_bar(self, bar, execution):
            state = original(self, bar, execution)
            self._record(OPEN, "entry_failed", "the call leg did not fill")
            return state

        monkeypatch.setattr(GexStraddleStrategy, "on_bar", on_bar)
        calls = {"n": 0}

        def positions(*_):
            calls["n"] += 1
            return [] if calls["n"] <= 2 else _straddle_rows(13)

        runner.connection.ib.positions = positions
        runner.run(max_cycles=2)
        assert runner.strategy.halted, "the drift waited for the timer"
        assert "+13" in runner.strategy.halt_reason

    def test_an_ordinary_poll_does_not_reconcile_early(self, tmp_path, monkeypatch):
        """Only a failure or a squared pair brings the check forward.
        (The stubbed execution fails every leg it is handed, so the bars
        here have their failures scrubbed to look like quiet ones.)"""
        from deltahedger.strategy import GexStraddleStrategy

        runner = build_runner(tmp_path, reconcile_seconds=10**6)
        patch_session(monkeypatch, runner)
        original = GexStraddleStrategy.on_bar

        def on_bar(self, bar, execution):
            state = original(self, bar, execution)
            self.events[:] = [e for e in self.events if not e.kind.endswith("_failed")]
            return state

        monkeypatch.setattr(GexStraddleStrategy, "on_bar", on_bar)
        calls = {"n": 0}

        def positions(*_):
            calls["n"] += 1
            return [] if calls["n"] <= 2 else _straddle_rows(13)

        runner.connection.ib.positions = positions
        runner.run(max_cycles=3)
        assert not runner.strategy.halted


class TestMarginIsCheckedAtConnect:
    def test_an_account_already_past_the_limit_is_halted_before_the_first_poll(
        self, tmp_path, monkeypatch
    ):
        runner = build_runner(tmp_path, reconcile_seconds=None)
        patch_session(monkeypatch, runner)
        values = {"NetLiquidation": 313_760.0, "FullInitMarginReq": 378_858.0}
        runner.connection.account_values = lambda: dict(values)
        runner.connection.net_liquidation = lambda: 313_760.0
        runner.connection.hedge_price = lambda: 5000.0
        runner.run(max_cycles=1)
        assert runner.strategy.halted
        assert "121%" in runner.strategy.halt_reason


class TestHedgeDriftAcrossContractMonths:
    def test_two_hedge_rows_are_one_position(self, tmp_path, monkeypatch):
        """IBKR reports one row per contract month; after a quarterly roll
        the hedge is two of them."""
        runner = build_runner(tmp_path, reconcile_seconds=0.001)
        patch_session(monkeypatch, runner)
        calls = {"n": 0}

        def positions(*_):
            calls["n"] += 1
            if calls["n"] <= 2:
                return []
            return [
                fake_position("FUT", "MES", -3, avg_cost=5000.0 * 5.0, expiry="20250620"),
                fake_position("FUT", "MES", -2, avg_cost=5100.0 * 5.0, expiry="20250919"),
            ]

        runner.connection.ib.positions = positions
        runner.run(max_cycles=4)
        hedge = runner.strategy.portfolio.hedge
        assert hedge.quantity == -5
        assert hedge.avg_price == pytest.approx(5040.0)
        assert not runner.strategy.halted
