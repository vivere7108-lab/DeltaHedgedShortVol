"""End-to-end tests.

The load-bearing one is ``test_zero_edge_produces_no_pnl``.  The synthetic
generator draws returns using the same volatility it reports as implied, so
a correctly delta-hedged short position has no edge by construction -- *if*
the strategy prices every strike off that same flat vol.  It does not by
default: ``vol.skew_slope`` (-1.5) prices out-of-the-money puts richer than
the generator's flat realized vol, which turns out to be a real, measurable,
*compounding* edge in its own right (traced empirically: +10% of equity over
120 days with zero other edge, shrinking to ~0 once the surface is flattened
to match the generator). That is a genuine property of selling skew-priced
options against a flat-vol process, not a bug -- but it means a test meant
to validate the hedge/greeks/accounting machinery has to flatten the vol
surface first, or it partly measures the skew assumption's own economic
effect instead. ``test_zero_edge_produces_no_pnl`` does that; a separate
test below documents the skew bias itself as an expected, tested property
rather than a silent gotcha. It is also why every backtest run against real
market data in this system carries this same skew-vs-true-market-skew
uncertainty -- see the README's "Known approximations".

If the hedger, the greeks, the delta-unit arithmetic or the P&L accounting
were wrong, the flattened-surface number would not come out near zero -- it
is the single strongest statement this suite makes about the system being
right.
"""

import pytest

from deltahedger.backtest import run_backtest
from deltahedger.config import Config
from deltahedger.data import build_source
from deltahedger.data.synthetic import SyntheticSource


def synthetic(days=10, **overrides) -> Config:
    cfg = Config()
    cfg.data.source = "synthetic"
    cfg.data.synthetic_days = days
    cfg.starting_equity = 250_000.0
    for dotted, value in overrides.items():
        section, _, attr = dotted.partition(".")
        setattr(getattr(cfg, section) if attr else cfg, attr or section, value)
    return cfg


class TestEndToEnd:
    def test_a_backtest_runs_and_produces_bars(self):
        result = run_backtest(synthetic())
        assert len(result.bars) > 0
        assert len(result.daily) == 10

    def test_it_opens_one_position_per_session(self):
        result = run_backtest(synthetic(days=7))
        assert result.metrics.entries == 7

    def test_every_entry_is_matched_by_an_exit(self):
        result = run_backtest(synthetic(days=7))
        kinds = result.events["kind"].value_counts()
        assert kinds.get("entry", 0) == kinds.get("exit", 0)

    def test_it_ends_flat(self):
        """No position may survive the last bar of the backtest."""
        result = run_backtest(synthetic())
        assert result.bars["put_contracts"].iloc[-1] == 0
        assert result.bars["hedge_contracts"].iloc[-1] == 0

    def test_it_is_deterministic(self):
        first = run_backtest(synthetic()).metrics.final_equity
        second = run_backtest(synthetic()).metrics.final_equity
        assert first == second

    def test_results_can_be_saved(self, tmp_path):
        run_backtest(synthetic(days=3)).save(tmp_path)
        for name in ("bars.csv", "events.csv", "fills.csv", "daily.csv", "summary.txt"):
            assert (tmp_path / name).exists(), name


class TestCorrectness:
    def test_zero_edge_produces_no_pnl(self):
        """Realised vol equals implied vol in the generator and the vol
        surface is flattened to match it (see the module docstring for why
        the default skewed surface is the wrong tool for this check), so a
        hedged short-vol book must break even once costs are removed."""
        cfg = synthetic(days=20)
        cfg.costs.enabled = False
        cfg.vol.skew_slope = 0.0
        source = SyntheticSource(cfg.data, cfg.source, vol_of_vol=0.0, vol_return_beta=0.0)
        result = run_backtest(cfg, source)
        pnl = result.metrics.final_equity - result.metrics.starting_equity
        assert abs(pnl) < 0.03 * cfg.starting_equity, (
            f"a zero-edge market produced ${pnl:,.0f}: the hedge, the greeks or "
            "the P&L accounting is wrong"
        )

    def test_selling_skew_against_a_flat_realized_process_is_a_real_bias(self):
        """The default vol.skew_slope (-1.5) prices out-of-the-money puts
        richer than the generator's flat realized vol -- a genuine,
        compounding edge that has nothing to do with market dynamics, worth
        keeping visible as an expected, tested property rather than a
        silent gotcha the next person re-discovers by surprise.
        """
        cfg = synthetic(days=60)
        cfg.costs.enabled = False
        source = SyntheticSource(cfg.data, cfg.source, vol_of_vol=0.0, vol_return_beta=0.0)
        result = run_backtest(cfg, source)
        pnl = result.metrics.final_equity - result.metrics.starting_equity
        assert pnl > 0.02 * cfg.starting_equity, (
            "expected the skew assumption's structural edge to show up over "
            "60 days against a flat-vol process; if this no longer holds, "
            "the module docstring's explanation may need revisiting"
        )

    def test_the_option_and_hedge_legs_offset_each_other(self):
        """Delta hedging converts the short put into a vol bet: the two legs
        must be strongly opposed, not independently profitable."""
        cfg = synthetic(days=20)
        cfg.costs.enabled = False
        m = run_backtest(cfg).metrics
        assert m.option_pnl * m.hedge_pnl < 0, "legs did not offset"

    def test_costs_only_ever_reduce_pnl(self):
        without = run_backtest(synthetic(days=10, **{"costs.enabled": False}))
        with_costs = run_backtest(synthetic(days=10))
        assert with_costs.metrics.final_equity < without.metrics.final_equity

    def test_more_buying_power_means_more_contracts(self):
        small = run_backtest(synthetic(days=5, **{"sizing.buying_power_pct": 0.05}))
        large = run_backtest(synthetic(days=5, **{"sizing.buying_power_pct": 0.40}))
        assert abs(large.bars["put_contracts"].min()) > abs(
            small.bars["put_contracts"].min()
        )

    def test_zero_position_when_buying_power_cannot_cover_a_contract(self):
        cfg = synthetic(days=5)
        cfg.starting_equity = 5_000.0
        result = run_backtest(cfg)
        assert result.metrics.entries == 0
        assert (result.events["kind"] == "entry_skipped").any()


class TestHedgeBehaviour:
    def test_net_delta_tracks_the_target(self):
        result = run_backtest(synthetic(days=10))
        held = result.bars[result.bars["put_contracts"] != 0]
        assert held["net_delta_units"].mean() == pytest.approx(20.0, abs=3.0)

    def test_the_residual_never_exceeds_half_a_hedge_contract(self, es):
        """The granularity bound the hedger promises, verified over a run."""
        result = run_backtest(synthetic(days=10))
        assert result.metrics.max_abs_delta_error <= es.hedge_quantum / 2 + 1e-6

    def test_unhedged_delta_is_far_larger(self):
        """Without the hedge the book runs the full short-put delta."""
        cfg = synthetic(days=5)
        cfg.hedge.max_hedge_contracts = 1
        cfg.hedge.min_hedge_contracts = 10_000  # effectively disables hedging
        result = run_backtest(cfg)
        held = result.bars[result.bars["put_contracts"] != 0]
        assert held["net_delta_units"].abs().max() > 100

    def test_a_wider_band_trades_less(self):
        tight = run_backtest(synthetic(days=10, **{"hedge.band": 5.0}))
        loose = run_backtest(synthetic(days=10, **{"hedge.band": 40.0}))
        assert loose.metrics.hedges < tight.metrics.hedges

    def test_a_band_below_half_the_quantum_is_inert(self, es):
        """Documented behaviour: MES granularity makes +/-3 and +/-5 identical."""
        narrow = run_backtest(synthetic(days=10, **{"hedge.band": 1.0}))
        default = run_backtest(synthetic(days=10, **{"hedge.band": 3.0}))
        at_quantum = run_backtest(synthetic(days=10, **{"hedge.band": 5.0}))
        assert narrow.metrics.hedges == default.metrics.hedges == at_quantum.metrics.hedges

    def test_the_hedge_is_flattened_with_the_option(self):
        result = run_backtest(synthetic(days=5))
        exits = result.events[result.events["kind"] == "exit"]
        assert len(exits) > 0
        for timestamp in exits["timestamp"]:
            after = result.bars[result.bars["timestamp"] >= timestamp]
            assert after.iloc[0]["hedge_contracts"] == 0


class TestWindowing:
    def test_a_date_window_is_respected(self):
        cfg = synthetic(days=20)
        cfg.start_date = "2025-01-06"
        cfg.end_date = "2025-01-10"
        result = run_backtest(cfg)
        assert result.bars["timestamp"].min().date().isoformat() >= "2025-01-06"
        assert result.bars["timestamp"].max().date().isoformat() <= "2025-01-10"

    def test_a_window_with_no_bars_returns_empty_metrics(self):
        cfg = synthetic(days=5)
        cfg.start_date = "2030-01-01"
        cfg.end_date = "2030-01-02"
        result = run_backtest(cfg)
        assert result.metrics.final_equity == cfg.starting_equity


class TestDataSources:
    def test_bars_are_time_ordered(self):
        cfg = synthetic(days=3)
        bars = list(build_source(cfg, cfg.source).bars())
        assert all(a.timestamp < b.timestamp for a, b in zip(bars, bars[1:]))

    def test_every_bar_is_timezone_aware(self):
        cfg = synthetic(days=2)
        assert all(b.timestamp.tzinfo is not None for b in build_source(cfg, cfg.source).bars())

    def test_out_of_order_bars_are_rejected(self):
        from datetime import datetime, timezone
        from deltahedger.data.base import MarketBar, ensure_sorted

        early = MarketBar(datetime(2025, 1, 2, 10, tzinfo=timezone.utc), 1, 1, 1, 1, 0.15)
        late = MarketBar(datetime(2025, 1, 2, 9, tzinfo=timezone.utc), 1, 1, 1, 1, 0.15)
        with pytest.raises(ValueError, match="ascending"):
            list(ensure_sorted([early, late]))

    def test_a_naive_timestamp_is_rejected(self):
        from datetime import datetime
        from deltahedger.data.base import MarketBar

        with pytest.raises(ValueError, match="timezone-aware"):
            MarketBar(datetime(2025, 1, 2, 10), 1, 1, 1, 1, 0.15)

    def test_an_unknown_source_is_rejected(self):
        cfg = synthetic()
        cfg.data.source = "carrier-pigeon"
        with pytest.raises(ValueError, match="unknown data source"):
            build_source(cfg, cfg.source)

    def test_csv_round_trips(self, tmp_path):
        cfg = synthetic(days=3)
        rows = ["timestamp,open,high,low,close,atm_iv"]
        for bar in build_source(cfg, cfg.source).bars():
            rows.append(
                f"{bar.timestamp.isoformat()},{bar.open},{bar.high},"
                f"{bar.low},{bar.close},{bar.atm_iv}"
            )
        path = tmp_path / "bars.csv"
        path.write_text("\n".join(rows))

        cfg.data.source = "csv"
        cfg.data.csv_path = str(path)
        assert len(list(build_source(cfg, cfg.source).bars())) == len(rows) - 1

    def test_a_csv_missing_a_column_is_rejected(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("timestamp,open,close\n2025-01-02T10:00:00,1,1\n")
        cfg = synthetic()
        cfg.data.source = "csv"
        cfg.data.csv_path = str(path)
        with pytest.raises(ValueError, match="missing column"):
            list(build_source(cfg, cfg.source).bars())
