"""End-to-end tests.

The load-bearing ones are in ``TestCorrectness``.  The synthetic generator
draws returns using the same volatility it reports as implied, so a
correctly delta-hedged straddle has no edge by construction -- in *either*
direction.  If the hedger, the greeks, the delta-unit arithmetic or the P&L
accounting were wrong, that would not hold.

One thing changed with the move to straddles and is worth stating plainly.
An ATM straddle carries several times the gamma of a 20-delta put, so
rebalancing only once per 5-minute bar leaves a much larger discrete-hedging
residual: a single 20-day run lands within a few percent of zero rather than
a fraction of one percent, and that residual is real, not a defect.  It is
path noise with mean zero, which is why the assertion below is made across a
panel of seeds rather than on any one of them -- averaging is what removes
the noise while still catching a genuine bias.
``test_the_hedging_residual_shrinks_with_rebalance_frequency`` pins the
other half of the claim: that the residual is a function of how often we
hedge, not of the accounting.
"""

import statistics
from datetime import time

import pytest

from deltahedger.backtest import run_backtest
from deltahedger.config import Config
from deltahedger.data import build_source
from deltahedger.data.synthetic import SyntheticSource
from deltahedger.gex import NEGATIVE, POSITIVE


def synthetic(days=10, **overrides) -> Config:
    cfg = Config()
    cfg.data.source = "synthetic"
    cfg.data.synthetic_days = days
    cfg.starting_equity = 250_000.0
    for dotted, value in overrides.items():
        section, _, attr = dotted.partition(".")
        setattr(getattr(cfg, section) if attr else cfg, attr or section, value)
    return cfg


def unlimited(cfg: Config) -> Config:
    """Lift the per-session entry cap and the daily loss limit.

    Both are right for trading and wrong for a test of the end-of-day
    rules: a book sized to the margin limit can trip a 5% daily loss limit
    inside an hour of generated data, and a flickering ungated regime can
    use up three entries by lunchtime -- after which ``_try_entry`` returns
    before it reaches the rule under test, and records nothing.
    """
    cfg.strategy.max_entries_per_session = 1_000
    cfg.strategy.daily_loss_limit_pct = None
    return cfg


def ungated(cfg: Config) -> Config:
    """Switch off all four stand-aside gates.

    The correctness tests below check the *arithmetic* -- the hedger, the
    P&L accounting, the sign of the greeks -- the same way
    ``costs.enabled = False`` isolates it from commissions and
    ``pin_implied_vol`` isolates it from vega drift.  The gates are a
    fourth thing that can differ between two otherwise-identical runs, and
    one of them (``persistence``) counts *bars*, which would silently
    confound a test that varies the bar size on purpose -- see
    ``test_the_hedging_residual_shrinks_with_rebalance_frequency``.  With a
    handful of gated trades over a run, a single differing entry can also
    swing a small-sample comparison either way, which is what the other
    three fixed here were actually catching.
    """
    cfg.gates.confidence = False
    cfg.gates.flip_distance = False
    cfg.gates.ensemble = False
    cfg.gates.persistence = False
    cfg.gates.entry_window = False
    return cfg


def pin_implied_vol(cfg):
    """Make realised vol equal implied vol exactly, and stay there.

    The shipped generator moves implied vol around (and leans it against
    returns), which is realistic but means entry IV and subsequent realised
    vol are not the same number. A straddle is a large vega position, so
    that difference shows up as a systematic P&L which belongs to the
    generator rather than to the strategy -- and it is big enough to swamp
    the arithmetic these tests exist to check. Same settings as
    configs/es_zero_edge.yaml.
    """
    cfg.data.synthetic_vol_of_vol = 0.0
    cfg.data.synthetic_vol_mean_reversion = 0.0
    cfg.data.synthetic_vol_return_beta = 0.0
    return SyntheticSource(cfg.data, cfg.source)


def zero_edge_residual(seed, days=10):
    cfg = ungated(synthetic(days=days, **{"costs.enabled": False}))
    cfg.data.synthetic_seed = seed
    m = run_backtest(cfg, source=pin_implied_vol(cfg)).metrics
    return (m.final_equity - m.starting_equity) / m.starting_equity


class TestEndToEnd:
    def test_a_backtest_runs_and_produces_bars(self):
        result = run_backtest(synthetic())
        assert len(result.bars) > 0
        assert len(result.daily) == 10

    def test_it_trades_on_most_sessions(self):
        """Not every session: a chain that reads neutral is stood aside."""
        result = run_backtest(synthetic(days=7))
        assert 0 < result.metrics.entries

    def test_every_entry_is_matched_by_an_exit_or_is_still_open(self):
        """A position rolled into tomorrow's series at 15:45 spans the end
        of a backtest window -- that is an honest "not finished yet", not a
        leak. At most one entry (the last one) may be unmatched, and only
        if the book is still holding it."""
        result = run_backtest(synthetic(days=7))
        kinds = result.events["kind"].value_counts()
        entries, exits = kinds.get("entry", 0), kinds.get("exit", 0)
        still_open = result.bars["straddle_contracts"].iloc[-1] != 0
        assert entries - exits == (1 if still_open else 0)

    def test_it_ends_flat_unless_a_position_was_rolled_into_tomorrow(self):
        """A position open at the last bar (16:00) can only be one rolled
        into the next session's series at the buffer -- today's has been
        closed by then -- and it must not be a leak, nor a hedge left
        unmatched to it."""
        result = run_backtest(synthetic())
        last = result.bars.iloc[-1]
        if last["straddle_contracts"] == 0:
            assert last["hedge_contracts"] == 0
        else:
            assert last["days_to_expiry"] == 1
            assert last["hedge_contracts"] != 0 or abs(last["net_delta_units"]) <= max(
                last["band_half_width"], 5.0
            )

    def test_todays_position_is_closed_at_the_buffer_and_rolled(self):
        """Every session ends the same way: an exit '15m to expiry' on the
        15:45 bar, and -- when the read allows and tomorrow is a session --
        an entry into the 1DTE series inside the roll window, never before."""
        result = run_backtest(unlimited(synthetic(days=10)))
        events = result.events
        exits = events[(events["kind"] == "exit") & events["detail"].str.contains("to expiry")]
        assert not exits.empty
        assert all(ts.time().isoformat() == "15:45:00" for ts in exits["timestamp"])
        rolls = events[(events["kind"] == "entry") & events["detail"].str.contains("1DTE")]
        assert not rolls.empty
        assert all(ts.time() >= time(15, 45) for ts in rolls["timestamp"])
        assert (rolls["timestamp"].dt.time == time(15, 45)).any()

    def test_no_position_is_carried_over_a_weekend(self):
        """The generated run starts on Thursday 2025-01-02: the Friday
        bells must leave the book flat, and the Friday roll must be
        refused for the weekend gap by name."""
        result = run_backtest(unlimited(synthetic(days=10)))
        bars = result.bars
        fridays = bars[bars["timestamp"].dt.weekday == 4]
        last_of_each = fridays.groupby(fridays["timestamp"].dt.date).tail(1)
        assert (last_of_each["straddle_contracts"] == 0).all()
        assert (last_of_each["hedge_contracts"] == 0).all()
        blocked = result.events[result.events["gate"] == "weekend_gap"]
        assert not blocked.empty
        assert blocked["detail"].str.contains("weekend or holiday").all()

    def test_an_event_blackout_closes_and_blocks(self):
        """A calendar with one event at 14:00 on the first session: the
        position is closed at 13:45, nothing is opened until after 14:15,
        and the bars in between are flagged."""
        cfg = unlimited(ungated(synthetic(days=1)))
        cfg.strategy.events = ["2025-01-02 14:00 test event"]
        result = run_backtest(cfg)
        bars = result.bars
        inside = bars[bars["event_blackout"] != ""]
        assert len(inside) == 7  # 13:45 .. 14:15 on 5-minute bars
        assert (inside["straddle_contracts"] == 0).all()
        assert result.metrics.bars_in_event_blackout == 7
        blocked = result.events[result.events["gate"] == "event_blackout"]
        assert len(blocked) == 7

    def test_both_regimes_are_exercised(self):
        """A run that only ever saw one regime has tested half the strategy;
        the generated chain is built to swing across both."""
        m = run_backtest(synthetic(days=20)).metrics
        assert m.long_gamma_trades > 0 and m.short_gamma_trades > 0

    def test_every_bar_carries_its_gex_read(self):
        """There is no bell to run out the clock on: once today's series
        is inside the buffer the traded series is tomorrow's, so every bar
        -- including the 16:00 one -- gets a profile. A gap here would mean
        the tenor selection or the blend broke, not that a series settled."""
        result = run_backtest(synthetic(days=5))
        assert result.bars["gex_total"].notna().all()
        assert set(result.bars["gex_regime"]) <= {POSITIVE, NEGATIVE, "neutral"}

    def test_it_is_deterministic(self):
        first = run_backtest(synthetic()).metrics.final_equity
        second = run_backtest(synthetic()).metrics.final_equity
        assert first == second

    def test_results_can_be_saved(self, tmp_path):
        run_backtest(synthetic(days=3)).save(tmp_path)
        for name in ("bars.csv", "events.csv", "fills.csv", "daily.csv", "summary.txt"):
            assert (tmp_path / name).exists(), name


class TestCorrectness:
    def test_zero_edge_produces_no_pnl_on_average(self):
        """Realised vol equals implied vol in the generator, so a hedged
        straddle must break even once costs are removed.

        Measured across a panel of seeds rather than on one: discrete
        hedging leaves a path-dependent residual of a few percent per run,
        which is real. What must not survive averaging is a *bias* -- and a
        sign error, a delta-unit slip or a P&L mis-attribution would all show
        up as one.
        """
        residuals = [zero_edge_residual(seed) for seed in range(1, 17)]
        mean = statistics.mean(residuals)
        standard_error = statistics.stdev(residuals) / len(residuals) ** 0.5
        assert abs(mean) < 3.0 * standard_error, (
            f"a zero-edge market produced a mean of {mean:+.3%} across "
            f"{len(residuals)} seeds ({abs(mean) / standard_error:.1f} standard "
            "errors from zero): the hedge, the greeks or the P&L accounting is "
            "biased"
        )

    def test_the_hedging_residual_shrinks_with_rebalance_frequency(self):
        """The other half of the claim above: the per-run residual is
        discrete-*time* hedging error, so hedging more often must shrink it.

        If it did not, the dispersion would be an accounting fault rather
        than a known property of rebalancing on a 5-minute grid.

        Confined to positions that open and close same-day (the roll is
        switched off), which isolates *intraday* discretisation from a
        second source of dispersion: the overnight gap the rolled 1DTE
        position spans. A position carried past the close is not
        rebalanced again until the next session's first bar -- the
        backtest's bar sources are RTH-only, so there is nothing to
        rebalance against overnight even in principle, unlike the live
        runner, which keeps polling and hedging around the clock. That gap
        is real risk, but it is not *discretisation* error and it does not
        shrink by sampling the session more finely, so mixing it into this
        comparison would test the wrong thing. See "A single run does not
        measure the hedge" in the README for the overnight-inclusive number.
        """
        def rms(bar_size):
            values = []
            for seed in (3, 7, 11, 19, 23):
                cfg = ungated(synthetic(days=10, **{"costs.enabled": False}))
                cfg.strategy.max_days_to_expiry = 0
                cfg.strategy.roll_at_expiry = False
                cfg.data.synthetic_seed = seed
                cfg.data.bar_size = bar_size
                m = run_backtest(cfg, source=pin_implied_vol(cfg)).metrics
                values.append(
                    (m.final_equity - m.starting_equity) / m.starting_equity
                )
            return (sum(v * v for v in values) / len(values)) ** 0.5

        # Gated, ``persistence`` counts bars rather than time, so changing
        # the bar size changes what "3 in a row" means in wall-clock terms
        # and would pick a different set of entries at each frequency --
        # confounding the one thing this test varies on purpose.
        assert rms("1 min") < rms("5 mins") < rms("15 mins")

    def test_the_overnight_gap_is_unbiased_even_though_it_is_unhedged(self):
        """The multi-day, overnight-inclusive residual is much larger than
        the intraday-only one above -- carrying a position through nights
        the backtest cannot rebalance against adds real dispersion -- but
        it must still average to zero. A biased overnight step (drift
        mismatched to the option pricer's clock) would show up here as a
        mean the intraday-only panel would never catch, because that panel
        never crosses a session boundary at all.
        """
        residuals = [zero_edge_residual(seed, days=15) for seed in range(101, 121)]
        mean = statistics.mean(residuals)
        standard_error = statistics.stdev(residuals) / len(residuals) ** 0.5
        assert abs(mean) < 3.0 * standard_error, (
            f"the overnight-inclusive zero-edge residual has a mean of "
            f"{mean:+.3%} across {len(residuals)} seeds "
            f"({abs(mean) / standard_error:.1f} standard errors from zero)"
        )

    def test_the_option_and_hedge_legs_offset_each_other(self):
        """Delta hedging converts the straddle into a pure vol bet: the two
        legs must be strongly opposed, not independently profitable."""
        cfg = ungated(synthetic(days=20))
        cfg.costs.enabled = False
        m = run_backtest(cfg, source=pin_implied_vol(cfg)).metrics
        assert m.option_pnl * m.hedge_pnl < 0, "legs did not offset"

    def test_the_two_regimes_carry_opposite_greeks(self):
        """The signature of the whole design: the long (negative-GEX) side
        owns gamma and pays theta, the short (positive-GEX) side is the
        mirror of it. If these ever shared a sign, the regime is not
        reaching the position.

        Run at a zero rate deliberately. Under a positive rate Black-76
        gives a deep in-the-money European put positive theta -- the
        discounted intrinsic grows into expiry -- which is correct and would
        make an exact sign assertion false for reasons that have nothing to
        do with this strategy.
        """
        cfg = ungated(synthetic(days=20))
        cfg.costs.enabled = False
        cfg.risk_free_rate = 0.0
        result = run_backtest(cfg, source=pin_implied_vol(cfg))

        long_bars = result.bars[result.bars["direction"] > 0]
        short_bars = result.bars[result.bars["direction"] < 0]
        assert not long_bars.empty and not short_bars.empty

        assert (long_bars["gamma_units"] > 0).all()
        assert (long_bars["theta_dollars"] < 0).all()
        assert (short_bars["gamma_units"] < 0).all()
        assert (short_bars["theta_dollars"] > 0).all()

    def test_vega_follows_the_direction_too(self):
        cfg = ungated(synthetic(days=20, **{"costs.enabled": False}))
        cfg.risk_free_rate = 0.0
        result = run_backtest(cfg, source=pin_implied_vol(cfg))
        assert (result.bars[result.bars["direction"] > 0]["vega_dollars"] > 0).all()
        assert (result.bars[result.bars["direction"] < 0]["vega_dollars"] < 0).all()

    def test_costs_only_ever_reduce_pnl(self):
        """With the gates on, a fee-driven change to one stop's timing can
        also change which entries a persistence or confidence read happens
        to catch -- a second-order effect real money would also feel, but
        not the one this test exists to isolate. Ungated, the only thing
        that differs between the two runs is the cost model."""
        without = run_backtest(ungated(synthetic(days=10, **{"costs.enabled": False})))
        with_costs = run_backtest(ungated(synthetic(days=10)))
        assert with_costs.metrics.final_equity < without.metrics.final_equity

    def test_more_buying_power_means_more_contracts(self):
        small = run_backtest(synthetic(days=5, **{"sizing.buying_power_pct": 0.05}))
        large = run_backtest(synthetic(days=5, **{"sizing.buying_power_pct": 0.40}))
        assert large.bars["straddle_contracts"].abs().max() > (
            small.bars["straddle_contracts"].abs().max()
        )

    def test_entries_are_declined_when_buying_power_cannot_cover_them(self):
        """Note this bites less hard than it did on a short put. A long
        straddle costs its debit, and an ATM 0DTE straddle late in the
        session is cheap, so a tiny account can still afford one -- what must
        hold is that the ones it cannot afford are declined with a reason,
        never silently downsized or taken anyway."""
        cfg = synthetic(days=5)
        cfg.starting_equity = 3_000.0
        result = run_backtest(cfg)
        skipped = result.events[result.events["kind"] == "entry_skipped"]
        assert skipped["detail"].str.contains("buying power supports").any()
        for row in result.events[result.events["kind"] == "entry"].itertuples():
            assert "of $" in row.detail  # every entry states the budget it fit inside


def uncapped(cfg: Config) -> Config:
    """Lift the per-order hedge cap so the band's residual bound is testable.

    A book sized to the margin limit can need more than the per-order cap
    in one bar near expiry; what the cap holds back is sent on the next
    pass. That is the right live behaviour and it is tested on its own
    below, but it would break a test of the bound the *band* promises.
    """
    cfg.hedge.max_hedge_contracts = 100_000
    return cfg


class TestHedgeBehaviour:
    def test_net_delta_is_held_at_neutral(self):
        """Under Whalley-Wilmott the band is a few hundred units on a book
        sized to the margin limit, so "neutral" means "well inside it" --
        the mean net delta over held bars sits within half the median band."""
        result = run_backtest(uncapped(synthetic(days=10)))
        held = result.bars[result.bars["straddle_contracts"] != 0]
        assert abs(held["net_delta_units"].mean()) <= 0.5 * result.metrics.band_half_width

    def test_the_residual_never_exceeds_the_binding_constraint(self, es):
        """The bound the hedger promises, verified bar by bar over a run.

        Two things can bind, and the wider one wins. The hedger stops once
        no whole contract lands closer to target, which caps the residual at
        half a contract; but it does not act at all inside the band, which
        caps it at the band. Under Whalley-Wilmott the band is a per-bar
        number, so the bound is checked against the band each bar was
        actually held to.
        """
        result = run_backtest(uncapped(synthetic(days=10)))
        held = result.bars[result.bars["straddle_contracts"] != 0]
        bound = held["band_half_width"].clip(lower=es.hedge_quantum / 2)
        assert (held["delta_error"].abs() <= bound + 1e-6).all()

    def test_the_per_order_cap_is_honoured_and_caught_up(self):
        """What the cap holds back is sent on the next pass rather than
        forgotten: no single hedge fill exceeds it, and the residual left
        behind is gone within a few bars."""
        cfg = synthetic(days=10)
        cfg.hedge.max_hedge_contracts = 100
        result = run_backtest(cfg)
        hedges = result.fills[result.fills["instrument"] == "hedge"]
        assert (hedges["quantity"].abs() <= 100).all()
        assert result.metrics.hedges > 0
        # The flatten on exit is sent in capped orders too, and says so.
        flattens = result.events[result.events["kind"] == "hedge_flatten"]
        assert flattens["detail"].str.contains("in \\d+ orders").any()

    def test_a_band_narrower_than_the_quantum_is_bounded_by_the_contract(self, es):
        """The other side of the same rule, under the fixed model."""
        result = run_backtest(uncapped(synthetic(
            days=10, **{"hedge.band": 1.0, "hedge.band_model": "fixed"}
        )))
        assert result.metrics.max_abs_delta_error <= es.hedge_quantum / 2 + 1e-6

    def test_neither_regime_is_hedged_more_tightly_than_the_other(self):
        """Under the fixed model, one band applied symmetrically: if the
        long side were held tighter than the short side the regime
        comparison would be measuring the hedger rather than the signal.
        (Under Whalley-Wilmott the band is even in gamma -- pinned in
        ``test_hedger`` -- but the two branches carry different gamma, so
        their bands differ in delta units by construction.)"""
        result = run_backtest(uncapped(synthetic(days=20, **{
            "hedge.band_model": "fixed", "hedge.band": 10.0,
            # At the old allocation: at the margin limit a 5-minute bar
            # moves a bigger book further outside a +/-10 band on the
            # heavier branch, which is size, not asymmetry in the hedger.
            "sizing.buying_power_pct": 0.15,
        })))
        held = result.bars[result.bars["straddle_contracts"] != 0]
        errors = held.groupby(held["direction"])["delta_error"].apply(
            lambda column: column.abs().mean()
        )
        assert len(errors) == 2
        assert errors.max() - errors.min() < 1.0

    def test_the_band_widens_with_the_gamma_of_the_book(self):
        """The Whalley-Wilmott property, end to end: a book with more gamma
        is held to a wider band in delta units, so a bigger allocation --
        more straddles, more gamma -- gets a wider median band."""
        small = run_backtest(synthetic(days=10, **{"sizing.buying_power_pct": 0.10}))
        large = run_backtest(synthetic(days=10, **{"sizing.buying_power_pct": 0.80}))
        assert large.metrics.band_half_width > small.metrics.band_half_width

    def test_a_higher_risk_aversion_hedges_more_often(self):
        timid = run_backtest(synthetic(days=10, **{"hedge.risk_aversion": 1.0}))
        bold = run_backtest(synthetic(days=10, **{"hedge.risk_aversion": 0.001}))
        assert timid.metrics.band_half_width < bold.metrics.band_half_width
        assert timid.metrics.hedges > bold.metrics.hedges

    def test_unhedged_delta_is_far_larger(self):
        """Without the hedge the book runs the full straddle delta."""
        cfg = synthetic(days=5)
        cfg.hedge.max_hedge_contracts = 1
        cfg.hedge.min_hedge_contracts = 10_000  # effectively disables hedging
        result = run_backtest(cfg)
        held = result.bars[result.bars["straddle_contracts"] != 0]
        assert held["net_delta_units"].abs().max() > 100

    def test_a_wider_fixed_band_trades_less(self):
        fixed = {"hedge.band_model": "fixed"}
        tight = run_backtest(synthetic(days=10, **fixed, **{"hedge.band": 5.0}))
        loose = run_backtest(synthetic(days=10, **fixed, **{"hedge.band": 40.0}))
        assert loose.metrics.hedges < tight.metrics.hedges

    def test_a_fixed_band_below_half_the_quantum_is_inert(self, es):
        """Documented behaviour: MES granularity makes +/-1, +/-3 and +/-5
        identical, which is why the fixed control is 10."""
        fixed = {"hedge.band_model": "fixed"}
        narrow = run_backtest(synthetic(days=10, **fixed, **{"hedge.band": 1.0}))
        middle = run_backtest(synthetic(days=10, **fixed, **{"hedge.band": 3.0}))
        at_quantum = run_backtest(synthetic(days=10, **fixed, **{"hedge.band": 5.0}))
        assert narrow.metrics.hedges == middle.metrics.hedges == at_quantum.metrics.hedges

    def test_the_hedge_is_flattened_with_the_straddle(self):
        result = run_backtest(synthetic(days=5))
        exits = result.events[result.events["kind"] == "exit"]
        assert len(exits) > 0
        for timestamp in exits["timestamp"]:
            after = result.bars[result.bars["timestamp"] >= timestamp]
            row = after.iloc[0]
            if row["straddle_contracts"] == 0:  # not immediately re-entered
                assert row["hedge_contracts"] == 0


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
