"""Entry, exit and hedging, driven bar by bar with hand-built data so each
rule can be triggered in isolation.

The GEX read is injected rather than generated here: a test about the stop
loss should not also depend on what a chain generator happened to produce.
``FixedRegime`` pins the regime so each rule can be exercised on its own.
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.broker.paper import SimulatedExecution
from deltahedger.chain import OptionQuote, select_atm_straddle
from deltahedger.config import Config, CostsConfig
from deltahedger.data.base import MarketBar
from deltahedger.gex import NEGATIVE, NEUTRAL, POSITIVE, StrikeOpenInterest
from deltahedger.pricing import black76
from deltahedger.strategy import GexStraddleStrategy

NY = ZoneInfo("America/New_York")
OPEN = datetime(2025, 6, 10, 9, 35, tzinfo=NY)


class FixedRegime:
    """An open-interest provider that pins the regime under test.

    Built from real strike-level open interest rather than by stubbing the
    calculator, so the classification still runs for real -- these tests
    exercise the same path production does, they just control its input.
    """

    def __init__(self, regime: str, center: float = 5000.0):
        self.regime = regime
        self.center = center
        self.calls = 0

    def open_interest(self, moment, future_price, expiry):
        self.calls += 1
        if self.regime == POSITIVE:
            call_oi, put_oi = 4000.0, 200.0
        elif self.regime == NEGATIVE:
            call_oi, put_oi = 200.0, 4000.0
        else:
            call_oi, put_oi = 2000.0, 2000.0
        return [
            StrikeOpenInterest(self.center + 5.0 * i, call_oi, put_oi)
            for i in range(-20, 21)
        ]


def make_cfg(**overrides) -> Config:
    cfg = Config()
    cfg.starting_equity = 500_000.0
    cfg.costs.enabled = False
    for dotted, value in overrides.items():
        section, _, attr = dotted.partition(".")
        setattr(getattr(cfg, section), attr, value)
    return cfg


def bar(minutes: int, price: float = 5000.0, iv: float = 0.15) -> MarketBar:
    return MarketBar(OPEN + timedelta(minutes=minutes), price, price, price, price, iv)


def drive(cfg: Config, bars, regime: str = POSITIVE, provider=None):
    provider = provider if provider is not None else FixedRegime(regime)
    strategy = GexStraddleStrategy(cfg, open_interest=provider)
    execution = SimulatedExecution(cfg.costs, cfg.source)
    for market_bar in bars:
        strategy.on_bar(market_bar, execution)
    return strategy


def kinds(strategy):
    return [e.kind for e in strategy.events]


def details(strategy, kind):
    return [e.detail for e in strategy.events if e.kind == kind]


class TestRegimeDirectsTheTrade:
    """The whole point of the change: the regime picks the side."""

    def test_positive_gex_sells_the_straddle(self):
        strategy = drive(make_cfg(), [bar(0)], regime=POSITIVE)
        assert "entry" in kinds(strategy)
        assert strategy.portfolio.straddle.quantity < 0
        assert strategy.portfolio.straddle.regime == POSITIVE

    def test_negative_gex_buys_the_straddle(self):
        strategy = drive(make_cfg(), [bar(0)], regime=NEGATIVE)
        assert "entry" in kinds(strategy)
        assert strategy.portfolio.straddle.quantity > 0
        assert strategy.portfolio.straddle.regime == NEGATIVE

    def test_a_neutral_read_stands_aside(self):
        strategy = drive(make_cfg(), [bar(0)], regime=NEUTRAL)
        assert strategy.portfolio.straddle is None
        assert "entry_skipped" in kinds(strategy)

    def test_the_entry_log_says_why_that_side_was_taken(self):
        for regime, phrase in ((POSITIVE, "collect theta"), (NEGATIVE, "scalp gamma")):
            detail = details(drive(make_cfg(), [bar(0)], regime=regime), "entry")[0]
            assert phrase in detail

    def test_without_an_open_interest_provider_nothing_is_traded(self):
        """No GEX read is a reason to stand aside, not to guess a side."""
        cfg = make_cfg()
        strategy = GexStraddleStrategy(cfg, open_interest=None)
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is None

    def test_the_position_is_an_atm_straddle(self):
        strategy = drive(make_cfg(), [bar(0)])
        position = strategy.portfolio.straddle
        assert abs(position.strike - 5000.0) <= 2.5
        assert abs(position.entry_delta) < 0.15  # both legs, near-offsetting


class TestZeroDte:
    def test_it_takes_todays_expiry(self):
        strategy = drive(make_cfg(), [bar(0)])
        assert strategy.portfolio.straddle.expiry == OPEN.date()

    def test_it_never_reaches_past_today(self):
        """0DTE means 0DTE: with no series listed today there is no trade."""
        cfg = make_cfg()
        saturday = datetime(2025, 6, 14, 10, 0, tzinfo=NY)
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        strategy.on_bar(
            MarketBar(saturday, 5000.0, 5000.0, 5000.0, 5000.0, 0.15),
            SimulatedExecution(cfg.costs, cfg.source),
        )
        assert strategy.portfolio.straddle is None
        assert "no 0DTE series is listed" in " ".join(details(strategy, "entry_skipped"))

    def test_the_default_config_is_zero_dte_at_both_ends(self):
        cfg = Config()
        assert cfg.strategy.min_days_to_expiry == 0
        assert cfg.strategy.max_days_to_expiry == 0


class TestEntry:
    def test_does_not_open_before_the_entry_window(self):
        cfg = make_cfg(**{"strategy.entry_time": time(10, 30)})
        assert "entry" not in kinds(drive(cfg, [bar(0)]))

    def test_does_not_open_after_the_cutoff(self):
        cfg = make_cfg(**{"strategy.entry_cutoff_time": time(9, 40)})
        assert "entry" not in kinds(drive(cfg, [bar(60)]))

    def test_opens_only_once_when_reentry_is_off(self):
        cfg = make_cfg(**{"strategy.reenter_after_exit": False})
        strategy = drive(cfg, [bar(m) for m in range(0, 60, 5)])
        assert kinds(strategy).count("entry") == 1

    def test_declines_when_buying_power_is_too_small(self):
        cfg = make_cfg()
        cfg.starting_equity = 2_000.0
        strategy = drive(cfg, [bar(0)])
        assert "entry_skipped" in kinds(strategy)
        assert strategy.portfolio.straddle is None

    def test_the_short_position_respects_the_margin_budget(self):
        cfg = make_cfg(**{"sizing.buying_power_pct": 0.15})
        strategy = drive(cfg, [bar(0)], regime=POSITIVE)
        budget = cfg.starting_equity * 0.15 * (1 - cfg.sizing.hedge_margin_reserve_pct)
        quote = _entry_quote(strategy)
        per = strategy.margin_model.straddle_requirement(quote, 5000.0, cfg.source, -1)
        assert abs(strategy.portfolio.straddle.quantity) * per <= budget * 1.001

    def test_the_long_position_never_spends_more_than_the_budget(self):
        """A long straddle is paid for in cash; the debit is the constraint."""
        cfg = make_cfg(**{"sizing.buying_power_pct": 0.15})
        strategy = drive(cfg, [bar(0)], regime=NEGATIVE)
        budget = cfg.starting_equity * 0.15 * (1 - cfg.sizing.hedge_margin_reserve_pct)
        position = strategy.portfolio.straddle
        debit = position.premium_at_risk(cfg.source.option.multiplier)
        assert 0 < debit <= budget * 1.001

    def test_open_interest_is_cached_rather_than_re_read_every_bar(self):
        provider = FixedRegime(POSITIVE)
        cfg = make_cfg(**{"gex.refresh_seconds": 3600.0})
        drive(cfg, [bar(m) for m in range(0, 60, 5)], provider=provider)
        assert provider.calls == 1

    def test_a_shorter_refresh_re_reads_open_interest(self):
        provider = FixedRegime(POSITIVE)
        cfg = make_cfg(**{"gex.refresh_seconds": 300.0})
        drive(cfg, [bar(m) for m in range(0, 60, 5)], provider=provider)
        assert provider.calls > 1


def _entry_quote(strategy):
    position = strategy.portfolio.straddle
    t = strategy.clock.time_to_expiry(OPEN, position.expiry)
    return select_atm_straddle(
        5000.0, position.expiry, t, 0.15, strategy.source, strategy.surface,
        strategy.cfg.risk_free_rate,
    )


class TestExits:
    def test_closes_before_expiry(self):
        bars = [bar(0)] + [bar(m) for m in range(5, 390, 5)]
        strategy = drive(make_cfg(**{"strategy.reenter_after_exit": False}), bars)
        assert "exit" in kinds(strategy)
        assert strategy.portfolio.straddle is None

    def test_the_timed_exit_fires_inside_the_configured_window(self):
        cfg = make_cfg(**{"strategy.close_before_expiry_minutes": 30})
        cfg.strategy.reenter_after_exit = False
        cfg.strategy.short_take_profit_pct = None  # isolate the timed exit
        cfg.strategy.short_stop_loss_premium_multiple = None
        bars = [bar(0)] + [bar(m) for m in range(5, 390, 5)]
        strategy = drive(cfg, bars)
        exit_event = next(e for e in strategy.events if e.kind == "exit")
        assert exit_event.timestamp.time() >= time(15, 25)

    def test_the_short_stop_fires_when_the_premium_runs_away(self):
        cfg = make_cfg(**{"strategy.short_stop_loss_premium_multiple": 1.5})
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        bars = [bar(0), bar(5, 4990.0), bar(10, 4930.0), bar(15, 4900.0)]
        exits = [e for e in drive(cfg, bars, regime=POSITIVE).events if e.kind == "exit"]
        assert exits and "stop" in exits[0].detail

    def test_the_short_target_fires_as_the_premium_decays(self):
        cfg = make_cfg(**{"strategy.short_take_profit_pct": 0.3})
        cfg.strategy.reenter_after_exit = False
        bars = [bar(0)] + [bar(m) for m in range(5, 240, 5)]
        exits = [e for e in drive(cfg, bars, regime=POSITIVE).events if e.kind == "exit"]
        assert exits and "target" in exits[0].detail

    def test_the_short_stop_can_be_disabled(self):
        cfg = make_cfg(**{"strategy.short_stop_loss_premium_multiple": None})
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.short_take_profit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        bars = [bar(0), bar(5, 4930.0), bar(10, 4900.0)]
        assert "exit" not in kinds(drive(cfg, bars, regime=POSITIVE))

    def test_decay_with_no_movement_does_stop_the_long_side_out(self):
        """Theta is only survivable if the market pays for it. In a dead flat
        market there is no gamma to scalp, the decay is a real loss, and the
        stop is right to fire."""
        cfg = make_cfg(**{"strategy.long_stop_loss_pct": 0.5})
        cfg.strategy.exit_on_regime_flip = False
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.reenter_after_exit = False
        bars = [bar(0)] + [bar(m) for m in range(5, 330, 5)]
        exits = [e for e in drive(cfg, bars, regime=NEGATIVE).events if e.kind == "exit"]
        assert exits and "stop" in exits[0].detail

    def test_a_scalped_long_survives_a_mark_that_would_trip_a_premium_stop(self):
        """The load-bearing asymmetry, and the reason the long side is judged
        on position P&L rather than on the mark.

        Here the straddle's mark falls by more than the stop threshold while
        the hedge scalps back more than the decay -- exactly the case a
        premium-decay stop gets wrong. It would close a winning gamma trade
        for having done the thing a long straddle is supposed to do.
        """
        cfg = make_cfg(**{"strategy.long_stop_loss_pct": 0.25})
        cfg.strategy.exit_on_regime_flip = False
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.long_take_profit_pct = None
        cfg.strategy.reenter_after_exit = False
        bars = [bar(0)] + [
            bar(m, 5000.0 + (10.0 if i % 2 else -10.0))
            for i, m in enumerate(range(5, 330, 5))
        ]
        strategy = drive(cfg, bars, regime=NEGATIVE)

        marks = [s.straddle_mark for s in strategy.bar_states if s.straddle_mark]
        assert marks[-1] < marks[0] * (1.0 - cfg.strategy.long_stop_loss_pct), (
            "the mark must fall past the threshold for this test to discriminate"
        )
        assert strategy.portfolio.hedge_realised > 0, "the hedge must have scalped"
        assert not [
            e for e in strategy.events if e.kind == "exit" and "stop" in e.detail
        ], "a scalped long was stopped out on decay it had already earned back"

    def test_the_long_stop_fires_on_position_pnl(self):
        cfg = make_cfg(**{"strategy.long_stop_loss_pct": 0.05})
        cfg.strategy.exit_on_regime_flip = False
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.reenter_after_exit = False
        bars = [bar(0)] + [bar(m) for m in range(5, 180, 5)]
        exits = [e for e in drive(cfg, bars, regime=NEGATIVE).events if e.kind == "exit"]
        assert exits and "stop" in exits[0].detail

    def test_the_daily_loss_limit_halts_the_session(self):
        cfg = make_cfg(**{"strategy.daily_loss_limit_pct": 0.002})
        cfg.strategy.short_stop_loss_premium_multiple = None
        cfg.strategy.short_take_profit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        bars = [bar(0), bar(5, 4900.0), bar(10, 4850.0), bar(15, 4860.0), bar(20, 4870.0)]
        strategy = drive(cfg, bars, regime=POSITIVE)
        exits = [e for e in strategy.events if e.kind == "exit"]
        assert exits and "daily loss limit" in exits[0].detail
        assert kinds(strategy).count("entry") == 1  # halted, no re-entry

    def test_the_hedge_is_flattened_on_exit(self):
        cfg = make_cfg(**{"strategy.short_stop_loss_premium_multiple": 1.5})
        cfg.strategy.reenter_after_exit = False
        bars = [bar(0), bar(5, 4980.0), bar(10, 4930.0), bar(15, 4900.0)]
        strategy = drive(cfg, bars, regime=POSITIVE)
        assert strategy.portfolio.hedge.quantity == 0

    def test_an_orphaned_hedge_is_closed_by_the_band_even_when_not_flattened(self):
        """A delta-neutral target changes what ``flatten_hedge_on_exit``
        buys.  Under the old +20 target an unflattened hedge could be
        carried; against a target of 0 it is a naked directional position
        and the band closes it on the spot.  The flag now only governs
        whether a sub-band residual -- under one hedge contract -- is left
        behind, so turning it off cannot strand real exposure."""
        cfg = make_cfg(**{"strategy.short_stop_loss_premium_multiple": 1.5})
        cfg.hedge.flatten_hedge_on_exit = False
        cfg.strategy.reenter_after_exit = False
        bars = [bar(0), bar(5, 4980.0), bar(10, 4930.0), bar(15, 4900.0)]
        strategy = drive(cfg, bars, regime=POSITIVE)
        residual = abs(strategy.portfolio.hedge_delta_units())
        assert residual <= cfg.hedge.band


class TestRegimeFlip:
    """Holding a position through a regime flip is holding the wrong side of
    dealer hedging, which is the one thing this strategy exists to avoid."""

    class Flipping:
        """Positive GEX for the first few reads, negative thereafter."""

        def __init__(self, switch_after: int = 1):
            self.switch_after = switch_after
            self.reads = 0

        def open_interest(self, moment, future_price, expiry):
            self.reads += 1
            regime = POSITIVE if self.reads <= self.switch_after else NEGATIVE
            return FixedRegime(regime).open_interest(moment, future_price, expiry)

    def test_a_flip_closes_the_position(self):
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.strategy.reenter_after_exit = False
        strategy = drive(cfg, [bar(0), bar(5), bar(10)], provider=self.Flipping())
        exits = [e for e in strategy.events if e.kind == "exit"]
        assert exits and "GEX flipped" in exits[0].detail

    def test_a_flip_can_be_traded_the_other_way(self):
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.strategy.reenter_after_exit = True
        strategy = drive(cfg, [bar(0), bar(5), bar(10)], provider=self.Flipping())
        entries = [e for e in strategy.events if e.kind == "entry"]
        assert len(entries) == 2
        assert entries[0].regime == POSITIVE and entries[1].regime == NEGATIVE
        assert strategy.portfolio.straddle.quantity > 0  # now long the straddle

    def test_the_flip_exit_can_be_turned_off(self):
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.strategy.exit_on_regime_flip = False
        strategy = drive(cfg, [bar(0), bar(5), bar(10)], provider=self.Flipping())
        assert not [
            e for e in strategy.events if e.kind == "exit" and "GEX flipped" in e.detail
        ]

    def test_entries_are_capped_per_session(self):
        """A spot level oscillating across the flip must not churn all day."""
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.strategy.max_entries_per_session = 2

        class Oscillating:
            def __init__(self):
                self.reads = 0

            def open_interest(self, moment, future_price, expiry):
                self.reads += 1
                regime = POSITIVE if self.reads % 2 else NEGATIVE
                return FixedRegime(regime).open_interest(moment, future_price, expiry)

        strategy = drive(cfg, [bar(m) for m in range(0, 60, 5)], provider=Oscillating())
        assert kinds(strategy).count("entry") == 2


class TestHedging:
    def test_a_short_straddle_is_held_delta_neutral(self):
        strategy = drive(make_cfg(), [bar(0), bar(5, 4995.0), bar(10, 4990.0)],
                         regime=POSITIVE)
        assert all(abs(s.net_delta_units) <= 10.0 for s in strategy.bar_states[1:])

    def test_a_long_straddle_is_held_delta_neutral(self):
        strategy = drive(make_cfg(), [bar(0), bar(5, 4995.0), bar(10, 4990.0)],
                         regime=NEGATIVE)
        assert all(abs(s.net_delta_units) <= 10.0 for s in strategy.bar_states[1:])

    def test_the_hedge_leans_the_opposite_way_in_the_two_regimes(self):
        """Short gamma and long gamma hedge in opposite directions after the
        same move -- if they did not, one of them is mis-signed."""
        bars = [bar(0), bar(5, 4995.0), bar(10, 4990.0)]
        short = drive(make_cfg(), bars, regime=POSITIVE).bar_states[-1]
        long = drive(make_cfg(), bars, regime=NEGATIVE).bar_states[-1]
        assert short.hedge_contracts != 0 and long.hedge_contracts != 0
        assert short.hedge_contracts * long.hedge_contracts < 0

    def test_a_long_straddle_carries_positive_gamma(self):
        state = drive(make_cfg(), [bar(0), bar(5)], regime=NEGATIVE).bar_states[-1]
        assert state.gamma_units > 0
        assert state.theta_dollars < 0  # paying for it

    def test_a_short_straddle_carries_negative_gamma(self):
        state = drive(make_cfg(), [bar(0), bar(5)], regime=POSITIVE).bar_states[-1]
        assert state.gamma_units < 0
        assert state.theta_dollars > 0  # being paid for it

    def test_no_hedge_happens_without_a_position(self):
        cfg = make_cfg(**{"strategy.entry_time": time(23, 0)})
        cfg.strategy.entry_cutoff_time = time(23, 30)
        assert "hedge" not in kinds(drive(cfg, [bar(0), bar(5)]))


class TestLegFills:
    """A straddle with one leg on is a naked option, not a straddle."""

    class DropsThePut(SimulatedExecution):
        def execute_option(self, quote, quantity, moment):
            if quote.right == "P" and quantity != 0:
                return None
            return super().execute_option(quote, quantity, moment)

    def test_a_failed_second_leg_leaves_the_book_flat(self):
        cfg = make_cfg()
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        strategy.on_bar(bar(0), self.DropsThePut(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is None
        assert "entry_failed" in kinds(strategy)

    def test_the_first_leg_is_unwound_rather_than_held(self):
        cfg = make_cfg()
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        strategy.on_bar(bar(0), self.DropsThePut(cfg.costs, cfg.source))
        quantities = [f.quantity for f in strategy.fills if f.instrument == "option"]
        assert sum(quantities) == 0, "the call leg was left on"


class TestCosts:
    def test_slippage_lowers_a_sale_and_raises_a_purchase(self, es):
        execution = SimulatedExecution(CostsConfig(option_slippage_ticks=2.0), es)
        quote = OptionQuote(4980.0, "P", OPEN.date(), 5.00, 0.15,
                            black76(5000.0, 4980.0, 0.001, 0.15, 0.0, "P"), 0.001)
        sell = execution.execute_option(quote, -1, OPEN)
        buy = execution.execute_option(quote, +1, OPEN)
        assert sell.price == pytest.approx(5.00 - 2 * es.option.tick_size)
        assert buy.price == pytest.approx(5.00 + 2 * es.option.tick_size)

    def test_fees_scale_with_contract_count(self, es):
        execution = SimulatedExecution(CostsConfig(), es)
        fill = execution.execute_hedge(-7, 5000.0, OPEN)
        assert fill.fees == pytest.approx(7 * CostsConfig().hedge_fees_per_contract)

    def test_a_straddle_pays_fees_on_both_legs(self):
        cfg = make_cfg()
        cfg.costs.enabled = True
        strategy = drive(cfg, [bar(0)])
        option_fills = [f for f in strategy.fills if f.instrument == "option"]
        assert len(option_fills) == 2
        contracts = abs(strategy.portfolio.straddle.quantity)
        assert strategy.portfolio.fees_paid == pytest.approx(
            2 * contracts * cfg.costs.option_fees_per_contract
        )

    def test_disabling_costs_removes_slippage_and_fees(self, es):
        execution = SimulatedExecution(CostsConfig(enabled=False), es)
        fill = execution.execute_hedge(-7, 5000.0, OPEN)
        assert fill.price == 5000.0 and fill.fees == 0.0

    def test_a_zero_quantity_order_is_not_sent(self, es):
        execution = SimulatedExecution(CostsConfig(), es)
        assert execution.execute_hedge(0, 5000.0, OPEN) is None

    def test_an_option_fill_price_never_goes_negative(self, es):
        execution = SimulatedExecution(CostsConfig(option_slippage_ticks=100.0), es)
        quote = OptionQuote(4980.0, "P", OPEN.date(), 0.05, 0.15,
                            black76(5000.0, 4980.0, 0.001, 0.15, 0.0, "P"), 0.001)
        assert execution.execute_option(quote, -1, OPEN).price >= 0.0
