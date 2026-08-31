"""Entry, exit and execution rules, driven bar by bar with hand-built data
so each rule can be triggered in isolation."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.broker.paper import SimulatedExecution
from deltahedger.chain import OptionQuote
from deltahedger.config import Config, CostsConfig
from deltahedger.data.base import MarketBar
from deltahedger.pricing import black76
from deltahedger.strategy import ShortVolStrategy

NY = ZoneInfo("America/New_York")
OPEN = datetime(2025, 6, 10, 9, 35, tzinfo=NY)


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


def drive(cfg: Config, bars):
    strategy = ShortVolStrategy(cfg)
    execution = SimulatedExecution(cfg.costs, cfg.source)
    for market_bar in bars:
        strategy.on_bar(market_bar, execution)
    return strategy


def kinds(strategy):
    return [e.kind for e in strategy.events]


class TestEntry:
    def test_opens_a_short_put_at_the_entry_time(self):
        strategy = drive(make_cfg(), [bar(0)])
        assert "entry" in kinds(strategy)
        assert strategy.portfolio.option.quantity < 0

    def test_sells_a_put_below_the_market(self):
        strategy = drive(make_cfg(), [bar(0)])
        assert strategy.portfolio.option.strike < 5000.0

    def test_targets_the_configured_delta(self):
        strategy = drive(make_cfg(), [bar(0)])
        assert abs(strategy.portfolio.option.entry_delta) == pytest.approx(0.20, abs=0.10)

    def test_does_not_open_before_the_entry_window(self):
        cfg = make_cfg(**{"strategy.entry_time": time(10, 30)})
        assert "entry" not in kinds(drive(cfg, [bar(0)]))

    def test_does_not_open_after_the_cutoff(self):
        cfg = make_cfg(**{"strategy.entry_cutoff_time": time(9, 40)})
        assert "entry" not in kinds(drive(cfg, [bar(60)]))

    def test_opens_only_once_per_session_by_default(self):
        strategy = drive(make_cfg(), [bar(m) for m in range(0, 60, 5)])
        assert kinds(strategy).count("entry") == 1

    def test_declines_when_buying_power_is_too_small(self):
        cfg = make_cfg()
        cfg.starting_equity = 5_000.0
        strategy = drive(cfg, [bar(0)])
        assert "entry_skipped" in kinds(strategy)
        assert strategy.portfolio.option is None

    def test_the_position_respects_the_buying_power_budget(self):
        cfg = make_cfg(**{"sizing.buying_power_pct": 0.15})
        strategy = drive(cfg, [bar(0)])
        budget = cfg.starting_equity * 0.15 * (1 - cfg.sizing.hedge_margin_reserve_pct)
        quote = _quote_for(strategy, 5000.0)
        margin = strategy.margin_model.short_option_margin(quote, 5000.0, cfg.source)
        assert abs(strategy.portfolio.option.quantity) * margin <= budget * 1.001


def _quote_for(strategy, future) -> OptionQuote:
    position = strategy.portfolio.option
    t = strategy.clock.time_to_expiry(OPEN, position.expiry)
    iv = strategy.surface.iv(future, position.strike, 0.15, t)
    greeks = black76(future, position.strike, t, iv, strategy.cfg.risk_free_rate, "P")
    return OptionQuote(position.strike, "P", position.expiry, greeks.price, iv, greeks, t)


class TestExits:
    def test_closes_before_expiry(self):
        bars = [bar(0)] + [bar(m) for m in range(5, 390, 5)]
        strategy = drive(make_cfg(), bars)
        assert "exit" in kinds(strategy)
        assert strategy.portfolio.option is None

    def test_the_timed_exit_fires_inside_the_configured_window(self):
        cfg = make_cfg(**{"strategy.close_before_expiry_minutes": 30})
        bars = [bar(0)] + [bar(m) for m in range(5, 390, 5)]
        strategy = drive(cfg, bars)
        exit_event = next(e for e in strategy.events if e.kind == "exit")
        assert exit_event.timestamp.time() >= time(15, 25)

    def test_the_stop_fires_on_a_selloff(self):
        cfg = make_cfg(**{"strategy.stop_loss_premium_multiple": 2.0})
        bars = [bar(0), bar(5, 4990.0), bar(10, 4930.0), bar(15, 4900.0)]
        strategy = drive(cfg, bars)
        exits = [e for e in strategy.events if e.kind == "exit"]
        assert exits and "stop" in exits[0].detail

    def test_the_stop_can_be_disabled(self):
        cfg = make_cfg(**{"strategy.stop_loss_premium_multiple": None})
        cfg.strategy.daily_loss_limit_pct = None  # isolate the stop rule
        bars = [bar(0), bar(5, 4930.0), bar(10, 4900.0)]
        assert "exit" not in kinds(drive(cfg, bars))

    def test_the_loss_limit_still_fires_when_the_stop_is_disabled(self):
        """The two exits are independent; disabling one must not disable the other."""
        cfg = make_cfg(**{"strategy.stop_loss_premium_multiple": None})
        bars = [bar(0), bar(5, 4930.0), bar(10, 4900.0)]
        exits = [e for e in drive(cfg, bars).events if e.kind == "exit"]
        assert exits and "daily loss limit" in exits[0].detail

    def test_the_take_profit_fires_on_a_rally(self):
        cfg = make_cfg(**{"strategy.take_profit_pct": 0.5})
        bars = [bar(0), bar(5, 5030.0), bar(10, 5060.0)]
        exits = [e for e in drive(cfg, bars).events if e.kind == "exit"]
        assert exits and "target" in exits[0].detail

    def test_the_daily_loss_limit_halts_the_session(self):
        cfg = make_cfg(**{"strategy.daily_loss_limit_pct": 0.005})
        cfg.strategy.stop_loss_premium_multiple = None
        cfg.strategy.reenter_after_exit = True
        bars = [bar(0), bar(5, 4900.0), bar(10, 4850.0), bar(15, 4860.0), bar(20, 4870.0)]
        strategy = drive(cfg, bars)
        exits = [e for e in strategy.events if e.kind == "exit"]
        assert exits and "daily loss limit" in exits[0].detail
        assert kinds(strategy).count("entry") == 1  # halted, no re-entry

    def test_the_hedge_is_flattened_on_exit(self):
        cfg = make_cfg(**{"strategy.stop_loss_premium_multiple": 2.0})
        bars = [bar(0), bar(5, 4980.0), bar(10, 4930.0), bar(15, 4900.0)]
        strategy = drive(cfg, bars)
        assert strategy.portfolio.hedge.quantity == 0

    def test_the_hedge_can_be_left_open_on_exit(self):
        cfg = make_cfg(**{"strategy.stop_loss_premium_multiple": 2.0})
        cfg.hedge.flatten_hedge_on_exit = False
        bars = [bar(0), bar(5, 4980.0), bar(10, 4930.0), bar(15, 4900.0)]
        strategy = drive(cfg, bars)
        assert strategy.portfolio.hedge.quantity != 0


class TestHedging:
    def test_it_hedges_the_short_put_delta_down_to_the_target(self):
        strategy = drive(make_cfg(), [bar(0), bar(5)])
        state = strategy.bar_states[-1]
        assert state.hedge_contracts < 0  # sold MES against the long delta
        assert abs(state.net_delta_units - 20.0) <= 5.0

    def test_a_falling_market_makes_the_book_longer_and_draws_a_sale(self):
        bars = [bar(0), bar(5, 4995.0), bar(10, 4985.0), bar(15, 4975.0)]
        strategy = drive(make_cfg(), bars)
        hedges = [e for e in strategy.events if e.kind == "hedge"]
        assert hedges, "a 25-point drop should have breached the band"
        assert all(abs(s.net_delta_units - 20.0) < 60 for s in strategy.bar_states[1:])

    def test_no_hedge_happens_without_a_position(self):
        cfg = make_cfg(**{"strategy.entry_time": time(23, 0)})
        cfg.strategy.entry_cutoff_time = time(23, 30)
        strategy = drive(cfg, [bar(0), bar(5)])
        assert "hedge" not in kinds(strategy)


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
