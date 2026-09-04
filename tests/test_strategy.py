"""Entry, exit and hedging, driven bar by bar with hand-built data so each
rule can be triggered in isolation.

The GEX read is injected rather than generated here: a test about the stop
loss should not also depend on what a chain generator happened to produce.
``FixedRegime`` pins the regime so each rule can be exercised on its own.

``make_cfg`` switches off three of the four stand-aside gates
(``ensemble``, ``persistence``, ``entry_window``) by default, so a test
about a stop-loss or a fill failure gets the same "read once, act
immediately" behaviour these tests were written against.  ``confidence``
and ``flip_distance`` stay on, because ``FixedRegime``'s POSITIVE/NEGATIVE
chains clear both by a wide margin and NEUTRAL depends on the confidence
gate to read as NEUTRAL at all.  ``TestGates`` turns each of the three off
gates on, one at a time, to test them directly.
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.broker.paper import SimulatedExecution
from deltahedger.chain import OptionQuote, select_atm_straddle
from deltahedger.config import Config, CostsConfig
from deltahedger.data.base import MarketBar
from deltahedger.gex import (
    GATE_CONFIDENCE,
    GATE_ENSEMBLE,
    GATE_ENTRY_WINDOW,
    GATE_FLIP_DISTANCE,
    GATE_NOWCAST,
    GATE_PERSISTENCE,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    EnsembleResult,
    StrikeFlow,
    StrikeOpenInterest,
)
from deltahedger.pricing import black76
from deltahedger.strategy import GexStraddleStrategy

NY = ZoneInfo("America/New_York")
#: Inside the default entry window (10:00-11:30) so a test that leaves the
#: entry-window gate on does not also have to move this.
OPEN = datetime(2025, 6, 10, 10, 0, tzinfo=NY)


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
    cfg.gates.ensemble = False
    cfg.gates.persistence = False
    cfg.gates.entry_window = False
    for dotted, value in overrides.items():
        section, _, attr = dotted.partition(".")
        setattr(getattr(cfg, section), attr, value)
    return cfg


def bar(minutes: int, price: float = 5000.0, iv: float = 0.15) -> MarketBar:
    return MarketBar(OPEN + timedelta(minutes=minutes), price, price, price, price, iv)


def session_bar(
    day_offset: int, minutes: int = 0, price: float = 5000.0, iv: float = 0.15
) -> MarketBar:
    """A bar on the trading day ``day_offset`` sessions after ``OPEN``.

    Multi-session tenors need bars that span real sessions to exercise
    anything time-dependent -- a stop measured on theta decay, or the DTE
    floor itself. ``bar()`` alone cannot reach past one day.
    """
    from deltahedger.session import next_trading_day

    day = OPEN.date()
    for _ in range(day_offset):
        day = next_trading_day(day)
    moment = datetime.combine(day, OPEN.timetz()) + timedelta(minutes=minutes)
    return MarketBar(moment, price, price, price, price, iv)


#: A tenor pinned to exactly 5 DTE, closing at 1 DTE. Time-decay tests use
#: this rather than the shipped 2-5 DTE range so the number of sessions a
#: position lives is deterministic and the test can drive exactly that many
#: bars instead of guessing how many the range's preference logic would
#: pick.
FIVE_DTE = {
    "strategy.min_days_to_expiry": 5,
    "strategy.max_days_to_expiry": 5,
    "strategy.prefer_min_days_to_expiry": 5,
    "strategy.prefer_max_days_to_expiry": 5,
    "strategy.close_at_days_to_expiry": 1,
}


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


class TestTenorSelection:
    def test_the_traded_expiry_is_inside_the_configured_dte_window(self):
        strategy = drive(make_cfg(), [bar(0)])
        position = strategy.portfolio.straddle
        dte = strategy.clock.days_to_expiry(OPEN, position.expiry)
        cfg = strategy.cfg.strategy
        assert cfg.min_days_to_expiry <= dte <= cfg.max_days_to_expiry

    def test_the_default_window_is_two_to_five_dte(self):
        cfg = Config()
        assert (cfg.strategy.min_days_to_expiry, cfg.strategy.max_days_to_expiry) == (2, 5)

    def test_it_prefers_the_expiry_closest_to_the_preferred_window(self):
        """From a Tuesday, 4 trading days out (Monday) is the nearest listed
        expiry inside ``(3, 4)`` -- Friday (3 days) is also inside the
        window, but 4 wins the tie-break toward the longer tenor."""
        strategy = drive(make_cfg(), [bar(0)])
        dte = strategy.clock.days_to_expiry(OPEN, strategy.portfolio.straddle.expiry)
        assert dte == 4

    def test_no_candidate_expiry_stands_aside_rather_than_reaching(self, es):
        """``_try_entry`` treats ``_traded_expiry`` returning ``None`` as a
        skip rather than reaching for the nearest or farthest series. The
        live path can hit this against a real chain's finite listing depth;
        this codebase's synthesised calendar never runs out, so the ``None``
        is produced directly rather than by exhausting it."""
        cfg = make_cfg()
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        strategy._traded_expiry = lambda moment: None
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is None
        assert "no expiry listed" in " ".join(details(strategy, "entry_skipped"))

    def test_an_open_position_keeps_its_own_expiry_rather_than_reselecting(self):
        """``_traded_expiry`` must answer ``on the book`` while a position is
        open, not ``what would be chosen now`` -- otherwise the exit and
        entry paths could disagree about which series is actually held."""
        strategy = drive(make_cfg(), [bar(0)])
        position = strategy.portfolio.straddle
        later = OPEN + timedelta(hours=1)
        assert strategy._traded_expiry(later) == position.expiry


class TestEntry:
    def test_does_not_open_before_the_entry_window(self):
        cfg = make_cfg(**{"strategy.entry_time": time(10, 30)})
        cfg.gates.entry_window = True
        assert "entry" not in kinds(drive(cfg, [bar(0)]))

    def test_does_not_open_after_the_cutoff(self):
        cfg = make_cfg(**{"strategy.entry_cutoff_time": time(9, 40)})
        cfg.gates.entry_window = True
        assert "entry" not in kinds(drive(cfg, [bar(60)]))

    def test_the_entry_window_gate_can_be_turned_off(self):
        """With the gate off, entries are not confined to the window at
        all -- ``make_cfg`` relies on exactly this for every other test."""
        cfg = make_cfg(**{"strategy.entry_time": time(10, 30)})
        assert "entry" in kinds(drive(cfg, [bar(0)]))

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
        """One read per expiry in the blend on the first bar, then nothing
        until the refresh timer expires -- the profile is rebuilt from the
        cached open interest on every bar in between."""
        provider = FixedRegime(POSITIVE)
        cfg = make_cfg(**{"gex.refresh_seconds": 3600.0})
        strategy = drive(cfg, [bar(m) for m in range(0, 60, 5)], provider=provider)
        assert provider.calls == len(strategy._books)

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


#: A same-day expiry, with the DTE floor disabled (close_days < min_days=0
#: means it can never trigger). Used only to test the seconds-based
#: backstop exit in isolation from the DTE floor that leads the ladder at
#: the shipped tenor -- see ``close_before_expiry_minutes`` in config.py.
ZERO_DTE = {
    "strategy.min_days_to_expiry": 0, "strategy.max_days_to_expiry": 0,
    "strategy.prefer_min_days_to_expiry": 0, "strategy.prefer_max_days_to_expiry": 0,
    "strategy.close_at_days_to_expiry": -1,
}


class TestExits:
    def test_closes_before_expiry(self):
        cfg = make_cfg(**{**ZERO_DTE, "strategy.reenter_after_exit": False})
        bars = [bar(0)] + [bar(m) for m in range(5, 390, 5)]
        strategy = drive(cfg, bars)
        assert "exit" in kinds(strategy)
        assert strategy.portfolio.straddle is None

    def test_the_timed_exit_fires_inside_the_configured_window(self):
        cfg = make_cfg(**{**ZERO_DTE, "strategy.close_before_expiry_minutes": 30})
        cfg.strategy.reenter_after_exit = False
        cfg.strategy.short_take_profit_pct = None  # isolate the timed exit
        cfg.strategy.short_stop_loss_premium_multiple = None
        bars = [bar(0)] + [bar(m) for m in range(5, 390, 5)]
        strategy = drive(cfg, bars)
        exit_event = next(e for e in strategy.events if e.kind == "exit")
        assert exit_event.timestamp.time() >= time(15, 25)

    def test_the_short_stop_fires_when_the_premium_runs_away(self):
        """At this tenor a straddle's mark is dominated by vega, not
        moneyness, so the move has to be large enough to push the fixed
        entry strike meaningfully in the money."""
        cfg = make_cfg(**{"strategy.short_stop_loss_premium_multiple": 1.5})
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        bars = [bar(0), bar(5, 4900.0), bar(10, 4800.0), bar(15, 4700.0)]
        exits = [e for e in drive(cfg, bars, regime=POSITIVE).events if e.kind == "exit"]
        assert exits and "stop" in exits[0].detail

    def test_the_short_target_fires_as_the_premium_decays(self):
        """Theta decay at this tenor is a multi-session effect, not a
        multi-hour one, so the test needs bars that span real sessions --
        see ``session_bar``. Pinned to exactly 5 DTE (``FIVE_DTE``) so the
        target is reached well before the 1DTE close-out floor would
        preempt it."""
        cfg = make_cfg(**{**FIVE_DTE, "strategy.short_take_profit_pct": 0.15})
        bars = [session_bar(d, m) for d in range(0, 4) for m in (0, 60, 120, 180, 240)]
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
        stop is right to fire.

        Reaching a 50% loss to decay alone needs a tenor much longer than
        the shipped 2-5 DTE range gives room for -- an ATM straddle's price
        scales roughly with sqrt(T), so most of a long-dated option's value
        decays only in its last few sessions, and the 5DTE-DTE gap this
        strategy actually holds through is too short a window for half the
        premium to bleed away before the close-out floor would act first.
        This test pins a 15DTE-only tenor purely to give decay the room to
        reach the threshold; it is testing the stop's arithmetic, not the
        shipped tenor.
        """
        tenor = {
            "strategy.min_days_to_expiry": 15, "strategy.max_days_to_expiry": 15,
            "strategy.prefer_min_days_to_expiry": 15,
            "strategy.prefer_max_days_to_expiry": 15,
            "strategy.close_at_days_to_expiry": 1,
        }
        cfg = make_cfg(**{**tenor, "strategy.long_stop_loss_pct": 0.5})
        cfg.strategy.exit_on_regime_flip = False
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.reenter_after_exit = False
        bars = [session_bar(d, m) for d in range(0, 13) for m in (0, 120, 240)]
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
        tenor = {
            "strategy.min_days_to_expiry": 15, "strategy.max_days_to_expiry": 15,
            "strategy.prefer_min_days_to_expiry": 15,
            "strategy.prefer_max_days_to_expiry": 15,
            "strategy.close_at_days_to_expiry": 1,
        }
        cfg = make_cfg(**{**tenor, "strategy.long_stop_loss_pct": 0.25})
        cfg.strategy.exit_on_regime_flip = False
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.long_take_profit_pct = None
        cfg.strategy.reenter_after_exit = False
        # 15 DTE for the same reason as the decay test above: at this tenor
        # a straddle's gamma is a fraction of what 0DTE carried, so the
        # oscillation has to be both wider and carried over many sessions
        # to scalp back enough to test the asymmetry at all.
        bars = [session_bar(0)]
        for day in range(0, 12):
            for i, minutes in enumerate((60, 120, 180, 240, 300)):
                price = 5000.0 + (40.0 if i % 2 else -40.0)
                bars.append(session_bar(day, minutes, price))
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
        cfg = make_cfg(**{**FIVE_DTE, "strategy.long_stop_loss_pct": 0.05})
        cfg.strategy.exit_on_regime_flip = False
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.reenter_after_exit = False
        bars = [session_bar(d, m) for d in range(0, 3) for m in (0, 60, 120, 180, 240)]
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
        bars = [bar(0), bar(5, 4900.0), bar(10, 4800.0), bar(15, 4700.0)]
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
        bars = [bar(0), bar(5, 4900.0), bar(10, 4800.0), bar(15, 4700.0)]
        strategy = drive(cfg, bars, regime=POSITIVE)
        residual = abs(strategy.portfolio.hedge_delta_units())
        assert residual <= cfg.hedge.band


class TestRegimeFlip:
    """Holding a position through a regime flip is holding the wrong side of
    dealer hedging, which is the one thing this strategy exists to avoid."""

    class Flipping:
        """Positive GEX for the first few bars, negative thereafter.

        Counted in distinct *bars* (moments) rather than provider calls: the
        front-expiry blend reads open interest once per expiry in it, so a
        single bar makes several calls, and counting those would flip the
        regime mid-bar rather than between bars.
        """

        def __init__(self, switch_after: int = 1):
            self.switch_after = switch_after
            self._moments: list = []

        def open_interest(self, moment, future_price, expiry):
            if not self._moments or self._moments[-1] != moment:
                self._moments.append(moment)
            regime = POSITIVE if len(self._moments) <= self.switch_after else NEGATIVE
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
            """Flips every bar, counted in distinct moments (see Flipping)."""

            def __init__(self):
                self._moments: list = []

            def open_interest(self, moment, future_price, expiry):
                if not self._moments or self._moments[-1] != moment:
                    self._moments.append(moment)
                regime = POSITIVE if len(self._moments) % 2 else NEGATIVE
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
        cfg.gates.entry_window = True
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
        # The position at this size is not delta-neutral out of the gate, so
        # the entry bar also carries a hedge fee -- isolate the option fees
        # rather than asserting the total.
        option_fees = sum(f.fees for f in option_fills)
        assert option_fees == pytest.approx(
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


class TestGates:
    """The four stand-aside gates, each exercised on its own.

    Three are proven directly against the strategy; the ensemble gate is
    proven by replacing ``strategy.gex.ensemble`` with a stub, because
    hand-building open interest that disagrees under a skew or sign
    perturbation is what ``deltahedger.gex.GexCalculator.ensemble`` itself
    is already tested against (``tests/test_gex.py``) -- here the question
    is only whether the strategy obeys the verdict.
    """

    class NearFlat:
        """Confidence ratio ~10%: above the old fixed 5% threshold, below
        the new ``gates.min_confidence_ratio`` default of 15%."""

        def open_interest(self, moment, future_price, expiry):
            return [
                StrikeOpenInterest(5000.0 + 5.0 * i, 1100.0, 900.0)
                for i in range(-20, 21)
            ]

    class Flipping:
        """Positive for the first ``switch_after`` bars, negative after,
        counted in distinct bars -- see ``TestRegimeFlip.Flipping``."""

        def __init__(self, switch_after: int):
            self.switch_after = switch_after
            self._moments: list = []

        def open_interest(self, moment, future_price, expiry):
            if not self._moments or self._moments[-1] != moment:
                self._moments.append(moment)
            regime = POSITIVE if len(self._moments) <= self.switch_after else NEGATIVE
            return FixedRegime(regime).open_interest(moment, future_price, expiry)

    # -- confidence --------------------------------------------------

    def test_confidence_gate_blocks_a_near_flat_book(self):
        cfg = make_cfg()
        cfg.gates.confidence = True
        strategy = drive(cfg, [bar(0)], provider=self.NearFlat())
        assert strategy.portfolio.straddle is None
        assert details(strategy, "entry_skipped")
        blocked = [e for e in strategy.events if e.gate == GATE_CONFIDENCE]
        assert blocked

    def test_confidence_gate_off_trades_the_same_book(self):
        cfg = make_cfg()
        cfg.gates.confidence = False
        strategy = drive(cfg, [bar(0)], provider=self.NearFlat())
        assert strategy.portfolio.straddle is not None

    # -- persistence ---------------------------------------------------

    def test_persistence_gate_blocks_entry_until_the_regime_holds(self):
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.gates.persistence = True
        cfg.gates.persistence_bars = 3
        strategy = drive(cfg, [bar(m) for m in range(0, 15, 5)], regime=POSITIVE)
        entries = [e for e in strategy.events if e.kind == "entry"]
        assert len(entries) == 1
        assert entries[0].timestamp == OPEN + timedelta(minutes=10)
        skipped = [e for e in strategy.events if e.gate == GATE_PERSISTENCE]
        assert len(skipped) == 2

    def test_a_flip_failing_the_persistence_check_does_not_close_the_position(self):
        """The load-bearing case: a regime flip that has not yet held long
        enough must defer the exit, not take it. Positive for the first 4
        bars -- long enough to confirm the entry -- then negative for the
        next 2, one short of the 3 the flip itself needs."""
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.gates.persistence = True
        cfg.gates.persistence_bars = 3
        provider = self.Flipping(switch_after=4)
        strategy = drive(cfg, [bar(m) for m in range(0, 30, 5)], provider=provider)

        assert "exit" not in kinds(strategy)
        deferred = [e for e in strategy.events if e.kind == "exit_deferred"]
        assert deferred and all(e.gate == GATE_PERSISTENCE for e in deferred)
        position = strategy.portfolio.straddle
        assert position is not None and position.regime == POSITIVE

    def test_the_flip_closes_once_the_new_regime_has_held_long_enough(self):
        """The other half: once the flip itself has held ``persistence_bars``
        bars, the deferred exit fires."""
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.gates.persistence = True
        cfg.gates.persistence_bars = 3
        provider = self.Flipping(switch_after=4)
        bars = [bar(m) for m in range(0, 45, 5)]  # 4 positive, 5 negative
        strategy = drive(cfg, bars, provider=provider)

        exits = [e for e in strategy.events if e.kind == "exit"]
        assert exits and "GEX flipped" in exits[0].detail

    def test_persistence_gate_off_acts_on_the_very_next_bar(self):
        cfg = make_cfg(**{"gex.refresh_seconds": 0.0})
        cfg.gates.persistence = False
        strategy = drive(cfg, [bar(0)], regime=POSITIVE)
        assert strategy.portfolio.straddle is not None

    # -- ensemble --------------------------------------------------------

    def test_ensemble_gate_blocks_entry_when_members_disagree(self):
        cfg = make_cfg()
        cfg.gates.ensemble = True
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        strategy.gex.ensemble = lambda *a, **k: EnsembleResult(
            False, NEUTRAL, (POSITIVE, NEUTRAL, NEGATIVE), "the ensemble disagrees"
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is None
        assert any(e.gate == GATE_ENSEMBLE for e in strategy.events)

    def test_ensemble_gate_allows_a_unanimous_entry(self):
        cfg = make_cfg()
        cfg.gates.ensemble = True
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        strategy.gex.ensemble = lambda *a, **k: EnsembleResult(
            True, POSITIVE, (POSITIVE,) * 9, "all 9 members agree"
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is not None

    def test_ensemble_gate_defers_a_flip_the_members_disagree_on(self):
        cfg = make_cfg(**{
            "strategy.short_stop_loss_premium_multiple": None,
            "gex.refresh_seconds": 0.0,
        })
        cfg.gates.ensemble = True
        cfg.strategy.daily_loss_limit_pct = None
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE))
        execution = SimulatedExecution(cfg.costs, cfg.source)
        strategy.on_bar(bar(0), execution)
        assert strategy.portfolio.straddle is not None

        strategy.open_interest = FixedRegime(NEGATIVE)
        strategy.gex.ensemble = lambda *a, **k: EnsembleResult(
            False, NEUTRAL, (POSITIVE, NEGATIVE), "the ensemble disagrees"
        )
        strategy.on_bar(bar(5), execution)
        assert strategy.portfolio.straddle is not None
        assert any(
            e.kind == "exit_deferred" and e.gate == GATE_ENSEMBLE
            for e in strategy.events
        )

    # -- entry window ------------------------------------------------

    def test_entry_window_gate_blocks_a_bar_outside_it(self):
        cfg = make_cfg(**{"strategy.entry_time": time(12, 0)})
        cfg.gates.entry_window = True
        strategy = drive(cfg, [bar(0)], regime=POSITIVE)
        assert strategy.portfolio.straddle is None

    def test_entry_window_gate_off_ignores_the_configured_window(self):
        cfg = make_cfg(**{"strategy.entry_time": time(12, 0)})
        cfg.gates.entry_window = False
        strategy = drive(cfg, [bar(0)], regime=POSITIVE)
        assert strategy.portfolio.straddle is not None

    # -- exits are never gated --------------------------------------

    def test_the_dte_floor_exit_is_never_gated(self):
        """A gate can delay a side change; it can never keep a position past
        its close-out floor."""
        cfg = make_cfg(**{**ZERO_DTE, "strategy.reenter_after_exit": False})
        cfg.gates.confidence = True
        cfg.gates.flip_distance = True
        cfg.gates.ensemble = True
        cfg.gates.persistence = True
        strategy = drive(cfg, [bar(0)] + [bar(m) for m in range(5, 390, 5)])
        assert "exit" in kinds(strategy)


class FixedFlow:
    """A nowcast provider returning the same flow rows for every expiry
    it's asked about, built from real strike-level flow so the calculator
    still runs for real -- the ``FixedRegime`` convention, applied here."""

    def __init__(self, rows: tuple[StrikeFlow, ...]):
        self.rows = rows
        self.calls = 0

    def flow_since(self, moment, expiry, since):
        self.calls += 1
        return self.rows


def _flow_rows(
    call_signed_volume: float = 0.0, put_signed_volume: float = 0.0,
    center: float = 5000.0,
) -> tuple[StrikeFlow, ...]:
    return tuple(
        StrikeFlow(center + 5.0 * i, call_signed_volume, put_signed_volume)
        for i in range(-20, 21)
    )


#: Against ``FixedRegime(POSITIVE)`` (call_oi=4000, put_oi=200 at every
#: strike): no flow at all leaves the nowcast equal to the base read, so it
#: always agrees with whatever regime the print picked.
_AGREEING_FLOW = _flow_rows()

#: Enough opposing call flow, at ``dealer_share=1.0``, to flip the read from
#: POSITIVE to NEGATIVE outright -- see ``tests/test_gex.py::TestNowcast
#: ::test_a_confident_disagreement_can_flip_the_regime`` for the same
#: arithmetic proven directly against the calculator.
_DISAGREEING_FLOW = _flow_rows(call_signed_volume=4000.0)

#: Enough call flow, at ``dealer_share=1.0``, to bring the effective call
#: and put OI to exactly the same number at every strike -- zero net GEX,
#: which reads NEUTRAL (no opinion), not a disagreement.
_NEUTRAL_FLOW = _flow_rows(call_signed_volume=3800.0)


class TestNowcast:
    """The flow nowcast's narrow authority: it can veto an entry the print
    picked, exit a position early, or haircut size when flow has no
    opinion -- it can never pick the side itself. ``FixedRegime`` always
    supplies the entry direction; only what the flow feed says changes."""

    def _cfg(self, **nowcast_overrides):
        cfg = make_cfg()
        cfg.nowcast.enabled = True
        cfg.nowcast.dealer_share = 1.0
        for attr, value in nowcast_overrides.items():
            setattr(cfg.nowcast, attr, value)
        return cfg

    # -- veto ----------------------------------------------------------

    def test_an_agreeing_flow_does_not_veto_the_entry(self):
        cfg = self._cfg()
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_AGREEING_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is not None

    def test_a_disagreeing_flow_vetoes_the_entry(self):
        cfg = self._cfg()
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_DISAGREEING_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is None
        blocked = [e for e in strategy.events if e.gate == GATE_NOWCAST]
        assert blocked and blocked[0].kind == "entry_skipped"

    def test_the_veto_can_be_turned_off(self):
        """The veto is a gate, not a rewiring of who picks the side: turning
        it off must still take the print's (short) direction, never the
        nowcast's disagreeing (long) one."""
        cfg = self._cfg(veto_enabled=False)
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_DISAGREEING_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is not None
        assert strategy.portfolio.straddle.quantity < 0  # still short: the print's side

    def test_nowcast_disabled_never_vetoes(self):
        cfg = self._cfg(enabled=False)
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_DISAGREEING_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle is not None

    # -- size haircut ----------------------------------------------------

    def test_a_neutral_flow_haircuts_the_size(self):
        cfg = self._cfg(size_haircut_when_unconfirmed=0.4)
        baseline = GexStraddleStrategy(make_cfg(), open_interest=FixedRegime(POSITIVE))
        baseline.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_NEUTRAL_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert abs(strategy.portfolio.straddle.quantity) < abs(baseline.portfolio.straddle.quantity)
        entry = details(strategy, "entry")[0]
        assert "40% sized" in entry and "no opinion" in entry

    def test_an_agreeing_flow_does_not_haircut_the_size(self):
        cfg = self._cfg(size_haircut_when_unconfirmed=0.4)
        baseline = GexStraddleStrategy(make_cfg(), open_interest=FixedRegime(POSITIVE))
        baseline.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_AGREEING_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle.quantity == baseline.portfolio.straddle.quantity

    def test_a_disagreeing_flow_is_vetoed_rather_than_haircut(self):
        """Disagreement is a full veto, never a partial size -- the two
        never compose."""
        cfg = self._cfg(veto_enabled=False, size_haircut_when_unconfirmed=0.4)
        baseline = GexStraddleStrategy(make_cfg(), open_interest=FixedRegime(POSITIVE))
        baseline.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_DISAGREEING_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle.quantity == baseline.portfolio.straddle.quantity

    def test_size_haircut_disabled_trades_full_size(self):
        cfg = self._cfg(size_haircut_enabled=False, size_haircut_when_unconfirmed=0.4)
        baseline = GexStraddleStrategy(make_cfg(), open_interest=FixedRegime(POSITIVE))
        baseline.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_NEUTRAL_FLOW),
        )
        strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
        assert strategy.portfolio.straddle.quantity == baseline.portfolio.straddle.quantity

    # -- early exit --------------------------------------------------------

    def test_a_disagreeing_flow_closes_an_open_position_early(self):
        """The print never flips (``FixedRegime`` is pinned POSITIVE
        throughout) -- only the flow read changes, from bar to bar."""
        cfg = self._cfg(refresh_seconds=0.0)
        cfg.strategy.short_stop_loss_premium_multiple = None
        cfg.strategy.short_take_profit_pct = None
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        nowcast = FixedFlow(_AGREEING_FLOW)
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE), nowcast=nowcast)
        execution = SimulatedExecution(cfg.costs, cfg.source)
        strategy.on_bar(bar(0), execution)
        assert strategy.portfolio.straddle is not None

        nowcast.rows = _DISAGREEING_FLOW
        strategy.on_bar(bar(5), execution)
        exits = [e for e in strategy.events if e.kind == "exit"]
        assert exits and "against the open short position" in exits[0].detail

    def test_an_agreeing_flow_does_not_close_the_position(self):
        cfg = self._cfg(refresh_seconds=0.0)
        cfg.strategy.short_stop_loss_premium_multiple = None
        cfg.strategy.short_take_profit_pct = None
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE), nowcast=FixedFlow(_AGREEING_FLOW),
        )
        execution = SimulatedExecution(cfg.costs, cfg.source)
        strategy.on_bar(bar(0), execution)
        strategy.on_bar(bar(5), execution)
        assert "exit" not in kinds(strategy)

    def test_exit_disabled_holds_through_a_disagreeing_flow(self):
        cfg = self._cfg(refresh_seconds=0.0, exit_enabled=False)
        cfg.strategy.short_stop_loss_premium_multiple = None
        cfg.strategy.short_take_profit_pct = None
        cfg.strategy.daily_loss_limit_pct = None
        cfg.strategy.exit_on_regime_flip = False
        nowcast = FixedFlow(_AGREEING_FLOW)
        strategy = GexStraddleStrategy(cfg, open_interest=FixedRegime(POSITIVE), nowcast=nowcast)
        execution = SimulatedExecution(cfg.costs, cfg.source)
        strategy.on_bar(bar(0), execution)
        nowcast.rows = _DISAGREEING_FLOW
        strategy.on_bar(bar(5), execution)
        assert "exit" not in kinds(strategy)
        assert strategy.portfolio.straddle is not None

    # -- reconciliation ----------------------------------------------------

    def test_reconciliation_logs_the_divergence_against_next_mornings_oi(self):
        cfg = self._cfg()
        cfg.gex.refresh_seconds = 0.0
        cfg.nowcast.refresh_seconds = 0.0
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE),
            nowcast=FixedFlow(_flow_rows(call_signed_volume=500.0)),
        )
        execution = SimulatedExecution(cfg.costs, cfg.source)
        strategy.on_bar(session_bar(0, 0), execution)
        strategy.on_bar(session_bar(1, 0), execution)
        events = [e for e in strategy.events if e.kind == "nowcast_reconciliation"]
        assert events
        assert "OI error" in events[0].detail
        assert "dealer_share=1" in events[0].detail

    def test_reconciliation_can_be_turned_off(self):
        cfg = self._cfg(reconciliation_enabled=False)
        cfg.gex.refresh_seconds = 0.0
        cfg.nowcast.refresh_seconds = 0.0
        strategy = GexStraddleStrategy(
            cfg, open_interest=FixedRegime(POSITIVE),
            nowcast=FixedFlow(_flow_rows(call_signed_volume=500.0)),
        )
        execution = SimulatedExecution(cfg.costs, cfg.source)
        strategy.on_bar(session_bar(0, 0), execution)
        strategy.on_bar(session_bar(1, 0), execution)
        assert "nowcast_reconciliation" not in kinds(strategy)

    # -- authority stays narrow --------------------------------------------

    def test_the_nowcast_never_supplies_the_entry_direction(self):
        """Even fully corroborating flow only ever agrees or defers -- the
        direction itself always traces back to the daily print, in both
        regimes."""
        for regime in (POSITIVE, NEGATIVE):
            cfg = self._cfg()
            strategy = GexStraddleStrategy(
                cfg, open_interest=FixedRegime(regime), nowcast=FixedFlow(_AGREEING_FLOW),
            )
            strategy.on_bar(bar(0), SimulatedExecution(cfg.costs, cfg.source))
            position = strategy.portfolio.straddle
            assert position is not None
            assert position.regime == regime
