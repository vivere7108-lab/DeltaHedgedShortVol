from datetime import time

import pytest
import yaml

from deltahedger.config import (
    Config,
    GatesConfig,
    GexConfig,
    HedgeConfig,
    SizingConfig,
    StrategyConfig,
)


class TestDefaults:
    def test_the_band_is_whalley_wilmott_about_neutral(self):
        """Hold the straddle delta-neutral under a Whalley-Wilmott band at
        a risk aversion of 0.01 per dollar; the fixed +/-10 is the control."""
        hedge = Config().hedge
        assert hedge.target == 0.0
        assert hedge.band_model == "whalley_wilmott" and not hedge.is_fixed
        assert hedge.risk_aversion == 0.01
        assert hedge.hedge_cost_per_contract is None  # derived from costs
        assert hedge.band == 10.0

    def test_buying_power_defaults_to_the_margin_limit_less_a_fifth(self):
        assert Config().sizing.buying_power_pct == 0.80

    def test_the_default_risk_source_is_es(self):
        assert Config().source.name == "ES"

    def test_the_default_tenor_is_todays_series_rolled_into_tomorrows(self):
        strategy = Config().strategy
        assert (strategy.min_days_to_expiry, strategy.max_days_to_expiry) == (0, 1)
        assert (
            strategy.prefer_min_days_to_expiry, strategy.prefer_max_days_to_expiry
        ) == (0, 0)
        assert strategy.close_at_days_to_expiry is None
        assert strategy.close_before_expiry_minutes == 15
        assert strategy.roll_at_expiry
        assert not strategy.hold_over_weekends

    def test_the_event_blackout_is_a_quarter_hour_each_side(self):
        strategy = Config().strategy
        assert strategy.event_blackout_minutes_before == 15
        assert strategy.event_blackout_minutes_after == 15
        assert strategy.events == [] and strategy.events_path is None

    def test_all_four_gates_are_on_by_default(self):
        gates = Config().gates
        assert gates.confidence and gates.flip_distance
        assert gates.ensemble and gates.persistence and gates.entry_window

    def test_the_overnight_band_widens_by_default(self):
        assert Config().hedge.overnight_band_multiplier > 1.0

    def test_gex_is_on_with_the_standard_dealer_convention(self):
        gex = Config().gex
        assert gex.enabled
        assert (gex.call_sign, gex.put_sign) == (1.0, -1.0)

    def test_open_interest_defaults_to_the_generated_surface(self):
        """A backtest must never silently reach for a live connection."""
        assert Config().data.open_interest == "synthetic"

    def test_the_fixed_model_is_selectable(self):
        assert HedgeConfig(band_model="fixed").is_fixed


class TestSerialisation:
    def test_yaml_round_trip(self, tmp_path):
        original = Config()
        original.hedge.target = 35.0
        original.sizing.buying_power_pct = 0.25
        original.strategy.entry_time = time(10, 15)
        path = tmp_path / "c.yaml"
        original.to_yaml(path)

        restored = Config.from_yaml(path)
        assert restored.hedge.target == 35.0
        assert restored.sizing.buying_power_pct == 0.25
        assert restored.strategy.entry_time == time(10, 15)

    def test_events_and_the_tenor_round_trip(self, tmp_path):
        original = Config()
        original.strategy.events = ["2026-09-16 14:00 FOMC statement"]
        original.strategy.hold_over_weekends = True
        original.strategy.close_at_days_to_expiry = None
        original.hedge.band_model = "fixed"
        path = tmp_path / "c.yaml"
        original.to_yaml(path)

        restored = Config.from_yaml(path)
        assert restored.strategy.events == ["2026-09-16 14:00 FOMC statement"]
        assert restored.strategy.hold_over_weekends
        assert restored.strategy.close_at_days_to_expiry is None
        assert restored.hedge.band_model == "fixed"

    def test_a_partial_file_keeps_the_other_defaults(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"hedge": {"band": 8.0}}))
        cfg = Config.from_yaml(path)
        assert cfg.hedge.band == 8.0
        assert cfg.hedge.target == 0.0
        assert cfg.sizing.buying_power_pct == 0.80

    def test_an_empty_file_gives_the_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert Config.from_yaml(path).hedge.band == 10.0

    def test_a_typo_is_rejected_rather_than_ignored(self, tmp_path):
        path = tmp_path / "typo.yaml"
        path.write_text(yaml.safe_dump({"hedge": {"targett": 5.0}}))
        with pytest.raises(ValueError, match="targett"):
            Config.from_yaml(path)

    def test_an_unknown_top_level_key_is_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"nonsense": 1}))
        with pytest.raises(ValueError, match="nonsense"):
            Config.from_yaml(path)

    @pytest.mark.parametrize("value", ["09:35", "9:35:00"])
    def test_times_parse_from_strings(self, value, tmp_path):
        path = tmp_path / "t.yaml"
        path.write_text(yaml.safe_dump({"strategy": {"entry_time": value}}))
        assert Config.from_yaml(path).strategy.entry_time == time(9, 35)


class TestValidation:
    def test_rejects_an_unknown_risk_source(self):
        with pytest.raises(KeyError, match="unknown risk source"):
            Config(risk_source="NOPE")

    @pytest.mark.parametrize("pct", [0.0, -0.1, 1.5])
    def test_rejects_an_impossible_buying_power(self, pct):
        with pytest.raises(ValueError, match="buying_power_pct"):
            SizingConfig(buying_power_pct=pct).validate()

    def test_rejects_a_negative_band(self):
        with pytest.raises(ValueError, match="band"):
            HedgeConfig(band=-1.0).validate()

    def test_rejects_an_unknown_margin_model(self):
        with pytest.raises(ValueError, match="margin_model"):
            SizingConfig(margin_model="vibes").validate()

    def test_rejects_an_inverted_dte_window(self):
        with pytest.raises(ValueError, match="min_days_to_expiry"):
            StrategyConfig(min_days_to_expiry=2, max_days_to_expiry=0).validate()

    def test_rejects_a_prefer_window_outside_the_dte_bounds(self):
        with pytest.raises(ValueError, match="prefer_days"):
            StrategyConfig(
                min_days_to_expiry=2, max_days_to_expiry=5,
                prefer_min_days_to_expiry=1, prefer_max_days_to_expiry=4,
            ).validate()

    def test_a_disabled_close_floor_is_allowed_at_any_tenor(self):
        StrategyConfig(
            min_days_to_expiry=0, max_days_to_expiry=0, close_at_days_to_expiry=None,
        ).validate()

    def test_rejects_a_negative_pre_settlement_buffer(self):
        with pytest.raises(ValueError, match="close_before_expiry_minutes"):
            StrategyConfig(close_before_expiry_minutes=-1).validate()

    def test_rejects_an_events_entry_that_is_not_a_list(self):
        with pytest.raises(ValueError, match="events"):
            StrategyConfig(events="2026-09-16 14:00").validate()

    def test_rejects_an_unknown_band_model(self):
        with pytest.raises(ValueError, match="band_model"):
            HedgeConfig(band_model="adaptive").validate()

    def test_rejects_a_non_positive_risk_aversion(self):
        with pytest.raises(ValueError, match="risk_aversion"):
            HedgeConfig(risk_aversion=-0.01).validate()

    def test_rejects_a_close_floor_at_or_above_the_dte_minimum(self):
        """A position entered at the floor would be immediately eligible to
        close, which is not a tenor -- it is a bug that looks like one."""
        with pytest.raises(ValueError, match="close_days"):
            StrategyConfig(
                min_days_to_expiry=2, max_days_to_expiry=5,
                prefer_min_days_to_expiry=3, prefer_max_days_to_expiry=4,
                close_at_days_to_expiry=2,
            ).validate()

    def test_rejects_an_overnight_band_narrower_than_the_day_band(self):
        with pytest.raises(ValueError, match="overnight_band_multiplier"):
            HedgeConfig(overnight_band_multiplier=0.5).validate()

    def test_rejects_an_out_of_range_confidence_ratio(self):
        with pytest.raises(ValueError, match="min_confidence_ratio"):
            GatesConfig(min_confidence_ratio=1.0).validate()

    def test_rejects_a_persistence_window_below_one_bar(self):
        with pytest.raises(ValueError, match="persistence_bars"):
            GatesConfig(persistence_bars=0).validate()

    def test_the_ensemble_must_include_the_traded_surface(self):
        with pytest.raises(ValueError, match="ensemble_skew_slope_deltas"):
            GatesConfig(ensemble_skew_slope_deltas=[-0.5, 0.5]).validate()

    def test_a_gates_section_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "gates.yaml"
        path.write_text(yaml.safe_dump({"gates": {"persistence_bars": 5}}))
        cfg = Config.from_yaml(path)
        assert cfg.gates.persistence_bars == 5

    def test_rejects_a_session_entry_cap_below_one(self):
        with pytest.raises(ValueError, match="max_entries_per_session"):
            StrategyConfig(max_entries_per_session=0).validate()

    def test_rejects_an_entry_window_that_closes_before_it_opens(self):
        with pytest.raises(ValueError, match="entry_cutoff_time"):
            StrategyConfig(entry_time=time(12, 0), entry_cutoff_time=time(9, 0)).validate()

    def test_rejects_negative_equity(self):
        with pytest.raises(ValueError, match="starting_equity"):
            Config(starting_equity=-1.0)

    def test_rejects_inverted_contract_limits(self):
        with pytest.raises(ValueError, match="max_straddles"):
            SizingConfig(min_straddles=5, max_straddles=2).validate()

    @pytest.mark.parametrize("kwargs,message", [
        ({"strike_width_pct": 0.0}, "strike_width_pct"),
        ({"flip_search_steps": 2}, "flip_search_steps"),
        ({"flip_search_pct": -0.1}, "flip_search_pct"),
        ({"min_hours_to_expiry": -1.0}, "min_hours_to_expiry"),
        ({"blend_max_expiries": 0}, "blend_max_expiries"),
    ])
    def test_rejects_impossible_gex_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            GexConfig(**kwargs).validate()

    def test_a_gex_section_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "g.yaml"
        path.write_text(yaml.safe_dump({"gex": {"call_sign": -1.0, "put_sign": 1.0}}))
        cfg = Config.from_yaml(path)
        assert (cfg.gex.call_sign, cfg.gex.put_sign) == (-1.0, 1.0)

    def test_a_gex_typo_is_rejected_rather_than_ignored(self, tmp_path):
        path = tmp_path / "g.yaml"
        path.write_text(yaml.safe_dump({"gex": {"call_signn": 1.0}}))
        with pytest.raises(ValueError, match="call_signn"):
            Config.from_yaml(path)
