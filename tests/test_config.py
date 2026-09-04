from datetime import time

import pytest
import yaml

from deltahedger.config import (
    Config,
    GatesConfig,
    GexConfig,
    HedgeConfig,
    NowcastConfig,
    SizingConfig,
    StrategyConfig,
)


class TestDefaults:
    def test_the_band_is_neutral_plus_or_minus_ten(self):
        """The fixed heuristic threshold for the forward walk: hold the
        straddle delta-neutral, one whole MES contract either side."""
        hedge = Config().hedge
        assert hedge.target == 0.0 and hedge.band == 10.0
        assert (hedge.lower, hedge.upper) == (-10.0, 10.0)

    def test_buying_power_defaults_to_fifteen_percent(self):
        assert Config().sizing.buying_power_pct == 0.15

    def test_the_default_risk_source_is_es(self):
        assert Config().source.name == "ES"

    def test_the_default_tenor_is_two_to_five_dte(self):
        strategy = Config().strategy
        assert (strategy.min_days_to_expiry, strategy.max_days_to_expiry) == (2, 5)
        assert (
            strategy.prefer_min_days_to_expiry, strategy.prefer_max_days_to_expiry
        ) == (3, 4)
        assert strategy.close_at_days_to_expiry == 1

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

    def test_nowcast_is_off_by_default(self):
        """The nowcast needs a paid Databento subscription; it must never
        turn itself on."""
        nowcast = Config().nowcast
        assert nowcast.enabled is False
        assert nowcast.dataset == "GLBX.MDP3"
        assert nowcast.parent_symbol == "ES.OPT"
        assert nowcast.dealer_share == 0.35
        assert nowcast.refresh_seconds == 1200.0
        assert nowcast.veto_enabled and nowcast.exit_enabled
        assert nowcast.size_haircut_enabled and nowcast.reconciliation_enabled

    def test_open_interest_defaults_to_the_generated_surface(self):
        """A backtest must never silently reach for a live connection."""
        assert Config().data.open_interest == "synthetic"

    @pytest.mark.parametrize("value,expected", [
        (-10.0, True), (0.0, True), (10.0, True), (-10.1, False), (10.1, False),
    ])
    def test_in_band(self, value, expected):
        assert Config().hedge.in_band(value) is expected


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

    def test_a_partial_file_keeps_the_other_defaults(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"hedge": {"band": 8.0}}))
        cfg = Config.from_yaml(path)
        assert cfg.hedge.band == 8.0
        assert cfg.hedge.target == 0.0
        assert cfg.sizing.buying_power_pct == 0.15

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

    def test_rejects_a_close_floor_at_or_above_the_dte_minimum(self):
        """A position entered at the floor would be immediately eligible to
        close, which is not a tenor -- it is a bug that looks like one."""
        with pytest.raises(ValueError, match="close_days"):
            StrategyConfig(
                min_days_to_expiry=2, max_days_to_expiry=5, close_at_days_to_expiry=2,
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

    @pytest.mark.parametrize("kwargs,message", [
        ({"dealer_share": -0.1}, "dealer_share"),
        ({"dealer_share": 3.1}, "dealer_share"),
        ({"refresh_seconds": 0.0}, "refresh_seconds"),
        ({"refresh_seconds": -1.0}, "refresh_seconds"),
        ({"backfill_days": 0}, "backfill_days"),
        ({"size_haircut_when_unconfirmed": -0.1}, "size_haircut_when_unconfirmed"),
        ({"size_haircut_when_unconfirmed": 1.1}, "size_haircut_when_unconfirmed"),
    ])
    def test_rejects_impossible_nowcast_settings(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            NowcastConfig(**kwargs).validate()

    def test_a_nowcast_section_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "n.yaml"
        path.write_text(yaml.safe_dump({"nowcast": {"enabled": True, "dealer_share": 0.5}}))
        cfg = Config.from_yaml(path)
        assert cfg.nowcast.enabled is True
        assert cfg.nowcast.dealer_share == 0.5

    def test_a_nowcast_typo_is_rejected_rather_than_ignored(self, tmp_path):
        path = tmp_path / "n.yaml"
        path.write_text(yaml.safe_dump({"nowcast": {"dealer_sharee": 0.5}}))
        with pytest.raises(ValueError, match="dealer_sharee"):
            Config.from_yaml(path)
