from datetime import time

import pytest
import yaml

from deltahedger.config import Config, HedgeConfig, SizingConfig, StrategyConfig


class TestDefaults:
    def test_the_band_is_twenty_plus_or_minus_three(self):
        hedge = Config().hedge
        assert hedge.target == 20.0 and hedge.band == 3.0
        assert (hedge.lower, hedge.upper) == (17.0, 23.0)

    def test_buying_power_defaults_to_fifteen_percent(self):
        assert Config().sizing.buying_power_pct == 0.15

    def test_the_default_risk_source_is_es(self):
        assert Config().source.name == "ES"

    def test_zero_dte_is_the_default_target(self):
        assert Config().strategy.min_days_to_expiry == 0

    @pytest.mark.parametrize("value,expected", [
        (17.0, True), (20.0, True), (23.0, True), (16.9, False), (23.1, False),
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
        assert cfg.hedge.target == 20.0
        assert cfg.sizing.buying_power_pct == 0.15

    def test_an_empty_file_gives_the_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert Config.from_yaml(path).hedge.target == 20.0

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

    def test_rejects_an_out_of_range_delta_target(self):
        with pytest.raises(ValueError, match="short_put_delta"):
            StrategyConfig(short_put_delta=1.5).validate()

    def test_rejects_an_entry_window_that_closes_before_it_opens(self):
        with pytest.raises(ValueError, match="entry_cutoff_time"):
            StrategyConfig(entry_time=time(12, 0), entry_cutoff_time=time(9, 0)).validate()

    def test_rejects_negative_equity(self):
        with pytest.raises(ValueError, match="starting_equity"):
            Config(starting_equity=-1.0)

    def test_rejects_inverted_contract_limits(self):
        with pytest.raises(ValueError, match="max_short_contracts"):
            SizingConfig(min_short_contracts=5, max_short_contracts=2).validate()
