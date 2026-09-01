import pytest

from deltahedger.instruments import (
    DELTA_UNITS_PER_FUTURE, ContractSpec, RiskSource, get_risk_source, register,
)


class TestDeltaUnits:
    def test_one_es_future_is_one_hundred_units(self, es):
        assert es.delta_units_per_contract(es.future) == pytest.approx(100.0)

    def test_one_mes_future_is_ten_units(self, es):
        assert es.delta_units_per_contract(es.hedge) == pytest.approx(10.0)

    def test_the_hedge_quantum_is_the_mes_size(self, es):
        assert es.hedge_quantum == pytest.approx(10.0)

    def test_a_twenty_delta_short_put_is_twenty_units(self, es):
        """The worked example from the strategy specification."""
        contracts, delta = -1, -0.20
        units = contracts * delta * es.delta_units_per_contract(es.option)
        assert units == pytest.approx(20.0)

    def test_the_reference_scale_is_one_percent_of_a_future(self):
        assert DELTA_UNITS_PER_FUTURE == 100.0


class TestContractSpec:
    def test_es_tick_is_twelve_fifty(self, es):
        assert es.future.tick_value == pytest.approx(12.50)

    def test_mes_tick_is_one_twenty_five(self, es):
        assert es.hedge.tick_value == pytest.approx(1.25)

    def test_es_option_tick_is_two_fifty(self, es):
        assert es.option.tick_value == pytest.approx(2.50)


class TestRegistry:
    def test_lookup_is_case_insensitive(self):
        assert get_risk_source("es") is get_risk_source("ES")

    def test_aliases_resolve(self):
        assert get_risk_source("EMINI").name == "ES"

    def test_an_unknown_symbol_raises_with_a_helpful_message(self):
        with pytest.raises(KeyError, match="registered"):
            get_risk_source("XYZ")

    def test_a_new_risk_source_can_be_registered(self):
        """The extension point for adding NQ, CL and the rest later."""
        spec = ContractSpec("NQ", "FUT", "CME", multiplier=20.0, tick_size=0.25)
        micro = ContractSpec("MNQ", "FUT", "CME", multiplier=2.0, tick_size=0.25)
        option = ContractSpec("NQ", "FOP", "CME", multiplier=20.0, tick_size=0.25)
        source = RiskSource(
            name="NQ", future=spec, option=option, hedge=micro,
            reference_multiplier=20.0, strike_increment=10.0,
        )
        register(source)
        try:
            found = get_risk_source("NQ")
            assert found.hedge_quantum == pytest.approx(10.0)
            assert found.delta_units_per_contract(found.future) == pytest.approx(100.0)
        finally:
            from deltahedger.instruments import REGISTRY
            REGISTRY.pop("NQ", None)
