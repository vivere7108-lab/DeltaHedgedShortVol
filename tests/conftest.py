import pytest

from deltahedger.config import Config
from deltahedger.instruments import get_risk_source


@pytest.fixture
def es():
    return get_risk_source("ES")


@pytest.fixture
def cfg():
    config = Config()
    config.data.source = "synthetic"
    config.data.synthetic_days = 5
    config.starting_equity = 250_000.0
    return config
