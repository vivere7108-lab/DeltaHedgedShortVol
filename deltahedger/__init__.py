"""Delta-hedged short-volatility trading on ES futures options.

Sell the shortest-dated (0DTE where listed) put on the ES future, size the
position from a buying-power allocation, and hold net delta inside a target
band by trading MES futures.

The same ``ShortVolStrategy`` runs the backtest and the live router, so a
forward test exercises validated logic rather than a reimplementation.
"""

__version__ = "0.1.0"

from .config import Config
from .hedger import DeltaHedger, HedgeDecision
from .instruments import RiskSource, get_risk_source, register
from .strategy import ShortVolStrategy

__all__ = [
    "Config",
    "DeltaHedger",
    "HedgeDecision",
    "RiskSource",
    "ShortVolStrategy",
    "get_risk_source",
    "register",
]
