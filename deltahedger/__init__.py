"""GEX-directed, delta-hedged 0DTE straddles on ES futures options.

Read dealer gamma exposure off the 0DTE chain, locate the gamma flip point,
and take the side dealer hedging is forced to supply: long the ATM straddle
and scalp gamma when dealers are short it, short the straddle and collect
theta when they are long it.  Either way the book is held delta-neutral by
trading MES futures against a fixed, heuristic delta band.

The same ``GexStraddleStrategy`` runs the backtest and the live router, so a
forward test exercises validated logic rather than a reimplementation.
"""

__version__ = "0.1.0"

from .config import Config
from .gex import GexCalculator, GexProfile, StrikeOpenInterest
from .hedger import DeltaHedger, HedgeDecision
from .instruments import RiskSource, get_risk_source, register
from .strategy import GexStraddleStrategy

__all__ = [
    "Config",
    "DeltaHedger",
    "GexCalculator",
    "GexProfile",
    "GexStraddleStrategy",
    "HedgeDecision",
    "RiskSource",
    "StrikeOpenInterest",
    "get_risk_source",
    "register",
]
