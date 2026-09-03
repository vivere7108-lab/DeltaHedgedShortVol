"""GEX-directed, delta-hedged 2-5 DTE straddles on ES futures options.

Read dealer gamma exposure off the front expiries, locate the gamma flip
point, and take the side dealer hedging is forced to supply: long the ATM
straddle and scalp gamma when dealers are short it, short the straddle and
collect theta when they are long it.  Either way the book is held
delta-neutral by trading MES futures against a fixed, heuristic delta band,
widened outside the regular session.

The traded series is a listed expiry two to five sessions out, closed at the
1DTE floor, so the position is never carried into the range where an ATM
straddle's gamma, its pin risk and the staleness of the open-interest print
all get worse together.  Four configurable gates decide whether a read is
worth acting on at all.

The same ``GexStraddleStrategy`` runs the backtest and the live router, so a
forward test exercises validated logic rather than a reimplementation.
"""

__version__ = "0.1.0"

from .chain import TenorPolicy
from .config import Config
from .gex import ExpiryBook, GexCalculator, GexProfile, StrikeOpenInterest
from .hedger import DeltaHedger, HedgeDecision
from .instruments import RiskSource, get_risk_source, register
from .strategy import GexStraddleStrategy

__all__ = [
    "Config",
    "DeltaHedger",
    "ExpiryBook",
    "GexCalculator",
    "GexProfile",
    "GexStraddleStrategy",
    "HedgeDecision",
    "RiskSource",
    "StrikeOpenInterest",
    "TenorPolicy",
    "get_risk_source",
    "register",
]
