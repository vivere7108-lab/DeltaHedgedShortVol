"""GEX-directed, delta-hedged 0DTE straddles on ES futures options.

Read dealer gamma exposure off the front of the curve, locate the gamma
flip point, and take the side dealer hedging is forced to supply: long the
ATM straddle and scalp gamma when dealers are short it, short the straddle
and collect theta when they are long it.  Either way the book is sized to
the margin limit less a buffer and held delta-neutral by trading MES
futures under a Whalley-Wilmott band -- a half-width that scales with the
book's gamma and the cost of hedging rather than a fixed number.

The traded series is today's expiry, closed a quarter of an hour before
settlement -- where an ATM straddle's gamma diverges -- and rolled into
tomorrow's at that moment, except into a weekend, a holiday, or the
blackout around a scheduled event such as an FOMC statement.  Four
configurable gates decide whether a read is worth acting on at all.

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
