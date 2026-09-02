"""Black-76 pricing and greeks for options on futures.

Options on ES are options on the *future*, so Black-76 is the right model:
the forward is the future price itself and there is no carry term inside the
d1/d2 expressions -- only the discount factor out front.

Everything here is written to survive the 0DTE case.  As ``T -> 0`` the
lognormal density blows up and gamma/theta diverge, so all functions clamp
the time argument at ``MIN_YEARS`` and fall through to the intrinsic-value
limits below it.  Without that clamp a 0DTE backtest produces infinities the
moment a bar lands on the expiry minute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

#: Floor on time-to-expiry, in years. ~30 seconds. Below this we use the
#: expiry limits rather than the model, which keeps greeks finite at 0DTE.
MIN_YEARS = 30.0 / (365.0 * 24.0 * 60.0 * 60.0)

#: Floor on volatility. A zero-vol input is degenerate in the same way T=0 is.
MIN_VOL = 1e-6

SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class Greeks:
    """Per-contract greeks, in the natural units of each.

    ``delta`` is per 1.00 move in the future (so -1..0 for a put).
    ``gamma`` is delta change per 1.00 move.
    ``vega`` is price change per 1 *volatility point* (0.01 of vol).
    ``theta`` is price change per calendar day.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float

    def scaled(self, quantity: float, multiplier: float) -> "Greeks":
        """Greeks for ``quantity`` contracts, in dollars (except delta/gamma)."""
        k = quantity * multiplier
        return Greeks(
            price=self.price * k,
            delta=self.delta * quantity,
            gamma=self.gamma * quantity,
            vega=self.vega * k,
            theta=self.theta * k,
        )


def _d1_d2(f: float, k: float, t: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _expiry_limit(f: float, k: float, right: str, discount: float) -> Greeks:
    """Greeks at (or below) the time floor: intrinsic value, binary delta."""
    if right == "P":
        intrinsic = max(k - f, 0.0)
        if f < k:
            delta = -1.0
        elif f > k:
            delta = 0.0
        else:
            delta = -0.5  # exactly at the strike, split the difference
    else:
        intrinsic = max(f - k, 0.0)
        if f > k:
            delta = 1.0
        elif f < k:
            delta = 0.0
        else:
            delta = 0.5
    return Greeks(price=intrinsic * discount, delta=delta, gamma=0.0, vega=0.0, theta=0.0)


def black76(
    f: float,
    k: float,
    t: float,
    sigma: float,
    r: float = 0.0,
    right: str = "P",
) -> Greeks:
    """Price and greeks of a European option on a future.

    Parameters
    ----------
    f : future (forward) price
    k : strike
    t : time to expiry in years
    sigma : annualised volatility, as a decimal (0.18 == 18 vol)
    r : continuously-compounded risk-free rate, for discounting only
    right : "P" for put, "C" for call
    """
    right = right.upper()
    if right not in ("P", "C"):
        raise ValueError(f"right must be 'P' or 'C', got {right!r}")
    if f <= 0.0 or k <= 0.0:
        raise ValueError(f"future and strike must be positive, got F={f}, K={k}")

    discount = math.exp(-r * max(t, 0.0))
    if t <= MIN_YEARS or sigma <= MIN_VOL:
        return _expiry_limit(f, k, right, discount)

    d1, d2 = _d1_d2(f, k, t, sigma)
    sqrt_t = math.sqrt(t)
    pdf_d1 = norm.pdf(d1)

    # Shared across rights.
    gamma = discount * pdf_d1 / (f * sigma * sqrt_t)
    vega = discount * f * pdf_d1 * sqrt_t
    time_decay = -discount * f * pdf_d1 * sigma / (2.0 * sqrt_t)

    if right == "P":
        price = discount * (k * norm.cdf(-d2) - f * norm.cdf(-d1))
        delta = -discount * norm.cdf(-d1)
        theta = time_decay + r * discount * (k * norm.cdf(-d2) - f * norm.cdf(-d1))
    else:
        price = discount * (f * norm.cdf(d1) - k * norm.cdf(d2))
        delta = discount * norm.cdf(d1)
        theta = time_decay - r * discount * (f * norm.cdf(d1) - k * norm.cdf(d2))

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega / 100.0,  # per volatility point rather than per 1.00 of vol
        theta=theta / 365.0,  # per calendar day rather than per year
    )


def implied_vol(
    price: float,
    f: float,
    k: float,
    t: float,
    r: float = 0.0,
    right: str = "P",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float | None:
    """Back out Black-76 volatility from a price by bisection.

    Bisection rather than Newton: vega collapses near expiry and Newton
    diverges exactly where 0DTE spends most of its time.  Returns ``None``
    when the price is outside the no-arbitrage bounds (which happens on stale
    or crossed quotes) so callers can skip the bar rather than trust a fit.
    """
    discount = math.exp(-r * max(t, 0.0))
    intrinsic = (max(k - f, 0.0) if right.upper() == "P" else max(f - k, 0.0)) * discount
    upper_bound = (k if right.upper() == "P" else f) * discount
    if price < intrinsic - tol or price > upper_bound + tol or t <= MIN_YEARS:
        return None

    lo, hi = MIN_VOL, 5.0
    if black76(f, k, t, hi, r, right).price < price:
        return None  # beyond 500 vol, treat as unfittable

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if black76(f, k, t, mid, r, right).price > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def year_fraction(seconds: float) -> float:
    """Convert a duration in seconds to a year fraction, floored at MIN_YEARS."""
    return max(seconds / SECONDS_PER_YEAR, 0.0)


def black76_gamma(
    f: "float | np.ndarray",
    k: "float | np.ndarray",
    t: float,
    sigma: "float | np.ndarray",
    r: float = 0.0,
) -> "np.ndarray":
    """Gamma alone, vectorised over ``f``, ``k`` and ``sigma``.

    Gamma is identical for a call and a put on the same strike and expiry
    (put-call parity differs by a forward, which is linear in F), so the GEX
    profile needs one number per strike rather than two option prices.  This
    exists because a gamma-flip search reprices a whole chain across a grid
    of hypothetical spot levels on every bar -- doing that through
    ``black76`` and ``scipy.stats.norm`` is two orders of magnitude slower
    than it needs to be.

    Returns 0 where ``t`` or ``sigma`` is below the model floors, matching
    ``_expiry_limit``: at expiry the delta is a step and gamma is a spike of
    zero width, which is not a number a hedger can act on.
    """
    f_arr = np.asarray(f, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if t <= MIN_YEARS:
        return np.zeros(np.broadcast(f_arr, k_arr, sigma_arr).shape)

    safe_sigma = np.maximum(sigma_arr, MIN_VOL)
    sqrt_t = math.sqrt(t)
    vol_sqrt_t = safe_sigma * sqrt_t
    d1 = (np.log(f_arr / k_arr) + 0.5 * safe_sigma * safe_sigma * t) / vol_sqrt_t
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_TWO_PI
    gamma = math.exp(-r * t) * pdf / (f_arr * vol_sqrt_t)
    return np.where(sigma_arr <= MIN_VOL, 0.0, gamma)
