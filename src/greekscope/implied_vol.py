"""Implied volatility solver (guarded Newton with a Brent fallback)."""

from __future__ import annotations

import math

from scipy.optimize import brentq

from ._validate import check_inputs, check_kind

__all__ = ["implied_vol"]

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 0.3989422804014327
_LO = 1e-6
_HI = 5.0


def _bs(kind: str, fs: float, pk: float, m: float, sqrt_t: float, sigma: float):
    """Scalar BS price and vega on forward-discounted quantities.

    ``fs = S e^{-qT}``, ``pk = K e^{-rT}``, ``m = ln(fs / pk)``.
    """
    sd = sigma * sqrt_t
    d1 = (m + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if kind == "call":
        p = fs * 0.5 * math.erfc(-d1 / _SQRT2) - pk * 0.5 * math.erfc(-d2 / _SQRT2)
    else:
        p = pk * 0.5 * math.erfc(d2 / _SQRT2) - fs * 0.5 * math.erfc(d1 / _SQRT2)
    vega = fs * sqrt_t * _INV_SQRT_2PI * math.exp(-0.5 * d1 * d1)
    return p, vega


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: str = "call",
    q: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Black-Scholes-Merton implied volatility of a European option price.

    The solver works on the out-of-the-money side of put-call parity (better
    conditioned), starts from a Corrado-Miller guess, takes Newton steps using
    vega, and falls back to :func:`scipy.optimize.brentq` on the tightest
    bracket found whenever vega is tiny or a step leaves the bracket.

    Parameters
    ----------
    price : float
        Observed option price.
    S, K : float
        Spot and strike (> 0).
    T : float
        Years to expiry (> 0).
    r : float
        Continuously compounded risk-free rate.
    kind : {"call", "put"}
    q : float, default 0.0
        Continuous dividend yield.
    tol : float, default 1e-10
        Convergence tolerance on sigma.
    max_iter : int, default 100
        Maximum Newton iterations before falling back to Brent.

    Returns
    -------
    float
        Implied volatility.  Returns ``0.0`` if the price equals the discounted
        intrinsic value (or implies a vol below 1e-6).

    Raises
    ------
    ValueError
        If inputs are invalid, ``T <= 0``, the price is outside the no-arbitrage
        bounds, or the implied vol exceeds 5.0 (500%).
    """
    kind = check_kind(kind)
    check_inputs(S, K, T, 0.0, r=r, q=q)
    if not T > 0:
        raise ValueError("implied_vol requires T > 0")
    price = float(price)
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")

    fs = S * math.exp(-q * T)
    pk = K * math.exp(-r * T)
    if kind == "call":
        lower, upper = max(fs - pk, 0.0), fs
    else:
        lower, upper = max(pk - fs, 0.0), pk
    slack = 1e-12 * max(S, K)
    if price < lower - slack:
        raise ValueError(
            f"price {price} is below the no-arbitrage lower bound {lower:.12g} "
            f"for a {kind}"
        )
    if price > upper + slack:
        raise ValueError(
            f"price {price} is above the no-arbitrage upper bound {upper:.12g} "
            f"for a {kind}"
        )

    # Work with the out-of-the-money option (forward vs. strike).
    parity = fs - pk  # C - P
    if parity >= 0.0:
        # OTM option is the put; call price converts via P = C - (fs - pk).
        otm, upper_otm = "put", pk
        target = price - parity if kind == "call" else price
    else:
        # OTM option is the call; put price converts via C = P + (fs - pk).
        otm, upper_otm = "call", fs
        target = price if kind == "call" else price + parity
    target = max(target, 0.0)
    if target <= 0.0:
        return 0.0
    if target >= upper_otm:
        raise ValueError("price implies an infinite volatility")

    m = math.log(fs / pk)
    sqrt_t = math.sqrt(T)

    def f(sig: float) -> float:
        return _bs(otm, fs, pk, m, sqrt_t, sig)[0] - target

    p_lo = _bs(otm, fs, pk, m, sqrt_t, _LO)[0]
    if target <= p_lo:
        return 0.0
    p_hi = _bs(otm, fs, pk, m, sqrt_t, _HI)[0]
    if target > p_hi:
        raise ValueError(f"implied volatility exceeds {_HI} (price too high)")
    if target == p_hi:
        return _HI

    # Corrado-Miller initial guess on the equivalent call price.
    c = target if otm == "call" else target + parity
    x = fs - pk
    a = c - 0.5 * x
    rad = a * a - x * x / math.pi
    if rad >= 0.0:
        guess = math.sqrt(2.0 * math.pi) / ((fs + pk) * sqrt_t) * (a + math.sqrt(rad))
    else:
        guess = math.sqrt(2.0 * math.pi / T) * c / fs
    if not math.isfinite(guess) or guess <= _LO:
        guess = 0.2
    sig = min(max(guess, 1e-3), 4.0)

    lo, hi = _LO, _HI
    for _ in range(max_iter):
        p, v = _bs(otm, fs, pk, m, sqrt_t, sig)
        diff = p - target
        if diff == 0.0:
            return sig
        if diff > 0.0:
            hi = sig
        else:
            lo = sig
        if v < 1e-14:
            break
        new = sig - diff / v
        if not (lo < new < hi):
            break
        if abs(new - sig) < tol:
            return new
        sig = new

    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=500))
