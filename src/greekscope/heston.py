"""Heston (1993) stochastic-volatility European option pricing.

Semi-analytic: Gil-Pelaez inversion of the characteristic function written in
the numerically stable "little Heston trap" form of Albrecher, Mayer,
Schoutens and Tistaert (2007), integrated with :func:`scipy.integrate.quad`.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.integrate import quad

from ._validate import check_inputs, check_kind

__all__ = ["heston_price"]


def _check_params(v0: float, kappa: float, theta: float, xi: float, rho: float) -> None:
    vals = (v0, kappa, theta, xi, rho)
    if not all(np.isfinite(vals)):
        raise ValueError("Heston parameters must be finite")
    if v0 < 0:
        raise ValueError(f"v0 must be >= 0, got {v0!r}")
    if kappa <= 0:
        raise ValueError(f"kappa must be > 0, got {kappa!r}")
    if theta < 0:
        raise ValueError(f"theta must be >= 0, got {theta!r}")
    if xi <= 0:
        raise ValueError(f"xi (vol of vol) must be > 0, got {xi!r}")
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho!r}")


def _log1p_c(w: complex) -> complex:
    """``log(1 + w)`` for complex ``w``, accurate for small ``|w|`` (series)."""
    if abs(w) < 0.1:
        total = 0j
        term = w
        for n in range(1, 40):
            total += term / n if n % 2 else -term / n
            term *= w
            if abs(term) < 1e-18 * (abs(total) + 1e-300):
                break
        return total
    return cmath.log(1.0 + w)


def _integrand(phi, j, lnS, lnK, T, r, q, v0, kappa, theta, xi, rho):
    """Real integrand of P_j (j = 1, 2) in the 'little trap' formulation.

    Uses ``t - d = xi^2 (2 u i phi - phi^2) / (t + d)`` (from ``t^2 - d^2``) so that
    nothing cancels catastrophically when ``xi`` is tiny.
    """
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa
    xi2 = xi * xi
    iphi = 1j * phi
    t = b - rho * xi * iphi
    num = 2.0 * u * iphi - phi * phi
    d = cmath.sqrt(t * t - xi2 * num)
    tpd = t + d
    tmd = xi2 * num / tpd  # = t - d
    g = tmd / tpd
    e = cmath.exp(-d * T)
    w = g * (1.0 - e) / (1.0 - g)  # (1 - g e)/(1 - g) = 1 + w
    C = (r - q) * iphi * T + kappa * theta * (tmd * T - 2.0 * _log1p_c(w)) / xi2
    D = num / tpd * (1.0 - e) / (1.0 - g * e)
    f = cmath.exp(C + D * v0 + iphi * lnS)
    return (cmath.exp(-iphi * lnK) * f / iphi).real


def _prob(j, lnS, lnK, T, r, q, v0, kappa, theta, xi, rho) -> float:
    args = (j, lnS, lnK, T, r, q, v0, kappa, theta, xi, rho)
    edges = [0.0, 5.0, 25.0, 100.0, 300.0]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        total += quad(_integrand, a, b, args=args, limit=200, epsabs=1e-13, epsrel=1e-11)[0]
    # Extend the upper limit if the tail has not decayed (short maturities).
    a, width = edges[-1], 300.0
    for _ in range(6):
        tail = quad(_integrand, a, a + width, args=args, limit=200, epsabs=1e-13, epsrel=1e-11)[0]
        total += tail
        if abs(tail) < 1e-11:
            break
        a += width
        width *= 2.0
    return 0.5 + total / math.pi


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    kind: str = "call",
    q: float = 0.0,
) -> float:
    """Heston stochastic-volatility price of a European option.

    The dynamics are ``dS = (r-q) S dt + sqrt(v) S dW1``,
    ``dv = kappa (theta - v) dt + xi sqrt(v) dW2``, ``corr(dW1, dW2) = rho``.
    The call is ``S e^{-qT} P1 - K e^{-rT} P2``; the put follows from put-call parity.

    Parameters
    ----------
    S, K : float
        Spot and strike (> 0).
    T : float
        Years to expiry (>= 0).
    r : float
        Risk-free rate.
    v0 : float
        Initial variance (>= 0).
    kappa : float
        Mean-reversion speed (> 0).
    theta : float
        Long-run variance (>= 0).
    xi : float
        Volatility of variance (> 0).
    rho : float
        Spot/variance correlation, in (-1, 1).
    kind : {"call", "put"}
    q : float, default 0.0
        Continuous dividend yield.

    Returns
    -------
    float
        Option price (``T == 0`` returns intrinsic value).

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    kind = check_kind(kind)
    check_inputs(S, K, T, 0.0, r=r, q=q)
    _check_params(v0, kappa, theta, xi, rho)
    S, K, T = float(S), float(K), float(T)
    fs = S * math.exp(-q * T)
    pk = K * math.exp(-r * T)
    if T == 0.0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    lnS, lnK = math.log(S), math.log(K)
    p = (lnS, lnK, T, r, q, v0, kappa, theta, xi, rho)
    p1 = _prob(1, *p)
    p2 = _prob(2, *p)
    call = fs * p1 - pk * p2
    call = max(call, fs - pk, 0.0)
    if kind == "call":
        return float(min(call, fs))
    put = call - fs + pk
    return float(min(max(put, 0.0), pk))
