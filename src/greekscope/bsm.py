"""Black-Scholes-Merton pricing for European options (continuous dividend yield)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr

from ._validate import _scalar_or_array, check_inputs, check_kind

__all__ = ["d1", "d2", "price"]


def _broadcast(S, K, T, r, sigma, q):
    return np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    )


def _d1_d2(S, K, T, r, sigma, q):
    """Unvalidated core. Returns broadcast arrays ``(S, K, T, r, sigma, q, sd, d1, d2)``.

    ``sd = sigma * sqrt(T)``. When ``sd == 0`` the d's are ``+/-inf`` (or 0 exactly
    at the forward), which makes ``ndtr`` return the correct step-function limits.
    """
    S, K, T, r, sigma, q = _broadcast(S, K, T, r, sigma, q)
    sd = sigma * np.sqrt(T)
    m = np.log(S / K) + (r - q) * T
    pos = sd > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe = np.where(pos, sd, 1.0)
        det = np.where(m > 0, np.inf, np.where(m < 0, -np.inf, 0.0))
        D1 = np.where(pos, (m + 0.5 * sd * sd) / safe, det)
        D2 = D1 - sd
    return S, K, T, r, sigma, q, sd, D1, D2


def d1(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0) -> Any:
    """Black-Scholes ``d1``.

    Parameters
    ----------
    S, K : float or array_like
        Spot and strike (> 0).
    T : float or array_like
        Years to expiry (>= 0).
    r : float or array_like
        Continuously compounded risk-free rate.
    sigma : float or array_like
        Volatility (>= 0).
    q : float or array_like, default 0.0
        Continuous dividend yield.

    Returns
    -------
    float or numpy.ndarray
        ``d1``. If ``sigma * sqrt(T) == 0`` the result is ``+inf``/``-inf``
        according to the sign of ``log(S/K) + (r-q) T`` (0 exactly at the forward).
    """
    check_inputs(S, K, T, sigma, r=r, q=q)
    return _scalar_or_array(_d1_d2(S, K, T, r, sigma, q)[7])


def d2(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0) -> Any:
    """Black-Scholes ``d2 = d1 - sigma sqrt(T)``. See :func:`d1` for parameters."""
    check_inputs(S, K, T, sigma, r=r, q=q)
    return _scalar_or_array(_d1_d2(S, K, T, r, sigma, q)[8])


def price(
    S: Any,
    K: Any,
    T: Any,
    r: Any,
    sigma: Any,
    kind: str = "call",
    q: Any = 0.0,
) -> Any:
    """Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    S, K : float or array_like
        Spot and strike (> 0).
    T : float or array_like
        Years to expiry (>= 0).
    r : float or array_like
        Continuously compounded risk-free rate.
    sigma : float or array_like
        Volatility (>= 0).
    kind : {"call", "put"}
        Option type (case-insensitive).
    q : float or array_like, default 0.0
        Continuous dividend yield.

    Returns
    -------
    float or numpy.ndarray
        Option price. For ``T == 0`` or ``sigma == 0`` the deterministic limit
        ``max(+/-(S e^{-qT} - K e^{-rT}), 0)`` is returned.

    Raises
    ------
    ValueError
        On invalid inputs or ``kind``.
    """
    kind = check_kind(kind)
    check_inputs(S, K, T, sigma, r=r, q=q)
    S, K, T, r, sigma, q, sd, D1, D2 = _d1_d2(S, K, T, r, sigma, q)
    fs = S * np.exp(-q * T)
    pk = K * np.exp(-r * T)
    if kind == "call":
        stoch = fs * ndtr(D1) - pk * ndtr(D2)
        det = np.maximum(fs - pk, 0.0)
    else:
        stoch = pk * ndtr(-D2) - fs * ndtr(-D1)
        det = np.maximum(pk - fs, 0.0)
    out = np.where(sd > 0, np.maximum(stoch, 0.0), det)
    return _scalar_or_array(out)
