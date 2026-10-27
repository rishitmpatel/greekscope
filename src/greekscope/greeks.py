"""Analytic Black-Scholes-Merton Greeks.

Conventions
-----------
* ``vega`` and ``rho`` are per unit (1.00) change of ``sigma`` / ``r`` (not per 1%).
* ``theta`` is per year and equals ``-dV/dT`` (calendar-time decay), so it is
  usually negative for long options.
* At ``T == 0``: delta is a step function (0.5 exactly at the money), while
  gamma, vega, theta and rho are 0.  With ``sigma == 0`` and ``T > 0`` the
  option is deterministic: delta is a (discounted) step function, gamma is 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr

from ._validate import _scalar_or_array, check_inputs, check_kind
from .bsm import _d1_d2
from .bsm import price as _bsm_price

__all__ = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "all_greeks"]

_INV_SQRT_2PI = 0.3989422804014327


def _pdf(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-0.5 * x * x) * _INV_SQRT_2PI


def _setup(S, K, T, r, sigma, q):
    check_inputs(S, K, T, sigma, r=r, q=q)
    return _d1_d2(S, K, T, r, sigma, q)


def delta(S: Any, K: Any, T: Any, r: Any, sigma: Any, kind: str = "call", q: Any = 0.0) -> Any:
    """Option delta ``dV/dS``.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array_like
        See :func:`greekscope.bsm.price`.
    kind : {"call", "put"}

    Returns
    -------
    float or numpy.ndarray
    """
    kind = check_kind(kind)
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    dq = np.exp(-q * T)
    out = dq * ndtr(D1) if kind == "call" else -dq * ndtr(-D1)
    return _scalar_or_array(out)


def gamma(S: Any, K: Any, T: Any, r: Any, sigma: Any, kind: str = "call", q: Any = 0.0) -> Any:
    """Option gamma ``d2V/dS2`` (identical for calls and puts; 0 when ``sigma sqrt(T) == 0``)."""
    check_kind(kind)
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    pos = sd > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.exp(-q * T) * _pdf(D1) / (S * np.where(pos, sd, 1.0))
    return _scalar_or_array(np.where(pos, g, 0.0))


def vega(S: Any, K: Any, T: Any, r: Any, sigma: Any, kind: str = "call", q: Any = 0.0) -> Any:
    """Option vega ``dV/dsigma`` per 1.00 change in volatility (0 at ``T == 0``)."""
    check_kind(kind)
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    v = S * np.exp(-q * T) * _pdf(D1) * np.sqrt(T)
    return _scalar_or_array(np.where(T > 0, v, 0.0))


def theta(S: Any, K: Any, T: Any, r: Any, sigma: Any, kind: str = "call", q: Any = 0.0) -> Any:
    """Option theta per year, defined as ``-dV/dT`` (0 at ``T == 0``)."""
    kind = check_kind(kind)
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    dq = np.exp(-q * T)
    dr = np.exp(-r * T)
    pos = sd > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -S * dq * _pdf(D1) * sigma / (2.0 * np.sqrt(np.where(T > 0, T, 1.0)))
    decay = np.where(pos, decay, 0.0)
    if kind == "call":
        th = decay - r * K * dr * ndtr(D2) + q * S * dq * ndtr(D1)
    else:
        th = decay + r * K * dr * ndtr(-D2) - q * S * dq * ndtr(-D1)
    return _scalar_or_array(np.where(T > 0, th, 0.0))


def rho(S: Any, K: Any, T: Any, r: Any, sigma: Any, kind: str = "call", q: Any = 0.0) -> Any:
    """Option rho ``dV/dr`` per 1.00 change in the rate (0 at ``T == 0``)."""
    kind = check_kind(kind)
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    base = K * T * np.exp(-r * T)
    out = base * ndtr(D2) if kind == "call" else -base * ndtr(-D2)
    return _scalar_or_array(out)


def vanna(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0) -> Any:
    """Vanna ``d2V/(dS dsigma) = d delta / d sigma`` (same for calls and puts)."""
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    pos = sd > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -np.exp(-q * T) * _pdf(D1) * D2 / np.where(pos, sigma, 1.0)
        v = np.where(pos & np.isfinite(D2), v, 0.0)
    return _scalar_or_array(v)


def volga(S: Any, K: Any, T: Any, r: Any, sigma: Any, q: Any = 0.0) -> Any:
    """Volga (vomma) ``d2V/dsigma2 = vega * d1 * d2 / sigma`` (same for calls and puts)."""
    S, K, T, r, sigma, q, sd, D1, D2 = _setup(S, K, T, r, sigma, q)
    pos = sd > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vg = S * np.exp(-q * T) * _pdf(D1) * np.sqrt(T)
        v = vg * D1 * D2 / np.where(pos, sigma, 1.0)
        v = np.where(pos & np.isfinite(D1) & np.isfinite(D2), v, 0.0)
    return _scalar_or_array(v)


def all_greeks(
    S: Any, K: Any, T: Any, r: Any, sigma: Any, kind: str = "call", q: Any = 0.0
) -> dict:
    """Price and all Greeks in one dict.

    Returns
    -------
    dict
        Keys ``price, delta, gamma, vega, theta, rho, vanna, volga`` (floats for
        scalar input, arrays otherwise).  Conventions as in the module docstring.
    """
    kind = check_kind(kind)
    return {
        "price": _bsm_price(S, K, T, r, sigma, kind, q),
        "delta": delta(S, K, T, r, sigma, kind, q),
        "gamma": gamma(S, K, T, r, sigma, kind, q),
        "vega": vega(S, K, T, r, sigma, kind, q),
        "theta": theta(S, K, T, r, sigma, kind, q),
        "rho": rho(S, K, T, r, sigma, kind, q),
        "vanna": vanna(S, K, T, r, sigma, q),
        "volga": volga(S, K, T, r, sigma, q),
    }
