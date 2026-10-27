"""Cox-Ross-Rubinstein binomial tree (European and American)."""

from __future__ import annotations

import math

import numpy as np

from ._validate import check_inputs, check_kind
from .bsm import price as _bsm_price

__all__ = ["crr_price"]


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    q: float = 0.0,
    steps: int = 500,
    american: bool = False,
) -> float:
    """Cox-Ross-Rubinstein binomial price.

    Backward induction is vectorised over the nodes of each time slice with
    numpy (only the loop over time steps is in Python).

    Parameters
    ----------
    S, K : float
        Spot and strike (> 0).
    T : float
        Years to expiry (>= 0).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (>= 0).
    kind : {"call", "put"}
    q : float, default 0.0
        Continuous dividend yield.
    steps : int, default 500
        Number of time steps (>= 1).
    american : bool, default False
        Allow early exercise at every node.

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        On invalid inputs, or if ``steps`` is so small that the risk-neutral
        probability leaves [0, 1] (increase ``steps``).

    Notes
    -----
    ``T == 0`` returns intrinsic value.  ``sigma == 0`` returns the
    deterministic limit (European: closed form; American: best discounted
    intrinsic value along the deterministic forward path on the step grid).
    """
    kind = check_kind(kind)
    check_inputs(S, K, T, sigma, r=r, q=q)
    if int(steps) != steps or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    n = int(steps)
    sgn = 1.0 if kind == "call" else -1.0

    if T == 0.0:
        return float(max(sgn * (S - K), 0.0))

    dt = T / n
    if sigma == 0.0:
        if not american:
            return float(_bsm_price(S, K, T, r, 0.0, kind, q))
        t = np.arange(n + 1) * dt
        path = S * np.exp((r - q) * t)
        return float(np.max(np.exp(-r * t) * np.maximum(sgn * (path - K), 0.0)))

    vol_dt = sigma * math.sqrt(dt)
    u = math.exp(vol_dt)
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"risk-neutral probability {p:.4g} outside [0, 1]; increase steps"
        )
    disc = math.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)

    j = np.arange(n + 1, dtype=float)
    V = np.maximum(sgn * (S * np.exp((2.0 * j - n) * vol_dt) - K), 0.0)
    for i in range(n - 1, -1, -1):
        V = pu * V[1:] + pd * V[:-1]
        if american:
            nodes = S * np.exp((2.0 * j[: i + 1] - i) * vol_dt)
            V = np.maximum(V, sgn * (nodes - K))
    return float(V[0])
