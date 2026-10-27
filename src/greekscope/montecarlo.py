"""Monte Carlo pricing: European (exact GBM) and arithmetic-average Asian options."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from ._validate import check_inputs, check_kind

__all__ = ["MCResult", "mc_price", "asian_mc_price"]


@dataclass
class MCResult:
    """Monte Carlo estimate.

    Attributes
    ----------
    price : float
        Estimated price.
    std_error : float
        Standard error of ``price`` (accounts for antithetic pairing and the
        control variate).
    n_paths : int
        Number of simulated paths actually used (twice the number of pairs when
        antithetic variates are on).
    """

    price: float
    std_error: float
    n_paths: int


def _combine(y: np.ndarray, x: np.ndarray | None, ex: float | None) -> tuple[float, float]:
    """Mean and standard error of ``y`` optionally adjusted by control ``x`` (E[x]=ex)."""
    m = y.size
    if x is not None:
        vx = float(np.var(x, ddof=1)) if m > 1 else 0.0
        if vx > 1e-300:
            beta = float(np.cov(y, x, ddof=1)[0, 1]) / vx
            y = y - beta * (x - ex)
    est = float(np.mean(y))
    se = float(np.std(y, ddof=1) / math.sqrt(m)) if m > 1 else 0.0
    return est, se


def _check_sim(n_paths: int, antithetic: bool) -> int:
    if int(n_paths) != n_paths or n_paths < 2:
        raise ValueError(f"n_paths must be an integer >= 2, got {n_paths!r}")
    return int(n_paths) // 2 if antithetic else int(n_paths)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    q: float = 0.0,
    n_paths: int = 200_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Monte Carlo price of a European option using exact GBM terminal sampling.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        See :func:`greekscope.bsm.price`.
    kind : {"call", "put"}
    n_paths : int, default 200_000
        Total number of paths.  With ``antithetic=True`` this is
        ``2 * (n_paths // 2)`` (an odd count is rounded down).
    antithetic : bool, default True
        Pair each normal draw ``Z`` with ``-Z``; the estimator and standard
        error are computed over the pair averages.
    control_variate : bool, default True
        Use the discounted terminal stock ``e^{-rT} S_T`` (known mean
        ``S e^{-qT}``) as a control, with the optimal coefficient estimated
        from the same sample (applied to the pair averages when antithetic).
    seed : int or None
        Seed for :func:`numpy.random.default_rng`.

    Returns
    -------
    MCResult
        ``price``, ``std_error`` and ``n_paths``.
    """
    kind = check_kind(kind)
    check_inputs(S, K, T, sigma, r=r, q=q)
    m = _check_sim(n_paths, antithetic)
    rng = np.random.default_rng(seed)
    sgn = 1.0 if kind == "call" else -1.0
    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    z = rng.standard_normal(m)

    def sample(zz):
        st = S * np.exp(drift + vol * zz)
        return disc * np.maximum(sgn * (st - K), 0.0), disc * st

    y, x = sample(z)
    if antithetic:
        y2, x2 = sample(-z)
        y, x = 0.5 * (y + y2), 0.5 * (x + x2)
        used = 2 * m
    else:
        used = m
    ex = S * math.exp(-q * T)
    est, se = _combine(y, x if control_variate else None, ex)
    return MCResult(price=est, std_error=se, n_paths=used)


def _geometric_asian_price(
    S: float, K: float, T: float, r: float, sigma: float, kind: str, q: float, n: int
) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian option."""
    sgn = 1.0 if kind == "call" else -1.0
    disc = math.exp(-r * T)
    mu = math.log(S) + (r - q - 0.5 * sigma * sigma) * T * (n + 1) / (2.0 * n)
    var = sigma * sigma * T * (n + 1) * (2 * n + 1) / (6.0 * n * n)
    if var <= 0.0:
        return disc * max(sgn * (math.exp(mu) - K), 0.0)
    s = math.sqrt(var)
    d1 = (mu + var - math.log(K)) / s
    d2 = d1 - s
    fwd = math.exp(mu + 0.5 * var)
    return disc * sgn * (fwd * float(ndtr(sgn * d1)) - K * float(ndtr(sgn * d2)))


def asian_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    q: float = 0.0,
    n_steps: int = 252,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Monte Carlo price of an arithmetic-average Asian option.

    The average is taken over ``n_steps`` equally spaced monitoring dates
    ``T/n, 2T/n, ..., T``.  Paths are simulated exactly on that grid.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        See :func:`greekscope.bsm.price`.
    kind : {"call", "put"}
    n_steps : int, default 252
        Number of monitoring dates.
    n_paths : int, default 100_000
        Total paths (``2 * (n_paths // 2)`` with antithetic variates).
    seed : int or None
        RNG seed.
    antithetic : bool, default True
        Antithetic pairs; statistics are computed over pair averages.
    control_variate : bool, default True
        Use the discretely monitored geometric-average Asian option, whose price
        is known in closed form, as a control variate.

    Returns
    -------
    MCResult
    """
    kind = check_kind(kind)
    check_inputs(S, K, T, sigma, r=r, q=q)
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    n = int(n_steps)
    m = _check_sim(n_paths, antithetic)
    rng = np.random.default_rng(seed)
    sgn = 1.0 if kind == "call" else -1.0
    dt = T / n
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)
    ln_s = math.log(S)

    chunk = max(1, min(m, 2_000_000 // n))
    ys, xs = [], []
    done = 0
    while done < m:
        c = min(chunk, m - done)
        z = rng.standard_normal((c, n))
        parts_y, parts_x = [], []
        for zz in (z, -z) if antithetic else (z,):
            logs = ln_s + np.cumsum(drift + vol * zz, axis=1)
            arith = np.exp(logs).mean(axis=1)
            geo = np.exp(logs.mean(axis=1))
            parts_y.append(disc * np.maximum(sgn * (arith - K), 0.0))
            parts_x.append(disc * np.maximum(sgn * (geo - K), 0.0))
        ys.append(np.mean(parts_y, axis=0))
        xs.append(np.mean(parts_x, axis=0))
        done += c
    y = np.concatenate(ys)
    x = np.concatenate(xs)
    ex = _geometric_asian_price(S, K, T, r, sigma, kind, q, n)
    est, se = _combine(y, x if control_variate else None, ex)
    return MCResult(price=est, std_error=se, n_paths=2 * m if antithetic else m)
