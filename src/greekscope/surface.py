"""Implied-volatility surface construction and interpolation.

Quotes ``(K, T, price, kind)`` are inverted with :func:`greekscope.implied_vol.implied_vol`;
quotes that cannot be inverted (arbitrage violations, ``T <= 0``, ...) are stored as
``NaN`` and ignored by the interpolation.

Interpolation scheme for :meth:`VolSurface.iv`
----------------------------------------------
1. Within each expiry slice, vol is linear in strike between observed strikes and
   flat outside them.
2. Across expiries, *total variance* ``w = iv**2 * T`` is linear in ``T``
   between the two bracketing slices evaluated at the query strike.
3. Before the first / after the last expiry the *vol* is held flat.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from ._validate import check_inputs, check_kind
from .bsm import price as _bsm_price
from .implied_vol import implied_vol as _implied_vol

__all__ = ["VolSurface", "implied_vol_surface", "synthetic_quotes", "parametric_vol"]

_KEY_DIGITS = 10
_MIN_VOL = 0.01


class VolSurface:
    """Gridded implied-volatility surface with arbitrage-agnostic interpolation.

    Parameters
    ----------
    strikes : array_like, shape (n_strikes,)
        Strictly increasing strikes.
    expiries : array_like, shape (n_expiries,)
        Strictly increasing expiries in years (> 0).
    ivs : array_like, shape (n_expiries, n_strikes)
        Implied vols; ``NaN`` where no valid quote exists.  Row ``i`` is the
        smile for ``expiries[i]``.
    S, r, q : float, optional
        Market data the surface was built with (informational).

    Attributes
    ----------
    strikes, expiries : numpy.ndarray
    ivs : numpy.ndarray
        2-D, rows indexed by expiry and columns by strike.
    """

    def __init__(
        self,
        strikes: Any,
        expiries: Any,
        ivs: Any,
        S: float | None = None,
        r: float | None = None,
        q: float = 0.0,
    ) -> None:
        self.strikes = np.asarray(strikes, dtype=float).ravel()
        self.expiries = np.asarray(expiries, dtype=float).ravel()
        self.ivs = np.array(ivs, dtype=float)
        if self.ivs.shape != (len(self.expiries), len(self.strikes)):
            raise ValueError(
                f"ivs must have shape (n_expiries, n_strikes) = "
                f"({len(self.expiries)}, {len(self.strikes)}), got {self.ivs.shape}"
            )
        if len(self.strikes) == 0 or len(self.expiries) == 0:
            raise ValueError("surface needs at least one strike and one expiry")
        if np.any(~np.isfinite(self.strikes)) or np.any(self.strikes <= 0) or np.any(np.diff(self.strikes) <= 0):
            raise ValueError("strikes must be finite, positive and strictly increasing")
        if np.any(~np.isfinite(self.expiries)) or np.any(self.expiries <= 0) or np.any(np.diff(self.expiries) <= 0):
            raise ValueError("expiries must be finite, positive and strictly increasing")
        self.S, self.r, self.q = S, r, q

    # ------------------------------------------------------------------ helpers
    def _slice_iv(self, i: int, K: float) -> float:
        """Vol of expiry slice ``i`` at strike ``K`` (linear, flat extrapolation); NaN if empty."""
        row = self.ivs[i]
        ok = np.isfinite(row)
        if not ok.any():
            return math.nan
        return float(np.interp(K, self.strikes[ok], row[ok]))

    def _iv_scalar(self, K: float, T: float) -> float:
        if not (math.isfinite(K) and K > 0):
            raise ValueError(f"K must be finite and > 0, got {K!r}")
        if not (math.isfinite(T) and T > 0):
            raise ValueError(f"T must be finite and > 0, got {T!r}")
        live = [i for i in range(len(self.expiries)) if np.isfinite(self.ivs[i]).any()]
        if not live:
            raise ValueError("surface has no valid implied vols")
        ts = self.expiries[live]
        if T <= ts[0]:
            return self._slice_iv(live[0], K)
        if T >= ts[-1]:
            return self._slice_iv(live[-1], K)
        j = int(np.searchsorted(ts, T))  # ts[j-1] < T <= ts[j]
        i0, i1 = live[j - 1], live[j]
        t0, t1 = ts[j - 1], ts[j]
        v1 = self._slice_iv(i1, K)
        if T == t1:
            return v1
        v0 = self._slice_iv(i0, K)
        w0, w1 = v0 * v0 * t0, v1 * v1 * t1
        w = w0 + (w1 - w0) * (T - t0) / (t1 - t0)
        return math.sqrt(max(w, 0.0) / T)

    # --------------------------------------------------------------- public API
    def iv(self, K: Any, T: Any) -> Any:
        """Interpolated implied volatility at strike ``K`` and expiry ``T``.

        Parameters
        ----------
        K, T : float or array_like
            Query strike(s) and expiry(ies) in years, both > 0 (broadcast together).

        Returns
        -------
        float or numpy.ndarray
            Volatility; at an observed ``(K, T)`` node it equals the observed value.

        Raises
        ------
        ValueError
            If ``K`` or ``T`` is not positive, or the surface holds no valid vols.
        """
        Ka, Ta = np.broadcast_arrays(np.asarray(K, dtype=float), np.asarray(T, dtype=float))
        if Ka.ndim == 0:
            return self._iv_scalar(float(Ka), float(Ta))
        out = np.empty(Ka.shape)
        for idx in np.ndindex(Ka.shape):
            out[idx] = self._iv_scalar(float(Ka[idx]), float(Ta[idx]))
        return out

    def smile(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        """Volatility smile at expiry ``T``.

        For a listed expiry this returns the observed ``(strikes, ivs)`` (strikes
        with a missing quote are dropped).  For any other ``T`` it returns every
        surface strike with the interpolated vol (see module docs).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Strikes and implied vols.
        """
        T = float(T)
        hit = np.where(np.isclose(self.expiries, T, rtol=0.0, atol=1e-12))[0]
        if len(hit):
            row = self.ivs[hit[0]]
            ok = np.isfinite(row)
            return self.strikes[ok].copy(), row[ok].copy()
        return self.strikes.copy(), np.asarray(self.iv(self.strikes, T), dtype=float)

    def __repr__(self) -> str:
        n = int(np.isfinite(self.ivs).sum())
        return (
            f"VolSurface({len(self.expiries)} expiries x {len(self.strikes)} strikes, "
            f"{n} valid quotes)"
        )


def _key(x: float) -> float:
    return round(float(x), _KEY_DIGITS)


def implied_vol_surface(quotes: Iterable[tuple], S: float, r: float, q: float = 0.0) -> VolSurface:
    """Build a :class:`VolSurface` from option quotes.

    Parameters
    ----------
    quotes : iterable of (K, T, price, kind)
        Market option prices; ``kind`` is ``"call"`` or ``"put"``.  If several
        quotes share ``(K, T)`` (e.g. a call and a put) their implied vols are averaged.
    S : float
        Spot.
    r, q : float
        Risk-free rate and dividend yield (continuous).

    Returns
    -------
    VolSurface
        Quotes whose implied vol cannot be computed (price outside the
        no-arbitrage bounds, ``T <= 0``, ...) are left as ``NaN``.

    Raises
    ------
    ValueError
        On invalid ``S``/``r``/``q``, malformed quotes or an empty quote list.
    """
    check_inputs(S, 1.0, 0.0, 0.0, r=r, q=q)
    rows = []
    for item in quotes:
        try:
            K, T, price, kind = item
            K, T, price = float(K), float(T), float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"each quote must be a (K, T, price, kind) tuple, got {item!r}") from exc
        kind = check_kind(kind)
        if not (math.isfinite(K) and K > 0):
            raise ValueError(f"quote strike must be finite and > 0, got {K!r}")
        if not (math.isfinite(T) and T > 0):
            raise ValueError(f"quote expiry must be finite and > 0, got {T!r}")
        rows.append((K, T, price, kind))
    if not rows:
        raise ValueError("no quotes supplied")

    strikes = sorted({_key(k) for k, *_ in rows})
    expiries = sorted({_key(t) for _, t, *_ in rows})
    kidx = {k: i for i, k in enumerate(strikes)}
    tidx = {t: i for i, t in enumerate(expiries)}
    acc = np.zeros((len(expiries), len(strikes)))
    cnt = np.zeros_like(acc)
    for K, T, price, kind in rows:
        try:
            iv = _implied_vol(price, S, K, T, r, kind, q)
        except ValueError:
            continue
        if not math.isfinite(iv):
            continue
        i, j = tidx[_key(T)], kidx[_key(K)]
        acc[i, j] += iv
        cnt[i, j] += 1
    ivs = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1), np.nan)
    return VolSurface(strikes, expiries, ivs, S=float(S), r=float(r), q=float(q))


def parametric_vol(
    K: Any,
    T: Any,
    S: float,
    r: float,
    base_vol: float = 0.2,
    skew: float = -0.1,
    curvature: float = 0.05,
    q: float = 0.0,
) -> Any:
    """Parametric smile used by :func:`synthetic_quotes`.

    ``iv(K, T) = base_vol + skew * m + curvature * m**2`` with
    ``m = ln(K / F)`` and ``F = S exp((r - q) T)``, floored at 0.01.

    Parameters
    ----------
    K, T : float or array_like
        Strike(s) and expiry(ies).
    S, r, q : float
        Spot, rate and dividend yield (define the forward ``F``).
    base_vol, skew, curvature : float
        Smile coefficients.

    Returns
    -------
    float or numpy.ndarray
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    m = np.log(K / (S * np.exp((r - q) * T)))
    out = np.maximum(base_vol + skew * m + curvature * m * m, _MIN_VOL)
    return float(out) if out.ndim == 0 else out


def synthetic_quotes(
    S: float,
    r: float,
    strikes: Iterable[float],
    expiries: Iterable[float],
    base_vol: float = 0.2,
    skew: float = -0.1,
    curvature: float = 0.05,
    q: float = 0.0,
) -> list[tuple[float, float, float, str]]:
    """Generate BSM option quotes from a parametric smile (for demos and tests).

    The vol at each node is ``base_vol + skew*ln(K/F) + curvature*ln(K/F)**2``
    (``F = S e^{(r-q)T}``, floored at 0.01).  Out-of-the-money options are
    quoted: puts for ``K < F`` and calls for ``K >= F``.

    Parameters
    ----------
    S, r, q : float
        Spot, rate and dividend yield.
    strikes, expiries : iterable of float
        Strikes (> 0) and expiries in years (> 0); the full grid is generated.
    base_vol, skew, curvature : float
        Smile coefficients, see :func:`parametric_vol`.

    Returns
    -------
    list of (K, T, price, kind)
        Ordered by expiry, then strike.
    """
    check_inputs(S, 1.0, 0.0, 0.0, r=r, q=q)
    ks = [float(k) for k in strikes]
    ts = [float(t) for t in expiries]
    for k in ks:
        check_inputs(S, k, 0.0, 0.0)
    for t in ts:
        if not (math.isfinite(t) and t > 0):
            raise ValueError(f"expiries must be finite and > 0, got {t!r}")
    out = []
    for T in sorted(ts):
        F = S * math.exp((r - q) * T)
        for K in sorted(ks):
            sig = parametric_vol(K, T, S, r, base_vol, skew, curvature, q)
            kind = "put" if K < F else "call"
            out.append((K, T, float(_bsm_price(S, K, T, r, sig, kind, q)), kind))
    return out
