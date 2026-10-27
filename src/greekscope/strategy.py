"""Multi-leg option strategies: payoff, mark-to-model value, Greeks and risk summary.

A :class:`Strategy` is a list of :class:`Leg` objects (calls, puts and/or stock)
priced with Black-Scholes-Merton at a single volatility.  Factory functions
build the usual textbook structures (straddle, iron condor, ...).

Conventions
-----------
* Quantities are signed: ``+`` long, ``-`` short.  One unit is one unit of the
  underlying (no contract multiplier) unless ``multiplier`` is given.
* A leg's ``premium`` is the price paid *per unit* (always >= 0); the sign of
  ``qty`` decides whether it is a debit or a credit.  When ``premium`` is
  ``None`` it defaults to the BSM price at construction time (spot ``S``, full
  time to the leg's expiry).
* Stock legs: ``strike`` is the entry price and the P&L is ``qty * (S - strike)``;
  their delta is ``qty`` and all other Greeks are 0.
* All P&L figures are net of premiums.

Mixed expiries
--------------
A leg may carry its own ``expiry`` (in years from now).  The strategy horizon
``T`` is the date at which :meth:`Strategy.payoff` is evaluated.  Legs with
``expiry <= T`` are settled at intrinsic value; legs with ``expiry > T`` are
still alive and are valued with BSM at the remaining time ``expiry - T`` (at
the strategy's ``sigma`` and ``r``/``q``).  This is how calendar spreads work:
the near leg expires at ``T`` while the far leg keeps its time value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq

from ._validate import check_inputs, check_kind
from .bsm import price as _bsm_price
from . import greeks as _greeks

__all__ = [
    "Leg",
    "Strategy",
    "long_call",
    "long_put",
    "covered_call",
    "protective_put",
    "bull_call_spread",
    "bear_put_spread",
    "straddle",
    "strangle",
    "iron_condor",
    "butterfly",
    "calendar_spread",
]

_KINDS = ("call", "put", "stock")
_EPS_T = 1e-12
_TINY_S = 1e-12  # BSM needs S > 0; used to evaluate the S -> 0 limit


@dataclass
class Leg:
    """One position in a strategy.

    Parameters
    ----------
    kind : {"call", "put", "stock"}
        Instrument type (case-insensitive).
    strike : float
        Strike for options; entry price for stock.  Must be > 0.
    qty : float
        Signed quantity (+ long, - short).
    premium : float, optional
        Price paid per unit (>= 0).  ``None`` means "BSM price at construction"
        for options (resolved by :class:`Strategy`); stock legs must leave it ``None``.
    expiry : float, optional
        Years from now to this leg's expiry.  ``None`` means the strategy's ``T``.
    """

    kind: str
    strike: float
    qty: float = 1.0
    premium: float | None = None
    expiry: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind.strip().lower() not in _KINDS:
            raise ValueError(f"leg kind must be one of {_KINDS}, got {self.kind!r}")
        self.kind = self.kind.strip().lower()
        _finite(self.strike, "strike")
        if not self.strike > 0:
            raise ValueError(f"leg strike must be > 0, got {self.strike!r}")
        _finite(self.qty, "qty")
        if self.premium is not None:
            _finite(self.premium, "premium")
            if self.kind == "stock":
                raise ValueError("stock legs take their cost from 'strike'; premium must be None")
            if self.premium < 0:
                raise ValueError(f"premium must be >= 0 (use a negative qty for a credit), got {self.premium!r}")
        if self.expiry is not None:
            _finite(self.expiry, "expiry")
            if self.expiry < 0:
                raise ValueError(f"expiry must be >= 0, got {self.expiry!r}")


def _finite(x: Any, name: str) -> None:
    try:
        ok = math.isfinite(float(x))
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{name} must be a finite number, got {x!r}")


def _fmt_money(x: float) -> str:
    if x == math.inf:
        return "unlimited"
    if x == -math.inf:
        return "unlimited"
    return f"{x:,.4f}"


class Strategy:
    """A portfolio of option/stock legs on one underlying.

    Parameters
    ----------
    legs : iterable of Leg
        The positions.  Legs are copied; premiums that are ``None`` are
        replaced by the BSM price at ``S`` (see module docs).
    S : float
        Current spot (> 0).
    T : float
        Strategy horizon in years (>= 0): the date at which :meth:`payoff` is
        evaluated and the default expiry of legs without their own ``expiry``.
    r, sigma, q : float
        Rate, volatility and dividend yield used for all BSM valuations.
    multiplier : float, default 1.0
        Scales P&L, :meth:`value`, :meth:`payoff`, :meth:`greeks`,
        :meth:`max_profit` and :meth:`max_loss` (e.g. 100 for equity option
        contracts).  ``net_premium`` and ``breakevens`` stay per unit.
    name : str, optional
        Label used in :meth:`summary`.

    Raises
    ------
    ValueError
        On empty ``legs`` or invalid market inputs.
    """

    def __init__(
        self,
        legs: Iterable[Leg],
        S: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        multiplier: float = 1.0,
        name: str | None = None,
    ) -> None:
        check_inputs(S, 1.0, T, sigma, r=r, q=q)
        _finite(multiplier, "multiplier")
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier!r}")
        legs = list(legs)
        if not legs:
            raise ValueError("a strategy needs at least one leg")
        for lg in legs:
            if not isinstance(lg, Leg):
                raise ValueError(f"legs must be Leg instances, got {type(lg).__name__}")
        self.S = float(S)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.multiplier = float(multiplier)
        self.name = name
        self.legs: list[Leg] = []
        for lg in legs:
            lg = replace(lg)  # defensive copy; also re-runs validation
            if lg.kind != "stock" and lg.premium is None:
                lg.premium = float(
                    _bsm_price(self.S, lg.strike, self._expiry(lg), self.r, self.sigma, lg.kind, self.q)
                )
            self.legs.append(lg)

    # ------------------------------------------------------------------ helpers
    def _expiry(self, leg: Leg) -> float:
        return self.T if leg.expiry is None else float(leg.expiry)

    def _option_legs(self) -> list[Leg]:
        return [lg for lg in self.legs if lg.kind != "stock"]

    @property
    def net_premium(self) -> float:
        """Net premium paid per unit of underlying (positive = debit, negative = credit)."""
        return float(sum(lg.qty * lg.premium for lg in self._option_legs()))

    @staticmethod
    def _check_spot(S_arr: Any) -> np.ndarray:
        S = np.asarray(S_arr, dtype=float)
        if not np.all(np.isfinite(S)) or np.any(S < 0):
            raise ValueError("spot values must be finite and >= 0")
        return S

    def _value(self, S: np.ndarray, t_elapsed: float) -> np.ndarray:
        """Unscaled mark-to-model P&L at ``t_elapsed`` years from now."""
        total = np.zeros_like(S, dtype=float)
        for lg in self.legs:
            if lg.kind == "stock":
                total += lg.qty * (S - lg.strike)
                continue
            rem = self._expiry(lg) - t_elapsed
            if rem <= _EPS_T:
                intrinsic = np.maximum(S - lg.strike, 0.0) if lg.kind == "call" else np.maximum(lg.strike - S, 0.0)
                mark = intrinsic
            else:
                mark = _bsm_price(np.maximum(S, _TINY_S), lg.strike, rem, self.r, self.sigma, lg.kind, self.q)
            total += lg.qty * (mark - lg.premium)
        return total

    @staticmethod
    def _out(x: np.ndarray) -> Any:
        return float(x) if np.ndim(x) == 0 else x

    # --------------------------------------------------------------- public API
    def value(self, S_arr: Any, t_elapsed: float = 0.0) -> Any:
        """Mark-to-model P&L after ``t_elapsed`` years.

        Every option leg is valued with BSM at its remaining time
        ``expiry - t_elapsed`` (intrinsic once expired); premiums paid are
        subtracted, so the result is a P&L (zero-cost entry at the premiums).

        Parameters
        ----------
        S_arr : float or array_like
            Underlying price(s), >= 0.
        t_elapsed : float, default 0.0
            Years elapsed since entry (>= 0).

        Returns
        -------
        float or numpy.ndarray
        """
        _finite(t_elapsed, "t_elapsed")
        if t_elapsed < 0:
            raise ValueError(f"t_elapsed must be >= 0, got {t_elapsed!r}")
        S = self._check_spot(S_arr)
        return self._out(self.multiplier * self._value(S, float(t_elapsed)))

    def payoff(self, ST: Any) -> Any:
        """P&L at the strategy horizon ``T`` (net of premiums).

        Equivalent to ``value(ST, t_elapsed=T)``.  Legs expiring at (or before)
        ``T`` pay intrinsic value; longer-dated legs are valued with BSM at
        their remaining time (see module docs).
        """
        S = self._check_spot(ST)
        return self._out(self.multiplier * self._value(S, self.T))

    def greeks(self, S: Any = None, t_elapsed: float = 0.0) -> dict:
        """Aggregate Greeks (qty- and multiplier-weighted sums over the legs).

        Parameters
        ----------
        S : float or array_like, optional
            Spot at which to evaluate; defaults to the construction spot.  Must be > 0.
        t_elapsed : float, default 0.0
            Years elapsed since entry.

        Returns
        -------
        dict
            Keys ``delta, gamma, vega, theta, rho`` (floats for scalar ``S``,
            arrays otherwise), with the core conventions: vega/rho per 1.00
            change, theta per year.  Stock contributes only to delta.
        """
        _finite(t_elapsed, "t_elapsed")
        if t_elapsed < 0:
            raise ValueError(f"t_elapsed must be >= 0, got {t_elapsed!r}")
        spot = self.S if S is None else S
        shape = np.shape(spot)
        out = {k: np.zeros(shape) for k in ("delta", "gamma", "vega", "theta", "rho")}
        check_inputs(spot, 1.0, 0.0, 0.0)  # spot must be finite and > 0
        for lg in self.legs:
            if lg.kind == "stock":
                out["delta"] = out["delta"] + lg.qty
                continue
            rem = max(self._expiry(lg) - t_elapsed, 0.0)
            args = (spot, lg.strike, rem, self.r, self.sigma, lg.kind, self.q)
            for k in out:
                out[k] = out[k] + lg.qty * getattr(_greeks, k)(*args)
        return {k: self._out(self.multiplier * v) for k, v in out.items()}

    # ------------------------------------------------------------ risk summary
    def _grid_upper(self) -> float:
        ref = [self.S] + [lg.strike for lg in self.legs]
        return 3.0 * max(ref)

    def breakevens(self) -> list[float]:
        """Spot levels at the horizon where the P&L is zero, sorted ascending.

        The payoff is sampled on a dense grid from 0 to ``3 * max(strikes, S)``
        (grid points include every strike); sign changes are refined with
        Brent's method.  Roots closer together than the grid spacing may be
        missed, and a payoff that is identically zero on an interval yields no
        breakeven there.

        Returns
        -------
        list of float
        """
        hi = self._grid_upper()
        strikes = [lg.strike for lg in self.legs]
        xs = np.unique(np.concatenate([np.linspace(0.0, hi, 6001), strikes]))
        fs = self._value(xs, self.T)

        def f(x: float) -> float:
            return float(self._value(np.asarray(x), self.T))

        roots: list[float] = []
        n = len(xs)
        scale = max(1.0, float(np.max(np.abs(fs))))
        zero_tol = 1e-12 * scale
        for i in range(n - 1):
            a, b = fs[i], fs[i + 1]
            if abs(a) <= zero_tol:
                # isolated exact zero (not part of a flat zero stretch)
                left_zero = i > 0 and abs(fs[i - 1]) <= zero_tol
                right_zero = abs(b) <= zero_tol
                if not left_zero and not right_zero:
                    roots.append(float(xs[i]))
                continue
            if abs(b) <= zero_tol:
                continue  # handled when i advances
            if (a < 0) != (b < 0):
                roots.append(float(brentq(f, xs[i], xs[i + 1], xtol=1e-12, rtol=1e-14)))
        if abs(fs[-1]) <= zero_tol and n > 1 and abs(fs[-2]) > zero_tol:
            roots.append(float(xs[-1]))
        roots.sort()
        dedup: list[float] = []
        for x in roots:
            if not dedup or abs(x - dedup[-1]) > 1e-9 * (1.0 + abs(x)):
                dedup.append(x)
        return dedup

    def _is_piecewise_linear(self) -> bool:
        """True when every option leg is settled (intrinsic) at the horizon."""
        return all(self._expiry(lg) <= self.T + _EPS_T for lg in self._option_legs())

    def _extremes(self) -> tuple[float, float]:
        """Unscaled (worst, best) P&L over spot in [0, inf)."""
        if self._is_piecewise_linear():
            kinks = [lg.strike for lg in self._option_legs()]
            pts = np.array([0.0] + kinks)
            vals = self._value(pts, self.T)
            slope = sum(lg.qty for lg in self.legs if lg.kind in ("call", "stock"))
            tol = 1e-12 * max(1.0, sum(abs(lg.qty) for lg in self.legs))
            best = math.inf if slope > tol else float(np.max(vals))
            worst = -math.inf if slope < -tol else float(np.min(vals))
            return worst, best
        # Mixed expiries: not piecewise linear.  Approximate on a grid and detect
        # unbounded tails from the numerical slope far out of the money.
        hi = 10.0 * self._grid_upper() / 3.0
        strikes = [lg.strike for lg in self.legs]
        xs = np.unique(np.concatenate([np.linspace(0.0, hi, 20001), strikes]))
        vals = self._value(xs, self.T)
        far = np.array([100.0 * hi, 200.0 * hi])
        fv = self._value(far, self.T)
        slope = (fv[1] - fv[0]) / (far[1] - far[0])
        tol = 1e-8
        best = math.inf if slope > tol else float(np.max(vals))
        worst = -math.inf if slope < -tol else float(np.min(vals))
        return worst, best

    def max_profit(self) -> float:
        """Best achievable P&L at the horizon over all spot levels ``>= 0``.

        ``inf`` when the payoff grows without bound as spot rises (net long
        calls/stock).  Exact (check kinks at 0 and the strikes) when every
        option leg expires by the horizon; otherwise a grid approximation over
        ``[0, 10 * max(strikes, S)]``.
        """
        return self.multiplier * self._extremes()[1]

    def max_loss(self) -> float:
        """Worst achievable P&L at the horizon (a negative number for a real loss).

        ``-inf`` when losses are unbounded (net short calls/stock as spot rises).
        If the strategy can never lose money the value is the (non-negative)
        minimum P&L.  Exactness as for :meth:`max_profit`.
        """
        return self.multiplier * self._extremes()[0]

    # --------------------------------------------------------------- reporting
    def _header(self) -> str:
        label = self.name or "Strategy"
        return (
            f"{label}  (S={self.S:g}, T={self.T:g}, r={self.r:.2%}, "
            f"sigma={self.sigma:.2%}, q={self.q:.2%}"
            + (f", x{self.multiplier:g}" if self.multiplier != 1.0 else "")
            + ")"
        )

    def _legs_table(self) -> str:
        rows = [("Leg", "Side", "Type", "Strike", "Qty", "Expiry", "Premium")]
        for i, lg in enumerate(self.legs, 1):
            side = "long" if lg.qty > 0 else "short" if lg.qty < 0 else "flat"
            if lg.kind == "stock":
                rows.append((str(i), side, "stock", f"{lg.strike:.4f}", f"{lg.qty:+g}", "-", "-"))
            else:
                rows.append(
                    (str(i), side, lg.kind, f"{lg.strike:.4f}", f"{lg.qty:+g}", f"{self._expiry(lg):.4f}", f"{lg.premium:.4f}")
                )
        widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        lines = []
        for k, r in enumerate(rows):
            cells = [r[c].ljust(widths[c]) if c in (1, 2) else r[c].rjust(widths[c]) for c in range(len(r))]
            lines.append("  ".join(cells))
            if k == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self._header()}\n{self._legs_table()}"

    def summary(self) -> str:
        """Readable text report: legs, net premium, breakevens, max profit/loss, Greeks."""
        npm = self.net_premium
        kind = "debit" if npm > 0 else "credit" if npm < 0 else "even"
        be = self.breakevens()
        mp, ml = self.max_profit(), self.max_loss()
        approx = "" if self._is_piecewise_linear() else "  (grid approximation: mixed expiries)"
        g = self.greeks()
        lines = [
            self._header(),
            "",
            self._legs_table(),
            "",
            f"Net premium     {npm:>14,.4f}  ({kind}, per unit)",
            "Breakevens      " + (", ".join(f"{x:,.4f}" for x in be) if be else "none"),
            f"Max profit      {_fmt_money(mp):>14}{approx}",
            f"Max loss        {_fmt_money(ml):>14}{approx}",
            "",
            "Greeks (sum over legs)",
        ]
        for k in ("delta", "gamma", "vega", "theta", "rho"):
            lines.append(f"  {k:<8}{g[k]:>14.6f}")
        return "\n".join(lines)

    __str__ = summary


# ====================================================================== factories
def _need_increasing(**strikes: float) -> None:
    vals = list(strikes.values())
    for v, n in zip(vals, strikes):
        _finite(v, n)
        if not v > 0:
            raise ValueError(f"{n} must be > 0, got {v!r}")
    for (n1, a), (n2, b) in zip(list(strikes.items())[:-1], list(strikes.items())[1:]):
        if not a < b:
            raise ValueError(f"strikes must satisfy {' < '.join(strikes)}; got {n1}={a:g}, {n2}={b:g}")


def long_call(S: float, T: float, r: float, sigma: float, K: float, q: float = 0.0) -> Strategy:
    """Long one call struck at ``K``."""
    return Strategy([Leg("call", K, 1.0)], S, T, r, sigma, q, name="Long call")


def long_put(S: float, T: float, r: float, sigma: float, K: float, q: float = 0.0) -> Strategy:
    """Long one put struck at ``K``."""
    return Strategy([Leg("put", K, 1.0)], S, T, r, sigma, q, name="Long put")


def covered_call(S: float, T: float, r: float, sigma: float, K: float, q: float = 0.0) -> Strategy:
    """Long one share (entry at ``S``) and short one call struck at ``K``."""
    legs = [Leg("stock", S, 1.0), Leg("call", K, -1.0)]
    return Strategy(legs, S, T, r, sigma, q, name="Covered call")


def protective_put(S: float, T: float, r: float, sigma: float, K: float, q: float = 0.0) -> Strategy:
    """Long one share (entry at ``S``) and long one put struck at ``K``."""
    legs = [Leg("stock", S, 1.0), Leg("put", K, 1.0)]
    return Strategy(legs, S, T, r, sigma, q, name="Protective put")


def bull_call_spread(S: float, T: float, r: float, sigma: float, K_low: float, K_high: float, q: float = 0.0) -> Strategy:
    """Long call at ``K_low``, short call at ``K_high`` (``K_low < K_high``)."""
    _need_increasing(K_low=K_low, K_high=K_high)
    legs = [Leg("call", K_low, 1.0), Leg("call", K_high, -1.0)]
    return Strategy(legs, S, T, r, sigma, q, name="Bull call spread")


def bear_put_spread(S: float, T: float, r: float, sigma: float, K_low: float, K_high: float, q: float = 0.0) -> Strategy:
    """Long put at ``K_high``, short put at ``K_low`` (``K_low < K_high``)."""
    _need_increasing(K_low=K_low, K_high=K_high)
    legs = [Leg("put", K_high, 1.0), Leg("put", K_low, -1.0)]
    return Strategy(legs, S, T, r, sigma, q, name="Bear put spread")


def straddle(S: float, T: float, r: float, sigma: float, K: float | None = None, q: float = 0.0) -> Strategy:
    """Long call and long put at the same strike ``K`` (default: at the money, ``K = S``)."""
    K = S if K is None else K
    legs = [Leg("call", K, 1.0), Leg("put", K, 1.0)]
    return Strategy(legs, S, T, r, sigma, q, name="Long straddle")


def strangle(
    S: float,
    T: float,
    r: float,
    sigma: float,
    K_put: float | None = None,
    K_call: float | None = None,
    q: float = 0.0,
) -> Strategy:
    """Long put at ``K_put`` and long call at ``K_call`` (``K_put < K_call``).

    Defaults are 10% out of the money on each side (``0.9 S`` and ``1.1 S``).
    """
    K_put = 0.9 * S if K_put is None else K_put
    K_call = 1.1 * S if K_call is None else K_call
    _need_increasing(K_put=K_put, K_call=K_call)
    legs = [Leg("put", K_put, 1.0), Leg("call", K_call, 1.0)]
    return Strategy(legs, S, T, r, sigma, q, name="Long strangle")


def iron_condor(
    S: float,
    T: float,
    r: float,
    sigma: float,
    K_put_low: float,
    K_put_high: float,
    K_call_low: float,
    K_call_high: float,
    q: float = 0.0,
) -> Strategy:
    """Short iron condor: long put ``K_put_low``, short put ``K_put_high``,
    short call ``K_call_low``, long call ``K_call_high``.

    Requires ``K_put_low < K_put_high <= K_call_low < K_call_high``
    (equality of the middle strikes gives an iron butterfly).
    """
    for n, v in (("K_put_low", K_put_low), ("K_put_high", K_put_high), ("K_call_low", K_call_low), ("K_call_high", K_call_high)):
        _finite(v, n)
        if not v > 0:
            raise ValueError(f"{n} must be > 0, got {v!r}")
    if not (K_put_low < K_put_high <= K_call_low < K_call_high):
        raise ValueError(
            "strikes must satisfy K_put_low < K_put_high <= K_call_low < K_call_high; got "
            f"{K_put_low:g}, {K_put_high:g}, {K_call_low:g}, {K_call_high:g}"
        )
    legs = [
        Leg("put", K_put_low, 1.0),
        Leg("put", K_put_high, -1.0),
        Leg("call", K_call_low, -1.0),
        Leg("call", K_call_high, 1.0),
    ]
    return Strategy(legs, S, T, r, sigma, q, name="Iron condor")


def butterfly(
    S: float,
    T: float,
    r: float,
    sigma: float,
    K_low: float,
    K_mid: float,
    K_high: float,
    kind: str = "call",
    q: float = 0.0,
) -> Strategy:
    """Long butterfly: +1 at ``K_low``, -2 at ``K_mid``, +1 at ``K_high`` (calls or puts)."""
    kind = check_kind(kind)
    _need_increasing(K_low=K_low, K_mid=K_mid, K_high=K_high)
    legs = [Leg(kind, K_low, 1.0), Leg(kind, K_mid, -2.0), Leg(kind, K_high, 1.0)]
    return Strategy(legs, S, T, r, sigma, q, name=f"Long {kind} butterfly")


def calendar_spread(
    S: float,
    T: float,
    r: float,
    sigma: float,
    K: float,
    T_far: float,
    kind: str = "call",
    q: float = 0.0,
) -> Strategy:
    """Short the near option (expiry ``T``), long the far option (expiry ``T_far > T``), same strike.

    The strategy horizon is the near expiry, so :meth:`Strategy.payoff` values the
    far leg with BSM at the remaining time ``T_far - T``.
    """
    kind = check_kind(kind)
    _finite(T_far, "T_far")
    if not T_far > T:
        raise ValueError(f"T_far must be > T; got T={T:g}, T_far={T_far:g}")
    legs = [Leg(kind, K, -1.0), Leg(kind, K, 1.0, expiry=T_far)]
    return Strategy(legs, S, T, r, sigma, q, name=f"Long {kind} calendar spread")
