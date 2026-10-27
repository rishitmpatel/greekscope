"""Command-line interface: ``greekscope price | iv | strategy``.

Examples
--------
::

    greekscope price --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2
    greekscope price --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method american
    greekscope iv --price 10.45 --S 100 --K 100 --T 1 --r 0.05
    greekscope strategy iron_condor --S 100 --T 0.5 --r 0.05 --sigma 0.2 \\
        --K1 80 --K2 90 --K3 110 --K4 120

Invalid input prints ``greekscope: error: ...`` to stderr and exits with status 2.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence

from . import strategy as _st
from .binomial import crr_price
from .bsm import price as bsm_price
from .greeks import all_greeks
from .heston import heston_price
from .implied_vol import implied_vol
from .montecarlo import mc_price

__all__ = ["main", "build_parser"]

_METHODS = ("bsm", "crr", "american", "mc", "heston")
_HESTON_ARGS = ("v0", "kappa", "theta", "xi", "rho")

_STRATEGIES = (
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
)

_STRATEGY_HELP = """\
strike arguments by strategy:
  long_call, long_put, covered_call, protective_put   --K
  straddle                                            --K (default: S)
  bull_call_spread, bear_put_spread                   --K1 (low) --K2 (high)
  strangle                                            --K1 (put) --K2 (call)   (default 0.9 S / 1.1 S)
  butterfly                                           --K1 --K2 --K3 [--kind call|put]
  iron_condor                                         --K1 --K2 --K3 --K4  (put_low, put_high, call_low, call_high)
  calendar_spread                                     --K --T-far [--kind call|put]
"""


class _CLIError(ValueError):
    """Bad command-line input (reported with exit status 2)."""


def _table(rows: Sequence[tuple[str, str]], title: str | None = None) -> str:
    w = max(len(k) for k, _ in rows)
    vw = max(len(v) for _, v in rows)
    lines = [title, "-" * (w + 2 + vw)] if title else []
    lines += [f"{k:<{w}}  {v:>{vw}}" for k, v in rows]
    return "\n".join(lines)


def _add_market(p: argparse.ArgumentParser, sigma: bool = True) -> None:
    p.add_argument("--S", type=float, required=True, help="spot price")
    p.add_argument("--T", type=float, required=True, help="years to expiry")
    p.add_argument("--r", type=float, required=True, help="continuous risk-free rate (e.g. 0.05)")
    if sigma:
        p.add_argument("--sigma", type=float, required=True, help="volatility (e.g. 0.2)")
    p.add_argument("--q", type=float, default=0.0, help="continuous dividend yield (default 0)")


def build_parser() -> argparse.ArgumentParser:
    """Create the ``argparse`` parser used by :func:`main`."""
    ap = argparse.ArgumentParser(
        prog="greekscope",
        description="Option pricing, Greeks, implied volatility and strategy analysis.",
    )
    sub = ap.add_subparsers(dest="command", required=True, metavar="{price,iv,strategy}")

    p = sub.add_parser("price", help="price an option (and show Greeks for BSM)")
    _add_market(p, sigma=False)
    p.add_argument("--sigma", type=float, default=None, help="volatility (required except for --method heston)")
    p.add_argument("--K", type=float, required=True, help="strike")
    p.add_argument("--kind", default="call", help="call or put (default call)")
    p.add_argument("--method", choices=_METHODS, default="bsm", help="pricing method (default bsm)")
    p.add_argument("--steps", type=int, default=500, help="CRR tree steps (default 500)")
    p.add_argument("--paths", type=int, default=200_000, help="Monte Carlo paths (default 200000)")
    p.add_argument("--seed", type=int, default=None, help="Monte Carlo seed")
    for name in _HESTON_ARGS:
        p.add_argument(f"--{name}", type=float, default=None, help="Heston parameter (method heston)")
    p.set_defaults(func=_cmd_price)

    p = sub.add_parser("iv", help="implied volatility from an option price")
    p.add_argument("--price", type=float, required=True, help="observed option price")
    p.add_argument("--S", type=float, required=True, help="spot price")
    p.add_argument("--K", type=float, required=True, help="strike")
    p.add_argument("--T", type=float, required=True, help="years to expiry")
    p.add_argument("--r", type=float, required=True, help="continuous risk-free rate")
    p.add_argument("--q", type=float, default=0.0, help="continuous dividend yield (default 0)")
    p.add_argument("--kind", default="call", help="call or put (default call)")
    p.set_defaults(func=_cmd_iv)

    p = sub.add_parser(
        "strategy",
        help="analyse a multi-leg strategy",
        description="Breakevens, max profit/loss and Greeks of a standard option strategy.",
        epilog=_STRATEGY_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", help="strategy: " + ", ".join(_STRATEGIES))
    _add_market(p)
    p.add_argument("--K", type=float, default=None, help="single strike")
    for i in range(1, 5):
        p.add_argument(f"--K{i}", type=float, default=None, help=f"strike {i} (ascending)")
    p.add_argument("--T-far", dest="T_far", type=float, default=None, help="far expiry (calendar_spread)")
    p.add_argument("--kind", default="call", help="call or put (butterfly, calendar_spread; default call)")
    p.add_argument("--multiplier", type=float, default=1.0, help="contract multiplier (default 1)")
    p.set_defaults(func=_cmd_strategy)
    return ap


# --------------------------------------------------------------------- commands
def _cmd_price(a: argparse.Namespace) -> str:
    if a.method == "heston":
        missing = [f"--{n}" for n in _HESTON_ARGS if getattr(a, n) is None]
        if missing:
            raise _CLIError("method heston requires " + ", ".join(missing))
        px = heston_price(a.S, a.K, a.T, a.r, a.v0, a.kappa, a.theta, a.xi, a.rho, a.kind, a.q)
        return _table([("Method", "heston"), ("Price", f"{px:.6f}")], f"{a.kind.lower()} option")
    if a.sigma is None:
        raise _CLIError(f"--sigma is required for method {a.method}")
    head = [("Spot", f"{a.S:g}"), ("Strike", f"{a.K:g}"), ("T (years)", f"{a.T:g}"),
            ("Rate", f"{a.r:g}"), ("Div yield", f"{a.q:g}"), ("Sigma", f"{a.sigma:g}")]
    ref = float(bsm_price(a.S, a.K, a.T, a.r, a.sigma, a.kind, a.q))
    title = f"{a.kind.lower()} option, method {a.method}"
    if a.method == "bsm":
        g = all_greeks(a.S, a.K, a.T, a.r, a.sigma, a.kind, a.q)
        rows = head + [(k.capitalize(), f"{g[k]:.6f}") for k in ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga")]
        out = _table(rows, title)
        return out + "\n\n(vega and rho per 1.00 change; theta per year)"
    if a.method in ("crr", "american"):
        px = crr_price(a.S, a.K, a.T, a.r, a.sigma, a.kind, a.q, steps=a.steps, american=(a.method == "american"))
        rows = head + [("Steps", str(a.steps)), ("Price", f"{px:.6f}"), ("BSM (European)", f"{ref:.6f}"),
                       ("Difference", f"{px - ref:+.6f}")]
        return _table(rows, title)
    res = mc_price(a.S, a.K, a.T, a.r, a.sigma, a.kind, a.q, n_paths=a.paths, seed=a.seed)
    rows = head + [("Paths", str(res.n_paths)), ("Price", f"{res.price:.6f} +/- {res.std_error:.6f}"),
                   ("BSM (European)", f"{ref:.6f}")]
    return _table(rows, title)


def _cmd_iv(a: argparse.Namespace) -> str:
    sig = implied_vol(a.price, a.S, a.K, a.T, a.r, a.kind, a.q)
    return _table([("Price", f"{a.price:g}"), ("Spot", f"{a.S:g}"), ("Strike", f"{a.K:g}"),
                   ("T (years)", f"{a.T:g}"), ("Implied vol", f"{sig:.6f}  ({sig:.4%})")],
                  f"{a.kind.lower()} option")


def _need(a: argparse.Namespace, *names: str) -> list[float]:
    vals = [getattr(a, n) for n in names]
    if any(v is None for v in vals):
        flags = " ".join("--" + n.replace("_", "-") for n in names)
        raise _CLIError(f"strategy {a.name} requires {flags}")
    return vals


def _cmd_strategy(a: argparse.Namespace) -> str:
    name = a.name.strip().lower().replace("-", "_")
    if name not in _STRATEGIES:
        raise _CLIError(f"unknown strategy {a.name!r}; choose from: {', '.join(_STRATEGIES)}")
    m = (a.S, a.T, a.r, a.sigma)
    q = a.q
    if name in ("long_call", "long_put", "covered_call", "protective_put"):
        (K,) = _need(a, "K")
        s = getattr(_st, name)(*m, K, q=q)
    elif name == "straddle":
        s = _st.straddle(*m, K=a.K, q=q)
    elif name == "strangle":
        s = _st.strangle(*m, K_put=a.K1, K_call=a.K2, q=q)
    elif name in ("bull_call_spread", "bear_put_spread"):
        k1, k2 = _need(a, "K1", "K2")
        s = getattr(_st, name)(*m, k1, k2, q=q)
    elif name == "butterfly":
        k1, k2, k3 = _need(a, "K1", "K2", "K3")
        s = _st.butterfly(*m, k1, k2, k3, kind=a.kind, q=q)
    elif name == "iron_condor":
        k1, k2, k3, k4 = _need(a, "K1", "K2", "K3", "K4")
        s = _st.iron_condor(*m, k1, k2, k3, k4, q=q)
    else:  # calendar_spread
        K, T_far = _need(a, "K", "T_far")
        s = _st.calendar_spread(*m, K, T_far, kind=a.kind, q=q)
    if a.multiplier != 1.0:
        s = _st.Strategy(s.legs, s.S, s.T, s.r, s.sigma, s.q, multiplier=a.multiplier, name=s.name)
    return s.summary()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments (default ``sys.argv[1:]``).

    Returns
    -------
    int
        0 on success, 2 on invalid input (argparse errors raise ``SystemExit(2)``).
    """
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], str] = args.func
    try:
        out = func(args)
    except ValueError as exc:  # includes _CLIError and all library validation errors
        print(f"greekscope: error: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
