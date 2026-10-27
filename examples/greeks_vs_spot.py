"""Greeks as functions of spot, for several maturities (single options and strategies)."""

from __future__ import annotations

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    raise SystemExit(
        "This example needs matplotlib. Install it with:  pip install matplotlib  "
        "(or  pip install 'greekscope[plot]')"
    )

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#6b7280",
)
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.8,
    "axes.edgecolor": "#9ca3af", "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": "#374151", "legend.frameon": False,
})

from greekscope import (
    butterfly, delta, gamma, iron_condor, rho, straddle, theta, vanna, vega,
)

K, R, SIGMA = 100.0, 0.03, 0.25
MATURITIES = [(0.05, BLUE), (0.25, GREEN), (1.0, ORANGE)]
spots = np.linspace(60, 140, 401)

# ---------------------------------------------------------------- single options
PANELS = [
    ("Delta", delta, True, "Delta  (dV/dS)"),
    ("Gamma", gamma, False, "Gamma  (d2V/dS2)"),
    ("Vega", vega, False, "Vega  (per 1.00 change in vol)"),
    ("Theta", theta, True, "Theta  (per year)"),
    ("Rho", rho, True, "Rho  (per 1.00 change in rate)"),
    ("Vanna", vanna, False, "Vanna  (d delta / d sigma)"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
for ax, (title, fn, kind_dep, ylab) in zip(axes.flat, PANELS):
    for T, colour in MATURITIES:
        if kind_dep:
            ax.plot(spots, fn(spots, K, T, R, SIGMA, "call"), color=colour, lw=2)
            ax.plot(spots, fn(spots, K, T, R, SIGMA, "put"), color=colour, lw=2, ls="--")
        elif fn is vanna:
            ax.plot(spots, fn(spots, K, T, R, SIGMA), color=colour, lw=2)
        else:
            ax.plot(spots, fn(spots, K, T, R, SIGMA, "call"), color=colour, lw=2)
    ax.axvline(K, color=GREY, lw=0.9, ls=":")
    ax.set_title(title)
    ax.set_ylabel(ylab)
for ax in axes[-1]:
    ax.set_xlabel("Spot price")
handles = [plt.Line2D([], [], color=c, lw=2, label=f"T = {T:g} y") for T, c in MATURITIES]
handles += [
    plt.Line2D([], [], color="#374151", lw=2, label="call"),
    plt.Line2D([], [], color="#374151", lw=2, ls="--", label="put"),
    plt.Line2D([], [], color=GREY, lw=1, ls=":", label=f"strike = {K:g}"),
]
fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    f"Black-Scholes-Merton Greeks vs. spot  (K = {K:g}, r = {R:.0%}, sigma = {SIGMA:.0%}; "
    "gamma, vega and vanna are identical for calls and puts)",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
path = OUT / "greeks_vs_spot.png"
fig.savefig(path)
print(f"wrote {path}")

# ------------------------------------------------------------------- strategies
S0, T0 = 100.0, 0.25
STRATS = [
    (straddle(S0, T0, R, SIGMA), BLUE),
    (butterfly(S0, T0, R, SIGMA, 90, 100, 110), GREEN),
    (iron_condor(S0, T0, R, SIGMA, 80, 90, 110, 120), RED),
]
spots2 = np.linspace(70, 130, 301)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
for ax, key in zip(axes, ("delta", "gamma", "vega", "theta")):
    for st, colour in STRATS:
        ax.plot(spots2, st.greeks(S=spots2)[key], color=colour, lw=2, label=st.name)
    ax.axhline(0, color="#374151", lw=0.8)
    ax.axvline(S0, color=GREY, lw=0.9, ls=":")
    ax.set_title(key.capitalize())
    ax.set_xlabel("Spot price")
axes[0].set_ylabel("Position Greek (per unit of underlying)")
axes[0].legend(loc="best", fontsize=8.5)
fig.suptitle(
    f"Strategy Greeks vs. spot  (T = {T0:g} y, r = {R:.0%}, sigma = {SIGMA:.0%})",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
path = OUT / "greeks_strategies.png"
fig.savefig(path)
print(f"wrote {path}")
