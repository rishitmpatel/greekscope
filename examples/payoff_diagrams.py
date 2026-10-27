"""Payoff diagrams for the standard option strategies (expiry payoff vs. value today)."""

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
    bear_put_spread, bull_call_spread, butterfly, calendar_spread, covered_call,
    iron_condor, protective_put, straddle, strangle,
)

S, T, R, SIGMA = 100.0, 0.25, 0.04, 0.25

STRATS = [
    covered_call(S, T, R, SIGMA, 105),
    protective_put(S, T, R, SIGMA, 95),
    bull_call_spread(S, T, R, SIGMA, 95, 110),
    bear_put_spread(S, T, R, SIGMA, 90, 105),
    straddle(S, T, R, SIGMA),
    strangle(S, T, R, SIGMA, 90, 110),
    iron_condor(S, T, R, SIGMA, 80, 90, 110, 120),
    butterfly(S, T, R, SIGMA, 90, 100, 110),
    calendar_spread(S, T, R, SIGMA, 100, 2 * T),
]


def panel(ax, st):
    xs = np.linspace(55, 145, 600)
    pay = st.payoff(xs)
    today = st.value(xs, 0.0)
    ax.axhline(0, color="#374151", lw=0.9)
    ax.fill_between(xs, pay, 0, where=pay >= 0, color=GREEN, alpha=0.18, lw=0)
    ax.fill_between(xs, pay, 0, where=pay < 0, color=RED, alpha=0.18, lw=0)
    ax.plot(xs, pay, color=BLUE, lw=2, label="P&L at expiry")
    ax.plot(xs, today, color=ORANGE, lw=1.6, ls="--", label="P&L today")
    ax.axvline(S, color=GREY, lw=0.9, ls=":", label=f"Spot = {S:g}")
    be = [b for b in st.breakevens() if xs[0] <= b <= xs[-1]]
    ax.plot(be, [0] * len(be), "o", color="white", mec="#111827", mew=1.5, ms=7, zorder=5, label="Breakeven")
    ax.set_title(st.name)
    ax.set_xlim(xs[0], xs[-1])
    lo, hi = float(min(pay.min(), today.min())), float(max(pay.max(), today.max()))
    pad = 0.12 * max(hi - lo, 1.0)
    ax.set_ylim(lo - pad, hi + pad)


fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
for ax, st in zip(axes.flat, STRATS):
    panel(ax, st)
for ax in axes[-1]:
    ax.set_xlabel("Underlying price")
for ax in axes[:, 0]:
    ax.set_ylabel("Profit / loss per share")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    f"Option strategy payoffs  (S = {S:g}, T = {T:g} y, r = {R:.0%}, sigma = {SIGMA:.0%}, premiums at BSM prices)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
path = OUT / "payoff_diagrams.png"
fig.savefig(path)
print(f"wrote {path}")
