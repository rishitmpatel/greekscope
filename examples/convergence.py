"""Convergence of the CRR tree and Monte Carlo to the Black-Scholes-Merton price."""

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

from greekscope import bsm_price, crr_price, mc_price

S, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.05, 0.2
exact = bsm_price(S, K, T, R, SIGMA, "call")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# (a) CRR price vs steps -- the familiar odd/even sawtooth around the BSM value
steps = np.arange(5, 201)
tree = np.array([crr_price(S, K, T, R, SIGMA, "call", steps=int(n)) for n in steps])
ax = axes[0]
ax.plot(steps, tree, color=BLUE, lw=1.4, label="CRR (European call)")
ax.axhline(exact, color=RED, lw=1.5, ls="--", label=f"BSM = {exact:.4f}")
ax.set_xlabel("Tree steps")
ax.set_ylabel("Price")
ax.set_title("CRR tree oscillates around BSM")
ax.legend()

# (b) CRR absolute error, log-log, with 1/N reference
steps_b = np.unique(np.round(np.logspace(1, 3.3, 40)).astype(int))
err = np.array([abs(crr_price(S, K, T, R, SIGMA, "call", steps=int(n)) - exact) for n in steps_b])
ax = axes[1]
ax.loglog(steps_b, err, "o-", color=BLUE, ms=4, lw=1.2, label="|CRR - BSM|")
ref = err[0] * steps_b[0] / steps_b
ax.loglog(steps_b, ref, color=GREY, ls=":", lw=1.6, label="O(1/N) reference")
ax.set_xlabel("Tree steps N")
ax.set_ylabel("Absolute error")
ax.set_title("CRR error decays roughly like 1/N")
ax.legend()

# (c) Monte Carlo standard error vs paths for three variance-reduction settings
paths = np.unique(np.round(np.logspace(3, 6, 10)).astype(int))
configs = [
    ("plain", dict(antithetic=False, control_variate=False), RED),
    ("antithetic", dict(antithetic=True, control_variate=False), ORANGE),
    ("antithetic + control variate", dict(antithetic=True, control_variate=True), BLUE),
]
ax = axes[2]
for label, kw, colour in configs:
    se, miss = [], []
    for n in paths:
        res = mc_price(S, K, T, R, SIGMA, "call", n_paths=int(n), seed=12345, **kw)
        se.append(res.std_error)
        miss.append(abs(res.price - exact))
    ax.loglog(paths, se, "o-", color=colour, ms=4, lw=1.5, label=f"std. error: {label}")
    ax.loglog(paths, np.maximum(miss, 1e-7), "x", color=colour, ms=6, alpha=0.7)
ref = se[0] * np.sqrt(paths[0] / paths)
ax.loglog(paths, ref, color=GREY, ls=":", lw=1.6, label="O(1/sqrt(n)) reference")
ax.plot([], [], "x", color="#374151", label="actual |MC - BSM| (same colour)")
ax.set_xlabel("Monte Carlo paths n")
ax.set_ylabel("Error")
ax.set_title("Monte Carlo: 1/sqrt(n) and variance reduction")
ax.legend(fontsize=8, loc="lower left")

fig.suptitle(
    f"Convergence to Black-Scholes-Merton  (S = {S:g}, K = {K:g}, T = {T:g} y, r = {R:.0%}, sigma = {SIGMA:.0%})",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
path = OUT / "convergence.png"
fig.savefig(path)
print(f"wrote {path}")
