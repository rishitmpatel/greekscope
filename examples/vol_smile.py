"""Implied-volatility smile and surface recovered from synthetic option quotes."""

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

from greekscope import implied_vol_surface, parametric_vol, synthetic_quotes

S, R, Q = 100.0, 0.03, 0.01
STRIKES = np.arange(60, 145, 5.0)
EXPIRIES = [0.1, 0.25, 0.5, 1.0, 2.0]
PARAMS = dict(base_vol=0.20, skew=-0.10, curvature=0.5)

quotes = synthetic_quotes(S, R, STRIKES, EXPIRIES, q=Q, **PARAMS)
# make the demo less perfect: drop a few illiquid quotes and corrupt one price
quotes = [x for x in quotes if not (x[1] == 0.5 and x[0] in (85.0, 90.0, 125.0))]
quotes = [(K, T, 1e6 if (K, T) == (135.0, 0.1) else p, k) for K, T, p, k in quotes]
surf = implied_vol_surface(quotes, S, R, Q)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))

# ---- left: smiles per expiry, markers = recovered quote vols, lines = interpolation
cols = [BLUE, SKY, GREEN, ORANGE, RED]
fine = np.linspace(STRIKES[0], STRIKES[-1], 300)
for T, c in zip(EXPIRIES, cols):
    ks, ivs = surf.smile(T)
    ax1.plot(ks, 100 * ivs, "o", color=c, ms=5)
    ax1.plot(fine, 100 * surf.iv(fine, T), color=c, lw=1.8, label=f"T = {T:g} y")
    ax1.plot(fine, 100 * parametric_vol(fine, T, S, R, q=Q, **PARAMS), color=c, lw=0.8, ls=":", alpha=0.8)
Tmid = 0.75
ax1.plot(fine, 100 * surf.iv(fine, Tmid), color="#111827", lw=1.8, ls="--", label=f"T = {Tmid:g} y (interpolated)")
ax1.set_xlabel("Strike")
ax1.set_ylabel("Implied volatility (%)")
ax1.set_title("Smiles recovered from quotes")
h, l = ax1.get_legend_handles_labels()
h += [plt.Line2D([], [], marker="o", ls="", color=GREY), plt.Line2D([], [], ls=":", color=GREY)]
l += ["recovered quote vols", "true parametric smile"]
ax1.legend(h, l, fontsize=8.5, loc="upper center")
ax1.text(0.98, 0.55, "gaps: dropped quotes (T=0.5) and one\ninvalid price (K=135, T=0.1) are skipped\nand interpolated over",
         transform=ax1.transAxes, fontsize=8, color="#374151", ha="right", va="top")

# ---- right: interpolated surface heatmap
ks = np.linspace(STRIKES[0], STRIKES[-1], 150)
ts = np.linspace(EXPIRIES[0], EXPIRIES[-1], 150)
Z = 100 * surf.iv(ks[None, :], ts[:, None])
im = ax2.pcolormesh(ks, ts, Z, shading="auto", cmap="viridis")
cs = ax2.contour(ks, ts, Z, levels=8, colors="white", linewidths=0.6, alpha=0.8)
ax2.clabel(cs, fmt="%.1f", fontsize=7)
kk, tt = np.meshgrid(surf.strikes, surf.expiries)
ok = np.isfinite(surf.ivs)
ax2.plot(kk[ok], tt[ok], ".", color="white", ms=3, alpha=0.9)
ax2.set_xlabel("Strike")
ax2.set_ylabel("Expiry (years)")
ax2.set_title("Interpolated surface (linear in total variance across T)")
cb = fig.colorbar(im, ax=ax2)
cb.set_label("Implied volatility (%)")
ax2.grid(False)

fig.suptitle(
    f"Implied-volatility surface from synthetic quotes  (S = {S:g}, r = {R:.0%}, q = {Q:.0%})",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
path = OUT / "vol_smile.png"
fig.savefig(path)
print(f"wrote {path}")
