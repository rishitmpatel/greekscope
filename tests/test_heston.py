import math

import numpy as np
import pytest
from scipy.integrate import quad

from greekscope import bsm
from greekscope.heston import heston_price

BASE = dict(S=100.0, K=100.0, T=0.5, r=0.03, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)


def hp(**over):
    p = {**BASE, **over}
    kind = p.pop("kind", "call")
    q = p.pop("q", 0.0)
    return heston_price(p["S"], p["K"], p["T"], p["r"], p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"], kind, q)


@pytest.mark.parametrize("K", [70, 100, 130])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity(K, T, q):
    c, p = hp(K=K, T=T, q=q), hp(K=K, T=T, q=q, kind="put")
    r = BASE["r"]
    assert c - p == pytest.approx(100 * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("K", [80, 100, 120])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_small_xi_reduces_to_bsm(K, kind):
    sigma = 0.25
    got = hp(K=K, kind=kind, q=0.01, v0=sigma**2, theta=sigma**2, xi=1e-4, rho=-0.5)
    assert got == pytest.approx(bsm.price(100, K, 0.5, 0.03, sigma, kind, 0.01), abs=1e-3)


def test_no_arbitrage_bounds_and_monotone_in_strike():
    prices = [hp(K=k) for k in np.linspace(60, 160, 21)]
    assert all(np.diff(prices) < 0)
    for k, c in zip(np.linspace(60, 160, 21), prices):
        assert max(100 - k * math.exp(-0.03 * 0.5), 0) - 1e-9 <= c <= 100


def test_rho_sign_skew():
    # negative correlation => OTM puts richer / OTM calls cheaper than with positive correlation
    assert hp(K=80, kind="put", rho=-0.8) > hp(K=80, kind="put", rho=0.8)
    assert hp(K=120, rho=-0.8) < hp(K=120, rho=0.8)


def test_expiry_returns_intrinsic():
    assert heston_price(110, 100, 0.0, 0.03, 0.04, 2, 0.04, 0.3, -0.7) == 10.0
    assert heston_price(110, 100, 0.0, 0.03, 0.04, 2, 0.04, 0.3, -0.7, "put") == 0.0


def test_zero_initial_variance_and_zero_theta_ok():
    c = hp(v0=0.0, theta=0.0)
    # no variance anywhere: deterministic forward payoff
    assert c == pytest.approx(max(100 - 100 * math.exp(-0.03 * 0.5), 0.0), abs=1e-6)
    assert math.isfinite(hp(v0=0.0))


@pytest.mark.parametrize(
    "over",
    [dict(v0=-0.01), dict(kappa=0.0), dict(kappa=-1.0), dict(theta=-0.01), dict(xi=0.0),
     dict(xi=-0.1), dict(rho=1.0), dict(rho=-1.0), dict(rho=1.5), dict(S=-1.0), dict(K=0.0),
     dict(T=-1.0), dict(kind="straddle"), dict(v0=float("nan"))],
)
def test_invalid_params_raise(over):
    with pytest.raises(ValueError):
        hp(**over)


# ---- independent reference: Lewis (2001) single-integral formula with the ORIGINAL
# ---- (Heston 1993) characteristic function. Different integral representation and a
# ---- different cf algebra from the implementation -> a genuine cross-check where
# ---- the original form is numerically safe (moderate maturities).
def _cf_original(u, T, r, q, v0, kappa, theta, xi, rho):
    iu = 1j * u
    t = kappa - rho * xi * iu
    d = np.sqrt(t * t + xi**2 * (iu + u * u))
    g = (t + d) / (t - d)
    e = np.exp(d * T)
    C = (r - q) * iu * T + kappa * theta / xi**2 * ((t + d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (t + d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def lewis_call(S, K, T, r, q, v0, kappa, theta, xi, rho):
    F = S * math.exp((r - q) * T)
    k = math.log(F / K)

    def f(u):
        z = u - 0.5j
        ph = _cf_original(z, T, r, q, v0, kappa, theta, xi, rho) * np.exp(-1j * z * (r - q) * T)
        return (np.exp(1j * u * k) * ph).real / (u * u + 0.25)

    I = sum(quad(f, a, b, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
            for a, b in [(0, 10), (10, 50), (50, 200), (200, 1000)])
    return S * math.exp(-q * T) - math.sqrt(F * K) * math.exp(-r * T) / math.pi * I


@pytest.mark.parametrize(
    "S,K,T,r,q,v0,kappa,theta,xi,rho",
    [
        (100, 100, 1.0, 0.0, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7),
        (100, 100, 0.5, 0.03, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7),
        (100, 80, 0.75, 0.02, 0.01, 0.09, 1.0, 0.06, 0.5, -0.3),
        (100, 120, 0.25, 0.01, 0.0, 0.02, 3.0, 0.05, 0.4, 0.3),
    ],
)
def test_matches_lewis_reference(S, K, T, r, q, v0, kappa, theta, xi, rho):
    ref = lewis_call(S, K, T, r, q, v0, kappa, theta, xi, rho)
    got = heston_price(S, K, T, r, v0, kappa, theta, xi, rho, "call", q)
    assert got == pytest.approx(ref, abs=1e-7)


def test_original_cf_trap_regime_is_handled():
    # Feller-violating, rho=-0.9, T=1: the ORIGINAL Heston cf is already inaccurate here
    # (Lewis with it gives 4.36198), whereas the little-trap cf gives 4.36614 -- confirmed by
    # Euler Monte Carlo (4.370 +/- 0.006 with 1000 steps, converging from above).
    got = heston_price(100, 105, 1.0, 0.05, 0.05, 0.8, 0.03, 0.6, -0.9, "call", 0.03)
    assert got == pytest.approx(4.36613931354, abs=1e-7)


def test_hardcoded_benchmark():
    # S=K=100, T=1, r=q=0, v0=theta=0.04, kappa=1.5, xi=0.3, rho=-0.7.
    # 7.5402666388187 was obtained independently by (a) this implementation (P1/P2 Gil-Pelaez,
    # little-trap cf) and (b) the Lewis single-integral formula with the original Heston cf;
    # both agree to ~1e-14. Also confirmed by Monte Carlo in test_matches_euler_monte_carlo.
    assert heston_price(100, 100, 1.0, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7) == pytest.approx(7.5402666388187, abs=1e-8)
    # T=0.5, r=3%: 6.26468092893878 (same two-method confirmation)
    assert hp() == pytest.approx(6.26468092893878, abs=1e-8)


def test_long_maturity_stable():
    # the original (trap-prone) formulation breaks for long T / large xi; ours must stay sane
    c = heston_price(100, 100, 10.0, 0.03, 0.04, 0.5, 0.04, 1.0, -0.7)
    p = heston_price(100, 100, 10.0, 0.03, 0.04, 0.5, 0.04, 1.0, -0.7, "put")
    assert 0 < c < 100 and 0 < p < 100 * math.exp(-0.3)
    assert c - p == pytest.approx(100 - 100 * math.exp(-0.3), abs=1e-7)


def test_short_maturity_stable():
    c = heston_price(100, 100, 0.02, 0.01, 0.04, 2.0, 0.04, 0.5, -0.5)
    assert c == pytest.approx(bsm.price(100, 100, 0.02, 0.01, 0.2, "call"), rel=0.02)


def _euler_ft_mc(S, K, T, r, q, v0, kappa, theta, xi, rho, kind, n_paths=200_000, n_steps=250, seed=2024):
    """Euler full-truncation Heston simulation of log-spot with antithetic pairs.

    Returns ``(price, std_error)`` where the standard error is computed over pair averages.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sq = math.sqrt(dt)
    m = n_paths // 2
    sgn = 1.0 if kind == "call" else -1.0
    ln_s = np.full(m, math.log(S))
    ln_s2 = ln_s.copy()
    v = np.full(m, v0)
    v2 = v.copy()
    c = math.sqrt(1 - rho * rho)
    for _ in range(n_steps):
        z1 = rng.standard_normal(m)
        z2 = rho * z1 + c * rng.standard_normal(m)
        vp = np.maximum(v, 0.0)
        ln_s += (r - q - 0.5 * vp) * dt + np.sqrt(vp) * sq * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sq * z2
        vp2 = np.maximum(v2, 0.0)  # antithetic partner uses (-z1, -z2)
        ln_s2 += (r - q - 0.5 * vp2) * dt - np.sqrt(vp2) * sq * z1
        v2 += kappa * (theta - vp2) * dt - xi * np.sqrt(vp2) * sq * z2
    y = 0.5 * (np.maximum(sgn * (np.exp(ln_s) - K), 0.0) + np.maximum(sgn * (np.exp(ln_s2) - K), 0.0))
    y *= math.exp(-r * T)
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(m))


@pytest.mark.slow
@pytest.mark.parametrize(
    "kw,kind",
    [
        (dict(S=100, K=100, T=0.5, r=0.03, q=0.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7), "call"),
        (dict(S=100, K=100, T=1.0, r=0.0, q=0.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.7), "call"),
        (dict(S=100, K=90, T=1.0, r=0.02, q=0.01, v0=0.06, kappa=1.0, theta=0.04, xi=0.5, rho=-0.5), "put"),
    ],
)
def test_matches_euler_monte_carlo(kw, kind):
    mc, se = _euler_ft_mc(**kw, kind=kind)
    ana = heston_price(kw["S"], kw["K"], kw["T"], kw["r"], kw["v0"], kw["kappa"], kw["theta"],
                       kw["xi"], kw["rho"], kind, kw["q"])
    # 3 standard errors plus a small allowance for Euler discretisation bias
    assert abs(mc - ana) < 3 * se + 0.03, (mc, ana, se)
