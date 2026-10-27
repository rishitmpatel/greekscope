import math

import numpy as np
import pytest

from greekscope import bsm, greeks as g

P = bsm.price
CASES = [
    # S, K, T, r, sigma, q
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.03),
    (85.0, 100.0, 0.5, 0.02, 0.35, 0.04),
    (120.0, 100.0, 2.0, 0.03, 0.25, 0.015),
    (100.0, 110.0, 0.25, -0.01, 0.4, 0.02),
    (70.0, 100.0, 3.0, 0.06, 0.15, 0.05),
]
KINDS = ["call", "put"]


def close(a, b, rtol=1e-5, atol=1e-7):
    return a == pytest.approx(b, rel=rtol, abs=atol)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_first_order_vs_finite_difference(S, K, T, r, sigma, q, kind):
    hS, hs, hr, hT = 1e-4 * S, 1e-5, 1e-6, 1e-5
    fd_delta = (P(S + hS, K, T, r, sigma, kind, q) - P(S - hS, K, T, r, sigma, kind, q)) / (2 * hS)
    fd_vega = (P(S, K, T, r, sigma + hs, kind, q) - P(S, K, T, r, sigma - hs, kind, q)) / (2 * hs)
    fd_rho = (P(S, K, T, r + hr, sigma, kind, q) - P(S, K, T, r - hr, sigma, kind, q)) / (2 * hr)
    fd_theta = -(P(S, K, T + hT, r, sigma, kind, q) - P(S, K, T - hT, r, sigma, kind, q)) / (2 * hT)
    assert close(g.delta(S, K, T, r, sigma, kind, q), fd_delta)
    assert close(g.vega(S, K, T, r, sigma, kind, q), fd_vega)
    assert close(g.rho(S, K, T, r, sigma, kind, q), fd_rho)
    assert close(g.theta(S, K, T, r, sigma, kind, q), fd_theta)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_second_order_vs_finite_difference(S, K, T, r, sigma, q, kind):
    hS, hs = 1e-4 * S, 1e-5
    # gamma = d delta / dS, volga = d vega / d sigma, vanna = d delta / d sigma = d vega / dS
    fd_gamma = (g.delta(S + hS, K, T, r, sigma, kind, q) - g.delta(S - hS, K, T, r, sigma, kind, q)) / (2 * hS)
    fd_volga = (g.vega(S, K, T, r, sigma + hs, kind, q) - g.vega(S, K, T, r, sigma - hs, kind, q)) / (2 * hs)
    fd_vanna_a = (g.delta(S, K, T, r, sigma + hs, kind, q) - g.delta(S, K, T, r, sigma - hs, kind, q)) / (2 * hs)
    fd_vanna_b = (g.vega(S + hS, K, T, r, sigma, kind, q) - g.vega(S - hS, K, T, r, sigma, kind, q)) / (2 * hS)
    assert close(g.gamma(S, K, T, r, sigma, kind, q), fd_gamma)
    assert close(g.volga(S, K, T, r, sigma, q), fd_volga)
    assert close(g.vanna(S, K, T, r, sigma, q), fd_vanna_a)
    assert close(g.vanna(S, K, T, r, sigma, q), fd_vanna_b)


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_second_order_vs_price_stencils(S, K, T, r, sigma, q):
    # direct second differences of PRICE (looser tolerance: stencil noise)
    h, hs = 1e-3 * S, 2e-4
    gam = (P(S + h, K, T, r, sigma, "call", q) - 2 * P(S, K, T, r, sigma, "call", q) + P(S - h, K, T, r, sigma, "call", q)) / h**2
    vol = (P(S, K, T, r, sigma + hs, "call", q) - 2 * P(S, K, T, r, sigma, "call", q) + P(S, K, T, r, sigma - hs, "call", q)) / hs**2
    van = (
        P(S + h, K, T, r, sigma + hs, "call", q) - P(S + h, K, T, r, sigma - hs, "call", q)
        - P(S - h, K, T, r, sigma + hs, "call", q) + P(S - h, K, T, r, sigma - hs, "call", q)
    ) / (4 * h * hs)
    assert g.gamma(S, K, T, r, sigma, "call", q) == pytest.approx(gam, rel=1e-4, abs=1e-6)
    assert g.volga(S, K, T, r, sigma, q) == pytest.approx(vol, rel=1e-4, abs=5e-5)
    assert g.vanna(S, K, T, r, sigma, q) == pytest.approx(van, rel=1e-4, abs=2e-5)


def test_known_atm_values():
    a = g.all_greeks(100, 100, 1, 0.05, 0.2)
    assert a["price"] == pytest.approx(10.450583572185565, abs=1e-10)
    assert a["delta"] == pytest.approx(0.6368306511756191, abs=1e-12)
    assert a["gamma"] == pytest.approx(0.018762017345846895, abs=1e-12)
    assert a["vega"] == pytest.approx(37.52403469169379, abs=1e-9)
    assert set(a) == {"price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga"}


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_call_put_relations(S, K, T, r, sigma, q):
    dq = math.exp(-q * T)
    assert g.delta(S, K, T, r, sigma, "call", q) - g.delta(S, K, T, r, sigma, "put", q) == pytest.approx(dq, abs=1e-12)
    assert g.gamma(S, K, T, r, sigma, "call", q) == g.gamma(S, K, T, r, sigma, "put", q)
    assert g.vega(S, K, T, r, sigma, "call", q) == g.vega(S, K, T, r, sigma, "put", q)
    # rho_c - rho_p = K T e^{-rT}
    assert g.rho(S, K, T, r, sigma, "call", q) - g.rho(S, K, T, r, sigma, "put", q) == pytest.approx(
        K * T * math.exp(-r * T), abs=1e-10
    )


def test_theta_sign_convention():
    # long ATM no-dividend option loses value as time passes
    assert g.theta(100, 100, 1, 0.05, 0.2, "call") < 0
    assert g.theta(100, 100, 1, 0.05, 0.2, "put") < 0
    # explicit calendar roll: price one day later (T smaller) minus today ~ theta/365
    dt = 1 / 365
    d = P(100, 100, 1 - dt, 0.05, 0.2, "call") - P(100, 100, 1, 0.05, 0.2, "call")
    assert d == pytest.approx(g.theta(100, 100, 1, 0.05, 0.2, "call") * dt, rel=2e-3)


def test_all_greeks_matches_individual_and_arrays():
    S = np.array([90.0, 100.0, 110.0])
    a = g.all_greeks(S, 100, 1, 0.05, 0.2, "put", 0.02)
    assert a["delta"].shape == (3,)
    for i, s in enumerate(S):
        assert a["theta"][i] == pytest.approx(g.theta(s, 100, 1, 0.05, 0.2, "put", 0.02), abs=1e-13)
        assert a["volga"][i] == pytest.approx(g.volga(s, 100, 1, 0.05, 0.2, 0.02), abs=1e-13)
    scalar = g.all_greeks(100, 100, 1, 0.05, 0.2)
    assert all(isinstance(v, float) for v in scalar.values())


@pytest.mark.parametrize("kind", KINDS)
def test_expiry_limits(kind):
    for S in (90.0, 100.0, 110.0):
        out = g.all_greeks(S, 100, 0.0, 0.05, 0.2, kind, 0.02)
        assert all(np.isfinite(v) for v in out.values())
        assert out["gamma"] == out["vega"] == out["theta"] == out["rho"] == 0.0
        assert out["vanna"] == out["volga"] == 0.0
    d_itm = g.delta(110, 100, 0, 0.05, 0.2, kind)
    d_otm = g.delta(90, 100, 0, 0.05, 0.2, kind)
    if kind == "call":
        assert (d_itm, d_otm) == (1.0, 0.0)
    else:
        assert (d_itm, d_otm) == (0.0, -1.0)
    assert abs(g.delta(100, 100, 0, 0.05, 0.2, kind)) in (0.0, 0.5, 1.0)
    assert g.delta(100, 100, 0, 0.0, 0.2, "call") == 0.5


def test_zero_vol_limits():
    # deterministic: call deep ITM delta = e^{-qT}, gamma 0, finite everything
    out = g.all_greeks(120, 100, 1.0, 0.05, 0.0, "call", 0.02)
    assert all(np.isfinite(v) for v in out.values())
    assert out["delta"] == pytest.approx(math.exp(-0.02), abs=1e-12)
    assert out["gamma"] == 0.0
    assert out["vega"] == pytest.approx(0.0, abs=1e-12)
    # theta = -dV/dT of deterministic value: V = S e^{-qT} - K e^{-rT}
    assert out["theta"] == pytest.approx(0.02 * 120 * math.exp(-0.02) - 0.05 * 100 * math.exp(-0.05), abs=1e-12)


def test_extreme_moneyness_no_warnings():
    # conftest turns warnings into errors
    for K in (1e-3, 1e6):
        out = g.all_greeks(100, K, 0.5, 0.05, 0.2, "call")
        assert all(np.isfinite(v) for v in out.values())


def test_invalid_inputs():
    with pytest.raises(ValueError):
        g.delta(-1, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        g.gamma(100, 100, 1, 0.05, -0.2)
    with pytest.raises(ValueError):
        g.vega(100, 100, 1, 0.05, 0.2, "straddle")
    with pytest.raises(ValueError):
        g.vanna(100, 0, 1, 0.05, 0.2)
