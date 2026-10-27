import math

import numpy as np
import pytest

from greekscope import bsm
from greekscope.montecarlo import MCResult, asian_mc_price, mc_price, _geometric_asian_price


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("cv", [True, False])
def test_mc_within_three_std_errors_of_bsm(kind, antithetic, cv):
    exact = bsm.price(100, 105, 1.0, 0.05, 0.25, kind, 0.02)
    res = mc_price(100, 105, 1.0, 0.05, 0.25, kind, 0.02, n_paths=100_000,
                   antithetic=antithetic, control_variate=cv, seed=12345)
    assert isinstance(res, MCResult)
    assert res.std_error > 0
    assert abs(res.price - exact) < 3 * res.std_error


def test_variance_reduction_orders():
    kw = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_paths=100_000, seed=7)
    plain = mc_price(**kw, antithetic=False, control_variate=False)
    anti = mc_price(**kw, antithetic=True, control_variate=False)
    cv = mc_price(**kw, antithetic=False, control_variate=True)
    both = mc_price(**kw, antithetic=True, control_variate=True)
    assert anti.std_error < plain.std_error
    assert cv.std_error < plain.std_error
    assert both.std_error < plain.std_error
    assert both.std_error <= 1.05 * min(anti.std_error, cv.std_error)


def test_std_error_calibration_coverage():
    # Repeating the experiment, the empirical spread of estimates should match the reported SE.
    exact = bsm.price(100, 100, 1.0, 0.05, 0.2, "call")
    ests, ses = [], []
    for seed in range(60):
        r = mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=4000, seed=seed)
        ests.append(r.price)
        ses.append(r.std_error)
    assert np.std(ests, ddof=1) == pytest.approx(np.mean(ses), rel=0.3)
    assert abs(np.mean(ests) - exact) < 3 * np.mean(ses) / math.sqrt(60) * 1.5 + 1e-3


def test_antithetic_se_uses_pairs():
    # Honest SE for antithetic must differ from the naive per-path SE (which would be too small
    # when payoffs are negatively correlated within pairs). Verify against empirical spread.
    ests, ses = [], []
    for seed in range(80):
        r = mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=2000, antithetic=True,
                     control_variate=False, seed=1000 + seed)
        ests.append(r.price)
        ses.append(r.std_error)
    assert np.std(ests, ddof=1) == pytest.approx(np.mean(ses), rel=0.25)


def test_seed_reproducibility_and_n_paths():
    a = mc_price(100, 100, 1, 0.05, 0.2, n_paths=1001, seed=3)
    b = mc_price(100, 100, 1, 0.05, 0.2, n_paths=1001, seed=3)
    assert a == b
    assert a.n_paths == 1000  # odd count rounded down to whole antithetic pairs
    assert mc_price(100, 100, 1, 0.05, 0.2, n_paths=1001, antithetic=False, seed=3).n_paths == 1001
    c = mc_price(100, 100, 1, 0.05, 0.2, n_paths=1000, seed=4)
    assert c.price != a.price


def test_degenerate_cases():
    r = mc_price(110, 100, 0.0, 0.05, 0.2, n_paths=100, seed=1)
    assert r.price == pytest.approx(10.0) and r.std_error == pytest.approx(0.0, abs=1e-12)
    r = mc_price(100, 90, 1.0, 0.05, 0.0, n_paths=100, seed=1)
    assert r.price == pytest.approx(bsm.price(100, 90, 1.0, 0.05, 0.0), abs=1e-10)
    assert math.isfinite(r.std_error)


def test_validation():
    with pytest.raises(ValueError):
        mc_price(100, 100, 1, 0.05, 0.2, n_paths=1)
    with pytest.raises(ValueError):
        mc_price(100, -1, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        mc_price(100, 100, 1, 0.05, 0.2, kind="x")
    with pytest.raises(ValueError):
        asian_mc_price(100, 100, 1, 0.05, 0.2, n_steps=0)


def test_asian_single_step_equals_european():
    exact = bsm.price(100, 100, 1.0, 0.05, 0.2, "call")
    r = asian_mc_price(100, 100, 1.0, 0.05, 0.2, n_steps=1, n_paths=100_000, seed=5)
    assert abs(r.price - exact) < 3 * r.std_error + 1e-9


def test_geometric_asian_closed_form_vs_simulation():
    # closed form used as the control variate must match an independent plain simulation
    S, K, T, r, sigma, q, n = 100, 100, 1.0, 0.05, 0.2, 0.02, 12
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200_000, n))
    dt = T / n
    logs = math.log(S) + np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z, axis=1)
    pay = math.exp(-r * T) * np.maximum(np.exp(logs.mean(axis=1)) - K, 0)
    cf = _geometric_asian_price(S, K, T, r, sigma, "call", q, n)
    assert abs(pay.mean() - cf) < 3 * pay.std() / math.sqrt(pay.size)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_asian_cv_and_plain_agree_and_cv_reduces_variance(kind):
    kw = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, kind=kind, q=0.01, n_steps=50, n_paths=40_000)
    cv = asian_mc_price(**kw, seed=11)
    plain = asian_mc_price(**kw, seed=12, control_variate=False)
    assert cv.std_error < plain.std_error / 5
    assert abs(cv.price - plain.price) < 4 * math.hypot(cv.std_error, plain.std_error)


def test_asian_bounds_and_ordering():
    # arithmetic Asian call is cheaper than the European call and above the geometric Asian call
    n = 252
    eu = bsm.price(100, 100, 1.0, 0.05, 0.2, "call")
    a = asian_mc_price(100, 100, 1.0, 0.05, 0.2, n_steps=n, n_paths=20_000, seed=2)
    g = _geometric_asian_price(100, 100, 1.0, 0.05, 0.2, "call", 0.0, n)
    assert g - 3 * a.std_error < a.price < eu
    # published-style sanity: ~5.7-5.8 for these inputs
    assert 5.5 < a.price < 6.0


def test_asian_degenerate_and_seed():
    r = asian_mc_price(100, 90, 1.0, 0.05, 0.0, n_steps=10, n_paths=10, seed=1)
    assert math.isfinite(r.price) and r.std_error == pytest.approx(0.0, abs=1e-9)
    a = asian_mc_price(100, 100, 1.0, 0.05, 0.2, n_steps=10, n_paths=1000, seed=9)
    b = asian_mc_price(100, 100, 1.0, 0.05, 0.2, n_steps=10, n_paths=1000, seed=9)
    assert a == b and a.n_paths == 1000
