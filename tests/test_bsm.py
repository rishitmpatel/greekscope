import math

import numpy as np
import pytest

from greekscope import bsm
from greekscope._validate import check_inputs, check_kind

CALL_REF = 10.450583572185565
PUT_REF = 5.573526022256971


def test_known_values_exact():
    assert bsm.price(100, 100, 1, 0.05, 0.2, "call") == pytest.approx(CALL_REF, abs=1e-10)
    assert bsm.price(100, 100, 1, 0.05, 0.2, "put") == pytest.approx(PUT_REF, abs=1e-10)


def test_default_kind_is_call_and_case_insensitive():
    assert bsm.price(100, 100, 1, 0.05, 0.2) == bsm.price(100, 100, 1, 0.05, 0.2, "CALL")
    assert bsm.price(100, 100, 1, 0.05, 0.2, " Put ") == pytest.approx(PUT_REF, abs=1e-10)


def test_d1_d2_values():
    d1 = bsm.d1(100, 100, 1, 0.05, 0.2)
    assert d1 == pytest.approx(0.35, abs=1e-12)
    assert bsm.d2(100, 100, 1, 0.05, 0.2) == pytest.approx(0.15, abs=1e-12)


@pytest.mark.parametrize("S", [60, 100, 140])
@pytest.mark.parametrize("K", [80, 100, 125])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(S, K, T, q):
    r, sigma = 0.04, 0.3
    c = bsm.price(S, K, T, r, sigma, "call", q)
    p = bsm.price(S, K, T, r, sigma, "put", q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_dividend_yield_matches_forward_shift():
    # With q, price equals the q=0 price at spot S e^{-qT} and rate r (no-div equivalence).
    S, K, T, r, sigma, q = 100, 95, 1.5, 0.05, 0.25, 0.04
    a = bsm.price(S, K, T, r, sigma, "call", q)
    b = bsm.price(S * math.exp(-q * T), K, T, r, sigma, "call", 0.0)
    assert a == pytest.approx(b, abs=1e-12)


def test_zero_time_is_intrinsic():
    assert bsm.price(110, 100, 0, 0.05, 0.2, "call") == 10.0
    assert bsm.price(110, 100, 0, 0.05, 0.2, "put") == 0.0
    assert bsm.price(90, 100, 0, 0.05, 0.2, "put") == 10.0
    assert bsm.price(100, 100, 0, 0.05, 0.2, "call") == 0.0


def test_zero_vol_is_discounted_forward_intrinsic():
    S, K, T, r, q = 100, 90, 2.0, 0.05, 0.01
    fs, pk = S * math.exp(-q * T), K * math.exp(-r * T)
    assert bsm.price(S, K, T, r, 0.0, "call", q) == pytest.approx(fs - pk, abs=1e-12)
    assert bsm.price(S, K, T, r, 0.0, "put", q) == 0.0
    assert bsm.price(S, 130, T, r, 0.0, "put", q) == pytest.approx(130 * math.exp(-r * T) - fs, abs=1e-12)
    # ATM-forward, sigma = 0: exactly zero, not NaN
    K_atm = S * math.exp((r - q) * T)
    assert bsm.price(100, K_atm, T, r, 0.0, "call", q) == pytest.approx(0.0, abs=1e-12)


def test_small_vol_continuity():
    big = bsm.price(100, 90, 1, 0.05, 1e-6, "call")
    zero = bsm.price(100, 90, 1, 0.05, 0.0, "call")
    assert big == pytest.approx(zero, abs=1e-8)


def test_array_broadcast_and_scalar_types():
    K = np.array([80.0, 100.0, 120.0])
    out = bsm.price(100, K, 1, 0.05, 0.2, "call")
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    for k, v in zip(K, out):
        assert v == pytest.approx(bsm.price(100, k, 1, 0.05, 0.2, "call"), abs=1e-13)
    assert isinstance(bsm.price(100, 100, 1, 0.05, 0.2), float)
    grid = bsm.price(np.array([[90.0], [110.0]]), K, 1, 0.05, 0.2)
    assert grid.shape == (2, 3)


def test_array_with_mixed_degenerate_entries_has_no_nan():
    T = np.array([0.0, 0.5, 1.0])
    sg = np.array([0.2, 0.0, 0.2])
    out = bsm.price(100, 100, T, 0.05, sg, "call")
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0


def test_monotonicity_in_sigma_and_bounds():
    sig = np.linspace(0.01, 2.0, 50)
    c = bsm.price(100, 100, 1, 0.05, sig, "call")
    assert np.all(np.diff(c) > 0)
    assert np.all(c < 100) and np.all(c > 0)


@pytest.mark.parametrize(
    "args",
    [
        (-1, 100, 1, 0.05, 0.2),
        (0, 100, 1, 0.05, 0.2),
        (100, 0, 1, 0.05, 0.2),
        (100, -5, 1, 0.05, 0.2),
        (100, 100, -0.1, 0.05, 0.2),
        (100, 100, 1, 0.05, -0.2),
        (float("nan"), 100, 1, 0.05, 0.2),
        (100, 100, 1, float("nan"), 0.2),
    ],
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        bsm.price(*args)


@pytest.mark.parametrize("kind", ["straddle", "", "calls", None, 1])
def test_bad_kind_raises(kind):
    with pytest.raises(ValueError):
        bsm.price(100, 100, 1, 0.05, 0.2, kind)
    with pytest.raises(ValueError):
        check_kind(kind)


def test_validate_helpers():
    assert check_kind("CALL") == "call"
    check_inputs(1, 1, 0, 0)
    with pytest.raises(ValueError, match="sigma"):
        check_inputs(1, 1, 1, -1)
    with pytest.raises(ValueError, match="T"):
        check_inputs(1, 1, -1, 1)
    with pytest.raises(ValueError):
        check_inputs(np.array([1.0, -1.0]), 1, 1, 1)
    with pytest.raises(ValueError):
        check_inputs("abc", 1, 1, 1)
