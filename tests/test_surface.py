import math

import numpy as np
import pytest

import greekscope
from greekscope.bsm import price as bsm
from greekscope.surface import VolSurface, implied_vol_surface, parametric_vol, synthetic_quotes

S, R, Q = 100.0, 0.03, 0.01
STRIKES = [70, 80, 90, 100, 110, 120, 140]
EXPIRIES = [0.1, 0.25, 0.5, 1.0, 2.0]
PARAMS = dict(base_vol=0.22, skew=-0.12, curvature=0.06)


@pytest.fixture(scope="module")
def surf():
    quotes = synthetic_quotes(S, R, STRIKES, EXPIRIES, q=Q, **PARAMS)
    return implied_vol_surface(quotes, S, R, Q)


def test_package_exports():
    assert greekscope.implied_vol_surface is implied_vol_surface
    assert greekscope.VolSurface is VolSurface


def test_synthetic_quotes_shape_and_kinds():
    quotes = synthetic_quotes(S, R, STRIKES, EXPIRIES, q=Q, **PARAMS)
    assert len(quotes) == len(STRIKES) * len(EXPIRIES)
    for K, T, px, kind in quotes:
        F = S * math.exp((R - Q) * T)
        assert kind == ("put" if K < F else "call")
        assert px > 0
        sig = parametric_vol(K, T, S, R, q=Q, **PARAMS)
        assert px == pytest.approx(bsm(S, K, T, R, sig, kind, Q))


def test_parametric_vol_floor_and_formula():
    F = 100.0
    assert parametric_vol(F, 1e-9, 100, 0.0, 0.2, -0.1, 0.05) == pytest.approx(0.2)
    m = math.log(80 / F)
    assert parametric_vol(80, 1e-9, 100, 0.0, 0.2, -0.1, 0.05) == pytest.approx(0.2 - 0.1 * m + 0.05 * m * m)
    assert parametric_vol(80, 1.0, 100, 0.0, 0.0, 0.0, 0.0) == 0.01


def test_surface_recovers_smile_at_nodes(surf):
    assert surf.ivs.shape == (len(EXPIRIES), len(STRIKES))
    assert not np.isnan(surf.ivs).any()
    np.testing.assert_allclose(surf.strikes, STRIKES)
    np.testing.assert_allclose(surf.expiries, EXPIRIES)
    for i, T in enumerate(EXPIRIES):
        for j, K in enumerate(STRIKES):
            expect = parametric_vol(K, T, S, R, q=Q, **PARAMS)
            assert surf.ivs[i, j] == pytest.approx(expect, abs=1e-6)
            assert surf.iv(K, T) == pytest.approx(expect, abs=1e-6)


def test_smile_at_listed_expiry(surf):
    ks, ivs = surf.smile(0.5)
    np.testing.assert_allclose(ks, STRIKES)
    np.testing.assert_allclose(ivs, surf.ivs[2])


def test_interpolation_in_strike_is_linear(surf):
    a, b = surf.iv(90, 0.5), surf.iv(100, 0.5)
    assert surf.iv(95, 0.5) == pytest.approx(0.5 * (a + b))
    lo, hi = min(a, b), max(a, b)
    assert lo < surf.iv(93, 0.5) < hi


def test_interpolation_in_expiry_is_linear_in_total_variance(surf):
    K = 90
    v0, v1 = surf.iv(K, 0.5), surf.iv(K, 1.0)
    T = 0.7
    w = v0**2 * 0.5 + (v1**2 * 1.0 - v0**2 * 0.5) * (T - 0.5) / 0.5
    assert surf.iv(K, T) == pytest.approx(math.sqrt(w / T))
    assert min(v0, v1) - 1e-12 <= surf.iv(K, T) <= max(v0, v1) + 1e-12


def test_interpolated_close_to_true_smile(surf):
    # parametric smile is smooth: interpolation error between nodes is small
    for K, T in [(95, 0.4), (105, 0.8), (85, 1.5), (115, 0.2)]:
        true = parametric_vol(K, T, S, R, q=Q, **PARAMS)
        assert surf.iv(K, T) == pytest.approx(true, abs=2.5e-3)


def test_flat_extrapolation(surf):
    assert surf.iv(10, 0.5) == pytest.approx(surf.iv(70, 0.5))
    assert surf.iv(1000, 0.5) == pytest.approx(surf.iv(140, 0.5))
    assert surf.iv(100, 0.01) == pytest.approx(surf.iv(100, 0.1))
    assert surf.iv(100, 10.0) == pytest.approx(surf.iv(100, 2.0))


def test_array_query(surf):
    Ks = np.array([80.0, 95.0, 120.0])
    out = surf.iv(Ks, 0.5)
    assert out.shape == (3,)
    assert out[0] == pytest.approx(surf.iv(80.0, 0.5))
    grid = surf.iv(Ks[:, None], np.array([0.25, 1.0])[None, :])
    assert grid.shape == (3, 2)


def test_smile_between_expiries(surf):
    ks, ivs = surf.smile(0.7)
    np.testing.assert_allclose(ks, STRIKES)
    assert ivs.shape == ks.shape and np.all(np.isfinite(ivs))
    assert ivs[0] == pytest.approx(surf.iv(70, 0.7))


def test_bad_quotes_become_nan_and_are_skipped():
    quotes = synthetic_quotes(S, R, [90, 100, 110], [0.5, 1.0], q=Q, **PARAMS)
    bad = list(quotes)
    K, T, px, kind = bad[1]  # (100, 0.5)
    bad[1] = (K, T, 1000.0, kind)  # violates upper bound
    bad.append((95.0, 0.5, -1.0, "call"))  # negative price
    s = implied_vol_surface(bad, S, R, Q)
    assert 95.0 in s.strikes
    assert np.isnan(s.ivs[0, list(s.strikes).index(100.0)])
    assert np.isnan(s.ivs[0, list(s.strikes).index(95.0)])
    # interpolation just skips the gaps
    expect = 0.5 * (s.iv(90, 0.5) + s.iv(110, 0.5))
    assert s.iv(100, 0.5) == pytest.approx(
        np.interp(100, [90, 110], [s.iv(90, 0.5), s.iv(110, 0.5)])
    )
    assert expect == pytest.approx(s.iv(100, 0.5))
    ks, ivs = s.smile(0.5)
    assert 100.0 not in ks and 95.0 not in ks and np.all(np.isfinite(ivs))


def test_empty_slice_ignored_in_time_interpolation():
    quotes = synthetic_quotes(S, R, [90, 100, 110], [0.25, 1.0], q=Q, **PARAMS)
    quotes.append((100.0, 0.5, 1e9, "call"))  # invalid -> expiry 0.5 slice entirely NaN
    s = implied_vol_surface(quotes, S, R, Q)
    assert len(s.expiries) == 3
    assert np.isnan(s.ivs[1]).all()
    assert np.isfinite(s.iv(100, 0.5))
    ref = implied_vol_surface(synthetic_quotes(S, R, [90, 100, 110], [0.25, 1.0], q=Q, **PARAMS), S, R, Q)
    assert s.iv(100, 0.5) == pytest.approx(ref.iv(100, 0.5))


def test_call_put_duplicates_averaged():
    from greekscope.implied_vol import implied_vol

    p_call = bsm(S, 100, 0.5, R, 0.2, "call", Q)
    p_put = bsm(S, 100, 0.5, R, 0.2, "put", Q)
    s = implied_vol_surface([(100, 0.5, p_call, "call"), (100, 0.5, p_put, "put")], S, R, Q)
    assert s.ivs.shape == (1, 1)
    assert s.ivs[0, 0] == pytest.approx(0.2, abs=1e-8)
    assert implied_vol(p_call, S, 100, 0.5, R, "call", Q) == pytest.approx(0.2, abs=1e-8)


def test_unsorted_input_ok():
    quotes = synthetic_quotes(S, R, [110, 90, 100], [1.0, 0.25], q=Q, **PARAMS)
    s = implied_vol_surface(reversed(quotes), S, R, Q)
    np.testing.assert_allclose(s.strikes, [90, 100, 110])
    np.testing.assert_allclose(s.expiries, [0.25, 1.0])


def test_validation_errors():
    with pytest.raises(ValueError):
        implied_vol_surface([], S, R)
    with pytest.raises(ValueError):
        implied_vol_surface([(100, 1.0, 5.0)], S, R)
    with pytest.raises(ValueError):
        implied_vol_surface([(100, 1.0, 5.0, "straddle")], S, R)
    with pytest.raises(ValueError):
        implied_vol_surface([(100, 0.0, 5.0, "call")], S, R)
    with pytest.raises(ValueError):
        implied_vol_surface([(100, 1.0, 5.0, "call")], -1.0, R)
    with pytest.raises(ValueError):
        synthetic_quotes(S, R, [100], [0.0])
    with pytest.raises(ValueError):
        synthetic_quotes(S, R, [-5], [1.0])
    with pytest.raises(ValueError):
        VolSurface([100, 90], [1.0], [[0.2, 0.2]])
    with pytest.raises(ValueError):
        VolSurface([90, 100], [1.0], [[0.2]])
    s = implied_vol_surface(synthetic_quotes(S, R, [90, 100], [1.0]), S, R)
    with pytest.raises(ValueError):
        s.iv(0.0, 1.0)
    with pytest.raises(ValueError):
        s.iv(100.0, 0.0)
    empty = VolSurface([100.0], [1.0], [[np.nan]])
    with pytest.raises(ValueError):
        empty.iv(100.0, 1.0)


def test_repr(surf):
    assert "5 expiries x 7 strikes" in repr(surf)
