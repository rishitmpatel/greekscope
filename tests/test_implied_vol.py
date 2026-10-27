import itertools
import math

import numpy as np
import pytest

from greekscope import bsm
from greekscope.implied_vol import implied_vol


def test_known_value():
    assert implied_vol(10.450583572185565, 100, 100, 1, 0.05, "call") == pytest.approx(0.2, abs=1e-9)
    assert implied_vol(5.573526022256971, 100, 100, 1, 0.05, "put") == pytest.approx(0.2, abs=1e-9)


def test_roundtrip_grid():
    r, q = 0.03, 0.01
    sigmas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    strikes = [50, 75, 100, 125, 150, 200]
    expiries = [0.05, 0.25, 1.0, 2.5, 5.0]
    checked = 0
    for kind, K, T, sig in itertools.product(["call", "put"], strikes, expiries, sigmas):
        p = bsm.price(100, K, T, r, sig, kind, q)
        fs, pk = 100 * math.exp(-q * T), K * math.exp(-r * T)
        intrinsic = max(fs - pk, 0) if kind == "call" else max(pk - fs, 0)
        time_value = p - intrinsic
        # skip prices carrying no usable information (deep OTM / price ~ intrinsic)
        if p < 1e-12 or time_value < 1e-9:
            continue
        iv = implied_vol(p, 100, K, T, r, kind, q)
        checked += 1
        # compare via repricing (always well-posed) and via sigma where vega is sizeable
        assert bsm.price(100, K, T, r, iv, kind, q) == pytest.approx(p, abs=1e-9, rel=1e-9)
        vega = 100 * math.exp(-q * T) * math.sqrt(T) * math.exp(-0.5 * bsm.d1(100, K, T, r, sig, q) ** 2) / math.sqrt(2 * math.pi)
        if vega > 1e-3 and time_value > 1e-6:
            assert iv == pytest.approx(sig, abs=1e-7, rel=1e-7), (kind, K, T, sig)
    assert checked > 150


@pytest.mark.parametrize("kind", ["call", "put"])
def test_roundtrip_tight_for_reasonable_cases(kind):
    for K in (80, 100, 120):
        for T in (0.1, 1.0, 4.0):
            for sig in (0.05, 0.2, 0.8, 2.5):
                p = bsm.price(100, K, T, 0.02, sig, kind, 0.015)
                otm = min(bsm.price(100, K, T, 0.02, sig, "call", 0.015), bsm.price(100, K, T, 0.02, sig, "put", 0.015))
                if otm < 1e-9:  # no time value: price carries no usable information about sigma
                    continue
                assert implied_vol(p, 100, K, T, 0.02, kind, 0.015) == pytest.approx(sig, abs=1e-8, rel=1e-8)


def test_deep_itm_and_otm_use_fallback_cleanly():
    # very low vega region: Newton may bail out to brentq; result must still reprice
    p = bsm.price(100, 200, 0.05, 0.0, 0.3, "call")
    assert p > 0
    iv = implied_vol(p, 100, 200, 0.05, 0.0, "call")
    assert bsm.price(100, 200, 0.05, 0.0, iv, "call") == pytest.approx(p, rel=1e-8)
    p = bsm.price(100, 60, 5.0, 0.01, 3.0, "put")
    assert implied_vol(p, 100, 60, 5.0, 0.01, "put") == pytest.approx(3.0, abs=1e-7)


def test_price_at_intrinsic_returns_zero_vol():
    assert implied_vol(20.0, 120, 100, 1.0, 0.0, "call") == 0.0
    assert implied_vol(0.0, 80, 100, 1.0, 0.0, "call") == 0.0


@pytest.mark.parametrize("price,kind", [(-0.01, "call"), (4.0, "call"), (110.5, "call"), (100.5, "put")])
def test_arbitrage_violations_raise(price, kind):
    # S=110, K=100, r=0: call bounds [10, 110]; put bounds [0, 100]
    with pytest.raises(ValueError):
        implied_vol(price, 110, 100, 1.0, 0.0, kind)


def test_below_intrinsic_put_and_above_bounds():
    with pytest.raises(ValueError):
        implied_vol(5.0, 90, 100, 1.0, 0.0, "put")  # intrinsic 10
    with pytest.raises(ValueError):
        implied_vol(100.0, 100, 100, 1.0, 0.05, "put")  # > K e^{-rT} = 95.1


def test_vol_above_cap_raises():
    p = bsm.price(100, 100, 1.0, 0.0, 6.0, "call")
    with pytest.raises(ValueError):
        implied_vol(p, 100, 100, 1.0, 0.0, "call")


def test_invalid_inputs():
    with pytest.raises(ValueError):
        implied_vol(5, 100, 100, 0.0, 0.05)  # T == 0
    with pytest.raises(ValueError):
        implied_vol(5, -1, 100, 1.0, 0.05)
    with pytest.raises(ValueError):
        implied_vol(5, 100, 100, 1.0, 0.05, "foo")
    with pytest.raises(ValueError):
        implied_vol(float("nan"), 100, 100, 1.0, 0.05)


def test_returns_python_float():
    assert isinstance(implied_vol(10.45, 100, 100, 1.0, 0.05), float)


def test_max_iter_zero_falls_back_to_brent():
    p = bsm.price(100, 110, 1.0, 0.03, 0.37, "call")
    assert implied_vol(p, 100, 110, 1.0, 0.03, "call", max_iter=0) == pytest.approx(0.37, abs=1e-9)
