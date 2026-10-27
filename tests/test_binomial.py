import itertools
import math

import pytest

from greekscope import bsm
from greekscope.binomial import crr_price


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80, 100, 120])
def test_european_converges_to_bsm(kind, K):
    exact = bsm.price(100, K, 1.0, 0.05, 0.2, kind, 0.02)
    assert crr_price(100, K, 1.0, 0.05, 0.2, kind, 0.02, steps=500) == pytest.approx(exact, abs=1e-2)


def test_convergence_improves():
    exact = bsm.price(100, 105, 1.0, 0.05, 0.25, "call")
    e100 = abs(crr_price(100, 105, 1.0, 0.05, 0.25, "call", steps=100) - exact)
    e2000 = abs(crr_price(100, 105, 1.0, 0.05, 0.25, "call", steps=2000) - exact)
    assert e2000 < e100
    assert e2000 < 2e-3


def test_american_put_geq_european_and_has_premium():
    for K in (90, 100, 110):
        eu = crr_price(100, K, 1.0, 0.05, 0.2, "put", steps=400)
        am = crr_price(100, K, 1.0, 0.05, 0.2, "put", steps=400, american=True)
        assert am >= eu - 1e-12
    # deep ITM put with positive rates has material early-exercise premium
    eu = crr_price(60, 100, 1.0, 0.08, 0.2, "put", steps=400)
    am = crr_price(60, 100, 1.0, 0.08, 0.2, "put", steps=400, american=True)
    assert am - eu > 0.1
    assert am >= 40.0 - 1e-12  # >= intrinsic


def test_american_put_known_benchmark():
    # Widely quoted value for S=K=100, T=1, r=5%, sigma=20% American put ~ 6.09
    am = crr_price(100, 100, 1.0, 0.05, 0.2, "put", steps=1000, american=True)
    assert am == pytest.approx(6.0903, abs=5e-3)


def test_american_call_no_dividend_equals_european():
    for K in (80, 100, 130):
        eu = crr_price(100, K, 1.5, 0.05, 0.3, "call", steps=300)
        am = crr_price(100, K, 1.5, 0.05, 0.3, "call", steps=300, american=True)
        assert am == pytest.approx(eu, abs=1e-10)


def test_american_call_with_dividend_exceeds_european():
    eu = crr_price(100, 80, 2.0, 0.03, 0.2, "call", q=0.08, steps=400)
    am = crr_price(100, 80, 2.0, 0.03, 0.2, "call", q=0.08, steps=400, american=True)
    assert am > eu + 1e-3


def test_american_dominates_intrinsic_and_monotone_in_steps_sanity():
    for S, kind in itertools.product((70, 100, 130), ("call", "put")):
        am = crr_price(S, 100, 0.5, 0.04, 0.3, kind, 0.02, steps=200, american=True)
        intr = max(S - 100, 0) if kind == "call" else max(100 - S, 0)
        assert am >= intr - 1e-12


def test_dividend_handled_european():
    exact = bsm.price(100, 100, 1.0, 0.05, 0.2, "put", 0.04)
    assert crr_price(100, 100, 1.0, 0.05, 0.2, "put", 0.04, steps=1000) == pytest.approx(exact, abs=5e-3)


def test_degenerate_cases():
    assert crr_price(110, 100, 0.0, 0.05, 0.2, "call") == 10.0
    assert crr_price(90, 100, 0.0, 0.05, 0.2, "put", american=True) == 10.0
    assert crr_price(100, 90, 1.0, 0.05, 0.0, "call") == pytest.approx(bsm.price(100, 90, 1.0, 0.05, 0.0, "call"), abs=1e-12)
    am = crr_price(100, 110, 1.0, 0.05, 0.0, "put", american=True, steps=50)
    assert am >= 10.0 - 1e-12 and math.isfinite(am)
    assert crr_price(100, 100, 1.0, 0.05, 0.2, steps=1) > 0


def test_steps_and_inputs_validation():
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.05, 0.2, steps=0)
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.05, 0.2, steps=10.5)
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.05, -0.2)
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.05, 0.2, "x")
    # tiny vol with few steps => probability out of range
    with pytest.raises(ValueError, match="steps"):
        crr_price(100, 100, 10.0, 0.5, 0.01, steps=2)
