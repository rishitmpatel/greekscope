import math

import numpy as np
import pytest

import greekscope
from greekscope import greeks as G
from greekscope.bsm import price as bsm
from greekscope.strategy import (
    Leg,
    Strategy,
    bear_put_spread,
    bull_call_spread,
    butterfly,
    calendar_spread,
    covered_call,
    iron_condor,
    long_call,
    long_put,
    protective_put,
    straddle,
    strangle,
)

S, T, R, SIG = 100.0, 0.5, 0.05, 0.2


def test_package_exports():
    assert greekscope.straddle is straddle
    assert "iron_condor" in greekscope.__all__
    assert greekscope.Strategy is Strategy


def test_default_premium_is_bsm():
    st = Strategy([Leg("call", 105, 1), Leg("put", 95, -2)], S, T, R, SIG)
    assert st.legs[0].premium == pytest.approx(bsm(S, 105, T, R, SIG, "call"))
    assert st.legs[1].premium == pytest.approx(bsm(S, 95, T, R, SIG, "put"))
    assert st.net_premium == pytest.approx(st.legs[0].premium - 2 * st.legs[1].premium)


def test_user_legs_not_mutated():
    leg = Leg("call", 100, 1)
    Strategy([leg], S, T, R, SIG)
    assert leg.premium is None


def test_explicit_premium_respected():
    st = Strategy([Leg("call", 100, 1, premium=3.0)], S, T, R, SIG)
    assert st.payoff(100.0) == pytest.approx(-3.0)
    assert st.payoff(110.0) == pytest.approx(7.0)


def test_straddle_breakevens():
    st = straddle(S, T, R, SIG)
    prem = st.net_premium
    be = st.breakevens()
    assert len(be) == 2
    assert be[0] == pytest.approx(100 - prem, abs=1e-8)
    assert be[1] == pytest.approx(100 + prem, abs=1e-8)
    assert be == sorted(be)


def test_straddle_off_atm_breakevens():
    st = straddle(S, T, R, SIG, K=110)
    prem = st.net_premium
    assert st.breakevens() == pytest.approx([110 - prem, 110 + prem], abs=1e-8)


def test_straddle_extremes():
    st = straddle(S, T, R, SIG)
    assert st.max_profit() == math.inf
    assert st.max_loss() == pytest.approx(-st.net_premium)


def test_bull_call_spread_extremes():
    st = bull_call_spread(S, T, R, SIG, 95, 110)
    debit = st.net_premium
    assert debit > 0
    assert st.max_profit() == pytest.approx(15 - debit)
    assert st.max_loss() == pytest.approx(-debit)
    assert st.breakevens() == pytest.approx([95 + debit], abs=1e-8)
    # capped payoff
    assert st.payoff(500.0) == pytest.approx(15 - debit)
    assert st.payoff(0.0) == pytest.approx(-debit)


def test_bear_put_spread_extremes():
    st = bear_put_spread(S, T, R, SIG, 90, 105)
    debit = st.net_premium
    assert st.max_profit() == pytest.approx(15 - debit)
    assert st.max_loss() == pytest.approx(-debit)
    assert st.breakevens() == pytest.approx([105 - debit], abs=1e-8)


def test_iron_condor_bounded():
    st = iron_condor(S, T, R, SIG, 80, 90, 110, 120)
    credit = -st.net_premium
    assert credit > 0
    assert st.max_profit() == pytest.approx(credit)
    assert st.max_loss() == pytest.approx(-(10 - credit))
    assert np.isfinite(st.max_profit()) and np.isfinite(st.max_loss())
    assert st.breakevens() == pytest.approx([90 - credit, 110 + credit], abs=1e-8)
    # tails are equal to the max loss
    assert st.payoff(1.0) == pytest.approx(st.max_loss())
    assert st.payoff(1000.0) == pytest.approx(st.max_loss())


def test_butterfly_extremes():
    st = butterfly(S, T, R, SIG, 90, 100, 110)
    debit = st.net_premium
    assert st.max_profit() == pytest.approx(10 - debit)
    assert st.max_loss() == pytest.approx(-debit)
    be = st.breakevens()
    assert be == pytest.approx([90 + debit, 110 - debit], abs=1e-8)
    put_fly = butterfly(S, T, R, SIG, 90, 100, 110, kind="put")
    # call and put butterflies have the same cost by parity
    assert put_fly.net_premium == pytest.approx(debit, abs=1e-10)


def test_strangle():
    st = strangle(S, T, R, SIG, 90, 110)
    prem = st.net_premium
    assert st.breakevens() == pytest.approx([90 - prem, 110 + prem], abs=1e-8)
    d = strangle(S, T, R, SIG)
    assert d.legs[0].strike == pytest.approx(90) and d.legs[1].strike == pytest.approx(110)


def test_long_call_long_put():
    c = long_call(S, T, R, SIG, 100)
    assert c.max_profit() == math.inf
    assert c.max_loss() == pytest.approx(-c.net_premium)
    assert c.breakevens() == pytest.approx([100 + c.net_premium], abs=1e-8)
    p = long_put(S, T, R, SIG, 100)
    assert p.max_profit() == pytest.approx(100 - p.net_premium)
    assert p.max_loss() == pytest.approx(-p.net_premium)
    assert p.breakevens() == pytest.approx([100 - p.net_premium], abs=1e-8)


def test_covered_call():
    st = covered_call(S, T, R, SIG, 105)
    prem = -st.net_premium  # credit received
    assert st.max_loss() == pytest.approx(-(S - prem))
    assert st.max_profit() == pytest.approx((105 - S) + prem)
    assert st.breakevens() == pytest.approx([S - prem], abs=1e-8)
    assert np.isfinite(st.max_loss())


def test_protective_put():
    st = protective_put(S, T, R, SIG, 95)
    prem = st.net_premium
    assert st.max_profit() == math.inf
    assert st.max_loss() == pytest.approx(-(S - 95) - prem)
    assert st.breakevens() == pytest.approx([S + prem], abs=1e-8)


def test_short_call_unbounded_loss():
    st = Strategy([Leg("call", 100, -1)], S, T, R, SIG)
    assert st.max_loss() == -math.inf
    assert st.max_profit() == pytest.approx(st.legs[0].premium)


def test_short_stock_unbounded_loss():
    st = Strategy([Leg("stock", 100, -1)], S, T, R, SIG)
    assert st.max_loss() == -math.inf
    assert st.max_profit() == pytest.approx(100.0)


def test_greeks_equal_sum_of_legs():
    st = iron_condor(S, T, R, SIG, 80, 90, 110, 120, q=0.01)
    g = st.greeks()
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        fn = getattr(G, name)
        exp = sum(lg.qty * fn(S, lg.strike, T, R, SIG, lg.kind, 0.01) for lg in st.legs)
        assert g[name] == pytest.approx(exp, rel=1e-12, abs=1e-12)
    assert set(g) == {"delta", "gamma", "vega", "theta", "rho"}


def test_stock_leg_greeks():
    st = covered_call(S, T, R, SIG, 105)
    call = Leg("call", 105)
    g = st.greeks()
    assert g["delta"] == pytest.approx(1 - G.delta(S, 105, T, R, SIG, "call"))
    assert g["gamma"] == pytest.approx(-G.gamma(S, 105, T, R, SIG))
    only_stock = Strategy([Leg("stock", 90, 3)], S, T, R, SIG)
    gs = only_stock.greeks()
    assert gs["delta"] == 3 and gs["gamma"] == gs["vega"] == gs["theta"] == gs["rho"] == 0
    assert call.premium is None


def test_greeks_array_spot_and_multiplier():
    st = straddle(S, T, R, SIG)
    spots = np.array([90.0, 100.0, 110.0])
    g = st.greeks(S=spots)
    assert g["delta"].shape == (3,)
    assert g["delta"][1] == pytest.approx(st.greeks()["delta"])
    big = Strategy(st.legs, S, T, R, SIG, multiplier=100)
    assert big.greeks()["vega"] == pytest.approx(100 * st.greeks()["vega"])
    assert big.max_loss() == pytest.approx(100 * st.max_loss())
    assert big.breakevens() == pytest.approx(st.breakevens())


@pytest.mark.parametrize(
    "make",
    [
        lambda: straddle(S, T, R, SIG),
        lambda: iron_condor(S, T, R, SIG, 80, 90, 110, 120),
        lambda: covered_call(S, T, R, SIG, 105, q=0.02),
        lambda: calendar_spread(S, 0.25, R, SIG, 100, 0.5),
    ],
)
def test_value_at_expiry_equals_payoff(make):
    st = make()
    grid = np.linspace(0, 250, 51)
    np.testing.assert_allclose(st.value(grid, t_elapsed=st.T), st.payoff(grid), atol=1e-12)


def test_value_today_is_zero_at_spot():
    st = iron_condor(S, T, R, SIG, 80, 90, 110, 120)
    assert st.value(S, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_value_before_expiry_smoother_than_payoff():
    st = long_call(S, T, R, SIG, 100)
    assert st.value(100.0, 0.25) > st.payoff(100.0)  # time value remains
    arr = st.value(np.array([80.0, 100.0, 120.0]), 0.1)
    assert arr.shape == (3,)
    assert np.all(np.diff(arr) > 0)


def test_input_validation():
    with pytest.raises(ValueError):
        Strategy([], S, T, R, SIG)
    with pytest.raises(ValueError):
        Strategy([Leg("call", 100)], -1.0, T, R, SIG)
    with pytest.raises(ValueError):
        Leg("swap", 100)
    with pytest.raises(ValueError):
        Leg("call", -5)
    with pytest.raises(ValueError):
        Leg("stock", 100, premium=1.0)
    with pytest.raises(ValueError):
        Leg("call", 100, expiry=-1)
    with pytest.raises(ValueError):
        straddle(S, T, R, SIG).value(-5.0)
    with pytest.raises(ValueError):
        straddle(S, T, R, SIG).value(5.0, t_elapsed=-1)
    with pytest.raises(ValueError):
        Strategy([Leg("call", 100)], S, T, R, SIG, multiplier=0)


def test_strike_ordering_validation():
    with pytest.raises(ValueError):
        bull_call_spread(S, T, R, SIG, 110, 100)
    with pytest.raises(ValueError):
        bear_put_spread(S, T, R, SIG, 100, 100)
    with pytest.raises(ValueError):
        strangle(S, T, R, SIG, 110, 90)
    with pytest.raises(ValueError):
        butterfly(S, T, R, SIG, 90, 110, 100)
    with pytest.raises(ValueError):
        butterfly(S, T, R, SIG, 90, 100, 110, kind="straddle")
    with pytest.raises(ValueError):
        iron_condor(S, T, R, SIG, 90, 80, 110, 120)
    with pytest.raises(ValueError):
        iron_condor(S, T, R, SIG, 80, 90, 85, 120)
    iron_condor(S, T, R, SIG, 80, 100, 100, 120)  # iron butterfly allowed


def test_zero_vol_and_zero_time():
    st = straddle(S, 0.0, R, SIG)
    assert st.net_premium == pytest.approx(0.0)  # ATM at T=0 has no intrinsic
    st0 = bull_call_spread(S, T, R, 0.0, 95, 110)
    assert np.isfinite(st0.max_profit())
    assert np.all(np.isfinite(st0.value(np.linspace(0, 200, 11), 0.2)))


def test_calendar_spread_mixed_expiry():
    st = calendar_spread(S, 0.25, R, SIG, 100, 0.5)
    near, far = st.legs
    assert near.premium == pytest.approx(bsm(S, 100, 0.25, R, SIG, "call"))
    assert far.premium == pytest.approx(bsm(S, 100, 0.5, R, SIG, "call"))
    # payoff at T: short near intrinsic, long far valued with BSM at remaining 0.25
    for ST in (80.0, 100.0, 125.0):
        exp = (-(max(ST - 100, 0) - near.premium)) + (bsm(ST, 100, 0.25, R, SIG, "call") - far.premium)
        assert st.payoff(ST) == pytest.approx(exp)
    # peak near the strike, losing on both wings
    assert st.payoff(100.0) > st.payoff(70.0)
    assert st.payoff(100.0) > st.payoff(140.0)
    be = st.breakevens()
    assert len(be) == 2 and be[0] < 100 < be[1]
    assert st.max_profit() > 0 and np.isfinite(st.max_profit())
    assert st.max_loss() == pytest.approx(-st.net_premium, rel=1e-3)
    with pytest.raises(ValueError):
        calendar_spread(S, 0.5, R, SIG, 100, 0.25)
    g = st.greeks()
    assert g["vega"] > 0  # long calendar is long vega


def test_calendar_leg_expiry_before_horizon_settles_intrinsic():
    st = Strategy([Leg("call", 100, 1, expiry=0.1)], S, 0.5, R, SIG)
    assert st.payoff(120.0) == pytest.approx(20 - st.legs[0].premium)


def test_repr_and_summary():
    st = iron_condor(S, T, R, SIG, 80, 90, 110, 120)
    rep = repr(st)
    assert "Iron condor" in rep and "short" in rep and "110.0000" in rep
    text = st.summary()
    for needle in ("Breakevens", "Max profit", "Max loss", "delta", "credit"):
        assert needle in text
    assert str(st) == text
    assert "unlimited" in straddle(S, T, R, SIG).summary()
    assert "approximation" in calendar_spread(S, 0.25, R, SIG, 100, 0.5).summary()
    assert "stock" in covered_call(S, T, R, SIG, 105).summary()


def test_no_breakevens_when_always_profitable_or_loss():
    st = Strategy([Leg("call", 100, 1, premium=0.0)], S, T, R, SIG)
    # payoff >= 0 everywhere, touching zero over a flat stretch: no sign change
    assert st.breakevens() == []
