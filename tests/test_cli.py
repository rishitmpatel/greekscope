import re
import runpy
import subprocess
import sys

import pytest

from greekscope.bsm import price as bsm
from greekscope.cli import main

BASE = ["--S", "100", "--T", "1", "--r", "0.05"]


def run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def value(out, label):
    m = re.search(rf"^{re.escape(label)}\s+(\S+)", out, re.M)
    assert m, f"{label!r} not found in:\n{out}"
    return float(m.group(1))


def test_price_bsm_with_greeks(capsys):
    code, out, err = run(capsys, ["price", *BASE, "--K", "100", "--sigma", "0.2"])
    assert code == 0 and err == ""
    assert value(out, "Price") == pytest.approx(10.450584, abs=1e-6)
    for g in ("Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga"):
        assert g in out
    assert value(out, "Delta") == pytest.approx(0.636831, abs=1e-6)


def test_price_put_default_method_case_insensitive(capsys):
    code, out, _ = run(capsys, ["price", *BASE, "--K", "100", "--sigma", "0.2", "--kind", "PUT"])
    assert code == 0
    assert value(out, "Price") == pytest.approx(5.573526, abs=1e-6)


def test_price_crr(capsys):
    code, out, _ = run(capsys, ["price", *BASE, "--K", "100", "--sigma", "0.2", "--method", "crr", "--steps", "400"])
    assert code == 0
    assert value(out, "Price") == pytest.approx(10.4506, abs=1e-2)
    assert "BSM" in out


def test_price_american_put_has_early_exercise_premium(capsys):
    code, out, _ = run(
        capsys, ["price", *BASE, "--K", "100", "--sigma", "0.2", "--kind", "put", "--method", "american"]
    )
    assert code == 0
    assert value(out, "Price") > bsm(100, 100, 1, 0.05, 0.2, "put") + 0.3


def test_price_mc_reports_std_error(capsys):
    code, out, _ = run(
        capsys,
        ["price", *BASE, "--K", "100", "--sigma", "0.2", "--method", "mc", "--paths", "50000", "--seed", "7"],
    )
    assert code == 0
    m = re.search(r"Price\s+(\S+) \+/- (\S+)", out)
    assert m
    px, se = float(m.group(1)), float(m.group(2))
    assert se > 0 and abs(px - 10.450584) < 4 * se


def test_price_mc_seed_reproducible(capsys):
    argv = ["price", *BASE, "--K", "100", "--sigma", "0.2", "--method", "mc", "--paths", "20000", "--seed", "3"]
    a = run(capsys, argv)[1]
    b = run(capsys, argv)[1]
    assert a == b


def test_price_heston(capsys):
    argv = ["price", "--S", "100", "--K", "100", "--T", "0.5", "--r", "0.03", "--method", "heston",
            "--v0", "0.04", "--kappa", "2", "--theta", "0.04", "--xi", "0.3", "--rho", "-0.7"]
    code, out, _ = run(capsys, argv)
    assert code == 0
    assert 3.0 < value(out, "Price") < 8.0


def test_price_heston_missing_params(capsys):
    code, out, err = run(capsys, ["price", *BASE, "--K", "100", "--method", "heston", "--v0", "0.04"])
    assert code == 2 and out == ""
    assert "--kappa" in err and "--xi" in err


def test_price_missing_sigma(capsys):
    code, _, err = run(capsys, ["price", *BASE, "--K", "100"])
    assert code == 2 and "--sigma" in err


def test_price_bad_input_exit_2(capsys):
    code, out, err = run(capsys, ["price", "--S", "-5", "--K", "100", "--T", "1", "--r", "0.05", "--sigma", "0.2"])
    assert code == 2 and out == "" and err.startswith("greekscope: error:")
    code, _, err = run(capsys, ["price", *BASE, "--K", "100", "--sigma", "0.2", "--kind", "straddle"])
    assert code == 2 and "kind" in err
    code, _, err = run(capsys, ["price", *BASE, "--K", "100", "--sigma", "nan"])
    assert code == 2


def test_argparse_errors_exit_2(capsys):
    with pytest.raises(SystemExit) as e:
        main(["price", "--S", "100"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["price", *BASE, "--K", "100", "--sigma", "0.2", "--method", "bogus"])
    assert e.value.code == 2
    capsys.readouterr()


def test_iv_roundtrip(capsys):
    p = bsm(100, 105, 0.75, 0.04, 0.3, "put", 0.01)
    code, out, _ = run(
        capsys,
        ["iv", "--price", repr(p), "--S", "100", "--K", "105", "--T", "0.75", "--r", "0.04", "--q", "0.01", "--kind", "put"],
    )
    assert code == 0
    m = re.search(r"Implied vol\s+(\S+)", out)
    assert float(m.group(1)) == pytest.approx(0.3, abs=1e-6)


def test_iv_arbitrage_price_exit_2(capsys):
    code, _, err = run(capsys, ["iv", "--price", "1000", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05"])
    assert code == 2 and "no-arbitrage" in err


STRAT = ["--S", "100", "--T", "0.5", "--r", "0.05", "--sigma", "0.2"]


@pytest.mark.parametrize(
    "name,extra",
    [
        ("long_call", ["--K", "100"]),
        ("long_put", ["--K", "100"]),
        ("covered_call", ["--K", "105"]),
        ("protective_put", ["--K", "95"]),
        ("bull_call_spread", ["--K1", "95", "--K2", "110"]),
        ("bear_put_spread", ["--K1", "90", "--K2", "105"]),
        ("straddle", ["--K", "100"]),
        ("straddle", []),
        ("strangle", ["--K1", "90", "--K2", "110"]),
        ("strangle", []),
        ("iron_condor", ["--K1", "80", "--K2", "90", "--K3", "110", "--K4", "120"]),
        ("butterfly", ["--K1", "90", "--K2", "100", "--K3", "110"]),
        ("butterfly", ["--K1", "90", "--K2", "100", "--K3", "110", "--kind", "put"]),
        ("calendar_spread", ["--K", "100", "--T-far", "1.0"]),
        ("bull-call-spread", ["--K1", "95", "--K2", "110"]),
    ],
)
def test_strategy_subcommands(capsys, name, extra):
    code, out, err = run(capsys, ["strategy", name, *STRAT, *extra])
    assert code == 0 and err == ""
    for needle in ("Breakevens", "Max profit", "Max loss", "delta", "vega"):
        assert needle in out


def test_strategy_straddle_values(capsys):
    _, out, _ = run(capsys, ["strategy", "straddle", *STRAT, "--K", "100"])
    prem = bsm(100, 100, 0.5, 0.05, 0.2, "call") + bsm(100, 100, 0.5, 0.05, 0.2, "put")
    assert f"{100 - prem:,.4f}" in out and f"{100 + prem:,.4f}" in out
    assert "unlimited" in out


def test_strategy_multiplier(capsys):
    _, one, _ = run(capsys, ["strategy", "bull_call_spread", *STRAT, "--K1", "95", "--K2", "110"])
    _, hundred, _ = run(
        capsys, ["strategy", "bull_call_spread", *STRAT, "--K1", "95", "--K2", "110", "--multiplier", "100"]
    )
    a = float(re.search(r"Max profit\s+(\S+)", one).group(1).replace(",", ""))
    b = float(re.search(r"Max profit\s+(\S+)", hundred).group(1).replace(",", ""))
    assert b == pytest.approx(100 * a, rel=1e-4)


def test_strategy_errors(capsys):
    code, _, err = run(capsys, ["strategy", "iron_condor", *STRAT, "--K1", "80"])
    assert code == 2 and "--K2" in err
    code, _, err = run(capsys, ["strategy", "bull_call_spread", *STRAT, "--K1", "110", "--K2", "100"])
    assert code == 2 and "K_low < K_high" in err
    code, _, err = run(capsys, ["strategy", "long_call", *STRAT])
    assert code == 2 and "--K" in err
    code, _, err = run(capsys, ["strategy", "wheel", *STRAT, "--K", "100"])
    assert code == 2 and "unknown strategy" in err
    code, _, err = run(capsys, ["strategy", "calendar_spread", *STRAT, "--K", "100"])
    assert code == 2 and "--T-far" in err
    code, _, err = run(capsys, ["strategy", "straddle", "--S", "100", "--T", "0.5", "--r", "0.05", "--sigma", "-1"])
    assert code == 2


def test_python_dash_m():
    res = subprocess.run(
        [sys.executable, "-m", "greekscope", "iv", "--price", "10.45", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Implied vol" in res.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "greekscope", "price", "--S", "-1", "--K", "1", "--T", "1", "--r", "0", "--sigma", "0.2"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2 and "error" in bad.stderr


def test_main_module_runs_in_process(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["greekscope", "iv", "--price", "10.45", "--S", "100", "--K", "100", "--T", "1", "--r", "0.05"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("greekscope", run_name="__main__")
    assert e.value.code == 0
    assert "Implied vol" in capsys.readouterr().out
