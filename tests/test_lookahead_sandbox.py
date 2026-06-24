import numpy as np
import pandas as pd

from penrose.brain import Claim
from penrose.data.contract import DataBundle, Series


def _bundle():
    idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    values = np.linspace(1.0, 4.0, len(idx))
    return DataBundle(series={
        "price": Series("price", pd.Series(values, index=idx), "unit-test", "level")
    })


def _sandbox_result(bundle):
    s = bundle.get("price").data
    net = pd.Series((s - s.mean()).to_numpy(), index=s.index)
    positions = pd.Series(np.ones(len(net)), index=net.index)
    return {"ok": True, "net": net, "positions": positions,
            "bars_per_year": 252.0, "n_trades": len(net)}


def test_sandbox_runtime_lookahead_rejects_future_dependent_module(tmp_path, monkeypatch):
    from penrose.pipeline import impl_gen
    from penrose.pipeline import sandbox

    module_id = "sandbox-lookahead"
    impl = tmp_path / "impl.py"
    impl.write_text(
        "import pandas as pd\n"
        "__module_id__ = 'sandbox-lookahead'\n"
        "__strategy_class__ = 'unit-test'\n"
        "def run(bundle, claim, cost_frac):\n"
        "    s = bundle.get('price').data\n"
        "    net = pd.Series((s - s.mean()).to_numpy(), index=s.index)\n"
        "    pos = pd.Series(1.0, index=net.index)\n"
        "    return {'ok': True, 'net': net, 'positions': pos, 'bars_per_year': 252.0, 'n_trades': len(net)}\n"
    )
    bundle = _bundle()
    claim = Claim("lookahead", "future-dependent mean", "", "", "", "unit", "span", "")
    monkeypatch.setattr(sandbox, "run_in_container",
                        lambda module_path, b, c, cost_frac: _sandbox_result(b))

    ok, reason = impl_gen._validate_module(
        impl, module_id, bundle, claim, 0.0, prerun_result=_sandbox_result(bundle))

    assert ok is False
    assert "look-ahead" in reason
    assert "overlapping early dates changed" in reason


def test_static_scan_allows_ambiguous_alignment_idioms():
    from penrose.pipeline import impl_gen

    benign = """
import pandas as pd
__module_id__ = 'benign-align'
__strategy_class__ = 'unit-test'
def run(bundle, claim, cost_frac):
    s = bundle.get('price').data
    first_diff = s.iloc[1:] - s[:-1].to_numpy()
    train, test = s[:-20], s[-20:]
    rev = s.iloc[::-1].cumsum().iloc[::-1]
    return {'ok': False, 'reason': 'data_unavailable: unit test'}
"""

    assert impl_gen._scan_code_text(benign) is None


def test_static_scan_still_rejects_unambiguous_future_leaks():
    from penrose.pipeline import impl_gen

    for code in [
        "x = s.shift(-1)",
        "x = s.shift(periods=-2)",
        "x = np.roll(s, -1)",
        "x = s.iloc[i + 1]",
    ]:
        reason = impl_gen._scan_code_text(code)
        assert reason is not None
        assert "look-ahead" in reason


def test_sandbox_lookahead_skip_is_visible(tmp_path, monkeypatch, capsys):
    from penrose.pipeline import impl_gen
    from penrose.pipeline import sandbox

    impl = tmp_path / "impl.py"
    impl.write_text(
        "import pandas as pd\n"
        "__module_id__ = 'sandbox-skip'\n"
        "__strategy_class__ = 'unit-test'\n"
        "def run(bundle, claim, cost_frac):\n"
        "    return {'ok': False, 'reason': 'data_unavailable: unit test'}\n"
    )
    bundle = _bundle()
    claim = Claim("skip", "sandbox skip", "", "", "", "unit", "span", "")
    monkeypatch.setattr(sandbox, "run_in_container",
                        lambda module_path, b, c, cost_frac: {"ok": False, "reason": "docker timeout"})
    meta = {}

    reason = impl_gen._sandbox_lookahead_check(
        impl, bundle, claim, 0.0, _sandbox_result(bundle)["net"], meta)

    assert reason is None
    assert meta["dynamic_lookahead_check"] == "skipped"
    assert "docker timeout" in meta["dynamic_lookahead_skip_reason"]
    assert "dynamic sandbox look-ahead check did not run" in capsys.readouterr().err
