from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from penrose.brain import Claim
from penrose.pipeline import p7_backtest, robustness, run as runmod, stages


def _claim() -> Claim:
    return Claim(
        claim_id="param-fragility",
        statement="parameter fragility unit claim",
        mechanism="unit",
        scope="unit",
        horizon="unit",
        source_id="unit",
        source_span="unit",
        claimed_metric_quote="unit",
        source_type="generated_hypothesis",
    )


def _series(edge: float, n: int = 120) -> pd.Series:
    return pd.Series(np.full(n, edge, dtype=float), index=pd.date_range("2024-01-01", periods=n))


class StableModule:
    __supports_param_override__ = True

    def run(self, bundle, claim, cost_frac, param_override=None):
        edge = float((param_override or {}).get("edge", 0.001))
        return {
            "ok": True,
            "net": _series(edge),
            "positions": pd.Series(1.0, index=_series(edge).index),
            "bars_per_year": 252.0,
        }


class FragileModule:
    __supports_param_override__ = True

    def run(self, bundle, claim, cost_frac, param_override=None):
        magic = (param_override or {}).get("magic")
        edge = 0.01 if magic == "declared" else -0.01
        return {
            "ok": True,
            "net": _series(edge),
            "positions": pd.Series(1.0, index=_series(edge).index),
            "bars_per_year": 252.0,
        }


class UnmarkedModule(StableModule):
    __supports_param_override__ = False


def _survivor_bt(parameter_fragility=None) -> dict:
    bt = {
        "psr": 0.98,
        "dsr": 0.98,
        "edge_t": 3.0,
        "n_oos": 1200,
        "bars_per_year": 252.0,
        "oos_sharpe": 1.2,
        "is_sharpe": 1.0,
        "n_trials": 1,
        "three_fold": {"folds": [1, 1, 1], "consistent": True},
        "capacity_usd": 1e6,
        "bootstrap": {"edge_ci_includes_zero": False},
        "permutation": {"p_value": 0.01},
        "regime": {"fragile": False},
        "walk_forward": {},
        "capacity_ci": {},
        "cpcv": {},
    }
    if parameter_fragility is not None:
        bt["parameter_fragility"] = parameter_fragility
    return bt


def test_parameter_fragility_stable_module_not_fragile():
    spec = {"param_grid": {"edge": [0.001, 0.002, 0.003, 0.004]}}

    out = robustness.parameter_fragility(
        StableModule(), None, _claim(), 0.0, spec, p7_backtest.OOS_FRAC)

    assert out["ran"] is True
    assert out["n_configs"] == 4
    assert out["positive_frac"] == 1.0
    assert out["fragile"] is False


def test_parameter_fragility_fragile_module_is_fragile():
    spec = {"param_grid": {"magic": ["declared", "a", "b", "c"]}}

    out = robustness.parameter_fragility(
        FragileModule(), None, _claim(), 0.0, spec, p7_backtest.OOS_FRAC)

    assert out["ran"] is True
    assert out["n_configs"] == 4
    assert out["positive_frac"] == 0.25
    assert out["fragile"] is True


def test_parameter_fragility_does_not_fire_without_grid_marker_or_survivor(monkeypatch):
    claim = _claim()
    spec = {"param_grid": {"edge": [0.001, 0.002, 0.003, 0.004]}}

    assert robustness.parameter_fragility(
        StableModule(), None, claim, 0.0, {}, p7_backtest.OOS_FRAC)["ran"] is False
    assert robustness.parameter_fragility(
        UnmarkedModule(), None, claim, 0.0, spec, p7_backtest.OOS_FRAC)["ran"] is False

    called = {"n": 0}

    def fake_fragility(*args, **kwargs):
        called["n"] += 1
        return {"ran": True, "fragile": True, "positive_frac": 0.0, "n_configs": 4}

    monkeypatch.setattr(runmod.robustness, "parameter_fragility", fake_fragility)
    bt = {}
    attached = runmod._maybe_attach_parameter_fragility(
        bt, StableModule(), None, claim, 0.0, spec,
        SimpleNamespace(verdict="kill", kill_reason="no_oos_edge"))

    assert attached is False
    assert called["n"] == 0
    assert "parameter_fragility" not in bt


def test_p8_verdict_parameter_fragility_downgrades_survivor_only():
    fragile = {"ran": True, "fragile": True, "positive_frac": 0.25, "n_configs": 4}

    dec = stages.p8_verdict(_claim(), _survivor_bt(fragile), {}, False)

    assert dec.verdict == "kill"
    assert dec.kill_reason == "parameter_fragile"
    assert "positive_frac=0.25 across 4 configs" in dec.rationale
    assert dec.metrics["parameter_fragility"] == fragile

    dead = _survivor_bt(fragile)
    dead.update({"dsr": 0.1, "psr": 0.1, "edge_t": 0.1})
    prior = stages.p8_verdict(_claim(), dead, {}, False)

    assert prior.verdict == "kill"
    assert prior.kill_reason != "parameter_fragile"


def test_parameter_fragility_is_deterministic():
    spec = {"param_grid": {"edge": [0.001, -0.001, 0.002], "thr": [1, 2, 3, 4]}}

    a = robustness.parameter_fragility(
        StableModule(), None, _claim(), 0.0, spec, p7_backtest.OOS_FRAC)
    b = robustness.parameter_fragility(
        StableModule(), None, _claim(), 0.0, spec, p7_backtest.OOS_FRAC)

    assert a == b


class InertModule:
    """Q-1: claims override support but IGNORES it — every config is byte-identical."""
    __supports_param_override__ = True

    def run(self, bundle, claim, cost_frac, param_override=None):
        edge = 0.002  # constant regardless of param_override -> inert grid
        return {"ok": True, "net": _series(edge),
                "positions": pd.Series(1.0, index=_series(edge).index), "bars_per_year": 252.0}


def test_parameter_fragility_inert_grid_is_not_certified():
    """Q-1: a grid over a param the executor ignores yields identical edges and must NOT be certified
    parameter-stable — it lands in degenerate_grid (ran False), issuing no robustness pass."""
    spec = {"param_grid": {"unread_knob": [1, 2, 3, 4]}}
    out = robustness.parameter_fragility(
        InertModule(), None, _claim(), 0.0, spec, p7_backtest.OOS_FRAC)
    assert out["ran"] is False and out["reason"] == "degenerate_grid"
    assert "fragile" not in out  # no false robustness certification
